"""Turning config into a live provider, which is the only place a provider is chosen."""

from __future__ import annotations

import httpx

from palate.config import Settings, require_secret, resolve_secret
from palate.db.connect import Database
from palate.errors import ConfigError
from palate.hf.models import pin
from palate.providers.base import ChatProvider, EmbeddingProvider, Reranker
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.providers.chat.hf_inference import HFInferenceChat
from palate.providers.chat.ollama import BASE_URL as OLLAMA_URL
from palate.providers.chat.ollama import OllamaChat
from palate.providers.chat.openai_compat import OpenAICompatChat
from palate.providers.chat.openrouter import OpenRouterChat
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.embed.hf_inference import HFInferenceEmbedder
from palate.providers.embed.ollama import BASE_URL as OLLAMA_EMBED_URL
from palate.providers.embed.ollama import OllamaEmbedder
from palate.providers.embed.openai_compat import OpenAICompatEmbedder
from palate.providers.rerank.cache import ScoreCache
from palate.providers.rerank.identity import IdentityReranker
from palate.providers.rerank.llm import LLMReranker

# What `palate doctor` gets when the config says fake, so the wiring is still exercised.
FAKE_REPLY = "the fake provider is selected, so no model was called"


def build_chat(settings: Settings, *, client: httpx.AsyncClient) -> ChatProvider:
    """Build the configured chat provider. The client is shared and closed by the caller."""
    chat = settings.chat
    if chat.provider == "ollama":
        return OllamaChat(
            model=chat.model,
            client=client,
            base_url=chat.base_url or OLLAMA_URL,
            keep_alive=chat.keep_alive,
            num_ctx=chat.num_ctx,
            timeout_s=chat.timeout_s,
            parse_content_tool_calls=chat.content_toolcall_parse,
        )
    if chat.provider == "openrouter":
        return OpenRouterChat(
            model=chat.model,
            api_key=require_secret(chat.api_key_env),
            client=client,
            order=chat.openrouter_order,
            allow_fallbacks=chat.openrouter_allow_fallbacks,
            timeout_s=chat.timeout_s,
        )
    if chat.provider == "openai_compat":
        if not chat.base_url:
            raise ConfigError("chat.base_url is required when chat.provider is openai_compat")
        return OpenAICompatChat(
            base_url=chat.base_url,
            model=chat.model,
            # LM Studio and vLLM serve without a key, so a missing one is not an error.
            api_key=resolve_secret(chat.api_key_env),
            client=client,
            timeout_s=chat.timeout_s,
        )
    if chat.provider == "hf_inference":
        return HFInferenceChat(
            model=chat.model,
            api_key=resolve_secret(chat.api_key_env) or resolve_secret("HF_TOKEN"),
            provider=chat.hf_provider,
            timeout_s=chat.timeout_s,
        )
    return FakeChatProvider([ScriptedTurn(text=FAKE_REPLY)], loop_last=True, model=chat.model)


def build_embedder(settings: Settings, *, client: httpx.AsyncClient) -> EmbeddingProvider:
    """Build the configured embedding provider, which is never the chat provider."""
    embed = settings.embed
    if embed.provider == "ollama":
        return OllamaEmbedder(
            model=embed.model,
            client=client,
            base_url=embed.base_url or OLLAMA_EMBED_URL,
            max_batch=embed.batch_size,
        )
    if embed.provider == "sentence_transformers":
        # Imported here so the base install never touches torch by loading this module.
        from palate.providers.embed.sentence_transformers import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(
            pin=pin(embed.model),
            device=embed.device,
            batch_size=embed.batch_size,
            truncate_dim=embed.truncate_dim,
        )
    if embed.provider == "openai_compat":
        if not embed.base_url:
            raise ConfigError("embed.base_url is required when embed.provider is openai_compat")
        return OpenAICompatEmbedder(
            base_url=embed.base_url,
            model=embed.model,
            client=client,
            api_key=resolve_secret(embed.api_key_env),
            max_batch=embed.batch_size,
            truncate_dim=embed.truncate_dim,
        )
    if embed.provider == "hf_inference":
        return HFInferenceEmbedder(
            model=embed.model,
            api_key=resolve_secret(embed.api_key_env) or resolve_secret("HF_TOKEN"),
            max_batch=embed.batch_size,
        )
    return FakeEmbedder(max_batch=embed.batch_size)


def build_reranker(
    settings: Settings,
    *,
    db: Database | None = None,
    chat: ChatProvider | None = None,
    provider: str | None = None,
    alias: str | None = None,
    cache: bool = True,
) -> Reranker:
    """Build one reranker. provider and alias override the config so an eval arm can name its own."""
    rerank = settings.rerank
    kind = provider or rerank.provider
    store = ScoreCache(db, enabled=cache) if db is not None else None
    if kind == "cross_encoder":
        # Imported here so the base install never touches torch by loading this module.
        from palate.providers.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(pin=pin(alias or rerank.model), cache=store)
    if kind == "llm":
        if chat is None:
            raise ConfigError("rerank.provider is llm, which needs a chat provider")
        return LLMReranker(
            chat=chat,
            model=chat.model,
            window=rerank.llm_window,
            stride=rerank.llm_stride,
            cache=store,
        )
    return IdentityReranker()
