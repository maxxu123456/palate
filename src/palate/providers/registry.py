"""Turning config into a live provider, which is the only place a provider is chosen."""

from __future__ import annotations

from palate.config import Settings
from palate.db.connect import Database
from palate.errors import ConfigError
from palate.hf.models import pin
from palate.providers.base import ChatProvider, EmbeddingProvider, Reranker
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.rerank.cache import ScoreCache
from palate.providers.rerank.identity import IdentityReranker
from palate.providers.rerank.llm import LLMReranker

# What `palate doctor` gets when the config says fake, so the wiring is still exercised.
FAKE_REPLY = "the fake provider is selected, so no model was called"


def build_chat(settings: Settings) -> ChatProvider:
    """Build the configured chat provider."""
    chat = settings.chat
    if chat.provider == "transformers":
        # Imported here so `palate --help` does not wait for transformers to load.
        from palate.providers.chat.transformers_local import TransformersLocalChat

        return TransformersLocalChat(
            alias=chat.model,
            device=chat.device,
            context_window=chat.num_ctx,
            max_new_tokens=chat.max_tokens,
            timeout_s=chat.timeout_s,
            parse_content_tool_calls=chat.content_toolcall_parse,
        )
    return FakeChatProvider([ScriptedTurn(text=FAKE_REPLY)], loop_last=True, model=chat.model)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider, which is never the chat provider."""
    embed = settings.embed
    if embed.provider == "sentence_transformers":
        # Imported here so `palate --help` does not wait for torch to load.
        from palate.providers.embed.sentence_transformers import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(
            pin=pin(embed.model),
            device=embed.device,
            batch_size=embed.batch_size,
            truncate_dim=embed.truncate_dim,
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
        # Imported here so `palate --help` does not wait for torch to load.
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
