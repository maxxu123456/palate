"""Turning config into a live provider, which is the only place a provider is chosen."""

from __future__ import annotations

import httpx

from palate.config import Settings, require_secret, resolve_secret
from palate.errors import ConfigError
from palate.providers.base import ChatProvider
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.providers.chat.hf_inference import HFInferenceChat
from palate.providers.chat.ollama import BASE_URL as OLLAMA_URL
from palate.providers.chat.ollama import OllamaChat
from palate.providers.chat.openai_compat import OpenAICompatChat
from palate.providers.chat.openrouter import OpenRouterChat

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
