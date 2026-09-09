"""OpenRouter: the same wire format, plus attribution, real cost and provider routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from palate.providers.base import JSONObject, Message, ToolSchema
from palate.providers.chat.openai_compat import OpenAICompatChat

BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter ranks apps by these two headers, and they cost nothing to send.
ATTRIBUTION = {
    "HTTP-Referer": "https://github.com/maxxu123456/palate",
    "X-Title": "palate",
}


class OpenRouterChat(OpenAICompatChat):
    """Hundreds of models behind one key, with the billed cost in the response."""

    name = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr,
        client: httpx.AsyncClient,
        base_url: str = BASE_URL,
        order: Sequence[str] = (),
        allow_fallbacks: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            client=client,
            default_headers=ATTRIBUTION,
            **kwargs,
        )
        self.order = tuple(order)
        self.allow_fallbacks = allow_fallbacks

    def _to_wire(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema], **kwargs: Any
    ) -> JSONObject:
        body = dict(super()._to_wire(messages, tools, **kwargs))
        # Without this the response carries no cost and every number comes from a table.
        body["usage"] = {"include": True}
        if self.order:
            body["provider"] = {
                "order": list(self.order),
                "allow_fallbacks": self.allow_fallbacks,
            }
        return body
