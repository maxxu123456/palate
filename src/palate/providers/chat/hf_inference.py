"""Hugging Face Inference Providers, which route each request to an upstream."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import SecretStr

from palate.errors import ProviderUnavailable
from palate.extras import require
from palate.providers.base import (
    ChatCapabilities,
    ChatChunk,
    Completion,
    JSONObject,
    Message,
    ProviderHealth,
    ResponseFormat,
    SpanLike,
    ToolChoice,
    ToolSchema,
)
from palate.providers.chat.openai_compat import (
    chunk_from_payload,
    completion_from_payload,
    request_body,
)
from palate.providers.retry import DEFAULT_RETRY, RetryPolicy, with_retry
from palate.providers.tokens import count_tokens


def as_payload(value: Any) -> JSONObject:
    """The hub returns dataclasses, and every conversion below needs plain json."""
    if isinstance(value, dict):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return dict(vars(value))


class HFInferenceChat:
    """A separate adapter because provider routing and billing are not base_url settings."""

    name = "hf_inference"

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | None = None,
        provider: str = "auto",
        client: Any | None = None,
        timeout_s: float = 90.0,
        retry: RetryPolicy = DEFAULT_RETRY,
    ) -> None:
        self.model = model
        self.provider = provider
        self.timeout_s = timeout_s
        self.retry = retry
        if client is None:
            hub = require("hf", "huggingface_hub")
            client = hub.AsyncInferenceClient(
                model=model,
                provider=provider,
                api_key=api_key.get_secret_value() if api_key else None,
                timeout=timeout_s,
            )
        self.client = client

    def _to_wire(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema], **kwargs: Any
    ) -> JSONObject:
        body = request_body(self.model, messages, tools, **kwargs)
        # The hub client takes the model separately and rejects it in the body.
        body.pop("model", None)
        body.pop("stream", None)
        body.pop("stream_options", None)
        return body

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
        response_format: ResponseFormat | None = None,
        seed: int | None = None,
        timeout_s: float | None = None,
        span: SpanLike | None = None,
    ) -> Completion:
        """One turn through the router."""
        body = self._to_wire(
            messages,
            tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            seed=seed,
        )
        attempts = 0

        async def once() -> Completion:
            nonlocal attempts
            attempts += 1
            started = time.perf_counter()
            output = await self.client.chat_completion(**body)
            elapsed = (time.perf_counter() - started) * 1000
            return completion_from_payload(
                as_payload(output), model=self.model, latency_ms=elapsed, attempt=attempts
            )

        return await with_retry(once, policy=self.retry, span=span)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
        response_format: ResponseFormat | None = None,
        seed: int | None = None,
        timeout_s: float | None = None,
        span: SpanLike | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """The hub yields the same chunk shape the OpenAI dialect uses."""
        body = self._to_wire(
            messages,
            tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            seed=seed,
        )
        async for event in await self.client.chat_completion(stream=True, **body):
            yield chunk_from_payload(as_payload(event))

    async def capabilities(self) -> ChatCapabilities:
        """Routed upstreams vary, so the conservative answer is the honest one."""
        return ChatCapabilities(
            tools=True,
            parallel_tool_calls=False,
            tool_streaming=True,
            json_schema=True,
            streaming=True,
            context_window=None,
            max_output_tokens=None,
            schema_style="hf",
        )

    async def health(self) -> ProviderHealth:
        """A one token request, because the router has no cheap status route."""
        started = time.perf_counter()
        try:
            await self.client.chat_completion(
                messages=[{"role": "user", "content": "ping"}], max_tokens=1
            )
        except Exception as exc:
            return ProviderHealth(False, f"hf router refused ({exc})")
        elapsed = (time.perf_counter() - started) * 1000
        return ProviderHealth(True, f"hf router served {self.model}", elapsed)

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSchema] = ()) -> int:
        """Estimate only."""
        return count_tokens(messages, tools)

    async def aclose(self) -> None:
        """Close the hub client, which owns its own session."""
        closer = getattr(self.client, "close", None)
        if closer is None:
            return
        try:
            await closer()
        except TypeError as exc:
            raise ProviderUnavailable(
                "hf client close is not awaitable", provider=self.name, model=self.model
            ) from exc
