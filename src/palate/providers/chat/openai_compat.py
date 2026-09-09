"""Any host that speaks POST {base_url}/chat/completions."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx
import orjson
from pydantic import SecretStr

from palate.errors import ProviderTimeout, ProviderUnavailable
from palate.providers.base import (
    ChatCapabilities,
    ChatChunk,
    Completion,
    FinishReason,
    JSONObject,
    Message,
    ProviderHealth,
    ResponseFormat,
    SchemaStyle,
    SpanLike,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSchema,
    Usage,
    json_int,
    json_object,
)
from palate.providers.retry import DEFAULT_RETRY, RetryPolicy, classify, with_retry
from palate.providers.streamacc import parse_arguments
from palate.providers.tokens import count_tokens

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


def finish_reason(raw: str | None) -> FinishReason:
    """Map a provider finish reason onto ours, defaulting to stop."""
    return _FINISH.get(raw or "", "stop")


class OpenAICompatChat:
    """LM Studio, vLLM, Together, Groq, OpenAI and anything else with the same route."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None = None,
        client: httpx.AsyncClient,
        default_headers: Mapping[str, str] | None = None,
        timeout_s: float = 90.0,
        retry: RetryPolicy = DEFAULT_RETRY,
        schema_style: SchemaStyle = "openai",
        supports_tools: bool = True,
        supports_parallel_tool_calls: bool = True,
        context_window: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = client
        self.default_headers = dict(default_headers or {})
        self.timeout_s = timeout_s
        self.retry = retry
        self.schema_style = schema_style
        self.supports_tools = supports_tools
        self.supports_parallel_tool_calls = supports_parallel_tool_calls
        self.context_window = context_window

    def headers(self) -> dict[str, str]:
        """Auth plus whatever the subclass adds. Never a key in a query string."""
        headers = dict(self.default_headers)
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def _to_wire(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema], **kwargs: Any
    ) -> JSONObject:
        return request_body(
            self.model,
            messages,
            tools,
            schema_style=self.schema_style,
            supports_tools=self.supports_tools,
            supports_parallel_tool_calls=self.supports_parallel_tool_calls,
            **kwargs,
        )

    def _from_wire(self, payload: JSONObject, **kwargs: Any) -> Completion:
        return completion_from_payload(payload, model=self.model, **kwargs)

    def _chunk_from_wire(self, event: JSONObject) -> ChatChunk:
        return chunk_from_payload(event)

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
        """One non streaming turn."""
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
            payload = await self._post("/chat/completions", body, timeout_s=timeout_s)
            elapsed = (time.perf_counter() - started) * 1000
            return self._from_wire(payload, latency_ms=elapsed, attempt=attempts)

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
        """Server sent events, one ChatChunk per data line. Not retried once bytes arrive."""
        body = self._to_wire(
            messages,
            tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            seed=seed,
            stream=True,
        )
        request = self.client.build_request(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self.headers(),
            timeout=timeout_s or self.timeout_s,
        )
        response = await self._send(request, stream=True)
        try:
            async for line in response.aiter_lines():
                event = _sse_event(line)
                if event is None:
                    continue
                yield self._chunk_from_wire(event)
        finally:
            await response.aclose()

    async def capabilities(self) -> ChatCapabilities:
        """What this endpoint supports, from config rather than a guess at the model name."""
        return ChatCapabilities(
            tools=self.supports_tools,
            parallel_tool_calls=self.supports_parallel_tool_calls,
            tool_streaming=True,
            json_schema=True,
            streaming=True,
            context_window=self.context_window,
            max_output_tokens=None,
            schema_style=self.schema_style,
        )

    async def health(self) -> ProviderHealth:
        """Ask the endpoint what it serves."""
        started = time.perf_counter()
        try:
            response = await self.client.get(
                f"{self.base_url}/models", headers=self.headers(), timeout=10.0
            )
        except httpx.HTTPError as exc:
            return ProviderHealth(False, f"{self.base_url} unreachable ({exc})")
        elapsed = (time.perf_counter() - started) * 1000
        error = classify(response, provider=self.name, model=self.model)
        if error is not None:
            return ProviderHealth(False, str(error), elapsed)
        return ProviderHealth(True, f"{self.base_url} answered", elapsed)

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSchema] = ()) -> int:
        """Estimate, corrected from real usage after the call."""
        return count_tokens(messages, tools)

    async def aclose(self) -> None:
        """The client is shared, so closing it is the caller's job."""
        return None

    async def _post(
        self, path: str, body: JSONObject, *, timeout_s: float | None = None
    ) -> JSONObject:
        request = self.client.build_request(
            "POST",
            f"{self.base_url}{path}",
            json=body,
            headers=self.headers(),
            timeout=timeout_s or self.timeout_s,
        )
        response = await self._send(request)
        parsed: JSONObject = orjson.loads(response.content)
        return parsed

    async def _send(self, request: httpx.Request, *, stream: bool = False) -> httpx.Response:
        try:
            response = await self.client.send(request, stream=stream)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{self.name} timed out", provider=self.name, model=self.model, retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"{self.name} unreachable ({exc})",
                provider=self.name,
                model=self.model,
                retryable=True,
            ) from exc
        if response.status_code >= 400:
            if stream:
                await response.aread()
            error = classify(response, provider=self.name, model=self.model)
            await response.aclose()
            if error is not None:
                raise error
        return response


def request_body(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSchema],
    *,
    tool_choice: ToolChoice = "auto",
    temperature: float = 0.2,
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
    response_format: ResponseFormat | None = None,
    seed: int | None = None,
    stream: bool = False,
    schema_style: SchemaStyle = "openai",
    supports_tools: bool = True,
    supports_parallel_tool_calls: bool = True,
) -> JSONObject:
    """The exact bytes we send. Asserted against cassettes, because this is what breaks."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_wire(m) for m in messages],
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stop:
        body["stop"] = list(stop)
    if seed is not None:
        body["seed"] = seed
    if tools and supports_tools:
        body["tools"] = [_tool_to_wire(t, schema_style) for t in tools]
        body["tool_choice"] = _tool_choice_to_wire(tool_choice)
        if supports_parallel_tool_calls:
            body["parallel_tool_calls"] = True
    if response_format is not None and response_format.kind != "text":
        body["response_format"] = _response_format_to_wire(response_format)
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def completion_from_payload(
    payload: JSONObject,
    *,
    model: str,
    latency_ms: float = 0.0,
    ttft_ms: float | None = None,
    attempt: int = 1,
) -> Completion:
    """One /chat/completions body into a Completion."""
    choices = payload.get("choices")
    choice = json_object(choices[0]) if isinstance(choices, list) and choices else {}
    message = json_object(choice.get("message"))
    usage_block = json_object(payload.get("usage"))
    return Completion(
        content=message.get("content") or "",
        tool_calls=_tool_calls_from_wire(message.get("tool_calls")),
        finish_reason=finish_reason(choice.get("finish_reason")),
        usage=_usage_from_wire(usage_block),
        model=model,
        response_model=str(payload.get("model") or model),
        cost_usd=_cost_from_wire(usage_block),
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        attempt=attempt,
        raw=payload,
    )


def chunk_from_payload(event: JSONObject) -> ChatChunk:
    """One streamed event into a ChatChunk."""
    choices = event.get("choices")
    choice = json_object(choices[0]) if isinstance(choices, list) and choices else {}
    delta = json_object(choice.get("delta"))
    usage_block = event.get("usage")
    fragments = delta.get("tool_calls") or []
    return ChatChunk(
        delta_text=delta.get("content") or "",
        tool_call_delta=_delta_from_wire(fragments[0]) if fragments else None,
        finish_reason=(
            finish_reason(choice.get("finish_reason")) if choice.get("finish_reason") else None
        ),
        usage=_usage_from_wire(usage_block) if isinstance(usage_block, dict) else None,
    )


def _tool_to_wire(tool: ToolSchema, schema_style: SchemaStyle) -> JSONObject:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if schema_style == "openai_strict":
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _message_to_wire(message: Message) -> JSONObject:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        wire["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in message.tool_calls
        ]
    return wire


def _tool_choice_to_wire(choice: ToolChoice) -> Any:
    if isinstance(choice, tuple):
        return {"type": "function", "function": {"name": choice[1]}}
    return choice


def _response_format_to_wire(fmt: ResponseFormat) -> JSONObject:
    if fmt.kind == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": fmt.name, "schema": fmt.schema, "strict": True},
    }


def _tool_calls_from_wire(raw: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls = []
    for position, item in enumerate(raw):
        function = item.get("function") or {}
        arguments_json = function.get("arguments") or ""
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{position}"),
                name=str(function.get("name") or ""),
                arguments_json=arguments_json,
                arguments=parse_arguments(arguments_json),
            )
        )
    return tuple(calls)


def _delta_from_wire(fragment: Mapping[str, Any]) -> ToolCallDelta:
    function = fragment.get("function") or {}
    index = fragment.get("index")
    return ToolCallDelta(
        index=int(index) if isinstance(index, int) else None,
        id=fragment.get("id"),
        name=function.get("name"),
        arguments_fragment=function.get("arguments") or "",
    )


def _usage_from_wire(raw: Mapping[str, Any]) -> Usage:
    details = json_object(raw.get("prompt_tokens_details"))
    completion_details = json_object(raw.get("completion_tokens_details"))
    return Usage(
        input_tokens=json_int(raw.get("prompt_tokens")),
        output_tokens=json_int(raw.get("completion_tokens")),
        cached_input_tokens=json_int(details.get("cached_tokens")),
        reasoning_tokens=json_int(completion_details.get("reasoning_tokens")),
    )


def _cost_from_wire(usage: Mapping[str, Any]) -> float | None:
    cost = usage.get("cost")
    return float(cost) if isinstance(cost, int | float) else None


def _sse_event(line: str) -> JSONObject | None:
    """One data line into a payload, skipping keepalives and the terminator."""
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        parsed = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
