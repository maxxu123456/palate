"""Ollama's native /api/chat, not the OpenAI shim."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import httpx
import orjson

from palate.errors import ModelNotFound, ProviderTimeout, ProviderUnavailable
from palate.providers.base import (
    ChatCapabilities,
    ChatChunk,
    Completion,
    JSONObject,
    Message,
    ProviderHealth,
    ResponseFormat,
    SpanLike,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolSchema,
    Usage,
    json_int,
    json_object,
)
from palate.providers.chat.openai_compat import finish_reason
from palate.providers.contentcalls import extract_tool_calls
from palate.providers.retry import LOCAL_RETRY, RetryPolicy, classify, with_retry
from palate.providers.tokens import count_tokens

BASE_URL = "http://127.0.0.1:11434"


class OllamaChat:
    """The shim drops the token counts, keep_alive, num_ctx and schema format, so we do not use it."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        client: httpx.AsyncClient,
        base_url: str = BASE_URL,
        keep_alive: str = "10m",
        num_ctx: int | None = 8192,
        timeout_s: float = 180.0,
        retry: RetryPolicy = LOCAL_RETRY,
        parse_content_tool_calls: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.retry = retry
        self.parse_content_tool_calls = parse_content_tool_calls
        self._turn = 0

    def _to_wire(
        self,
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
    ) -> JSONObject:
        options: dict[str, Any] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = list(stop)
        if seed is not None:
            options["seed"] = seed
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
            "stream": stream,
            # Without this every call reloads a multi gigabyte model from disk.
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if tools and tool_choice != "none":
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        if response_format is not None and response_format.kind == "json_schema":
            body["format"] = response_format.schema
        elif response_format is not None and response_format.kind == "json_object":
            body["format"] = "json"
        return body

    def _from_wire(self, payload: JSONObject, *, turn: int) -> Completion:
        message = json_object(payload.get("message"))
        content = str(message.get("content") or "")
        calls = _tool_calls_from_wire(message.get("tool_calls"), turn=turn)
        if not calls and self.parse_content_tool_calls:
            calls, content = extract_tool_calls(content, turn=turn)
        reason = finish_reason(str(payload.get("done_reason") or "stop"))
        return Completion(
            content=content,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else reason,
            usage=Usage(
                input_tokens=json_int(payload.get("prompt_eval_count")),
                output_tokens=json_int(payload.get("eval_count")),
            ),
            model=self.model,
            response_model=str(payload.get("model") or self.model),
            raw=payload,
        )

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
        """One non streaming turn against /api/chat."""
        self._turn += 1
        turn = self._turn
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
            response = await self._send(body, stream=False, timeout_s=timeout_s)
            payload: JSONObject = orjson.loads(response.content)
            done = self._from_wire(payload, turn=turn)
            return replace(
                done, latency_ms=(time.perf_counter() - started) * 1000, attempt=attempts
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
        """NDJSON, one object per line, terminated by the object carrying the counts."""
        self._turn += 1
        turn = self._turn
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
        response = await self._send(body, stream=True, timeout_s=timeout_s)
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event: JSONObject = orjson.loads(line)
                yield _chunk_from_wire(event, turn=turn)
        finally:
            await response.aclose()

    async def capabilities(self) -> ChatCapabilities:
        """Tool results correlate by name only, which rules parallel calls out."""
        return ChatCapabilities(
            tools=True,
            parallel_tool_calls=False,
            tool_streaming=False,
            json_schema=True,
            streaming=True,
            context_window=self.num_ctx,
            max_output_tokens=None,
            schema_style="ollama",
        )

    async def health(self) -> ProviderHealth:
        """GET /api/tags, and say how to fix it when the daemon is not there."""
        started = time.perf_counter()
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
        except httpx.HTTPError:
            return ProviderHealth(
                False,
                f"no ollama at {self.base_url}, try: brew install ollama && ollama serve "
                f"&& ollama pull {self.model}",
            )
        elapsed = (time.perf_counter() - started) * 1000
        names = {
            str(m.get("model", "")).split(":")[0]
            for m in (orjson.loads(response.content).get("models") or [])
        }
        if self.model.split(":")[0] not in names:
            return ProviderHealth(False, f"ollama pull {self.model}", elapsed)
        return ProviderHealth(True, f"ollama serving {self.model}", elapsed)

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSchema] = ()) -> int:
        """Estimate only. The real counts come back on every response."""
        return count_tokens(messages, tools)

    async def aclose(self) -> None:
        """The client is shared, so closing it is the caller's job."""
        return None

    async def _send(
        self, body: JSONObject, *, stream: bool, timeout_s: float | None
    ) -> httpx.Response:
        request = self.client.build_request(
            "POST",
            f"{self.base_url}/api/chat",
            json=body,
            timeout=timeout_s or self.timeout_s,
        )
        try:
            response = await self.client.send(request, stream=stream)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                "ollama timed out", provider=self.name, model=self.model, retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"no ollama at {self.base_url} ({exc})",
                provider=self.name,
                model=self.model,
                retryable=True,
            ) from exc
        if response.status_code >= 400:
            if stream:
                await response.aread()
            status = response.status_code
            await response.aclose()
            if status == 404:
                raise ModelNotFound(
                    f"ollama pull {self.model}",
                    provider=self.name,
                    model=self.model,
                    pull_hint=f"ollama pull {self.model}",
                )
            error = classify(response, provider=self.name, model=self.model)
            if error is not None:
                raise error
        return response


def _message_to_wire(message: Message) -> JSONObject:
    # Ollama has no tool_call_id, so a result correlates by tool name and nothing else.
    if message.role == "tool":
        return {"role": "tool", "content": message.content, "tool_name": message.name or ""}
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments or {}}}
            for call in message.tool_calls
        ]
    return wire


def _tool_calls_from_wire(raw: Any, *, turn: int) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls = []
    for position, item in enumerate(raw):
        function = item.get("function") or {}
        arguments = function.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        calls.append(
            ToolCall(
                id=f"call_{turn}_{position}",
                name=str(function.get("name") or ""),
                arguments_json=orjson.dumps(arguments).decode(),
                arguments=arguments,
            )
        )
    return tuple(calls)


def _chunk_from_wire(event: JSONObject, *, turn: int) -> ChatChunk:
    message = json_object(event.get("message"))
    calls = _tool_calls_from_wire(message.get("tool_calls"), turn=turn)
    done = bool(event.get("done"))
    return ChatChunk(
        delta_text=str(message.get("content") or ""),
        # Tool calls arrive whole on the last chunk, so one delta carries everything.
        tool_call_delta=(
            ToolCallDelta(
                index=0,
                id=calls[0].id,
                name=calls[0].name,
                arguments_fragment=calls[0].arguments_json,
            )
            if calls
            else None
        ),
        finish_reason=(
            ("tool_calls" if calls else finish_reason(str(event.get("done_reason") or "stop")))
            if done
            else None
        ),
        usage=(
            Usage(
                input_tokens=json_int(event.get("prompt_eval_count")),
                output_tokens=json_int(event.get("eval_count")),
            )
            if done
            else None
        ),
    )
