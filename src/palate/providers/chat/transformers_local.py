"""A transformers pipeline in this process, so chat needs no daemon and no key."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Sequence
from functools import partial
from typing import Any

import anyio
from transformers import TextIteratorStreamer, pipeline, set_seed

from palate.errors import ConfigError, ProviderUnavailable
from palate.hf.cache import holds
from palate.hf.device import mps_limiter, resolve_device
from palate.hf.download import ensure_local
from palate.hf.models import ModelPin, pin
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
)
from palate.providers.contentcalls import extract_tool_calls
from palate.providers.tokens import count_tokens

# An empty queue for this long means generation died without closing the streamer.
STALL_TIMEOUT_S = 90.0

_DONE = object()


class TransformersLocalChat:
    """Weights load on the first call and stay on the instance, because loading costs seconds."""

    name = "transformers"

    def __init__(
        self,
        *,
        alias: str,
        device: str | None = None,
        dtype: str = "auto",
        context_window: int | None = 8192,
        max_new_tokens: int = 1024,
        timeout_s: float = STALL_TIMEOUT_S,
        parse_content_tool_calls: bool = True,
        limiter: anyio.CapacityLimiter | None = None,
        pipe: Any | None = None,
        streamer_factory: Callable[[Any, float], Any] | None = None,
    ) -> None:
        self.model = alias
        self.device = resolve_device(device)
        self.dtype = dtype
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.parse_content_tool_calls = parse_content_tool_calls
        self._limiter = limiter
        self._pipe: Any = pipe
        self._streamer_factory = streamer_factory
        self._pin: ModelPin | None = None
        self._lock = anyio.Lock()
        self._turn = 0

    @property
    def pin(self) -> ModelPin:
        """The models.toml entry, read on first use so a missing sha is not an import error."""
        if self._pin is None:
            self._pin = pin(self.model)
        return self._pin

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
        """One turn. Not retried: a forward pass that failed here fails the same way twice."""
        self._turn += 1
        turn = self._turn
        await self._ready()
        prompt = self._render(messages, tools, tool_choice=tool_choice)
        started = time.perf_counter()
        kwargs = self._generation(temperature=temperature, max_tokens=max_tokens)
        text = _cut(await self._generate(prompt, kwargs, seed=seed), stop)
        calls, content = self._calls_of(text, tools, turn=turn)
        return Completion(
            content=content,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=self._usage(prompt, text),
            model=self.model,
            response_model=self._response_model(),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

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
        """Text off the streamer as it is decoded, unless a tool call could be hiding in it."""
        self._turn += 1
        turn = self._turn
        await self._ready()
        prompt = self._render(messages, tools, tool_choice=tool_choice)
        streamer = self._streamer(timeout_s or self.timeout_s)
        kwargs = self._generation(temperature=temperature, max_tokens=max_tokens)
        # A call arrives as ordinary text, so releasing it token by token shows the user its json.
        withhold = bool(tools) and self.parse_content_tool_calls
        pieces: list[str] = []
        async with anyio.create_task_group() as group:
            group.start_soon(
                partial(self._generate, prompt, {**kwargs, "streamer": streamer}, seed)
            )
            step = iter(streamer)
            while True:
                # The queue read blocks, so it cannot happen on the event loop.
                piece = await anyio.to_thread.run_sync(partial(next, step, _DONE))
                if piece is _DONE:
                    break
                pieces.append(str(piece))
                if not withhold:
                    yield ChatChunk(delta_text=str(piece))
        text = _cut("".join(pieces), stop)
        calls, content = self._calls_of(text, tools, turn=turn)
        if withhold and content:
            yield ChatChunk(delta_text=content)
        for index, call in enumerate(calls):
            yield ChatChunk(
                tool_call_delta=ToolCallDelta(
                    index=index,
                    id=call.id,
                    name=call.name,
                    arguments_fragment=call.arguments_json,
                )
            )
        yield ChatChunk(
            finish_reason="tool_calls" if calls else "stop", usage=self._usage(prompt, text)
        )

    async def capabilities(self) -> ChatCapabilities:
        """Tools ride in the template, but nothing here can constrain what is decoded."""
        return ChatCapabilities(
            tools=True,
            parallel_tool_calls=False,
            tool_streaming=False,
            json_schema=False,
            streaming=True,
            context_window=self.context_window,
            max_output_tokens=self.max_new_tokens,
            schema_style="chat_template",
        )

    async def health(self) -> ProviderHealth:
        """Answered from the pin and the cache, because loading the weights to check is absurd."""
        if self._pipe is not None:
            return ProviderHealth(True, f"{self._response_model()} loaded on {self.device}")
        try:
            target = self.pin
        except ConfigError as exc:
            return ProviderHealth(False, str(exc))
        short = f"{target.repo_id}@{target.revision[:8]}"
        if not _cached(target):
            return ProviderHealth(False, f"{short} is not cached, the first call downloads it")
        return ProviderHealth(True, f"{short} on {self.device}")

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSchema] = ()) -> int:
        """Exact once the tokenizer is loaded. Loading it just to count would cost gigabytes."""
        if self._pipe is None:
            return count_tokens(messages, tools)
        return len(self._tokenizer().encode(self._render(messages, tools)))

    async def aclose(self) -> None:
        """Drop the pipeline, because torch does not hand unified memory back on its own."""
        self._pipe = None

    async def _ready(self) -> Any:
        """The pipeline, built once even when two turns race for it."""
        if self._pipe is not None:
            return self._pipe
        async with self._lock:
            if self._pipe is None:
                self._pipe = await anyio.to_thread.run_sync(self._build, limiter=self._gate())
        return self._pipe

    def _build(self) -> Any:
        # The snapshot is already the pinned commit, so nothing here resolves a revision.
        local = ensure_local(self.pin)
        return pipeline(
            task="text-generation", model=str(local), dtype=self.dtype, device=self.device
        )

    def _render(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        tool_choice: ToolChoice = "auto",
    ) -> str:
        """The prompt the model sees, tools included, from the checkpoint's own chat template."""
        offered = [_tool_to_template(t) for t in tools] if tool_choice != "none" else []
        return str(
            self._tokenizer().apply_chat_template(
                [_message_to_template(m) for m in messages],
                tools=offered or None,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    def _generation(self, *, temperature: float, max_tokens: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or self.max_new_tokens,
            "return_full_text": False,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        return kwargs

    async def _generate(self, prompt: str, kwargs: dict[str, Any], seed: int | None = None) -> str:
        # A forward pass on the event loop stalls every other stream on the worker.
        call = partial(self._blocking, prompt, kwargs, seed)
        return await anyio.to_thread.run_sync(call, limiter=self._gate())

    def _blocking(self, prompt: str, kwargs: dict[str, Any], seed: int | None) -> str:
        if seed is not None:
            set_seed(seed)
        out = self._pipe(prompt, **kwargs)
        return str(out[0]["generated_text"]) if out else ""

    def _streamer(self, timeout_s: float) -> Any:
        if self._streamer_factory is not None:
            return self._streamer_factory(self._tokenizer(), timeout_s)
        return TextIteratorStreamer(
            self._tokenizer(), skip_prompt=True, skip_special_tokens=True, timeout=timeout_s
        )

    def _calls_of(
        self, text: str, tools: Sequence[ToolSchema], *, turn: int
    ) -> tuple[tuple[ToolCall, ...], str]:
        if not tools or not self.parse_content_tool_calls:
            return (), text
        return extract_tool_calls(text, known=[t.name for t in tools], turn=turn)

    def _usage(self, prompt: str, text: str) -> Usage:
        encode = self._tokenizer().encode
        # The reply is re-encoded rather than counted as it was sampled, so this is close.
        return Usage(input_tokens=len(encode(prompt)), output_tokens=len(encode(text)), exact=False)

    def _tokenizer(self) -> Any:
        if self._pipe is None:
            raise ProviderUnavailable(
                "the pipeline is not loaded yet", provider=self.name, model=self.model
            )
        return self._pipe.tokenizer

    def _response_model(self) -> str:
        target = self._pin
        return f"{target.repo_id}@{target.revision[:12]}" if target is not None else self.model

    def _gate(self) -> anyio.CapacityLimiter:
        return self._limiter or mps_limiter()


def _message_to_template(message: Message) -> JSONObject:
    """One message as a template row. Arguments stay objects, which is what templates expect."""
    if message.role == "tool":
        # Chat templates carry no call id, so a result is correlated by name and order.
        return {"role": "tool", "name": message.name or "", "content": message.content}
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {"type": "function", "function": {"name": c.name, "arguments": c.arguments or {}}}
            for c in message.tool_calls
        ]
    return wire


def _tool_to_template(tool: ToolSchema) -> JSONObject:
    """The schema the registry rendered, in the envelope apply_chat_template reads."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _cached(target: ModelPin) -> bool:
    """Whether the cache holds this exact commit. A missing cache is an answer, not an error."""
    try:
        return holds(target.repo_id, target.revision)
    except Exception:
        return False


def _cut(text: str, stop: Sequence[str]) -> str:
    """Truncate at the first stop string, which generate only honours with its own tokenizer."""
    found = [text.index(s) for s in stop if s and s in text]
    return text[: min(found)] if found else text
