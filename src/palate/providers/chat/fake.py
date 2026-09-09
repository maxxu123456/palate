"""The scripted provider every offline test runs against."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

import anyio

from palate.providers.base import (
    ChatCapabilities,
    ChatChunk,
    Completion,
    FinishReason,
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
from palate.providers.tokens import count_tokens


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    """One canned reply, optionally asserting what the loop sent to get it."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raises: Exception | None = None
    usage: Usage = field(default_factory=Usage)
    finish_reason: FinishReason = "stop"
    delay_s: float = 0.0
    expect: Callable[[Sequence[Message], Sequence[ToolSchema]], None] | None = None


type Script = Sequence[ScriptedTurn] | Callable[[list[Message], list[ToolSchema]], Completion]


class FakeChatProvider:
    """Deterministic, offline, and records every request it was given."""

    name = "fake"

    def __init__(
        self,
        script: Script,
        *,
        loop_last: bool = False,
        capabilities: ChatCapabilities | None = None,
        model: str = "fake-model",
    ) -> None:
        self.script = script
        self.loop_last = loop_last
        self.model = model
        self._capabilities = capabilities or ChatCapabilities(context_window=8192)
        self._calls: list[tuple[list[Message], list[ToolSchema]]] = []

    @property
    def calls(self) -> list[tuple[list[Message], list[ToolSchema]]]:
        """Every (messages, tools) pair this provider was asked with."""
        return self._calls

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
        """The next scripted turn, or whatever the callable form decides."""
        sent = (list(messages), list(tools))
        self._calls.append(sent)
        if callable(self.script):
            return self.script(*sent)
        turn = self._next(len(self._calls) - 1)
        if turn.expect is not None:
            turn.expect(*sent)
        if turn.delay_s:
            await anyio.sleep(turn.delay_s)
        if turn.raises is not None:
            raise turn.raises
        return Completion(
            content=turn.text,
            tool_calls=turn.tool_calls,
            finish_reason="tool_calls" if turn.tool_calls else turn.finish_reason,
            usage=turn.usage,
            model=self.model,
            response_model=self.model,
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
        """The same turn, cut into one chunk per word so the consumer sees real deltas."""
        done = await self.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            seed=seed,
            timeout_s=timeout_s,
            span=span,
        )
        for position, word in enumerate(done.content.split(" ")):
            if not word:
                continue
            yield ChatChunk(delta_text=word if position == 0 else f" {word}")
        for index, call in enumerate(done.tool_calls):
            yield ChatChunk(
                tool_call_delta=ToolCallDelta(
                    index=index,
                    id=call.id,
                    name=call.name,
                    arguments_fragment=call.arguments_json,
                )
            )
        yield ChatChunk(finish_reason=done.finish_reason, usage=done.usage)

    async def capabilities(self) -> ChatCapabilities:
        """Whatever the test asked for."""
        return self._capabilities

    async def health(self) -> ProviderHealth:
        """Always up, which is the point of a fake."""
        return ProviderHealth(True, "fake provider", 0.0)

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSchema] = ()) -> int:
        """The same estimate the real adapters use."""
        return count_tokens(messages, tools)

    async def aclose(self) -> None:
        """Nothing to close."""
        return None

    def _next(self, position: int) -> ScriptedTurn:
        turns = list(self.script) if not callable(self.script) else []
        if not turns:
            return ScriptedTurn()
        if position < len(turns):
            return turns[position]
        if self.loop_last:
            return turns[-1]
        raise AssertionError(f"fake provider ran out of script at call {position + 1}")
