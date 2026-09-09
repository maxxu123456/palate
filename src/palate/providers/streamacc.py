"""Folding streamed tool call fragments back into whole calls."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import orjson

from palate.providers.base import (
    ChatChunk,
    FinishReason,
    JSONObject,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from palate.providers.tokens import estimate_text


@dataclass(slots=True)
class _Slot:
    id: str | None = None
    name: str = ""
    arguments: list[str] = field(default_factory=list)


class ToolCallAccumulator:
    """Folds index keyed deltas, and deltas from servers that send no index at all."""

    def __init__(self, *, turn: int = 0) -> None:
        self.turn = turn
        self._slots: dict[int, _Slot] = {}
        self._order: list[int] = []

    def add(self, delta: ToolCallDelta) -> None:
        """Absorb one fragment."""
        key = delta.index if delta.index is not None else self._implied_index(delta)
        slot = self._slots.get(key)
        if slot is None:
            slot = _Slot()
            self._slots[key] = slot
            self._order.append(key)
        if delta.id:
            slot.id = delta.id
        if delta.name:
            slot.name += delta.name
        if delta.arguments_fragment:
            slot.arguments.append(delta.arguments_fragment)

    def _implied_index(self, delta: ToolCallDelta) -> int:
        # No index means a new call starts whenever a name or id shows up.
        if not self._order or delta.name or delta.id:
            return len(self._order)
        return self._order[-1]

    def finish(self) -> tuple[ToolCall, ...]:
        """Every call seen, in arrival order, with the arguments parsed if they parse."""
        calls = []
        for position, key in enumerate(self._order):
            slot = self._slots[key]
            raw = "".join(slot.arguments)
            calls.append(
                ToolCall(
                    id=slot.id or f"call_{self.turn}_{position}",
                    name=slot.name,
                    arguments_json=raw,
                    arguments=parse_arguments(raw),
                )
            )
        return tuple(calls)

    def __len__(self) -> int:
        return len(self._order)


def parse_arguments(raw: str) -> JSONObject | None:
    """Parsed arguments, or None when the model emitted something that is not an object."""
    if not raw.strip():
        return {}
    try:
        parsed = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True, slots=True)
class StreamResult:
    """A finished stream, folded back into the shape a non streaming call returns."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage


async def accumulate(chunks: AsyncIterator[ChatChunk], *, turn: int = 0) -> StreamResult:
    """Drain a chat stream into one result, estimating usage when the stream drops it."""
    calls = ToolCallAccumulator(turn=turn)
    text: list[str] = []
    finish: FinishReason = "stop"
    usage: Usage | None = None
    async for chunk in chunks:
        if chunk.delta_text:
            text.append(chunk.delta_text)
        if chunk.tool_call_delta is not None:
            calls.add(chunk.tool_call_delta)
        if chunk.finish_reason is not None:
            finish = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
    content = "".join(text)
    if usage is None:
        # Nothing else knows what was generated, so the estimate is flagged as one.
        usage = Usage(output_tokens=estimate_text(content), exact=False)
    return StreamResult(content, calls.finish(), finish, usage)
