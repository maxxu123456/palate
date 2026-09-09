"""Rescuing tool calls from models that emit them as ordinary text."""

from __future__ import annotations

import re
from collections.abc import Sequence

import orjson

from palate.providers.base import JSONObject, ToolCall

_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.DOTALL)

_ARGUMENT_KEYS = ("arguments", "parameters", "args", "input")


def extract_tool_calls(
    content: str, *, known: Sequence[str] = (), turn: int = 0
) -> tuple[tuple[ToolCall, ...], str]:
    """Pull tool calls out of assistant text, returning them and the text that is left."""
    calls: list[ToolCall] = []
    remaining = content
    for pattern in (_TAGGED, _FENCED):
        for match in list(pattern.finditer(remaining)):
            call = _as_call(match.group(1), known=known, position=len(calls), turn=turn)
            if call is not None:
                calls.append(call)
                remaining = remaining.replace(match.group(0), "", 1)
    if not calls:
        bare = _as_call(remaining.strip(), known=known, position=0, turn=turn)
        if bare is not None:
            return (bare,), ""
    return tuple(calls), remaining.strip()


def _as_call(text: str, *, known: Sequence[str], position: int, turn: int) -> ToolCall | None:
    if not text.startswith("{"):
        return None
    try:
        parsed = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or (known and name not in known):
        return None
    arguments = _arguments(parsed)
    return ToolCall(
        id=f"call_{turn}_{position}",
        name=name,
        arguments_json=orjson.dumps(arguments).decode(),
        arguments=arguments,
        source="content",
    )


def _arguments(parsed: dict[str, object]) -> JSONObject:
    for key in _ARGUMENT_KEYS:
        value = parsed.get(key)
        if isinstance(value, dict):
            return value
        # Some models put the arguments object back into a json string.
        if isinstance(value, str):
            try:
                reparsed = orjson.loads(value)
            except orjson.JSONDecodeError:
                continue
            if isinstance(reparsed, dict):
                return reparsed
    return {}
