"""The tool table: what exists, what it looks like per provider, and how a call is dispatched."""

from __future__ import annotations

import difflib
import math
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import anyio
import orjson
from pydantic import BaseModel, ValidationError, ValidationInfo

from palate.errors import (
    PreferenceRefused,
    ProviderAuthError,
    ProviderBadRequest,
    StorageError,
    ToolDeadlineExceeded,
    ToolFailure,
)
from palate.hashing import canonical_json, sha256_hex
from palate.providers.base import JSONObject, SchemaStyle, ToolCall, ToolSchema
from palate.tools.context import ToolContext
from palate.tools.envelope import ToolErrorCode, ToolResult, failure
from palate.tools.schema import excerpt, render

# What the model sees of its own broken json, which is what a small model needs to self correct.
RAW_EXCERPT = 200


class ToolKind(StrEnum):
    """Reads run concurrently. Writes run serially, after every read."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class ToolSpec[A: BaseModel, R: BaseModel]:
    """One tool: its models, its handler, and the limits the loop enforces around it."""

    name: str
    description: str
    args_model: type[A]
    result_model: type[R]
    kind: ToolKind
    handler: Callable[[A, ToolContext], Awaitable[R]]
    examples: tuple[A, ...] = ()
    cacheable: bool = True
    cost_hint_ms: int = 50
    max_result_chars: int = 4000
    invalidates: frozenset[str] = frozenset()


def clamp(value: object, info: ValidationInfo, *, lo: int, hi: int) -> int:
    """Clamp rather than reject. A rejected call costs the model a whole turn."""
    try:
        number = int(cast("int", value))
    except (TypeError, ValueError):
        raise ValueError(f"expected a whole number, got {value!r}") from None
    kept = max(lo, min(hi, number))
    if kept != number:
        _note_clamp(info)
    return kept


def _note_clamp(info: ValidationInfo) -> None:
    context = info.context
    if not isinstance(context, dict) or not info.field_name:
        return
    seen = context.get("clamped")
    if isinstance(seen, list):
        seen.append(info.field_name)


def fingerprint(call: ToolCall) -> str:
    """Stable per (name, arguments), which is what repeat detection counts."""
    return sha256_hex(call.name.encode() + canonical_json(call.arguments or {}))


class ToolRegistry:
    """Every tool the agent may call, and the only place a call becomes a result."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec[Any, Any]] = {}

    def register(self, spec: ToolSpec[Any, Any]) -> None:
        """Add a tool. A duplicate name is a programming error, not a runtime condition."""
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name} is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec[Any, Any] | None:
        """One spec by name."""
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        """Every registered name, in registration order."""
        return tuple(self._specs)

    def suggest(self, name: str, n: int = 3) -> list[str]:
        """Closest registered names, for an unknown_tool error worth reading."""
        return difflib.get_close_matches(name, self.names(), n=n, cutoff=0.4)

    def schemas(
        self,
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] = (),
        style: SchemaStyle = "openai",
    ) -> list[ToolSchema]:
        """The tools offered on one request. exclude is how a failing tool is withdrawn."""
        wanted = [
            spec
            for spec in self._specs.values()
            if spec.name not in exclude and (include is None or spec.name in include)
        ]
        return [
            ToolSchema(
                name=spec.name,
                description=spec.description,
                parameters=render(spec.args_model, style=style),
                strict=style == "openai_strict",
            )
            for spec in wanted
        ]

    async def dispatch(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Run one call. Everything the handler raises becomes a typed result instead."""
        started = time.perf_counter()
        spec = self.get(call.name)
        if spec is None:
            return self._unknown(call)
        if call.arguments is None:
            return self._malformed(call)
        clamped: list[str] = []
        try:
            args = spec.args_model.model_validate(call.arguments, context={"clamped": clamped})
        except ValidationError as exc:
            return self._bad_arguments(call, spec, exc)
        meta: JSONObject = cast("JSONObject", {"clamped": sorted(set(clamped))}) if clamped else {}
        try:
            with anyio.fail_after(_budgeted(ctx.remaining_s())):
                result = await spec.handler(args, ctx)
        except (ProviderAuthError, ProviderBadRequest, StorageError):
            raise
        except (ToolDeadlineExceeded, TimeoutError):
            return self._timeout(call, started)
        except PreferenceRefused as exc:
            return failure(
                call.id,
                call.name,
                ToolErrorCode.REFUSED,
                str(exc),
                latency_ms=_ms(started),
            )
        except ToolFailure as exc:
            return failure(
                call.id,
                call.name,
                _code(exc.code),
                str(exc),
                hint=exc.hint,
                valid_values=exc.valid_values,
                latency_ms=_ms(started),
            )
        except Exception as exc:
            return failure(
                call.id,
                call.name,
                ToolErrorCode.UPSTREAM_ERROR,
                f"{type(exc).__name__}: {exc}",
                latency_ms=_ms(started),
            )
        return _envelope(call, result, meta=meta, latency_ms=_ms(started))

    def _unknown(self, call: ToolCall) -> ToolResult:
        close = self.suggest(call.name)
        hint = f"did you mean {', '.join(close)}" if close else "call one of the offered tools"
        return failure(
            call.id,
            call.name,
            ToolErrorCode.UNKNOWN_TOOL,
            f"no tool named {call.name!r}",
            hint=hint,
            valid_values=self.names(),
        )

    @staticmethod
    def _malformed(call: ToolCall) -> ToolResult:
        raw = call.arguments_json[:RAW_EXCERPT]
        return failure(
            call.id,
            call.name,
            ToolErrorCode.MALFORMED_JSON,
            "the arguments were not a json object",
            hint=f"you emitted: {raw}",
        )

    @staticmethod
    def _bad_arguments(
        call: ToolCall, spec: ToolSpec[Any, Any], exc: ValidationError
    ) -> ToolResult:
        first = exc.errors()[0]
        field_name = str(first["loc"][0]) if first["loc"] else ""
        rendered = render(spec.args_model, style="openai")
        if first["type"] == "extra_forbidden":
            properties = rendered.get("properties")
            real = sorted(properties) if isinstance(properties, dict) else []
            hint = f"{field_name} is not a parameter of {spec.name}. It takes: {', '.join(real)}"
        else:
            hint = str(first["msg"])
        return failure(
            call.id,
            call.name,
            ToolErrorCode.BAD_ARGUMENTS,
            _message(exc),
            hint=hint,
            schema_excerpt=excerpt(rendered, field_name),
        )

    @staticmethod
    def _timeout(call: ToolCall, started: float) -> ToolResult:
        return failure(
            call.id,
            call.name,
            ToolErrorCode.TIMEOUT,
            "the run ran out of time before this call finished",
            hint="answer with what you already have",
            latency_ms=_ms(started),
        )


def _message(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]]
    return "; ".join(parts) if parts else str(exc)


def _code(raw: str) -> ToolErrorCode:
    try:
        return ToolErrorCode(raw)
    except ValueError:
        return ToolErrorCode.UPSTREAM_ERROR


def _budgeted(seconds: float) -> float | None:
    """None means no timeout, which is what an unbounded context asks for."""
    return None if seconds == math.inf else seconds


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000.0)


def _envelope(
    call: ToolCall, result: BaseModel, *, meta: JSONObject, latency_ms: int
) -> ToolResult:
    payload = orjson.loads(result.model_dump_json())
    carried = payload.pop("meta", None)
    if isinstance(carried, dict):
        meta = {**carried, **meta}
    undo = payload.get("undo_token")
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=True,
        data=payload,
        meta=meta,
        latency_ms=latency_ms,
        undo_token=undo if isinstance(undo, str) else None,
    )
