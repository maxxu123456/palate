"""One shape for every tool result, success or failure, so the model learns it once."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import orjson

from palate.providers.base import JSONObject


class ToolErrorCode(StrEnum):
    """Every way a call can fail, as the model sees it."""

    UNKNOWN_TOOL = "unknown_tool"
    MALFORMED_JSON = "malformed_json"
    BAD_ARGUMENTS = "bad_arguments"
    NOT_FOUND = "not_found"
    EMPTY_RESULT = "empty_result"
    PRECONDITION_FAILED = "precondition_failed"
    TOO_MANY_CALLS = "too_many_calls"
    REPEATED_CALL = "repeated_call"
    UNSUPPORTED = "unsupported"
    UPSTREAM_ERROR = "upstream_error"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class ToolError:
    """Why a call failed, and what would make the next one work."""

    code: ToolErrorCode
    message: str
    retryable: bool
    hint: str = ""
    schema_excerpt: JSONObject | None = None
    valid_values: tuple[str, ...] = ()

    def as_json(self) -> JSONObject:
        """The error as it is serialised into the tool message."""
        out: JSONObject = {
            "code": str(self.code),
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.hint:
            out["hint"] = self.hint
        if self.schema_excerpt is not None:
            out["schema"] = self.schema_excerpt
        if self.valid_values:
            out["valid_values"] = list(self.valid_values)
        return out


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What one call produced. A tool error is data, never an exception."""

    call_id: str
    name: str
    ok: bool
    data: JSONObject | None = None
    error: ToolError | None = None
    meta: JSONObject = field(default_factory=dict)
    latency_ms: int = 0
    cache_hit: bool = False
    undo_token: str | None = None

    def render(self, max_chars: int) -> str:
        """Serialize for the tool message. Truncation is announced with an offset to resume."""
        payload: JSONObject = {"ok": self.ok}
        if self.data is not None:
            payload["data"] = self.data
        if self.error is not None:
            payload["error"] = self.error.as_json()
        if self.meta:
            payload["meta"] = self.meta
        text = orjson.dumps(payload).decode()
        if len(text) <= max_chars:
            return text
        notice = f'\n{{"truncated": true, "of": {len(text)}, "offset": {max_chars}}}'
        return text[: max(max_chars - len(notice), 0)] + notice

    def summary(self) -> str:
        """One line for the UI card, which is not the payload."""
        if self.error is not None:
            return f"{self.name} failed: {self.error.code}"
        rows = self.meta.get("returned", self.meta.get("count"))
        return f"{self.name} returned {rows}" if rows is not None else f"{self.name} ok"

    def with_error(self, error: ToolError) -> ToolResult:
        """The same call carrying a reshaped error, which is how escalation adds detail."""
        return ToolResult(
            call_id=self.call_id,
            name=self.name,
            ok=False,
            error=error,
            meta=self.meta,
            latency_ms=self.latency_ms,
            cache_hit=self.cache_hit,
        )


RETRYABLE = frozenset(
    {
        ToolErrorCode.BAD_ARGUMENTS,
        ToolErrorCode.MALFORMED_JSON,
        ToolErrorCode.TIMEOUT,
        ToolErrorCode.UPSTREAM_ERROR,
        ToolErrorCode.EMPTY_RESULT,
    }
)


def failure(
    call_id: str,
    name: str,
    code: ToolErrorCode,
    message: str,
    *,
    hint: str = "",
    schema_excerpt: JSONObject | None = None,
    valid_values: tuple[str, ...] = (),
    meta: JSONObject | None = None,
    latency_ms: int = 0,
) -> ToolResult:
    """A failed result, with retryability decided by the code rather than by the caller."""
    return ToolResult(
        call_id=call_id,
        name=name,
        ok=False,
        error=ToolError(
            code=code,
            message=message,
            retryable=code in RETRYABLE,
            hint=hint,
            schema_excerpt=schema_excerpt,
            valid_values=valid_values,
        ),
        meta=meta or {},
        latency_ms=latency_ms,
    )
