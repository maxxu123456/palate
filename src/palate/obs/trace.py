"""Spans, and the two ways a component reaches one: the argument, or the context variable."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import orjson

from palate.clock import now_iso
from palate.hashing import canonical_json, sha256_hex
from palate.ids import new_run_id, new_span_id
from palate.obs.cost import CostResult
from palate.obs.store import TraceRow, TraceStore, payload
from palate.providers.base import (
    Completion,
    Message,
    RerankReport,
    ToolCall,
    ToolSchema,
)

type SpanKind = Literal[
    "run", "llm", "embed", "rerank", "tool", "retrieval", "ground", "http", "db"
]

# OpenTelemetry GenAI names where they exist, so an exporter is an adapter and not a rewrite.
GEN_AI_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_TOKENS_IN = "gen_ai.usage.input_tokens"
GEN_AI_TOKENS_OUT = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"


@dataclass(slots=True)
class Span:
    """One timed node. Nothing here writes, the tracer does that when the span closes."""

    span_id: str
    run_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    seq: int
    started_ns: int
    attrs: dict[str, Any] = field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    error_type: str | None = None
    error_message: str | None = None
    ended_ns: int | None = None

    def set(self, **attrs: Any) -> None:
        """Attach attributes. A helper four frames down can do this without a parameter."""
        self.attrs.update(attrs)

    def event(self, name: str, **fields: Any) -> None:
        """A point in time inside the span, kept as one attribute list."""
        self.attrs.setdefault("events", []).append({"name": name, **fields})

    def fail(self, code: str, message: str) -> None:
        """Mark the span failed without raising, for a failure that is an expected outcome."""
        self.status = "error"
        self.error_type = code
        self.error_message = message[:500]

    @property
    def latency_ms(self) -> float:
        """Elapsed milliseconds, measured to now while the span is still open."""
        end = self.ended_ns if self.ended_ns is not None else time.time_ns()
        return (end - self.started_ns) / 1_000_000.0


class Tracer(Protocol):
    """What the loop, the providers and the recommender are allowed to call."""

    def run(
        self, kind: str, *, session_id: str | None = None, input_text: str | None = None
    ) -> Any: ...

    def span(self, name: str, kind: SpanKind, *, parent: Span | None = None) -> Any: ...

    def record_llm(self, span: Span, req: LLMRequestRecord, res: Completion) -> None: ...

    def record_rerank(self, span: Span, rep: RerankReport) -> None: ...

    def record_tool(self, span: Span, call: ToolCall, result: Any) -> None: ...

    def record_grounding(self, span: Span, report: Any) -> None: ...

    def finish_run(self, span: Span, **fields: Any) -> None: ...

    def flush(self, timeout_s: float = 5.0) -> None: ...


@dataclass(frozen=True, slots=True)
class LLMRequestRecord:
    """Everything about the request a trace row needs and a Completion does not carry."""

    provider: str
    model: str
    messages: Sequence[Message] = ()
    tools: Sequence[ToolSchema] = ()
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_sha: str | None = None
    prompt_template: str | None = None
    temperature: float | None = None
    seed: int | None = None


_current: ContextVar[Span | None] = ContextVar("palate_span", default=None)


def current_span() -> Span | None:
    """The innermost open span on this task, for a provider given no explicit one."""
    return _current.get()


class NullTracer:
    """Every method a no-op. The default in unit tests that do not assert on traces."""

    @contextmanager
    def run(
        self, kind: str, *, session_id: str | None = None, input_text: str | None = None
    ) -> Iterator[Span]:
        yield _blank("run", kind)

    @contextmanager
    def span(self, name: str, kind: SpanKind, *, parent: Span | None = None) -> Iterator[Span]:
        yield _blank(kind, name)

    def record_llm(self, span: Span, req: LLMRequestRecord, res: Completion) -> None:
        return None

    def record_rerank(self, span: Span, rep: RerankReport) -> None:
        return None

    def record_tool(self, span: Span, call: ToolCall, result: Any) -> None:
        return None

    def record_grounding(self, span: Span, report: Any) -> None:
        return None

    def finish_run(self, span: Span, **fields: Any) -> None:
        return None

    def flush(self, timeout_s: float = 5.0) -> None:
        return None


class SQLiteTracer:
    """Writes through the store, so nothing here touches the disk on the calling thread."""

    def __init__(
        self,
        store: TraceStore,
        *,
        git_sha: str | None = None,
        config_sha: str | None = None,
        price: Any = None,
    ) -> None:
        self.store = store
        self.git_sha = git_sha
        self.config_sha = config_sha
        self.price = price
        self._seq: dict[str, int] = {}

    @contextmanager
    def run(
        self, kind: str, *, session_id: str | None = None, input_text: str | None = None
    ) -> Iterator[Span]:
        """One runs row per user turn, written open first so a cancelled run still lands."""
        span = Span(
            span_id=new_span_id(),
            run_id=new_run_id(),
            parent_id=None,
            name=kind,
            kind="run",
            seq=0,
            started_ns=time.time_ns(),
        )
        self._submit_run(span, kind, session_id, input_text, status="running")
        self._submit_span(span)
        token = _current.set(span)
        try:
            yield span
        except BaseException as exc:
            span.fail(type(exc).__name__, str(exc))
            raise
        finally:
            _current.reset(token)
            span.ended_ns = time.time_ns()
            self._submit_span(span)
            self._submit_run(
                span,
                kind,
                session_id,
                input_text,
                status="error" if span.status == "error" else "ok",
            )

    @contextmanager
    def span(self, name: str, kind: SpanKind, *, parent: Span | None = None) -> Iterator[Span]:
        """A child of the given span, or of whatever is open on this task."""
        above = parent or current_span()
        run_id = above.run_id if above is not None else new_run_id()
        child = Span(
            span_id=new_span_id(),
            run_id=run_id,
            parent_id=above.span_id if above is not None else None,
            name=name,
            kind=kind,
            seq=self._next_seq(run_id),
            started_ns=time.time_ns(),
        )
        token = _current.set(child)
        try:
            yield child
        except BaseException as exc:
            child.fail(type(exc).__name__, str(exc))
            raise
        finally:
            _current.reset(token)
            child.ended_ns = time.time_ns()
            self._submit_span(child)

    def record_llm(self, span: Span, req: LLMRequestRecord, res: Completion) -> None:
        """One llm_calls row, with the payload written only when the mode says so."""
        mode = self.store.payloads
        sent = orjson.dumps([_message_json(m) for m in req.messages]).decode()
        messages_z, messages_sha = payload(sent, mode)
        response_z, response_sha = payload(res.content, mode)
        cost = self._cost(req, res)
        span.set(
            **{
                GEN_AI_OPERATION: "chat",
                GEN_AI_MODEL: req.model,
                GEN_AI_RESPONSE_MODEL: res.response_model,
                GEN_AI_TOKENS_IN: res.usage.input_tokens,
                GEN_AI_TOKENS_OUT: res.usage.output_tokens,
            }
        )
        if req.prompt_sha and req.prompt_template and req.prompt_name:
            self.store.submit(
                TraceRow(
                    "prompts",
                    {
                        "prompt_sha": req.prompt_sha,
                        "name": req.prompt_name,
                        "version": req.prompt_version or "1",
                        "template": req.prompt_template,
                        "first_seen": now_iso(),
                    },
                )
            )
        self.store.submit(
            TraceRow(
                "llm_calls",
                {
                    "span_id": span.span_id,
                    "run_id": span.run_id,
                    "provider": req.provider,
                    "request_model": req.model,
                    "response_model": res.response_model,
                    "prompt_name": req.prompt_name,
                    "prompt_version": req.prompt_version,
                    "prompt_sha": req.prompt_sha,
                    "messages_z": messages_z,
                    "messages_sha": messages_sha,
                    "response_z": response_z,
                    "response_sha": response_sha,
                    "tools_json": orjson.dumps([t.name for t in req.tools]).decode(),
                    "tool_calls_json": orjson.dumps(
                        [_call_json(c, mode) for c in res.tool_calls]
                    ).decode(),
                    "finish_reason": res.finish_reason,
                    "temperature": req.temperature,
                    "seed": req.seed,
                    "tokens_in": res.usage.input_tokens,
                    "tokens_out": res.usage.output_tokens,
                    "tokens_cached": res.usage.cached_input_tokens,
                    "tokens_exact": int(res.usage.exact),
                    "cost_usd": cost.usd,
                    "cost_source": cost.source,
                    "cache_hit": int(res.cache_hit),
                    "attempt": res.attempt,
                    "ttft_ms": res.ttft_ms,
                    "latency_ms": span.latency_ms,
                },
            )
        )

    def record_rerank(self, span: Span, rep: RerankReport) -> None:
        """One rerank_calls row. The dollars land in the same tree as the chat dollars."""
        self.store.submit(
            TraceRow(
                "rerank_calls",
                {
                    "span_id": span.span_id,
                    "run_id": span.run_id,
                    "model_key": rep.model_key,
                    "doc_version": str(span.attrs.get("doc_version", "")),
                    "n_pairs": rep.n_pairs,
                    "n_cache_hits": rep.cache_hits,
                    "cold_start": int(rep.cold_start),
                    "cost_usd": rep.cost_usd,
                    "cost_source": "provider",
                    "latency_ms": rep.elapsed_ms,
                },
            )
        )

    def record_tool(self, span: Span, call: ToolCall, result: Any) -> None:
        """One tool_calls row, with the exact validation text the model was handed."""
        rendered = orjson.dumps(result.data).decode() if result.data is not None else None
        error = result.error
        span.set(**{GEN_AI_TOOL_NAME: call.name})
        self.store.submit(
            TraceRow(
                "tool_calls",
                {
                    "span_id": span.span_id,
                    "run_id": span.run_id,
                    "turn": int(span.attrs.get("turn", 0)),
                    "seq_in_turn": int(span.attrs.get("seq_in_turn", 0)),
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "args_json": call.arguments_json if self.store.payloads == "full" else None,
                    "args_fingerprint": sha256_hex(
                        call.name.encode() + canonical_json(call.arguments or {})
                    ),
                    "args_valid": int(error is None or str(error.code) != "bad_arguments"),
                    "validation_error": None if error is None else error.message,
                    "result_json": rendered if self.store.payloads == "full" else None,
                    "result_sha": sha256_hex(rendered) if rendered else None,
                    "result_rows": _rows(result),
                    "result_bytes": len(rendered) if rendered else 0,
                    "truncated": int(bool(result.meta.get("truncated"))),
                    "cache_hit": int(result.cache_hit),
                    "ok": int(result.ok),
                    "error_code": None if error is None else str(error.code),
                    "latency_ms": float(result.latency_ms),
                },
            )
        )

    def record_grounding(self, span: Span, report: Any) -> None:
        """One grounding_claims row per claim, plus the ratio on the span."""
        span.set(
            grounded_ratio=report.grounded_ratio,
            nli_available=report.nli_available,
            method_counts=dict(report.method_counts),
        )
        for checked in report.claims:
            self.store.submit(
                TraceRow(
                    "grounding_claims",
                    {
                        "run_id": span.run_id,
                        "sentence": checked.claim.sentence,
                        "film_id": checked.claim.film_id,
                        "claim_kind": checked.claim.kind,
                        "method": checked.method,
                        "supported": int(checked.supported),
                        "evidence_source": checked.evidence[:200],
                        "score": checked.score,
                        "detail": None,
                    },
                )
            )

    def finish_run(self, span: Span, **fields: Any) -> None:
        """Fold the ledger totals into the runs row the context manager already opened."""
        span.set(**fields)

    def flush(self, timeout_s: float = 5.0) -> None:
        """Everything queued so far on disk."""
        self.store.flush(timeout_s)

    def _cost(self, req: LLMRequestRecord, res: Completion) -> CostResult:
        if self.price is None:
            return CostResult(
                res.cost_usd or 0.0, "provider" if res.cost_usd is not None else "unknown"
            )
        return CostResult(*self.price(req.provider, req.model, res.usage, res.cost_usd))

    def _next_seq(self, run_id: str) -> int:
        seq = self._seq.get(run_id, 0) + 1
        self._seq[run_id] = seq
        return seq

    def _submit_span(self, span: Span) -> None:
        self.store.submit(
            TraceRow(
                "spans",
                {
                    "span_id": span.span_id,
                    "run_id": span.run_id,
                    "parent_id": span.parent_id,
                    "name": span.name,
                    "kind": span.kind,
                    "seq": span.seq,
                    "started_at": now_iso(),
                    "ended_at": now_iso() if span.ended_ns is not None else None,
                    "latency_ms": span.latency_ms if span.ended_ns is not None else None,
                    "status": span.status,
                    "error_type": span.error_type,
                    "error_message": span.error_message,
                    "attrs_json": orjson.dumps(span.attrs, default=str).decode(),
                },
            )
        )

    def _submit_run(
        self,
        span: Span,
        kind: str,
        session_id: str | None,
        input_text: str | None,
        *,
        status: str,
    ) -> None:
        keep = self.store.payloads == "full"
        self.store.submit(
            TraceRow(
                "runs",
                {
                    "run_id": span.run_id,
                    "kind": kind,
                    "session_id": session_id,
                    "started_at": now_iso(),
                    "ended_at": now_iso() if span.ended_ns is not None else None,
                    "latency_ms": span.latency_ms if span.ended_ns is not None else None,
                    "status": status,
                    "error_type": span.error_type,
                    "error_message": span.error_message,
                    "input_text": input_text if keep else None,
                    "output_text": str(span.attrs.get("output_text", "")) if keep else None,
                    "turns": int(span.attrs.get("turns", 0)),
                    "total_tokens_in": int(span.attrs.get("total_tokens_in", 0)),
                    "total_tokens_out": int(span.attrs.get("total_tokens_out", 0)),
                    "total_cost_usd": float(span.attrs.get("total_cost_usd", 0.0)),
                    "cost_complete": int(span.attrs.get("cost_complete", 1)),
                    "git_sha": self.git_sha,
                    "config_sha": self.config_sha,
                },
            )
        )


def _blank(kind: SpanKind, name: str) -> Span:
    return Span(span_id="", run_id="", parent_id=None, name=name, kind=kind, seq=0, started_ns=0)


def _call_json(call: ToolCall, mode: str) -> dict[str, Any]:
    """The name always, the arguments only under full, because they carry what the user asked."""
    out: dict[str, Any] = {"name": call.name}
    if mode == "full":
        out["arguments"] = call.arguments_json
    return out


def _message_json(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "name": message.name,
    }


def _rows(result: Any) -> int | None:
    for key in ("returned", "count"):
        value = result.meta.get(key)
        if isinstance(value, int):
            return value
    return None
