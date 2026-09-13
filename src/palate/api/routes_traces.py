"""What the agent actually did, read back out of traces.db."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from functools import partial
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from palate.api.deps import State
from palate.db.connect import Database
from palate.obs import report
from palate.obs.replay import replay

router = APIRouter(prefix="/traces", tags=["traces"])

_LLM = (
    "select span_id, provider, request_model, response_model, prompt_name, prompt_version, "
    "finish_reason, tokens_in, tokens_out, tokens_cached, cost_usd, cost_source, cache_hit, "
    "attempt, ttft_ms, latency_ms from llm_calls where run_id = ? order by rowid"
)

_TOOLS = (
    "select span_id, turn, seq_in_turn, tool_name, args_fingerprint, args_valid, "
    "validation_error, result_rows, result_bytes, truncated, cache_hit, ok, error_code, "
    "latency_ms from tool_calls where run_id = ? order by turn, seq_in_turn"
)

_CLAIMS = (
    "select claim_id, sentence, film_id, claim_kind, method, supported, evidence_source, "
    "score, detail from grounding_claims where run_id = ? order by claim_id"
)

_RUN = "select run_id, kind, status from runs where run_id = ?"


class RunOut(BaseModel):
    run_id: str
    kind: str
    session_id: str | None = None
    started_at: str
    latency_ms: float | None = None
    status: str
    turns: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cost_complete: bool


class SpanOut(BaseModel):
    span_id: str
    parent_id: str | None = None
    name: str
    kind: str
    seq: int
    depth: int
    latency_ms: float | None = None
    status: str
    error_type: str | None = None


class RunDetail(BaseModel):
    run_id: str
    spans: list[SpanOut] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)


class CostOut(BaseModel):
    bucket: str
    calls: int
    tokens_in: int
    tokens_out: int
    usd: float
    unknown_calls: int
    unknown_tokens: int


class ReplayBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_override: str | None = None


class ReplayOut(BaseModel):
    span_id: str
    original_model: str
    new_model: str
    original_response: str
    new_response: str
    diff: str
    tokens_delta: tuple[int, int]
    cost_delta: float


def _rows(conn: sqlite3.Connection, sql: str, run_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, (run_id,))]


def detail(db: Database, run_id: str) -> RunDetail:
    """One run, whole. Payload text is present only where trace.payloads kept it."""
    conn = db.read()
    return RunDetail(
        run_id=run_id,
        spans=[SpanOut(**asdict(s)) for s in report.tree(db, run_id)],
        llm_calls=_rows(conn, _LLM, run_id),
        tool_calls=_rows(conn, _TOOLS, run_id),
        claims=_rows(conn, _CLAIMS, run_id),
    )


@router.get("")
async def runs(
    state: State,
    since: Annotated[str, Query()] = "24h",
    kind: Annotated[str, Query()] = "",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[RunOut]:
    """What ran, newest first."""
    db = state.require_traces()
    found = await anyio.to_thread.run_sync(
        partial(report.runs, db, window=since, kind=kind, limit=limit)
    )
    return [RunOut(**asdict(row)) for row in found]


@router.get("/costs")
async def costs(
    state: State,
    since: Annotated[str, Query()] = "7d",
    group_by: Annotated[str, Query()] = "model",
) -> list[CostOut]:
    """The rollup. The unpriced line is never folded into the total."""
    db = state.require_traces()
    found = await anyio.to_thread.run_sync(
        partial(report.costs, db, window=since, group_by=group_by)
    )
    return [CostOut(**asdict(row)) for row in found]


@router.get("/{run_id}")
async def run(run_id: str, state: State) -> RunDetail:
    """The span tree with its calls and the claims the answer was checked against."""
    db = state.require_traces()
    if db.read().execute(_RUN, (run_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"no run traced under {run_id}")
    return await anyio.to_thread.run_sync(partial(detail, db, run_id))


@router.post("/spans/{span_id}/replay")
async def reissue(span_id: str, body: ReplayBody, state: State) -> ReplayOut:
    """Send a stored call again against the provider configured right now, and diff it."""
    result = await replay(
        state.require_traces(),
        span_id,
        provider=state.chat,
        prompt_override=body.prompt_override,
    )
    return ReplayOut(
        span_id=result.span_id,
        original_model=result.original_model,
        new_model=result.new_model,
        original_response=result.original_response,
        new_response=result.new_response,
        diff=result.diff,
        tokens_delta=result.tokens_delta,
        cost_delta=result.cost_delta,
    )
