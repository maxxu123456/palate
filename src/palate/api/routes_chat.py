"""The streaming run, its cancel button, and the sessions it is remembered in."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from dataclasses import asdict

import anyio
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from palate.agent.budget import Budget
from palate.agent.events import AgentEvent
from palate.agent.state import AgentState, StopReason
from palate.api import sse
from palate.api.deps import AppState, State
from palate.ids import new_run_id
from palate.providers.base import Message

router = APIRouter(tags=["chat"])

_MESSAGES = (
    "select seq, run_id, role, channel, content, tool_name, created_at from messages "
    "where session_id = ? order by seq desc limit ?"
)


class BudgetOverride(BaseModel):
    """The knobs a caller may turn down. Nothing here may widen the configured budget."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = Field(default=None, ge=1, le=32)
    max_tool_calls: int | None = Field(default=None, ge=1, le=64)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    max_wall_s: float | None = Field(default=None, gt=0.0)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
    budget: BudgetOverride | None = None


class SessionOut(BaseModel):
    session_id: str
    title: str | None
    chat_provider: str
    chat_model: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    seq: int
    run_id: str | None
    role: str
    channel: str
    content: str
    tool_name: str | None
    created_at: str


def budget_for(state: AppState, asked: BudgetOverride | None) -> Budget:
    """The configured budget, narrowed by anything the caller asked for."""
    limits = Budget.from_settings(state.settings.agent)
    if asked is None:
        return limits
    return Budget(
        max_turns=min(limits.max_turns, asked.max_turns or limits.max_turns),
        max_tool_calls=min(limits.max_tool_calls, asked.max_tool_calls or limits.max_tool_calls),
        max_parallel_calls=limits.max_parallel_calls,
        max_prompt_tokens=limits.max_prompt_tokens,
        max_completion_tokens=limits.max_completion_tokens,
        max_cost_usd=min(limits.max_cost_usd, asked.max_cost_usd or limits.max_cost_usd),
        max_wall_s=min(limits.max_wall_s, asked.max_wall_s or limits.max_wall_s),
        max_consecutive_tool_errors=limits.max_consecutive_tool_errors,
        max_repeat=limits.max_repeat,
    )


async def run_events(
    state: AppState, session_id: str, body: ChatRequest
) -> AsyncIterator[AgentEvent]:
    """One turn, logged before it starts so a cancelled run still leaves a row."""
    run = AgentState(run_id=new_run_id(), session_id=session_id)
    asked = Message(role="user", content=body.message)
    await state.transcript.append(session_id, run.run_id, asked)
    state.sessions.start_run(run.run_id, session_id)
    opened = anyio.current_time()
    scope = anyio.CancelScope()
    state.runs.add(run.run_id, scope)
    try:
        with scope:
            async for event in state.loop.run(
                body.message,
                session_id=session_id,
                budget=budget_for(state, body.budget),
                ctx_factory=state.context,
                state=run,
            ):
                yield event
        if scope.cancel_called:
            run.stop_reason = StopReason.CANCELLED
    finally:
        state.runs.drop(run.run_id)
        # Closed with no await, so a disconnected client still leaves a finished row.
        state.sessions.finish_run(run, wall_ms=int((anyio.current_time() - opened) * 1000.0))
        if run.final_text:
            state.transcript.append_sync(
                session_id, run.run_id, Message(role="assistant", content=run.final_text)
            )


@router.post("/chat")
async def chat(body: ChatRequest, state: State) -> EventSourceResponse:
    """Stream one run. Every film named in it came from a tool result, not from the model."""
    session = state.sessions.open(
        provider=state.settings.chat.provider,
        model=state.settings.chat.model,
        session_id=body.session_id,
    )
    return sse.stream(run_events(state, session.session_id, body))


@router.post("/chat/{run_id}/cancel", status_code=204)
async def cancel(run_id: str, state: State) -> Response:
    """Stop a run in flight. Its row keeps stop_reason cancelled."""
    if not state.runs.cancel(run_id):
        raise HTTPException(status_code=404, detail=f"no run in flight with id {run_id}")
    return Response(status_code=204)


@router.get("/sessions")
async def sessions(state: State, limit: int = 20) -> list[SessionOut]:
    """Newest first, which is the order a sidebar wants them in."""
    return [SessionOut(**asdict(s)) for s in state.sessions.recent(limit=limit)]


@router.get("/sessions/{session_id}/messages")
async def messages(session_id: str, state: State, limit: int = 200) -> list[MessageOut]:
    """The stored transcript, oldest first."""
    rows = state.db.read().execute(_MESSAGES, (session_id, limit)).fetchall()
    return [_message(row) for row in reversed(rows)]


def _message(row: sqlite3.Row) -> MessageOut:
    return MessageOut(
        seq=int(row["seq"]),
        run_id=row["run_id"],
        role=str(row["role"]),
        channel=str(row["channel"]),
        content=str(row["content"]),
        tool_name=row["tool_name"],
        created_at=str(row["created_at"]),
    )
