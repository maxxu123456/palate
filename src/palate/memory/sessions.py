"""Sessions and the local run log, which is the half of tracing the user actually reads."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from palate.agent.state import AgentState
from palate.clock import now_iso
from palate.db.connect import Database
from palate.ids import new_session_id

_UPSERT = (
    "insert into sessions (session_id, title, chat_provider, chat_model, created_at, updated_at) "
    "values (?,?,?,?,?,?) on conflict(session_id) do update set "
    "updated_at = excluded.updated_at, title = coalesce(excluded.title, sessions.title)"
)

_START = (
    "insert into runs_local (run_id, session_id, user_message_id, phase, started_at) "
    "values (?,?,?,?,?)"
)

_FINISH = (
    "update runs_local set phase = ?, stop_reason = ?, turns = ?, tool_calls = ?, "
    "input_tokens = ?, output_tokens = ?, cost_usd = ?, wall_ms = ?, compactions = ?, "
    "error_code = ?, ended_at = ? where run_id = ?"
)


@dataclass(frozen=True, slots=True)
class Session:
    """One conversation, pinned to the provider and model it was held with."""

    session_id: str
    title: str | None
    chat_provider: str
    chat_model: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class RunRow:
    """One finished run as the local log holds it."""

    run_id: str
    session_id: str
    phase: str
    stop_reason: str | None
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_ms: int | None
    grounded_ratio: float | None
    started_at: str
    ended_at: str | None


class SessionStore:
    """The sessions and runs_local tables, which live in palate.db and not in traces.db."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def open(
        self,
        *,
        provider: str,
        model: str,
        session_id: str | None = None,
        title: str | None = None,
    ) -> Session:
        """Make a session exist, or touch the one named, and return it."""
        ident = session_id or new_session_id()
        stamp = now_iso()
        with self.db.write() as conn:
            conn.execute(_UPSERT, (ident, title, provider, model, stamp, stamp))
        found = self.get(ident)
        assert found is not None
        return found

    def get(self, session_id: str) -> Session | None:
        """One session by id."""
        row = (
            self.db.read()
            .execute("select * from sessions where session_id = ?", (session_id,))
            .fetchone()
        )
        return None if row is None else _session(row)

    def recent(self, limit: int = 20) -> list[Session]:
        """Newest first, for a listing."""
        rows = self.db.read().execute(
            "select * from sessions order by updated_at desc limit ?", (limit,)
        )
        return [_session(r) for r in rows]

    def start_run(
        self, run_id: str, session_id: str, *, user_message_id: int | None = None
    ) -> None:
        """Write the partial row first, so a cancelled run still leaves a trace."""
        with self.db.write() as conn:
            conn.execute(_START, (run_id, session_id, user_message_id, "plan", now_iso()))

    def finish_run(self, state: AgentState, *, wall_ms: int, error_code: str | None = None) -> None:
        """Close the row with what the ledger actually saw."""
        with self.db.write() as conn:
            conn.execute(
                _FINISH,
                (
                    str(state.phase),
                    None if state.stop_reason is None else str(state.stop_reason),
                    state.turn,
                    state.total_tool_calls,
                    state.ledger.input_tokens,
                    state.ledger.output_tokens,
                    state.ledger.cost_usd,
                    wall_ms,
                    state.compactions,
                    error_code,
                    now_iso(),
                    state.run_id,
                ),
            )

    def runs(self, session_id: str, *, limit: int = 20) -> Sequence[RunRow]:
        """The run log for one session, newest first."""
        rows = self.db.read().execute(
            "select * from runs_local where session_id = ? order by started_at desc limit ?",
            (session_id, limit),
        )
        return [_run(r) for r in rows]


def _session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=str(row["session_id"]),
        title=row["title"],
        chat_provider=str(row["chat_provider"]),
        chat_model=str(row["chat_model"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _run(row: sqlite3.Row) -> RunRow:
    return RunRow(
        run_id=str(row["run_id"]),
        session_id=str(row["session_id"]),
        phase=str(row["phase"]),
        stop_reason=row["stop_reason"],
        turns=int(row["turns"]),
        tool_calls=int(row["tool_calls"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        cost_usd=float(row["cost_usd"]),
        wall_ms=row["wall_ms"],
        grounded_ratio=row["grounded_ratio"],
        started_at=str(row["started_at"]),
        ended_at=row["ended_at"],
    )
