"""Conversation memory: what is replayed into the next run, and what is thrown away first."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import anyio
import orjson

from palate.clock import now_iso
from palate.db.connect import Database
from palate.providers.base import Message, Role, ToolCall
from palate.providers.tokens import estimate_text

DROPPED = "[{n} earlier turns dropped]"

_RECENT = (
    "select role, channel, content, tool_calls_json, tool_call_id, tool_name "
    "from messages where session_id = ? order by seq desc"
)

_NEXT_SEQ = "select coalesce(max(seq), -1) + 1 from messages where session_id = ?"

_INSERT = (
    "insert into messages (session_id, run_id, seq, role, channel, content, tool_calls_json, "
    "tool_call_id, tool_name, token_estimate, created_at) values (?,?,?,?,?,?,?,?,?,?,?)"
)


def cost(messages: Sequence[Message]) -> int:
    """Token estimate of a message list, which is what both budgets are measured in."""
    return sum(estimate_text(m.content) + estimate_text(m.name or "") + 4 for m in messages)


class Transcript:
    """The messages table, read newest first and written one row at a time."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def load(self, session_id: str, *, token_budget: int) -> list[Message]:
        """Newest first until the budget is spent. Tool results collapse before user text does."""
        rows = self.db.read().execute(_RECENT, (session_id,)).fetchall()
        kept: list[Message] = []
        spent = 0
        for row in rows:
            message = _message(row)
            spent += estimate_text(message.content) + 4
            if spent > token_budget and kept:
                break
            kept.append(message)
        return list(reversed(kept))

    def compact(self, messages: Sequence[Message], token_budget: int) -> list[Message]:
        """Collapse tool results first, then drop the oldest turns and say how many."""
        kept = list(messages)
        if cost(kept) <= token_budget:
            return kept
        kept = [collapse(m) if m.role == "tool" else m for m in kept]
        dropped = 0
        while len(kept) > 1 and cost(kept) > token_budget:
            kept.pop(0)
            dropped += 1
        if dropped:
            kept.insert(0, Message(role="system", content=DROPPED.format(n=dropped)))
        return kept

    async def append(self, session_id: str, run_id: str, msg: Message) -> int:
        """Persist one message off the event loop, returning its sequence number."""
        return cast(
            "int", await anyio.to_thread.run_sync(self.append_sync, session_id, run_id, msg)
        )

    def append_sync(self, session_id: str, run_id: str, msg: Message) -> int:
        """The same write, on this thread, for callers that are already synchronous."""
        calls = (
            orjson.dumps([_call_json(c) for c in msg.tool_calls]).decode()
            if msg.tool_calls
            else None
        )
        with self.db.write() as conn:
            seq = int(conn.execute(_NEXT_SEQ, (session_id,)).fetchone()[0])
            conn.execute(
                _INSERT,
                (
                    session_id,
                    run_id,
                    seq,
                    msg.role,
                    msg.channel,
                    msg.content,
                    calls,
                    msg.tool_call_id,
                    msg.name,
                    estimate_text(msg.content),
                    now_iso(),
                ),
            )
        return seq


def collapse(msg: Message) -> Message:
    """A tool result as one line. Tool results are most of the tokens and age worst."""
    body = _parse(msg.content)
    rows = _rows(body)
    ids = _ids(body)
    detail = f"{rows} rows" if rows is not None else ("ok" if body.get("ok") else "failed")
    if ids:
        detail += ", ids " + ",".join(str(i) for i in ids[:8])
    return replace(msg, content=f"[{msg.name or 'tool'} -> {detail}]")


def _parse(content: str) -> dict[str, Any]:
    try:
        parsed = orjson.loads(content)
    except orjson.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rows(body: dict[str, Any]) -> int | None:
    meta = body.get("meta")
    if isinstance(meta, dict):
        for key in ("returned", "count"):
            if isinstance(meta.get(key), int):
                return int(meta[key])
    data = body.get("data")
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return len(value)
    return None


def _ids(body: dict[str, Any]) -> list[int]:
    data = body.get("data")
    if not isinstance(data, dict):
        return []
    for value in data.values():
        if not isinstance(value, list):
            continue
        found = [
            int(item["film_id"]) for item in value if isinstance(item, dict) and "film_id" in item
        ]
        if found:
            return found
    return []


def _call_json(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments_json}


def _message(row: sqlite3.Row) -> Message:
    raw = row["tool_calls_json"]
    calls = tuple(
        ToolCall(
            id=str(c["id"]),
            name=str(c["name"]),
            arguments_json=str(c["arguments"]),
            arguments=_parse(str(c["arguments"])) or None,
        )
        for c in (orjson.loads(raw) if raw else [])
    )
    return Message(
        role=cast("Role", row["role"]),
        content=str(row["content"]),
        tool_calls=calls,
        tool_call_id=row["tool_call_id"],
        name=row["tool_name"],
        channel="thinking" if row["channel"] == "thinking" else "answer",
    )
