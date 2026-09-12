"""Three surfaces over traces.db, because they answer three different questions."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from palate.clock import now
from palate.db.connect import Database
from palate.obs.store import DROPPED_KEY, stats

_WINDOW = re.compile(r"^(\d+)([hdm])$")

_UNITS = {"h": "hours", "d": "days", "m": "minutes"}

_RUNS = (
    "select run_id, kind, session_id, started_at, latency_ms, status, turns, "
    "total_tokens_in, total_tokens_out, total_cost_usd, cost_complete from runs "
    "where started_at >= ? and (? = '' or kind = ?) order by started_at desc limit ?"
)

_SPANS = (
    "select span_id, parent_id, name, kind, seq, latency_ms, status, error_type, attrs_json "
    "from spans where run_id = ? order by seq"
)

_COSTS = (
    "select {group_by} as bucket, count(*) as calls, sum(tokens_in) as tokens_in, "
    "sum(tokens_out) as tokens_out, sum(coalesce(cost_usd, 0)) as usd, "
    "sum(case when cost_source = 'unknown' then 1 else 0 end) as unknown_calls, "
    "sum(case when cost_source = 'unknown' then tokens_in + tokens_out else 0 end) "
    "as unknown_tokens from llm_calls join spans using (span_id) "
    "where spans.started_at >= ? group by bucket order by usd desc"
)

GROUPS = {"model": "request_model", "provider": "provider", "run": "llm_calls.run_id"}


@dataclass(frozen=True, slots=True)
class RunRow:
    """One run as the listing shows it."""

    run_id: str
    kind: str
    session_id: str | None
    started_at: str
    latency_ms: float | None
    status: str
    turns: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cost_complete: bool


@dataclass(frozen=True, slots=True)
class SpanRow:
    """One span, with the depth the tree renders it at."""

    span_id: str
    parent_id: str | None
    name: str
    kind: str
    seq: int
    latency_ms: float | None
    status: str
    error_type: str | None
    depth: int = 0


@dataclass(frozen=True, slots=True)
class CostRow:
    """One bucket of the rollup, with the unknown half kept separate on purpose."""

    bucket: str
    calls: int
    tokens_in: int
    tokens_out: int
    usd: float
    unknown_calls: int
    unknown_tokens: int


@dataclass(frozen=True, slots=True)
class Status:
    """What palate traces status prints."""

    runs: int
    spans: int
    llm_calls: int
    dropped: int
    db_bytes: int
    oldest: str | None
    counters: dict[str, int] = field(default_factory=dict)


def since(window: str) -> str:
    """A window like 24h or 7d as the timestamp everything here filters on."""
    match = _WINDOW.match(window.strip())
    if match is None:
        raise ValueError(f"window {window!r} is not a number followed by h, d or m")
    delta = timedelta(**{_UNITS[match.group(2)]: int(match.group(1))})
    return (now() - delta).isoformat(timespec="microseconds")


def runs(db: Database, *, window: str = "24h", kind: str = "", limit: int = 50) -> list[RunRow]:
    """What ran."""
    rows = db.read().execute(_RUNS, (since(window), kind, kind, limit))
    return [
        RunRow(
            run_id=str(r["run_id"]),
            kind=str(r["kind"]),
            session_id=r["session_id"],
            started_at=str(r["started_at"]),
            latency_ms=r["latency_ms"],
            status=str(r["status"]),
            turns=int(r["turns"]),
            tokens_in=int(r["total_tokens_in"]),
            tokens_out=int(r["total_tokens_out"]),
            cost_usd=float(r["total_cost_usd"]),
            cost_complete=bool(r["cost_complete"]),
        )
        for r in rows
    ]


def tree(db: Database, run_id: str) -> list[SpanRow]:
    """The span tree for one run, in the order the run produced it."""
    rows = list(db.read().execute(_SPANS, (run_id,)))
    depth: dict[str | None, int] = {None: -1}
    out: list[SpanRow] = []
    for row in rows:
        parent = row["parent_id"]
        level = depth.get(parent, 0) + 1
        depth[str(row["span_id"])] = level
        out.append(
            SpanRow(
                span_id=str(row["span_id"]),
                parent_id=parent,
                name=str(row["name"]),
                kind=str(row["kind"]),
                seq=int(row["seq"]),
                latency_ms=row["latency_ms"],
                status=str(row["status"]),
                error_type=row["error_type"],
                depth=level,
            )
        )
    return out


def costs(db: Database, *, window: str = "7d", group_by: str = "model") -> list[CostRow]:
    """The rollup. The unknown line is never folded into the total."""
    column = GROUPS.get(group_by)
    if column is None:
        raise ValueError(f"group by one of {', '.join(GROUPS)}")
    rows = db.read().execute(_COSTS.format(group_by=column), (since(window),))
    return [
        CostRow(
            bucket=str(r["bucket"]),
            calls=int(r["calls"]),
            tokens_in=int(r["tokens_in"] or 0),
            tokens_out=int(r["tokens_out"] or 0),
            usd=float(r["usd"] or 0.0),
            unknown_calls=int(r["unknown_calls"] or 0),
            unknown_tokens=int(r["unknown_tokens"] or 0),
        )
        for r in rows
    ]


def status(db: Database) -> Status:
    """Queue losses, table sizes and the retention edge."""
    conn = db.read()
    counters = stats(conn)
    oldest = conn.execute("select min(started_at) from runs").fetchone()[0]
    return Status(
        runs=_count(conn, "runs"),
        spans=_count(conn, "spans"),
        llm_calls=_count(conn, "llm_calls"),
        dropped=counters.get(DROPPED_KEY, 0),
        db_bytes=db.path.stat().st_size if db.path.exists() else 0,
        oldest=oldest,
        counters=counters,
    )


def gc(db: Database, *, older_than: str = "30d") -> int:
    """Delete runs past the retention window. Cascades, and only ever touches traces.db."""
    cutoff = since(older_than)
    with db.write() as conn:
        cursor = conn.execute("delete from runs where started_at < ?", (cutoff,))
    return int(cursor.rowcount)


def bars(spans: Sequence[SpanRow], width: int = 24) -> Iterator[tuple[SpanRow, str]]:
    """A proportional bar per span, so the expensive one is visible without reading numbers."""
    longest = max((s.latency_ms or 0.0 for s in spans), default=0.0)
    for span in spans:
        share = (span.latency_ms or 0.0) / longest if longest else 0.0
        yield span, "#" * max(int(share * width), 1 if span.latency_ms else 0)


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"select count(*) from {table}").fetchone()[0])
