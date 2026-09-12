"""One background writer. Tracing never sits on the latency path of the thing it measures."""

from __future__ import annotations

import queue
import sqlite3
import threading
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from palate.db.connect import Database
from palate.hashing import sha256_hex

type PayloadMode = Literal["off", "hashed", "full"]

# Parents before children, so one batch never inserts a span before its run.
TABLE_ORDER = (
    "runs",
    "spans",
    "prompts",
    "llm_calls",
    "embed_calls",
    "rerank_calls",
    "tool_calls",
    "http_calls",
    "grounding_claims",
)

# Upserted rather than inserted, because a run and a span are written open and again closed.
UPSERT_KEY = {"runs": "run_id", "spans": "span_id", "prompts": "prompt_sha"}

DROPPED_KEY = "rows_dropped"


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One row bound for one table. The writer never interprets it."""

    table: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Flush:
    done: threading.Event


def compress(text: str) -> bytes:
    """zlib, because a transcript is mostly repeated system prompt."""
    return zlib.compress(text.encode("utf-8"), level=6)


def decompress(blob: bytes) -> str:
    """The stored payload back, for replay and the detail pane."""
    return zlib.decompress(blob).decode("utf-8")


def payload(text: str, mode: PayloadMode) -> tuple[bytes | None, str]:
    """The stored bytes and the hash. hashed keeps the proof without keeping the text."""
    digest = sha256_hex(text)
    if mode == "full":
        return compress(text), digest
    return None, ("" if mode == "off" else digest)


class TraceStore:
    """A bounded queue and one writer connection. A full queue drops and counts, never blocks."""

    def __init__(
        self,
        db: Database,
        *,
        queue_max: int = 10_000,
        flush_ms: int = 200,
        batch: int = 64,
        payloads: PayloadMode = "hashed",
    ) -> None:
        self.db = db
        self.payloads: PayloadMode = payloads
        self.batch = batch
        self.flush_s = flush_ms / 1000.0
        self._queue: queue.Queue[TraceRow | _Flush | None] = queue.Queue(maxsize=queue_max)
        self._dropped = 0
        self._lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(target=self._drain, name="palate-traces", daemon=True)
        self._worker.start()

    @property
    def dropped(self) -> int:
        """Rows the queue refused. A silently lossy trace store is worse than none."""
        return self._dropped

    def submit(self, row: TraceRow) -> None:
        """Non-blocking. Drops and counts on a full queue, never blocks a request."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def flush(self, timeout_s: float = 5.0) -> None:
        """Wait until everything queued so far is on disk, which is what tests assert on."""
        if self._closed:
            return
        marker = _Flush(threading.Event())
        try:
            self._queue.put(marker, timeout=timeout_s)
        except queue.Full:
            return
        marker.done.wait(timeout_s)

    def close(self) -> None:
        """Drain, record what was dropped, and stop the writer."""
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=5.0)
        if self._dropped:
            self._record_drops()

    def _drain(self) -> None:
        pending: list[TraceRow] = []
        while True:
            try:
                item = self._queue.get(timeout=self.flush_s)
            except queue.Empty:
                self._write(pending)
                pending = []
                continue
            if item is None:
                self._write(pending)
                return
            if isinstance(item, _Flush):
                self._write(pending)
                pending = []
                item.done.set()
                continue
            pending.append(item)
            if len(pending) >= self.batch:
                self._write(pending)
                pending = []

    def _write(self, rows: Sequence[TraceRow]) -> None:
        if not rows:
            return
        grouped: dict[str, list[TraceRow]] = {}
        for row in rows:
            grouped.setdefault(row.table, []).append(row)
        with self.db.write() as conn:
            for table in TABLE_ORDER:
                for row in grouped.get(table, ()):
                    conn.execute(*_statement(table, row.values))

    def _record_drops(self) -> None:
        with self.db.write() as conn:
            conn.execute(
                "insert into trace_stats (key, value) values (?, ?) "
                "on conflict(key) do update set value = trace_stats.value + excluded.value",
                (DROPPED_KEY, self._dropped),
            )


def _statement(table: str, values: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    columns = list(values)
    placeholders = ",".join("?" for _ in columns)
    sql = f"insert into {table} ({','.join(columns)}) values ({placeholders})"
    key = UPSERT_KEY.get(table)
    if key is not None:
        updates = ",".join(f"{c} = excluded.{c}" for c in columns if c != key)
        sql += f" on conflict({key}) do update set {updates}"
    return sql, tuple(values.values())


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Whatever the store has counted, for palate traces status."""
    return {str(r["key"]): int(r["value"]) for r in conn.execute("select * from trace_stats")}
