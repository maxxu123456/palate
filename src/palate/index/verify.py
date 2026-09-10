"""Reading an index back, and refusing one that a different embedder built."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from palate.db.connect import Database
from palate.errors import NoActiveIndex
from palate.index.fts import target_ids
from palate.index.vecstore import VecStore
from palate.providers.base import Embedder
from palate.providers.fingerprint import EmbeddingFingerprint, mismatch_message, require_match

TABLE_PREFIX = "vec_films_"


def table_name(index_id: str) -> str:
    """The runtime table one fingerprint owns."""
    return f"{TABLE_PREFIX}{index_id}"


@dataclass(frozen=True, slots=True)
class IndexRecord:
    """One embedding_indexes row, with the fingerprint already rebuilt."""

    index_id: str
    fingerprint: EmbeddingFingerprint
    table_name: str
    status: str
    n_vectors: int
    canary_text: str
    canary_vec: bytes
    canary_checked_at: str | None
    created_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """What `palate index verify` prints, per index."""

    index_id: str
    status: str
    n_expected: int
    n_present: int
    missing: int
    orphans: int
    dim: int
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def _record(row: sqlite3.Row) -> IndexRecord:
    return IndexRecord(
        index_id=str(row["index_id"]),
        fingerprint=EmbeddingFingerprint.from_row(dict(row)),
        table_name=str(row["table_name"]),
        status=str(row["status"]),
        n_vectors=int(row["n_vectors"]),
        canary_text=str(row["canary_text"]),
        canary_vec=bytes(row["canary_vec"]),
        canary_checked_at=None
        if row["canary_checked_at"] is None
        else str(row["canary_checked_at"]),
        created_at=str(row["created_at"]),
        completed_at=None if row["completed_at"] is None else str(row["completed_at"]),
    )


def load(db: Database, index_id: str) -> IndexRecord | None:
    """One index by id, or None when it was never built."""
    row = (
        db.read()
        .execute("select * from embedding_indexes where index_id = ?", (index_id,))
        .fetchone()
    )
    return None if row is None else _record(row)


def listing(db: Database) -> tuple[IndexRecord, ...]:
    """Every index ever built, newest first, because the tables are not in the migrations."""
    rows = db.read().execute("select * from embedding_indexes order by created_at desc")
    return tuple(_record(r) for r in rows)


def active_id(db: Database) -> str | None:
    """The index queries are served from, if any."""
    row = db.read().execute("select index_id from active_index where only_row = 1").fetchone()
    return None if row is None else str(row["index_id"])


def active(db: Database) -> IndexRecord:
    """The active index, raising rather than answering from nothing."""
    index_id = active_id(db)
    if index_id is None:
        raise NoActiveIndex("no embedding index is active, run: palate index build")
    record = load(db, index_id)
    if record is None:
        raise NoActiveIndex(f"active_index points at {index_id}, which has no row")
    return record


def require_active_match(db: Database, embedder: Embedder) -> IndexRecord:
    """The active index, checked against the current embedder once per open."""
    record = active(db)
    require_match(record.fingerprint, embedder.fingerprint)
    return record


async def drift(record: IndexRecord, embedder: Embedder) -> str:
    """Empty when this embedder built the index, the mismatch report when it did not."""
    current = await embedder.ready()
    if current.key == record.fingerprint.key:
        return ""
    return mismatch_message(record.fingerprint, current)


def activate(db: Database, index_id: str) -> None:
    """Point queries at a built index. A building index is never served."""
    record = load(db, index_id)
    if record is None:
        raise NoActiveIndex(f"no index {index_id}")
    if record.status != "ready":
        raise NoActiveIndex(f"index {index_id} is {record.status}, not ready")
    with db.write() as conn:
        conn.execute(
            "insert into active_index (only_row, index_id) values (1, ?) "
            "on conflict(only_row) do update set index_id = excluded.index_id",
            (index_id,),
        )


def verify(db: Database, index_id: str) -> VerifyReport:
    """Count, dimension, orphans and completeness for one index."""
    record = load(db, index_id)
    conn = db.read()
    if record is None:
        return VerifyReport(index_id, "missing", 0, 0, 0, 0, 0, ("no such index",))
    store = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
    problems: list[str] = []
    if not store.exists():
        return VerifyReport(
            index_id, record.status, 0, 0, 0, 0, record.fingerprint.dim, ("vector table is gone",)
        )
    expected = set(target_ids(conn))
    present = store.ids()
    missing = len(expected - present)
    orphans = len(present - expected)
    if record.status != "ready":
        problems.append(f"status is {record.status}")
    if missing:
        problems.append(f"{missing} films have no vector")
    if orphans:
        problems.append(f"{orphans} vectors belong to films that left the corpus")
    if record.n_vectors != len(present):
        problems.append(f"row says {record.n_vectors} vectors, the table holds {len(present)}")
    if len(record.canary_vec) != record.fingerprint.dim * 4:
        problems.append("the canary vector is the wrong width")
    return VerifyReport(
        index_id=index_id,
        status=record.status,
        n_expected=len(expected),
        n_present=len(present),
        missing=missing,
        orphans=orphans,
        dim=record.fingerprint.dim,
        problems=tuple(problems),
    )


def drop(db: Database, index_id: str) -> bool:
    """Delete an index and its table. The active one is never droppable."""
    record = load(db, index_id)
    if record is None:
        return False
    if active_id(db) == index_id:
        raise NoActiveIndex(f"index {index_id} is active, activate another one first")
    with db.write() as conn:
        VecStore(conn, table=record.table_name, dim=record.fingerprint.dim).drop()
        conn.execute("delete from embedding_indexes where index_id = ?", (index_id,))
    return True
