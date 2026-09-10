"""Building a vec0 index for one fingerprint, then flipping to it in one transaction."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import orjson

from palate.clock import now_iso
from palate.db.connect import Database
from palate.db.sqlvec import serialize_f32
from palate.errors import IndexIncomplete
from palate.index.fts import target_ids
from palate.index.vecstore import VecRow, VecStore
from palate.index.verify import VerifyReport, table_name, verify
from palate.providers.base import Embedder, Vector
from palate.providers.fingerprint import CANARY_TEXT, EmbeddingFingerprint, unpack_f32

BATCH = 64


@dataclass(frozen=True, slots=True)
class BuildReport:
    """What one build did, which is what the README quotes."""

    index_id: str
    table_name: str
    dim: int
    n_targets: int
    n_embedded: int
    n_cached: int
    n_vectors: int
    activated: bool
    report: VerifyReport


@dataclass(frozen=True, slots=True)
class _Pending:
    tmdb_id: int
    doc_sha: str
    text: str


_DOCS = (
    "select d.tmdb_id, d.doc_sha, d.full_text, e.doc_sha as embedded_sha from film_docs d "
    "left join film_embeddings e on e.tmdb_id = d.tmdb_id and e.index_id = ? "
    "where d.tmdb_id in (select value from json_each(?))"
)

_META = (
    "select f.tmdb_id, f.year, f.decade, f.runtime, f.vote_count, f.adult, f.original_language, "
    "coalesce(s.is_animation, 0) as is_animation, "
    "coalesce(s.is_documentary, 0) as is_documentary, "
    "coalesce(s.has_overview, 0) as has_overview, "
    "coalesce(m.eligible, 0) as in_corpus, "
    "case when u.watched_date is not null or u.rating_half is not null then 1 else 0 end "
    "as is_watched "
    "from films f "
    "left join film_stats s on s.tmdb_id = f.tmdb_id "
    "left join corpus_members m on m.tmdb_id = f.tmdb_id "
    "left join user_films u on u.tmdb_id = f.tmdb_id "
    "where f.tmdb_id in (select value from json_each(?))"
)


async def build(
    db: Database,
    embedder: Embedder,
    *,
    rebuild: bool = False,
    only_missing: bool = True,
    activate: bool = True,
    batch: int = BATCH,
) -> BuildReport:
    """Create vec_films_<index_id>, fill it, then flip active_index in one transaction."""
    fingerprint = await embedder.ready()
    index_id = fingerprint.key
    table = table_name(index_id)
    _open_index(db, fingerprint, table)

    conn = db.read()
    targets = target_ids(conn)
    pending = _split(conn, index_id, targets, rebuild=rebuild, only_missing=only_missing)
    metadata = _metadata(conn, targets)

    embedded = 0
    cached = 0
    size = max(1, min(batch, embedder.max_batch))
    for start in range(0, len(pending), size):
        chunk = pending[start : start + size]
        vectors, hits = await _vectors(db, embedder, index_id, chunk)
        cached += hits
        embedded += len(chunk) - hits
        _persist(db, index_id, table, fingerprint, chunk, vectors, metadata)

    await _store_canary(db, embedder, index_id)
    report = verify(db, index_id)
    activated = _finish(db, index_id, table, fingerprint, report, activate=activate)
    return BuildReport(
        index_id=index_id,
        table_name=table,
        dim=fingerprint.dim,
        n_targets=len(targets),
        n_embedded=embedded,
        n_cached=cached,
        n_vectors=report.n_present,
        activated=activated,
        report=report,
    )


def _open_index(db: Database, fingerprint: EmbeddingFingerprint, table: str) -> None:
    with db.write() as conn:
        row = fingerprint.to_row()
        conn.execute(
            "insert into embedding_indexes (index_id, provider, model_id, revision, dim, "
            "normalized, query_prompt, document_prompt, pooling, doc_template_version, "
            "table_name, canary_text, canary_vec, status, created_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?, 'building', ?) "
            # A ready index keeps serving while it is topped up. Only a broken one reopens.
            "on conflict(index_id) do update set status = case "
            "when embedding_indexes.status = 'ready' then 'ready' else 'building' end",
            (
                fingerprint.key,
                row["provider"],
                row["model_id"],
                row["revision"],
                row["dim"],
                int(row["normalized"]),
                row["query_prompt"],
                row["document_prompt"],
                row["pooling"],
                row["doc_template_version"],
                table,
                CANARY_TEXT,
                b"",
                now_iso(),
            ),
        )
        VecStore(conn, table=table, dim=fingerprint.dim).create()


def _split(
    conn: sqlite3.Connection,
    index_id: str,
    targets: Sequence[int],
    *,
    rebuild: bool,
    only_missing: bool,
) -> list[_Pending]:
    rows = conn.execute(_DOCS, (index_id, orjson.dumps(list(targets)).decode())).fetchall()
    pending: list[_Pending] = []
    for row in rows:
        moved = row["embedded_sha"] != row["doc_sha"]
        if rebuild or not only_missing or moved:
            pending.append(
                _Pending(int(row["tmdb_id"]), str(row["doc_sha"]), str(row["full_text"]))
            )
    return pending


def _metadata(conn: sqlite3.Connection, targets: Sequence[int]) -> dict[int, VecRow]:
    rows = conn.execute(_META, (orjson.dumps(list(targets)).decode(),))
    return {
        int(r["tmdb_id"]): VecRow(
            film_id=int(r["tmdb_id"]),
            embedding=(),
            decade=-1 if r["decade"] is None else int(r["decade"]),
            year=-1 if r["year"] is None else int(r["year"]),
            runtime=-1 if r["runtime"] is None else int(r["runtime"]),
            vote_count=int(r["vote_count"]),
            original_language=str(r["original_language"] or ""),
            in_corpus=int(r["in_corpus"]),
            is_watched=int(r["is_watched"]),
            is_adult=int(r["adult"]),
            is_animation=int(r["is_animation"]),
            is_documentary=int(r["is_documentary"]),
            has_overview=int(r["has_overview"]),
        )
        for r in rows
    }


async def _vectors(
    db: Database, embedder: Embedder, index_id: str, chunk: Sequence[_Pending]
) -> tuple[list[Vector], int]:
    conn = db.read()
    cached: dict[str, Vector] = {}
    for item in chunk:
        row = conn.execute(
            "select vec from embedding_cache where fingerprint_key = ? and text_sha = ?",
            (index_id, item.doc_sha),
        ).fetchone()
        if row is not None:
            cached[item.doc_sha] = unpack_f32(bytes(row["vec"]))
    missing = [item for item in chunk if item.doc_sha not in cached]
    if missing:
        batch = await embedder.embed_documents([item.text for item in missing])
        for item, vector in zip(missing, batch.vectors, strict=True):
            cached[item.doc_sha] = vector
    return [cached[item.doc_sha] for item in chunk], len(chunk) - len(missing)


def _persist(
    db: Database,
    index_id: str,
    table: str,
    fingerprint: EmbeddingFingerprint,
    chunk: Sequence[_Pending],
    vectors: Sequence[Vector],
    metadata: dict[int, VecRow],
) -> None:
    stamp = now_iso()
    with db.write() as conn:
        store = VecStore(conn, table=table, dim=fingerprint.dim)
        rows = []
        for item, vector in zip(chunk, vectors, strict=True):
            base = metadata.get(item.tmdb_id, VecRow(item.tmdb_id, ()))
            rows.append(
                VecRow(
                    film_id=item.tmdb_id,
                    embedding=vector,
                    decade=base.decade,
                    year=base.year,
                    runtime=base.runtime,
                    vote_count=base.vote_count,
                    original_language=base.original_language,
                    in_corpus=base.in_corpus,
                    is_watched=base.is_watched,
                    is_adult=base.is_adult,
                    is_animation=base.is_animation,
                    is_documentary=base.is_documentary,
                    has_overview=base.has_overview,
                )
            )
        store.upsert(rows)
        conn.executemany(
            "insert into embedding_cache (fingerprint_key, text_sha, vec, created_at) "
            "values (?,?,?,?) on conflict(fingerprint_key, text_sha) do update set "
            "hits = embedding_cache.hits + 1",
            [
                (index_id, i.doc_sha, serialize_f32(v), stamp)
                for i, v in zip(chunk, vectors, strict=True)
            ],
        )
        conn.executemany(
            "insert into film_embeddings (index_id, tmdb_id, doc_sha, embedded_at) "
            "values (?,?,?,?) on conflict(index_id, tmdb_id) do update set "
            "doc_sha = excluded.doc_sha, embedded_at = excluded.embedded_at",
            [(index_id, i.tmdb_id, i.doc_sha, stamp) for i in chunk],
        )


async def _store_canary(db: Database, embedder: Embedder, index_id: str) -> None:
    """The only thing that catches a remote host swapping weights under a stable name."""
    vector = await embedder.embed_query(CANARY_TEXT)
    with db.write() as conn:
        conn.execute(
            "update embedding_indexes set canary_text = ?, canary_vec = ?, "
            "canary_checked_at = ? where index_id = ?",
            (CANARY_TEXT, serialize_f32(vector), now_iso(), index_id),
        )


def _finish(
    db: Database,
    index_id: str,
    table: str,
    fingerprint: EmbeddingFingerprint,
    report: VerifyReport,
    *,
    activate: bool,
) -> bool:
    if report.missing:
        with db.write() as conn:
            conn.execute(
                "update embedding_indexes set status = 'failed' where index_id = ?", (index_id,)
            )
        raise IndexIncomplete(report.n_expected, report.n_present)
    with db.write() as conn:
        conn.execute(
            "update embedding_indexes set status = 'ready', completed_at = ?, n_vectors = ? "
            "where index_id = ?",
            (now_iso(), report.n_present, index_id),
        )
        if activate:
            conn.execute(
                "insert into active_index (only_row, index_id) values (1, ?) "
                "on conflict(only_row) do update set index_id = excluded.index_id",
                (index_id,),
            )
    return activate
