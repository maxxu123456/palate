"""film_docs and the BM25 channel over it."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from palate.clock import now_iso
from palate.db.connect import Database
from palate.index.documents import DOC_TEMPLATE_VERSION, RenderedDoc, load_inputs, render

# title, people, keyword, overview. Title outranks plot because a title match is a fact.
BM25_WEIGHTS = (3.0, 2.5, 2.0, 1.0)

BATCH = 500

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class DocsReport:
    """What one film_docs rebuild changed."""

    n_seen: int
    n_written: int
    by_kind: dict[str, int]

    @property
    def n_unchanged(self) -> int:
        return self.n_seen - self.n_written


@dataclass(frozen=True, slots=True)
class Bm25Hit:
    """One FTS5 match, already sign corrected."""

    tmdb_id: int
    score: float


_UPSERT = (
    "insert into film_docs (tmdb_id, doc_template_version, doc_kind, title_text, people_text, "
    "keyword_text, overview_text, full_text, overview_offset, doc_sha, built_at) "
    "values (?,?,?,?,?,?,?,?,?,?,?) on conflict(tmdb_id) do update set "
    "doc_template_version = excluded.doc_template_version, doc_kind = excluded.doc_kind, "
    "title_text = excluded.title_text, people_text = excluded.people_text, "
    "keyword_text = excluded.keyword_text, overview_text = excluded.overview_text, "
    "full_text = excluded.full_text, overview_offset = excluded.overview_offset, "
    "doc_sha = excluded.doc_sha, built_at = excluded.built_at"
)


def _row(tmdb_id: int, doc: RenderedDoc, built_at: str) -> tuple[object, ...]:
    return (
        tmdb_id,
        DOC_TEMPLATE_VERSION,
        doc.doc_kind,
        doc.title_text,
        doc.people_text,
        doc.keyword_text,
        doc.overview_text,
        doc.full_text,
        doc.overview_offset,
        doc.doc_sha,
        built_at,
    )


def target_ids(conn: sqlite3.Connection, *, corpus_only: bool = True) -> list[int]:
    """Films that should have a document. History stays in even when it is ineligible."""
    if corpus_only:
        sql = (
            "select tmdb_id from corpus_members where eligible = 1 "
            "union select tmdb_id from user_films order by tmdb_id"
        )
    else:
        sql = "select tmdb_id from films order by tmdb_id"
    return [int(r["tmdb_id"]) for r in conn.execute(sql)]


def rebuild(
    db: Database,
    *,
    ids: Sequence[int] | None = None,
    include_credits: bool = True,
    corpus_only: bool = True,
) -> DocsReport:
    """Render every target film and write only the documents whose text moved."""
    conn = db.read()
    targets = list(ids) if ids is not None else target_ids(conn, corpus_only=corpus_only)
    # The template version is half the key: a split that moves text between columns
    # leaves full_text, and therefore doc_sha, untouched.
    known = {
        int(r["tmdb_id"]): str(r["doc_sha"])
        for r in conn.execute(
            "select tmdb_id, doc_sha from film_docs where doc_template_version = ?",
            (DOC_TEMPLATE_VERSION,),
        )
    }
    built_at = now_iso()
    by_kind: dict[str, int] = {}
    written = 0
    for start in range(0, len(targets), BATCH):
        chunk = targets[start : start + BATCH]
        rows = []
        for item in load_inputs(conn, chunk):
            doc = render(item, include_credits=include_credits)
            by_kind[doc.doc_kind] = by_kind.get(doc.doc_kind, 0) + 1
            if known.get(item.film.tmdb_id) == doc.doc_sha:
                continue
            rows.append(_row(item.film.tmdb_id, doc, built_at))
        if rows:
            with db.write() as write_conn:
                write_conn.executemany(_UPSERT, rows)
            written += len(rows)
    return DocsReport(n_seen=len(targets), n_written=written, by_kind=by_kind)


def prune(db: Database) -> int:
    """Drop documents for films that left the target set, which also clears their tokens."""
    with db.write() as conn:
        cursor = conn.execute(
            "delete from film_docs where tmdb_id not in "
            "(select tmdb_id from corpus_members where eligible = 1 "
            "union select tmdb_id from user_films)"
        )
        return int(cursor.rowcount or 0)


def match_expression(text: str, *, exclude: Sequence[str] = ()) -> str:
    """Turn free text into an fts5 MATCH string, with negation handled here and nowhere else."""
    wanted: list[str] = []
    unwanted = [term for word in exclude for term in _WORD.findall(word)]
    for raw in text.split():
        negated = raw.startswith("-")
        words = _WORD.findall(raw)
        (unwanted if negated else wanted).extend(words)
    if not wanted:
        return ""
    positive = " OR ".join(f'"{w}"' for w in wanted)
    if not unwanted:
        return positive
    negative = " OR ".join(f'"{w}"' for w in unwanted)
    # fts5 binds NOT tighter than OR, so both sides need their own parentheses.
    return f"({positive}) NOT ({negative})"


def search(
    conn: sqlite3.Connection,
    text: str,
    *,
    limit: int = 300,
    exclude: Sequence[str] = (),
) -> tuple[Bm25Hit, ...]:
    """BM25 over film_docs, best first."""
    expression = match_expression(text, exclude=exclude)
    if not expression:
        return ()
    # bm25() is more negative the better the match, so the sign flips once, here.
    rows = conn.execute(
        "select rowid as tmdb_id, -bm25(films_fts, ?, ?, ?, ?) as score from films_fts "
        "where films_fts match ? order by score desc limit ?",
        (*BM25_WEIGHTS, expression, limit),
    )
    return tuple(Bm25Hit(int(r["tmdb_id"]), float(r["score"])) for r in rows)
