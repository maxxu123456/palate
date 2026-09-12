"""Scores kept across a sweep, keyed so a changed document cannot serve a stale one."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import orjson

from palate.clock import now_iso
from palate.db.connect import Database
from palate.hashing import sha256_hex

_SELECT = (
    "select film_id, score from rerank_cache where model_key = ? and query_sha = ? "
    "and doc_version = ? and film_id in (select value from json_each(?))"
)

_INSERT = (
    "insert or replace into rerank_cache (model_key, query_sha, doc_version, film_id, score, "
    "created_at) values (?,?,?,?,?,?)"
)


def query_sha(text: str) -> str:
    """The query half of the cache key. An unconditioned run hashes its synthesised query."""
    return sha256_hex(text)


class ScoreCache:
    """One table, two methods, and a counter so a cache nobody can see never ships."""

    def __init__(self, db: Database, *, enabled: bool = True) -> None:
        self.db = db
        self.enabled = enabled
        self.hits = 0

    def get(
        self, model_key: str, query: str, doc_version: str, film_ids: Sequence[int]
    ) -> dict[int, float]:
        """Whatever this model already scored for this query against this document version."""
        if not self.enabled or not film_ids:
            return {}
        rows = self.db.read().execute(
            _SELECT,
            (model_key, query_sha(query), doc_version, orjson.dumps(list(film_ids)).decode()),
        )
        found = {int(r["film_id"]): float(r["score"]) for r in rows}
        self.hits += len(found)
        return found

    def put(
        self, model_key: str, query: str, doc_version: str, scores: Mapping[int, float]
    ) -> None:
        """Write the scores a forward pass just paid for."""
        if not self.enabled or not scores:
            return
        stamp = now_iso()
        key = query_sha(query)
        with self.db.write() as conn:
            conn.executemany(
                _INSERT,
                [
                    (model_key, key, doc_version, film_id, float(score), stamp)
                    for film_id, score in scores.items()
                ],
            )
