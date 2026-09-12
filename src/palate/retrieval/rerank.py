"""Plugging a reranker into the pipeline: the text it reads and the query it is given."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence

import orjson

from palate.providers.base import RerankCandidate
from palate.taste.profile import TasteProfile

# A cross-encoder window is 512 tokens, so the tail of a long document never reaches it anyway.
MAX_DOC_CHARS = 1200

MODE_QUERY_TERMS = 12

# Two per mode. An exemplar further down the cluster describes the cluster less well.
EXEMPLARS_PER_MODE = 2

_TEXTS = (
    "select tmdb_id, full_text, keyword_text from film_docs "
    "where tmdb_id in (select value from json_each(?))"
)

COLD_QUERY = "films this viewer would rate highly"


def load_texts(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, str]:
    """The exact text that was embedded, which is the text a reranker should also read."""
    rows = conn.execute(_TEXTS, (orjson.dumps(list(ids)).decode(),))
    return {int(r["tmdb_id"]): str(r["full_text"])[:MAX_DOC_CHARS] for r in rows}


def candidates(
    ids: Sequence[int], scores: Mapping[int, float], texts: Mapping[int, str]
) -> tuple[RerankCandidate, ...]:
    """A film with no rendered document cannot be reranked, so it keeps its stage one place."""
    return tuple(
        RerankCandidate(film_id=i, text=texts[i], prior_score=float(scores.get(i, 0.0)))
        for i in ids
        if i in texts
    )


def mode_query(profile: TasteProfile, conn: sqlite3.Connection) -> str:
    """A query for a viewer who said nothing, built from the mode labels and their exemplars.

    Out of distribution for a MS MARCO cross-encoder, which is why the two modes are reported apart.
    """
    labels = [m.label for m in profile.modes if m.label]
    exemplars = [i for m in profile.modes for i in m.exemplars[:EXEMPLARS_PER_MODE]]
    terms = [*labels, *_descriptors(conn, exemplars)]
    return ", ".join(terms) if terms else COLD_QUERY


def _descriptors(conn: sqlite3.Connection, ids: Sequence[int]) -> list[str]:
    if not ids:
        return []
    counted: Counter[str] = Counter()
    for row in conn.execute(_TEXTS, (orjson.dumps(list(ids)).decode(),)):
        counted.update(word for word in str(row["keyword_text"]).split() if len(word) > 2)
    # Count first, then alphabet, so two runs on the same history build the same query.
    ranked = sorted(counted.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:MODE_QUERY_TERMS]]
