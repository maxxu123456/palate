"""Real queries out of the user's own reviews, with every name that gives the answer away cut."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import orjson

from palate.db.connect import Database
from palate.eval.split import Fold
from palate.providers.base import ChatProvider, Message

MIN_CHARS = 60

# A generated sentence is short by design, so it is held to a lower floor than a review.
SYNTH_MIN_CHARS = 20

# Cast beyond the fourth billing is not what a review names, and stripping it costs real words.
TOP_CAST = 4

MIN_TOKEN = 3

SYNTH_MAX_TOKENS = 120

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Cut from a title before it becomes a strip token, or "the" deletes half of every review.
_STOPWORDS = frozenset(
    {"the", "and", "for", "from", "with", "that", "this", "her", "his", "you", "les", "der", "una"}
)

SYNTH_PROMPT = (
    "Write one sentence describing the mood and texture of a film someone might be after. "
    "Use only the facts below. Never name a film, a person, a place or a franchise. "
    "Answer with the sentence and nothing else.\n"
)


@dataclass(frozen=True, slots=True)
class QueryCase:
    """One query and the single film that answers it, over the whole corpus."""

    query_id: str
    text: str
    target_tmdb_id: int
    source: Literal["review", "synthetic"]
    stripped_tokens: tuple[str, ...] = ()
    n_chars: int = 0


def _json(values: Sequence[int]) -> str:
    return orjson.dumps(list(values)).decode()


_REVIEWS = (
    "select u.tmdb_id as tmdb_id, u.review_text as review_text, f.title as title, "
    "f.original_title as original_title, f.collection_name as collection_name "
    "from user_films u join films f on f.tmdb_id = u.tmdb_id "
    "where u.review_text is not null and u.review_text != '' "
    "and u.tmdb_id in (select value from json_each(?)) order by u.tmdb_id"
)

_NAMES = (
    "select c.tmdb_id as tmdb_id, p.name as name from credits c "
    "join people p on p.person_id = c.person_id "
    "where c.tmdb_id in (select value from json_each(?)) "
    f"and (c.job = 'Director' or (c.credit_kind = 'cast' and c.ord < {TOP_CAST}))"
)

_FACTS = (
    "select f.tmdb_id as tmdb_id, f.title as title, f.original_title as original_title, "
    "f.collection_name as collection_name, coalesce(f.decade, 0) as decade, "
    "coalesce(f.original_language, '') as language, coalesce(f.runtime, 0) as runtime, "
    "coalesce(f.overview, '') as overview, "
    "(select group_concat(g.name, ', ') from film_genres fg join genres g "
    "  on g.genre_id = fg.genre_id where fg.tmdb_id = f.tmdb_id) as genres, "
    "(select group_concat(k.name, ', ') from film_keywords fk join keywords k "
    "  on k.keyword_id = fk.keyword_id where fk.tmdb_id = f.tmdb_id) as keywords "
    "from films f where f.tmdb_id in (select value from json_each(?)) order by f.tmdb_id"
)


def _people_tokens(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, set[str]]:
    """Surnames of the directors and the top billing, which reviews name constantly."""
    out: dict[int, set[str]] = {}
    for row in conn.execute(_NAMES, (_json(ids),)):
        parts = _WORD.findall(str(row["name"]).casefold())
        if parts:
            out.setdefault(int(row["tmdb_id"]), set()).add(parts[-1])
    return out


def _title_tokens(*titles: str | None) -> set[str]:
    words: set[str] = set()
    for title in titles:
        if title:
            words |= {w for w in _WORD.findall(title.casefold()) if len(w) >= MIN_TOKEN}
    return words - _STOPWORDS


def strip_identity(text: str, tokens: Sequence[str]) -> str:
    """Remove the words that turn a semantic query into a lexical lookup."""
    if not tokens:
        return " ".join(text.split())
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(tokens)) + r")\b", re.I)
    return " ".join(pattern.sub(" ", text).split())


def build_review_queries(
    db: Database, fold: Fold, *, min_chars: int = MIN_CHARS
) -> list[QueryCase]:
    """The user's own words about a held-out film, with the film's own names taken out."""
    conn = db.read()
    ids = list(fold.test)
    surnames = _people_tokens(conn, ids)
    out: list[QueryCase] = []
    for row in conn.execute(_REVIEWS, (_json(ids),)):
        tmdb_id = int(row["tmdb_id"])
        tokens = _title_tokens(row["title"], row["original_title"], row["collection_name"])
        tokens |= surnames.get(tmdb_id, set())
        text = strip_identity(str(row["review_text"]), sorted(tokens))
        if len(text) < min_chars:
            continue
        out.append(
            QueryCase(
                query_id=f"review:{tmdb_id}",
                text=text,
                target_tmdb_id=tmdb_id,
                source="review",
                stripped_tokens=tuple(sorted(tokens)),
                n_chars=len(text),
            )
        )
    return out


def review_survival(db: Database, fold: Fold, *, min_chars: int = MIN_CHARS) -> tuple[int, int]:
    """Surviving queries against reviews offered. Twenty survivors is an anecdote, not a metric."""
    conn = db.read()
    offered = conn.execute(
        "select count(*) from user_films where review_text is not null and review_text != '' "
        "and tmdb_id in (select value from json_each(?))",
        (_json(list(fold.test)),),
    ).fetchone()[0]
    return len(build_review_queries(db, fold, min_chars=min_chars)), int(offered)


def _facts_block(row: sqlite3.Row) -> str:
    parts = [
        f"decade: {int(row['decade']) or 'unknown'}",
        f"language: {row['language'] or 'unknown'}",
        f"runtime: {int(row['runtime']) or 'unknown'} minutes",
        f"genres: {row['genres'] or 'unknown'}",
        f"keywords: {row['keywords'] or 'none'}",
        f"premise: {row['overview'] or 'unknown'}",
    ]
    return "\n".join(parts)


async def build_synthetic_queries(
    db: Database,
    fold: Fold,
    chat: ChatProvider,
    *,
    n: int = 200,
    model: str,
    seed: int = 0,
    min_chars: int = SYNTH_MIN_CHARS,
) -> list[QueryCase]:
    """An LLM writes the query from metadata. It scales, it is circular, and it says so."""
    conn = db.read()
    ids = sorted(fold.test)
    if len(ids) > n:
        rng = np.random.default_rng(seed)
        ids = sorted(int(i) for i in rng.choice(ids, size=n, replace=False))
    surnames = _people_tokens(conn, ids)
    out: list[QueryCase] = []
    for row in conn.execute(_FACTS, (_json(ids),)):
        tmdb_id = int(row["tmdb_id"])
        reply = await chat.complete(
            [Message(role="user", content=SYNTH_PROMPT + _facts_block(row))],
            temperature=0.0,
            max_tokens=SYNTH_MAX_TOKENS,
            seed=seed,
        )
        tokens = _title_tokens(
            row["title"], row["original_title"], row["collection_name"]
        ) | surnames.get(tmdb_id, set())
        text = strip_identity(reply.content, sorted(tokens))
        if len(text) < min_chars:
            continue
        out.append(
            QueryCase(
                query_id=f"synth:{model}:{tmdb_id}",
                text=text,
                target_tmdb_id=tmdb_id,
                source="synthetic",
                stripped_tokens=tuple(sorted(tokens)),
                n_chars=len(text),
            )
        )
    return out
