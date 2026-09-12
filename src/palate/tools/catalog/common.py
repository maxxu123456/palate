"""Shapes and lookups every tool in the catalog shares."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence

import orjson
from pydantic import BaseModel, ConfigDict, Field

from palate.errors import ToolFailure
from palate.retrieval.evidence import RecommendedFilm
from palate.retrieval.vocab import Vocabulary
from palate.taste.profile import TasteProfile

# One sentence of the overview, which is a hook and not a plot summary.
HOOK_CHARS = 180

_SENTENCE = re.compile(r"(?<=[.!?])\s")

_OVERVIEWS = (
    "select tmdb_id, coalesce(overview, '') as overview from films "
    "where tmdb_id in (select value from json_each(?))"
)


class FilmRow(BaseModel):
    """One film as every read tool returns it."""

    model_config = ConfigDict(extra="forbid")

    film_id: int
    title: str
    year: int | None = None
    directors: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    original_language: str | None = None
    runtime: int | None = None
    score: float | None = None
    mode_label: str | None = None
    in_watchlist: bool = False
    hook: str | None = None


def ids_json(values: Sequence[object]) -> str:
    """A bind parameter for the json_each idiom every lookup here uses."""
    return orjson.dumps(list(values)).decode()


def hook_of(overview: str) -> str | None:
    """First sentence of the overview, trimmed. Nothing here is generated."""
    text = " ".join(overview.split())
    if not text:
        return None
    first = _SENTENCE.split(text, maxsplit=1)[0]
    return first if len(first) <= HOOK_CHARS else first[: HOOK_CHARS - 1] + "."


def hooks(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, str | None]:
    """One hook per film, read from the database rather than written by a model."""
    if not ids:
        return {}
    rows = conn.execute(_OVERVIEWS, (ids_json(ids),))
    return {int(r["tmdb_id"]): hook_of(str(r["overview"])) for r in rows}


def mode_labels(profile: TasteProfile | None) -> dict[int, str]:
    """Mode id to its cached label, which is never used in scoring."""
    if profile is None:
        return {}
    return {m.mode_id: m.label for m in profile.modes if m.label}


def as_rows(
    films: Sequence[RecommendedFilm],
    *,
    hook_by_id: Mapping[int, str | None],
    label_by_mode: Mapping[int, str],
) -> list[FilmRow]:
    """Recommender output as the model sees it, with no free text added."""
    return [
        FilmRow(
            film_id=film.tmdb_id,
            title=film.title,
            year=film.year,
            directors=list(film.directors),
            countries=list(film.countries),
            original_language=film.original_language,
            runtime=film.runtime,
            score=round(film.score, 4),
            mode_label=label_by_mode.get(film.mode_id if film.mode_id is not None else -1),
            in_watchlist=film.in_watchlist,
            hook=hook_by_id.get(film.tmdb_id),
        )
        for film in films
    ]


def resolve_kind(vocab: Vocabulary, kind: str, names: Sequence[str], *, field: str) -> list[str]:
    """Corpus ids for each name. An unknown one comes back carrying the corpus vocabulary."""
    known = vocab.counts(kind)
    out: list[str] = []
    for name in names:
        if name in known:
            out.append(name)
            continue
        found = [m for m in vocab.resolve(name, prefer=[kind]) if m.kind == kind]
        if not found:
            close = vocab.suggest(kind, name)
            raise ToolFailure(
                "bad_arguments",
                f"unknown {kind} {name!r} in {field}",
                hint=_did_you_mean(kind, name, close),
                valid_values=tuple(label for _, label, _ in close),
            )
        out.extend(found[0].ids)
    return out


def _did_you_mean(kind: str, name: str, close: Sequence[tuple[str, str, int]]) -> str:
    if not close:
        return f"no {kind} in this corpus is called {name!r}"
    listed = ", ".join(f"{entity} ({label}, {films} films)" for entity, label, films in close)
    return f"unknown {kind} {name!r}. Closest in this corpus: {listed}"


def as_ints(values: Sequence[str]) -> frozenset[int]:
    """Numeric corpus ids, which genres, keywords and people all use."""
    return frozenset(int(v) for v in values)
