"""The structured answer. The model supplies ids and reasons, never a film name."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from palate.db.connect import Database
from palate.ground.evidence_index import EvidenceIndex
from palate.providers.base import JSONObject, ResponseFormat
from palate.tools.schema import render

_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_ROW = (
    "select f.tmdb_id, f.title, f.year, f.runtime, "
    "coalesce(u.rating_half, 0) as rating_half, u.watched_date "
    "from films f left join user_films u on u.tmdb_id = f.tmdb_id "
    "where f.tmdb_id in (select value from json_each(?))"
)

_DIRECTORS = (
    "select c.tmdb_id, p.name from credits c join people p on p.person_id = c.person_id "
    "where c.job = 'Director' and c.tmdb_id in (select value from json_each(?)) "
    "order by c.tmdb_id, c.ord"
)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    film_id: int
    why: str = Field(min_length=10, max_length=400)
    evidence_refs: list[str] = Field(default_factory=list, max_length=6)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preamble: str = Field(default="", max_length=600)
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=5)
    caveats: list[str] = Field(default_factory=list, max_length=3)
    could_not_check: list[str] = Field(default_factory=list, max_length=3)


@dataclass(frozen=True, slots=True)
class FilmFacts:
    """The row the prose is written from, which is the database and not the model."""

    film_id: int
    title: str
    year: int | None
    runtime: int | None
    directors: tuple[str, ...]
    watched: bool


@dataclass(frozen=True, slots=True)
class ResolveReport:
    """What the server dropped before the user saw anything, and why."""

    answer: StructuredAnswer
    facts: dict[int, FilmFacts]
    unresolvable_ids: tuple[int, ...]
    unseen_ids: tuple[int, ...]
    watched_ids: tuple[int, ...]

    @property
    def constraint_violation(self) -> bool:
        """Recommending a watched film is a bug, not a metric to be averaged."""
        return bool(self.watched_ids)

    @property
    def unresolvable_id_rate(self) -> float:
        """Share of the ids the model named that the corpus does not hold."""
        named = len(self.answer.recommendations) + len(self.unresolvable_ids)
        return len(self.unresolvable_ids) / named if named else 0.0


def response_format() -> ResponseFormat:
    """The json schema a provider that supports one is given."""
    schema: JSONObject = render(StructuredAnswer, style="plain")
    return ResponseFormat(kind="json_schema", schema=schema, name="palate_answer")


def parse(raw: str, *, provider_json: bool) -> StructuredAnswer:
    """The object, from a clean json body or from one fenced block inside prose."""
    text = raw if provider_json else _body(raw)
    return StructuredAnswer.model_validate_json(text)


def parse_error(exc: ValidationError) -> str:
    """The pydantic text, which is what the one repair retry is given verbatim."""
    return "\n".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])


def _body(raw: str) -> str:
    fenced = _FENCE.search(raw)
    if fenced is not None:
        return fenced.group(1)
    found = _OBJECT.search(raw)
    return found.group(0) if found is not None else raw


def facts_for(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, FilmFacts]:
    """Title, year, runtime and directors by id. Nothing here came from the model."""
    if not ids:
        return {}
    payload = orjson.dumps(list(ids)).decode()
    directors: dict[int, list[str]] = {}
    for row in conn.execute(_DIRECTORS, (payload,)):
        directors.setdefault(int(row["tmdb_id"]), []).append(str(row["name"]))
    out: dict[int, FilmFacts] = {}
    for row in conn.execute(_ROW, (payload,)):
        film_id = int(row["tmdb_id"])
        out[film_id] = FilmFacts(
            film_id=film_id,
            title=str(row["title"]),
            year=row["year"],
            runtime=row["runtime"],
            directors=tuple(directors.get(film_id, ())),
            watched=bool(row["rating_half"]) or row["watched_date"] is not None,
        )
    return out


def resolve(answer: StructuredAnswer, db: Database, evidence: EvidenceIndex) -> ResolveReport:
    """Drop what the corpus cannot confirm, before a single word reaches the user."""
    named = [r.film_id for r in answer.recommendations]
    facts = facts_for(db.read(), named)
    kept: list[Recommendation] = []
    unresolvable: list[int] = []
    unseen: list[int] = []
    watched: list[int] = []
    for recommendation in answer.recommendations:
        film = facts.get(recommendation.film_id)
        if film is None:
            unresolvable.append(recommendation.film_id)
            continue
        if film.watched:
            watched.append(recommendation.film_id)
            continue
        if not evidence.seen(recommendation.film_id):
            unseen.append(recommendation.film_id)
            continue
        kept.append(recommendation)
    return ResolveReport(
        answer=answer.model_copy(update={"recommendations": kept}),
        facts=facts,
        unresolvable_ids=tuple(unresolvable),
        unseen_ids=tuple(unseen),
        watched_ids=tuple(watched),
    )


def assemble(answer: StructuredAnswer, facts: dict[int, FilmFacts]) -> str:
    """Render prose. Title, year and director come from the database by id, never the model."""
    lines: list[str] = []
    if answer.preamble.strip():
        lines.extend([answer.preamble.strip(), ""])
    for place, recommendation in enumerate(answer.recommendations, start=1):
        film = facts.get(recommendation.film_id)
        if film is None:
            continue
        lines.append(f"{place}. {_headline(film)}")
        lines.append(f"   {recommendation.why.strip()}")
    _block(lines, "Caveats", answer.caveats)
    _block(lines, "Could not check", answer.could_not_check)
    return "\n".join(lines).strip()


def _headline(film: FilmFacts) -> str:
    bits = [film.title]
    if film.year:
        bits.append(str(film.year))
    if film.directors:
        bits.append(", ".join(film.directors[:2]))
    if film.runtime:
        bits.append(f"{film.runtime} min")
    return "  ".join(bits)


def _block(lines: list[str], title: str, rows: Sequence[str]) -> None:
    if not rows:
        return
    lines.extend(["", f"{title}:"])
    lines.extend(f"- {row.strip()}" for row in rows)


def films_json(answer: StructuredAnswer, facts: dict[int, FilmFacts]) -> tuple[JSONObject, ...]:
    """The recommendation list as the UI receives it, assembled server side."""
    out: list[JSONObject] = []
    for recommendation in answer.recommendations:
        film = facts.get(recommendation.film_id)
        if film is None:
            continue
        out.append(
            {
                "film_id": film.film_id,
                "title": film.title,
                "year": film.year,
                "runtime": film.runtime,
                "directors": list(film.directors),
                "why": recommendation.why.strip(),
                "evidence_refs": list(recommendation.evidence_refs),
            }
        )
    return tuple(out)
