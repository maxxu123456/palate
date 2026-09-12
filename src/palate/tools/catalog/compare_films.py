"""compare_films: set algebra with no model inside, so the comparison is citable facts."""

from __future__ import annotations

import sqlite3
from functools import partial

import anyio
from pydantic import BaseModel, ConfigDict, Field

from palate.errors import ToolFailure
from palate.tools.catalog.common import ids_json
from palate.tools.catalog.get_film import FilmRecord, records
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "Set algebra between two films: shared keywords, shared crew and cast, deltas on runtime, "
    "decade, language and country, plus what the user rated of each director. Returns facts, "
    "never prose."
)

_BY_DIRECTOR = (
    "select p.name, f.tmdb_id, f.title, u.rating_half from credits c "
    "join people p on p.person_id = c.person_id "
    "join films f on f.tmdb_id = c.tmdb_id "
    "join user_films u on u.tmdb_id = c.tmdb_id "
    "where c.job = 'Director' and u.rating_half is not null "
    "and p.name in (select value from json_each(?)) "
    "order by u.rating_half desc limit 20"
)


class CompareFilmsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    film_id_a: int
    film_id_b: int


class RatedByDirector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    director: str
    film_id: int
    title: str
    your_rating: float


class CompareFilmsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: FilmRecord
    b: FilmRecord
    shared_keywords: list[str] = Field(default_factory=list)
    shared_people: list[str] = Field(default_factory=list)
    shared_countries: list[str] = Field(default_factory=list)
    runtime_delta: int | None = None
    year_delta: int | None = None
    same_decade: bool = False
    same_language: bool = False
    rated_by_same_directors: list[RatedByDirector] = Field(default_factory=list)
    meta: dict[str, object] = Field(default_factory=dict)


def compare(conn: sqlite3.Connection, film_id_a: int, film_id_b: int) -> CompareFilmsResult:
    """Two records and the intersections between them, computed rather than described."""
    found = {r.film_id: r for r in records(conn, [film_id_a, film_id_b])}
    a, b = found.get(film_id_a), found.get(film_id_b)
    if a is None or b is None:
        missing = [i for i in (film_id_a, film_id_b) if i not in found]
        raise ToolFailure("not_found", f"no film in the corpus has id {missing}")
    people_a = {*a.directors, *a.writers, *a.cast}
    people_b = {*b.directors, *b.writers, *b.cast}
    directors = sorted(set(a.directors) & set(b.directors))
    rated = conn.execute(_BY_DIRECTOR, (ids_json(directors),)) if directors else ()
    return CompareFilmsResult(
        a=a,
        b=b,
        shared_keywords=sorted(set(a.keywords) & set(b.keywords)),
        shared_people=sorted(people_a & people_b),
        shared_countries=sorted(set(a.countries) & set(b.countries)),
        runtime_delta=_delta(a.runtime, b.runtime),
        year_delta=_delta(a.year, b.year),
        same_decade=_decade(a.year) == _decade(b.year) and a.year is not None,
        same_language=a.original_language == b.original_language,
        rated_by_same_directors=[
            RatedByDirector(
                director=str(r["name"]),
                film_id=int(r["tmdb_id"]),
                title=str(r["title"]),
                your_rating=int(r["rating_half"]) / 2.0,
            )
            for r in rated
        ],
        meta={"count": 2, "returned": 2},
    )


def _delta(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left - right


def _decade(year: int | None) -> int | None:
    return None if year is None else (year // 10) * 10


async def handler(args: CompareFilmsArgs, ctx: ToolContext) -> CompareFilmsResult:
    """Both records plus the set algebra between them."""
    ctx.check_deadline()
    conn = ctx.require_db().read()
    return await anyio.to_thread.run_sync(partial(compare, conn, args.film_id_a, args.film_id_b))


SPEC = ToolSpec(
    name="compare_films",
    description=DESCRIPTION,
    args_model=CompareFilmsArgs,
    result_model=CompareFilmsResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(CompareFilmsArgs(film_id_a=1398, film_id_b=551),),
    cost_hint_ms=40,
    max_result_chars=6000,
)
