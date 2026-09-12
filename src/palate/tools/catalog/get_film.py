"""get_film: the record the model must read before it describes anything."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from functools import partial

import anyio
from pydantic import BaseModel, ConfigDict, Field

from palate.errors import ToolFailure
from palate.tools.catalog.common import ids_json
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "Full record for up to 5 films by TMDB id. Returns the overview with character offsets, "
    "which is the only permitted source for a plot claim. Call this before describing any film."
)

CAST_SHOWN = 8

_FILMS = (
    "select f.tmdb_id, f.title, f.original_title, f.year, f.runtime, f.original_language, "
    "coalesce(f.overview, '') as overview, coalesce(f.tagline, '') as tagline, "
    "f.vote_average, f.vote_count, f.collection_name, "
    "coalesce(d.overview_offset, -1) as overview_offset, "
    "u.rating_half, u.watched_date, coalesce(u.in_watchlist, 0) as in_watchlist "
    "from films f "
    "left join film_docs d on d.tmdb_id = f.tmdb_id "
    "left join user_films u on u.tmdb_id = f.tmdb_id "
    "where f.tmdb_id in (select value from json_each(?))"
)

_CREDITS = (
    "select c.tmdb_id, c.credit_kind, coalesce(c.job, '') as job, c.ord, p.name "
    "from credits c join people p on p.person_id = c.person_id "
    "where c.tmdb_id in (select value from json_each(?)) order by c.tmdb_id, c.ord"
)

_KEYWORDS = (
    "select fk.tmdb_id, k.name from film_keywords fk "
    "join keywords k on k.keyword_id = fk.keyword_id "
    "where fk.tmdb_id in (select value from json_each(?)) order by fk.tmdb_id, k.name"
)

_GENRES = (
    "select fg.tmdb_id, g.name from film_genres fg join genres g on g.genre_id = fg.genre_id "
    "where fg.tmdb_id in (select value from json_each(?)) order by fg.tmdb_id, g.name"
)

_COUNTRIES = (
    "select fc.tmdb_id, co.name from film_countries fc "
    "join countries co on co.iso_3166_1 = fc.iso_3166_1 "
    "where fc.tmdb_id in (select value from json_each(?)) order by fc.tmdb_id, co.name"
)


class GetFilmArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    film_ids: list[int] = Field(min_length=1, max_length=5)


class FilmRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    film_id: int
    title: str
    original_title: str | None = None
    year: int | None = None
    runtime: int | None = None
    original_language: str | None = None
    overview: str = ""
    overview_offset: int = -1
    overview_length: int = 0
    tagline: str = ""
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    collection: str | None = None
    vote_average: float | None = None
    vote_count: int = 0
    your_rating: float | None = None
    watched_date: str | None = None
    in_watchlist: bool = False


class GetFilmResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    films: list[FilmRecord]
    meta: dict[str, object] = Field(default_factory=dict)


def _grouped(
    conn: sqlite3.Connection, sql: str, ids: Sequence[int]
) -> dict[int, list[sqlite3.Row]]:
    out: dict[int, list[sqlite3.Row]] = {}
    for row in conn.execute(sql, (ids_json(ids),)):
        out.setdefault(int(row["tmdb_id"]), []).append(row)
    return out


def records(conn: sqlite3.Connection, ids: Sequence[int]) -> list[FilmRecord]:
    """Every field from the database, so nothing about a film comes from the model."""
    credits = _grouped(conn, _CREDITS, ids)
    keywords = _grouped(conn, _KEYWORDS, ids)
    genres = _grouped(conn, _GENRES, ids)
    countries = _grouped(conn, _COUNTRIES, ids)
    out: list[FilmRecord] = []
    for row in conn.execute(_FILMS, (ids_json(ids),)):
        tmdb_id = int(row["tmdb_id"])
        people = credits.get(tmdb_id, [])
        overview = str(row["overview"])
        out.append(
            FilmRecord(
                film_id=tmdb_id,
                title=str(row["title"]),
                original_title=row["original_title"],
                year=row["year"],
                runtime=row["runtime"],
                original_language=row["original_language"],
                overview=overview,
                overview_offset=int(row["overview_offset"]),
                overview_length=len(overview),
                tagline=str(row["tagline"]),
                genres=[str(r["name"]) for r in genres.get(tmdb_id, [])],
                keywords=[str(r["name"]) for r in keywords.get(tmdb_id, [])],
                countries=[str(r["name"]) for r in countries.get(tmdb_id, [])],
                directors=[str(r["name"]) for r in people if r["job"] == "Director"],
                writers=[str(r["name"]) for r in people if r["job"] in ("Writer", "Screenplay")],
                cast=[str(r["name"]) for r in people if r["credit_kind"] == "cast"][:CAST_SHOWN],
                collection=row["collection_name"],
                vote_average=row["vote_average"],
                vote_count=int(row["vote_count"]),
                your_rating=None if row["rating_half"] is None else int(row["rating_half"]) / 2.0,
                watched_date=row["watched_date"],
                in_watchlist=bool(row["in_watchlist"]),
            )
        )
    return out


async def handler(args: GetFilmArgs, ctx: ToolContext) -> GetFilmResult:
    """Read the record. An id the corpus does not hold is a not_found, never a guess."""
    ctx.check_deadline()
    conn = ctx.require_db().read()
    found = await anyio.to_thread.run_sync(partial(records, conn, args.film_ids))
    missing = sorted(set(args.film_ids) - {f.film_id for f in found})
    if not found:
        raise ToolFailure(
            "not_found",
            f"no film in the corpus has id {missing}",
            hint="check the id against a search result, and do not recommend a film you invented",
        )
    return GetFilmResult(
        films=found, meta={"returned": len(found), "missing_ids": missing, "count": len(found)}
    )


SPEC = ToolSpec(
    name="get_film",
    description=DESCRIPTION,
    args_model=GetFilmArgs,
    result_model=GetFilmResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(GetFilmArgs(film_ids=[1398]),),
    cost_hint_ms=30,
    max_result_chars=6000,
)
