"""get_ratings: the only permitted source for any claim about what the user has seen."""

from __future__ import annotations

import sqlite3
from functools import partial
from typing import Any, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec, clamp

DESCRIPTION = (
    "The user's real ratings, filtered. This is the ONLY permitted source for any claim about "
    "what the user has seen or liked. The 1 and 2 star films are reachable here and carry most "
    "of the information."
)

_ORDERS = {
    "rating_desc": "u.rating_half desc, f.title",
    "rating_asc": "u.rating_half asc, f.title",
    "watched_desc": "coalesce(u.watched_date, u.logged_date) desc, f.title",
}

_BASE = (
    "select f.tmdb_id, f.title, f.year, f.original_language, f.runtime, u.rating_half, "
    "u.is_rewatch, coalesce(u.watched_date, u.logged_date) as watched_at from user_films u "
    "join films f on f.tmdb_id = u.tmdb_id where u.rating_half is not null"
)

_DIRECTOR = (
    " and exists (select 1 from credits c join people p on p.person_id = c.person_id "
    "where c.tmdb_id = f.tmdb_id and c.job = 'Director' and p.name like ?)"
)

_KEYWORD = (
    " and exists (select 1 from film_keywords fk join keywords k on k.keyword_id = fk.keyword_id "
    "where fk.tmdb_id = f.tmdb_id and k.name like ?)"
)


class GetRatingsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    director: str | None = Field(default=None, max_length=120)
    keyword: str | None = Field(default=None, max_length=80)
    language: str | None = Field(default=None, max_length=3)
    decade: int | None = Field(default=None, ge=1880, le=2030)
    rating_min: float | None = Field(default=None, ge=0.5, le=5.0)
    rating_max: float | None = Field(default=None, ge=0.5, le=5.0)
    order_by: Literal["rating_desc", "rating_asc", "watched_desc"] = "rating_desc"
    limit: int = 20

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, value: object, info: ValidationInfo) -> int:
        """Clamp rather than reject. A rejected call costs the model a whole turn."""
        return clamp(value, info, lo=1, hi=50)


class RatedFilm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    film_id: int
    title: str
    year: int | None = None
    original_language: str | None = None
    runtime: int | None = None
    your_rating: float
    is_rewatch: bool = False
    watched_at: str | None = None


class GetRatingsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratings: list[RatedFilm]
    meta: dict[str, Any] = Field(default_factory=dict)


def query(args: GetRatingsArgs) -> tuple[str, list[Any]]:
    """The one SELECT this tool runs, built from whatever clauses were asked for."""
    sql = _BASE
    params: list[Any] = []
    if args.director:
        sql += _DIRECTOR
        params.append(f"%{args.director}%")
    if args.keyword:
        sql += _KEYWORD
        params.append(f"%{args.keyword}%")
    if args.language:
        sql += " and f.original_language = ?"
        params.append(args.language)
    if args.decade is not None:
        sql += " and f.decade = ?"
        params.append((args.decade // 10) * 10)
    if args.rating_min is not None:
        sql += " and u.rating_half >= ?"
        params.append(round(args.rating_min * 2))
    if args.rating_max is not None:
        sql += " and u.rating_half <= ?"
        params.append(round(args.rating_max * 2))
    sql += f" order by {_ORDERS[args.order_by]} limit ?"
    params.append(args.limit)
    return sql, params


def read(conn: sqlite3.Connection, args: GetRatingsArgs) -> GetRatingsResult:
    """Rows straight from user_films, with the rating back in stars."""
    sql, params = query(args)
    rows = conn.execute(sql, params).fetchall()
    ratings = [
        RatedFilm(
            film_id=int(r["tmdb_id"]),
            title=str(r["title"]),
            year=r["year"],
            original_language=r["original_language"],
            runtime=r["runtime"],
            your_rating=int(r["rating_half"]) / 2.0,
            is_rewatch=bool(r["is_rewatch"]),
            watched_at=r["watched_at"],
        )
        for r in rows
    ]
    mean = sum(r.your_rating for r in ratings) / len(ratings) if ratings else 0.0
    return GetRatingsResult(
        ratings=ratings,
        meta={
            "count": len(ratings),
            "returned": len(ratings),
            "mean_rating": round(mean, 3),
            "truncated": len(ratings) == args.limit,
        },
    )


async def handler(args: GetRatingsArgs, ctx: ToolContext) -> GetRatingsResult:
    """One indexed read off the event loop."""
    ctx.check_deadline()
    conn = ctx.require_db().read()
    return await anyio.to_thread.run_sync(partial(read, conn, args))


SPEC = ToolSpec(
    name="get_ratings",
    description=DESCRIPTION,
    args_model=GetRatingsArgs,
    result_model=GetRatingsResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(GetRatingsArgs(director="Andrei Tarkovsky", limit=10),),
    cost_hint_ms=20,
    max_result_chars=4000,
)
