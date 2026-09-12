"""check_watched: belt and braces on corpus exclusion, and it catches invented titles."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from functools import partial

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from palate.tools.catalog.common import ids_json
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "Batch membership test against the user's history and watchlist. Accepts ids or titles. "
    "A title that resolves to nothing means the film is not in the corpus and must not be "
    "recommended."
)

_BY_ID = (
    "select f.tmdb_id, f.title, f.year, u.rating_half, u.watched_date, "
    "coalesce(u.in_watchlist, 0) as in_watchlist from films f "
    "left join user_films u on u.tmdb_id = f.tmdb_id "
    "where f.tmdb_id in (select value from json_each(?))"
)

_BY_TITLE = (
    "select f.tmdb_id, f.title, f.year, u.rating_half, u.watched_date, "
    "coalesce(u.in_watchlist, 0) as in_watchlist from films f "
    "left join user_films u on u.tmdb_id = f.tmdb_id "
    "where f.title = ? collate nocase or f.original_title = ? collate nocase "
    "order by f.vote_count desc limit 1"
)


class CheckWatchedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    film_ids: list[int] = Field(default_factory=list, max_length=25)
    titles: list[str] = Field(default_factory=list, max_length=25)

    @model_validator(mode="after")
    def _something(self) -> CheckWatchedArgs:
        if not self.film_ids and not self.titles:
            raise ValueError("pass film_ids or titles")
        return self


class WatchStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asked: str
    film_id: int | None = None
    title: str | None = None
    year: int | None = None
    in_corpus: bool = False
    watched: bool = False
    your_rating: float | None = None
    in_watchlist: bool = False


class CheckWatchedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    films: list[WatchStatus]
    meta: dict[str, object] = Field(default_factory=dict)


def _status(asked: str, row: sqlite3.Row | None) -> WatchStatus:
    if row is None:
        return WatchStatus(asked=asked)
    rating = row["rating_half"]
    return WatchStatus(
        asked=asked,
        film_id=int(row["tmdb_id"]),
        title=str(row["title"]),
        year=row["year"],
        in_corpus=True,
        watched=rating is not None or row["watched_date"] is not None,
        your_rating=None if rating is None else int(rating) / 2.0,
        in_watchlist=bool(row["in_watchlist"]),
    )


def lookup(
    conn: sqlite3.Connection, film_ids: Sequence[int], titles: Sequence[str]
) -> CheckWatchedResult:
    """Ids and titles both, with a title nobody can resolve reported rather than guessed."""
    found = {int(r["tmdb_id"]): r for r in conn.execute(_BY_ID, (ids_json(film_ids),))}
    out = [_status(str(i), found.get(i)) for i in film_ids]
    for title in titles:
        out.append(_status(title, conn.execute(_BY_TITLE, (title, title)).fetchone()))
    unknown = [s.asked for s in out if not s.in_corpus]
    return CheckWatchedResult(
        films=out,
        meta={
            "count": len(out),
            "returned": len(out),
            "not_in_corpus": unknown,
            "watched": sum(1 for s in out if s.watched),
        },
    )


async def handler(args: CheckWatchedArgs, ctx: ToolContext) -> CheckWatchedResult:
    """One read per title plus one for every id."""
    ctx.check_deadline()
    conn = ctx.require_db().read()
    return await anyio.to_thread.run_sync(partial(lookup, conn, args.film_ids, args.titles))


SPEC = ToolSpec(
    name="check_watched",
    description=DESCRIPTION,
    args_model=CheckWatchedArgs,
    result_model=CheckWatchedResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(CheckWatchedArgs(titles=["Stalker"]),),
    cost_hint_ms=20,
    max_result_chars=3000,
)
