"""filter_films: predicates are predicates, and routing them through an embedder wastes a turn."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from palate.retrieval.candidates import HardFilters, from_preferences
from palate.retrieval.evidence import load_cards
from palate.retrieval.recommend import NO_PREFERENCES, RecommendRequest, build_store
from palate.tools.catalog.common import (
    FilmRow,
    as_ints,
    as_rows,
    hook_of,
    hooks,
    ids_json,
    mode_labels,
    resolve_kind,
)
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec, clamp

DESCRIPTION = (
    "Structural query over the corpus with no embedding call. Use for decade, runtime, language, "
    "country, genre, keyword or director constraints with no mood component. Returns only "
    "unwatched films."
)

type Order = Literal["taste", "vote_count", "vote_average", "year_desc", "year_asc", "runtime_asc"]

_SORTS: dict[str, str] = {
    "vote_count": "vote_count desc",
    "vote_average": "vote_average desc",
    "year_desc": "year desc",
    "year_asc": "year asc",
    "runtime_asc": "runtime asc",
}

_PAGE = (
    "select tmdb_id, coalesce(overview, '') as overview from films "
    "where tmdb_id in (select value from json_each(?)) order by {sort} limit ? offset ?"
)


class FilterFilmsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    include_genres: list[str] = Field(default_factory=list, max_length=5)
    exclude_genres: list[str] = Field(default_factory=list, max_length=5)
    include_keywords: list[str] = Field(default_factory=list, max_length=8)
    exclude_keywords: list[str] = Field(default_factory=list, max_length=8)
    include_languages: list[str] = Field(default_factory=list, max_length=10)
    exclude_languages: list[str] = Field(default_factory=list, max_length=10)
    include_countries: list[str] = Field(default_factory=list, max_length=10)
    exclude_countries: list[str] = Field(default_factory=list, max_length=10)
    director: str | None = Field(default=None, max_length=120)
    year_min: int | None = Field(default=None, ge=1888, le=2030)
    year_max: int | None = Field(default=None, ge=1888, le=2030)
    runtime_min: int | None = Field(default=None, ge=1, le=900)
    runtime_max: int | None = Field(default=None, ge=1, le=900)
    order_by: Order = "taste"
    apply_preferences: bool = True
    limit: int = 20
    offset: int = Field(default=0, ge=0)

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, value: object, info: ValidationInfo) -> int:
        """Clamp rather than reject. A rejected call costs the model a whole turn."""
        return clamp(value, info, lo=1, hi=50)

    @model_validator(mode="after")
    def _ranges(self) -> FilterFilmsArgs:
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError("year_min must be <= year_max")
        if self.runtime_min and self.runtime_max and self.runtime_min > self.runtime_max:
            raise ValueError("runtime_min must be <= runtime_max")
        return self


class FilterFilmsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    films: list[FilmRow]
    meta: dict[str, Any] = Field(default_factory=dict)


def hard_filters(ctx: ToolContext, args: FilterFilmsArgs) -> HardFilters:
    """Every clause the model asked for, resolved against the corpus vocabulary."""
    vocab = ctx.require_vocab()
    people = (
        resolve_kind(vocab, "person", [args.director], field="director") if args.director else []
    )
    return HardFilters(
        year_min=args.year_min,
        year_max=args.year_max,
        runtime_min=args.runtime_min,
        runtime_max=args.runtime_max,
        include_languages=frozenset(
            resolve_kind(vocab, "language", args.include_languages, field="include_languages")
        ),
        exclude_languages=frozenset(
            resolve_kind(vocab, "language", args.exclude_languages, field="exclude_languages")
        ),
        include_genres=as_ints(
            resolve_kind(vocab, "genre", args.include_genres, field="include_genres")
        ),
        exclude_genres=as_ints(
            resolve_kind(vocab, "genre", args.exclude_genres, field="exclude_genres")
        ),
        include_keywords=as_ints(
            resolve_kind(vocab, "keyword", args.include_keywords, field="include_keywords")
        ),
        exclude_keywords=as_ints(
            resolve_kind(vocab, "keyword", args.exclude_keywords, field="exclude_keywords")
        ),
        include_countries=frozenset(
            resolve_kind(vocab, "country", args.include_countries, field="include_countries")
        ),
        exclude_countries=frozenset(
            resolve_kind(vocab, "country", args.exclude_countries, field="exclude_countries")
        ),
        include_people=as_ints(people),
    )


async def handler(args: FilterFilmsArgs, ctx: ToolContext) -> FilterFilmsResult:
    """Taste order goes through the ranker. Every other order is a SQL sort."""
    ctx.check_deadline()
    filters = hard_filters(ctx, args)
    if args.order_by == "taste":
        answer = await ctx.require_recommender().recommend(
            RecommendRequest(
                n=args.limit,
                filters=filters,
                apply_preferences=args.apply_preferences,
                offset=args.offset,
            )
        )
        ids = [f.tmdb_id for f in answer.films]
        rows = as_rows(
            answer.films,
            hook_by_id=hooks(ctx.require_db().read(), ids),
            label_by_mode=mode_labels(ctx.profile),
        )
        meta = {
            "count": answer.diagnostics.after_filter,
            "returned": len(rows),
            "excluded_watched": answer.diagnostics.excluded_watched,
            "removed_by_clause": {
                k: v for k, v in answer.diagnostics.removed_by_clause.items() if v
            },
            "order_by": args.order_by,
        }
        return FilterFilmsResult(films=rows, meta=meta)
    return await anyio.to_thread.run_sync(partial(_sorted_page, args, ctx, filters))


def _sorted_page(
    args: FilterFilmsArgs, ctx: ToolContext, filters: HardFilters
) -> FilterFilmsResult:
    db = ctx.require_db()
    store = build_store(db, ctx.settings.retrieval)
    stated = (
        ctx.require_prefs().as_filter(ctx.session_id)
        if args.apply_preferences and ctx.prefs is not None
        else NO_PREFERENCES
    )
    merged = filters.merge(from_preferences(store.conn, stated))
    allowed = store.allow(merged)
    sql = _PAGE.format(sort=_SORTS[args.order_by])
    rows = store.conn.execute(
        sql, (ids_json(sorted(allowed.ids)), args.limit, args.offset)
    ).fetchall()
    ids = [int(r["tmdb_id"]) for r in rows]
    overview = {int(r["tmdb_id"]): str(r["overview"]) for r in rows}
    cards = load_cards(store.conn, ids)
    films = [
        FilmRow(
            film_id=tmdb_id,
            title=cards[tmdb_id].title,
            year=cards[tmdb_id].year,
            directors=list(cards[tmdb_id].directors),
            countries=list(cards[tmdb_id].countries),
            original_language=cards[tmdb_id].original_language,
            runtime=cards[tmdb_id].runtime,
            in_watchlist=cards[tmdb_id].in_watchlist,
            hook=hook_of(overview[tmdb_id]),
        )
        for tmdb_id in ids
        if tmdb_id in cards
    ]
    meta = {
        "count": len(allowed.ids),
        "returned": len(films),
        "excluded_watched": allowed.excluded_watched,
        "removed_by_clause": {k: v for k, v in allowed.removed_by_clause.items() if v},
        "order_by": args.order_by,
    }
    return FilterFilmsResult(films=films, meta=meta)


SPEC = ToolSpec(
    name="filter_films",
    description=DESCRIPTION,
    args_model=FilterFilmsArgs,
    result_model=FilterFilmsResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(FilterFilmsArgs(exclude_languages=["ru"], year_max=1990, limit=20),),
    cost_hint_ms=120,
    max_result_chars=4000,
)
