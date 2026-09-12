"""search_films: the semantic path, with watched films excluded at the source."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from palate.retrieval.candidates import HardFilters
from palate.retrieval.recommend import RecommendRequest
from palate.tools.catalog.common import FilmRow, as_ints, as_rows, hooks, mode_labels, resolve_kind
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec, clamp

DESCRIPTION = (
    "Search the candidate corpus by meaning, with optional hard metadata filters. Returns only "
    "films the user has NOT already watched. Use this when the request is about mood, style, "
    "pace or subject. Use filter_films instead when the request is purely structural, for "
    "example a decade or a language."
)


class SearchFilmsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=400)
    similar_to_film_ids: list[int] = Field(default_factory=list, max_length=5)
    include_genres: list[str] = Field(default_factory=list, max_length=5)
    exclude_genres: list[str] = Field(default_factory=list, max_length=5)
    include_keywords: list[str] = Field(default_factory=list, max_length=8)
    exclude_keywords: list[str] = Field(default_factory=list, max_length=8)
    include_languages: list[str] = Field(default_factory=list, max_length=10)
    exclude_languages: list[str] = Field(default_factory=list, max_length=10)
    exclude_countries: list[str] = Field(default_factory=list, max_length=10)
    year_min: int | None = Field(default=None, ge=1888, le=2030)
    year_max: int | None = Field(default=None, ge=1888, le=2030)
    runtime_min: int | None = Field(default=None, ge=1, le=900)
    runtime_max: int | None = Field(default=None, ge=1, le=900)
    min_vote_count: int = Field(default=0, ge=0)
    apply_preferences: bool = True
    rerank: Literal["auto", "none", "cross_encoder", "llm"] = "auto"
    limit: int = 12
    offset: int = Field(default=0, ge=0)

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, value: object, info: ValidationInfo) -> int:
        """Clamp rather than reject. A rejected call costs the model a whole turn."""
        return clamp(value, info, lo=1, hi=50)

    @model_validator(mode="after")
    def _ranges(self) -> SearchFilmsArgs:
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError("year_min must be <= year_max")
        if self.runtime_min and self.runtime_max and self.runtime_min > self.runtime_max:
            raise ValueError("runtime_min must be <= runtime_max")
        return self


class SearchFilmsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    films: list[FilmRow]
    meta: dict[str, Any] = Field(default_factory=dict)


def hard_filters(ctx: ToolContext, args: SearchFilmsArgs) -> HardFilters:
    """Free text filter targets resolved against the corpus before any query runs."""
    vocab = ctx.require_vocab()
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
        exclude_countries=frozenset(
            resolve_kind(vocab, "country", args.exclude_countries, field="exclude_countries")
        ),
        min_vote_count=args.min_vote_count,
    )


async def handler(args: SearchFilmsArgs, ctx: ToolContext) -> SearchFilmsResult:
    """Embed the query, rank the pool, and say what the filters cost."""
    ctx.check_deadline()
    answer = await ctx.require_recommender().recommend(
        RecommendRequest(
            query_text=args.query,
            n=args.limit,
            filters=hard_filters(ctx, args),
            similar_to=tuple(args.similar_to_film_ids),
            rerank=args.rerank,
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
    return SearchFilmsResult(films=rows, meta=meta_of(answer, len(rows)))


def meta_of(answer: Any, returned: int) -> dict[str, Any]:
    """The diagnostics the agent needs to say what it narrowed, in numbers."""
    diagnostics = answer.diagnostics
    return {
        "count": diagnostics.after_filter,
        "returned": returned,
        "excluded_watched": diagnostics.excluded_watched,
        "excluded_by_preference": diagnostics.excluded_by_preference,
        "removed_by_clause": {k: v for k, v in diagnostics.removed_by_clause.items() if v},
        "removed_top10_by_clause": dict(diagnostics.removed_top10_by_clause),
        "most_restrictive_clause": diagnostics.most_restrictive_clause,
        "prefilter_path": diagnostics.prefilter_path,
        "elapsed_ms": round(diagnostics.elapsed_ms, 1),
        "degraded": list(answer.degraded),
    }


SPEC = ToolSpec(
    name="search_films",
    description=DESCRIPTION,
    args_model=SearchFilmsArgs,
    result_model=SearchFilmsResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(SearchFilmsArgs(query="slow contemplative science fiction", limit=10),),
    cost_hint_ms=400,
    max_result_chars=4000,
)
