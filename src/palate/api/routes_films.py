"""The agent free surface: look a film up, ask for a list, read the profile."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from palate.api.deps import AppState, State
from palate.errors import PalateError
from palate.index import verify
from palate.retrieval.evidence import FilmCard, RecommendedFilm, load_cards
from palate.retrieval.recommend import RecommendRequest, RecommendResponse
from palate.retrieval.vocab import Kind
from palate.tools.catalog import compare_films, get_film, get_taste_profile, resolve_vocabulary
from palate.tools.catalog.common import hook_of
from palate.tools.catalog.search_films import SearchFilmsArgs, hard_filters, meta_of
from palate.tools.context import ToolContext

router = APIRouter(tags=["films"])

NO_PROFILE = "no taste profile yet, run: palate profile build"

# Title lookup for a search field. Ordering by votes puts the film people meant first.
_TITLES = (
    "select tmdb_id from films where title like ? escape '\\' "
    "or original_title like ? escape '\\' order by vote_count desc limit ?"
)

_COUNTS = {
    "films": "select count(*) from films",
    "eligible": "select count(*) from corpus_members where eligible = 1",
    "rated": "select count(*) from user_films where rating_half is not null",
}


class FilmBrief(BaseModel):
    """One row of a listing, with every field read from the database."""

    film_id: int
    title: str
    year: int | None = None
    directors: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    original_language: str | None = None
    runtime: int | None = None
    in_watchlist: bool = False
    hook: str | None = None


class EvidenceOut(BaseModel):
    kind: str
    text: str
    source_table: str
    source_id: str
    value: float | str | None = None
    span: tuple[int, int] | None = None


class RecommendedOut(FilmBrief):
    score: float = 0.0
    confidence: float = 0.0
    mode_label: str | None = None
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    evidence: list[EvidenceOut] = Field(default_factory=list)


class RecommendBody(BaseModel):
    """The same arguments search_films takes, from a form instead of from a model."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    search: SearchFilmsArgs


class RecommendOut(BaseModel):
    films: list[RecommendedOut]
    pool_size: int
    condition: str
    profile_id: str
    degraded: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ModeDetail(BaseModel):
    mode_id: int
    polarity: str
    label: str | None
    n_members: int
    coherence: float
    confidence: float
    mean_signal: float
    exemplars: list[FilmBrief] = Field(default_factory=list)
    members: list[FilmBrief] = Field(default_factory=list)


class Health(BaseModel):
    ok: bool
    chat: dict[str, Any]
    embeddings: dict[str, Any]
    index: dict[str, Any]
    profile: dict[str, Any]
    corpus: dict[str, int]


def context(state: AppState, session_id: str | None = None) -> ToolContext:
    """A context for the routes that reuse a tool's own query, with no run behind it."""
    return ToolContext(
        session_id=session_id or "",
        run_id="",
        settings=state.settings,
        db=state.db,
        recommender=state.recommender(session_id),
        profile=state.profile,
        prefs=state.prefs,
        vocab=state.vocab,
    )


def brief(card: FilmCard) -> FilmBrief:
    """One card as a listing row. The hook is the overview's first sentence, not a summary."""
    return FilmBrief(
        film_id=card.tmdb_id,
        title=card.title,
        year=card.year,
        directors=list(card.directors),
        countries=list(card.countries),
        original_language=card.original_language,
        runtime=card.runtime,
        in_watchlist=card.in_watchlist,
        hook=hook_of(card.overview),
    )


def briefs(state: AppState, ids: list[int]) -> list[FilmBrief]:
    """Cards for a list of ids, in the order asked, skipping any the corpus does not hold."""
    cards = load_cards(state.db.read(), ids)
    return [brief(cards[i]) for i in ids if i in cards]


def recommended(film: RecommendedFilm, card: FilmCard | None) -> RecommendedOut:
    """One ranked film with the evidence a reader can argue with."""
    return RecommendedOut(
        film_id=film.tmdb_id,
        title=film.title,
        year=film.year,
        directors=list(film.directors),
        countries=list(film.countries),
        original_language=film.original_language,
        runtime=film.runtime,
        in_watchlist=film.in_watchlist,
        hook=None if card is None else hook_of(card.overview),
        score=round(film.score, 4),
        confidence=round(film.confidence, 4),
        mode_label=film.mode_label,
        feature_contributions={k: round(v, 4) for k, v in film.feature_contributions.items()},
        evidence=[
            EvidenceOut(
                kind=row.kind,
                text=row.text,
                source_table=row.source_table,
                source_id=row.source_id,
                value=row.value,
                span=row.span,
            )
            for row in film.evidence
        ],
    )


def response_of(state: AppState, answer: RecommendResponse) -> RecommendOut:
    """The ranked list plus the diagnostics that say what the filters cost."""
    cards = load_cards(state.db.read(), [f.tmdb_id for f in answer.films])
    films = [recommended(f, cards.get(f.tmdb_id)) for f in answer.films]
    return RecommendOut(
        films=films,
        pool_size=answer.pool_size,
        condition=answer.condition,
        profile_id=answer.profile_id,
        degraded=list(answer.degraded),
        diagnostics=meta_of(answer, len(films)),
    )


def title_ids(state: AppState, q: str, limit: int) -> list[int]:
    """Ids whose title contains the text, with the wildcards the user typed neutralised."""
    needle = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    rows = state.db.read().execute(_TITLES, (needle, needle, limit)).fetchall()
    return [int(row["tmdb_id"]) for row in rows]


@router.get("/films/search")
async def search(
    state: State,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[FilmBrief]:
    """Title match over the corpus. No embedding call, which is what a lookup field wants."""
    ids = await anyio.to_thread.run_sync(partial(title_ids, state, q, limit))
    return briefs(state, ids)


@router.get("/films/{tmdb_id}")
async def film(tmdb_id: int, state: State) -> get_film.FilmRecord:
    """The full record, with the overview offset a plot claim is later checked against."""
    found = await anyio.to_thread.run_sync(partial(get_film.records, state.db.read(), [tmdb_id]))
    if not found:
        raise HTTPException(status_code=404, detail=f"no film in the corpus has id {tmdb_id}")
    return found[0]


@router.get("/films/{film_id_a}/compare/{film_id_b}")
async def compare(film_id_a: int, film_id_b: int, state: State) -> compare_films.CompareFilmsResult:
    """Set algebra between two films, computed rather than described."""
    return await anyio.to_thread.run_sync(
        partial(compare_films.compare, state.db.read(), film_id_a, film_id_b)
    )


@router.post("/recommend")
async def recommend(body: RecommendBody, state: State) -> RecommendOut:
    """The listing, ranked. The same path the search tool takes, with no model in it."""
    ctx = context(state, body.session_id)
    args = body.search
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
    return response_of(state, answer)


@router.get("/profile")
async def profile(state: State) -> get_taste_profile.TasteProfileResult:
    """Modes, tier, and the top and bottom entities with the support behind each."""
    if state.profile is None:
        raise PalateError(NO_PROFILE)
    return get_taste_profile.summarise(state.profile, get_taste_profile.TasteProfileArgs())


@router.get("/profile/modes/{mode_id}")
async def mode(mode_id: int, state: State) -> ModeDetail:
    """One cluster of the history, with the films that put it there."""
    if state.profile is None:
        raise PalateError(NO_PROFILE)
    for polarity, group in (("like", state.profile.modes), ("dislike", state.profile.anti_modes)):
        for one in group:
            if one.mode_id != mode_id:
                continue
            return ModeDetail(
                mode_id=one.mode_id,
                polarity=polarity,
                label=one.label,
                n_members=one.n_members,
                coherence=round(float(one.coherence), 4),
                confidence=round(float(one.confidence), 4),
                mean_signal=round(float(one.mean_signal), 4),
                exemplars=briefs(state, list(one.exemplars)),
                members=briefs(state, [m.tmdb_id for m in one.members]),
            )
    raise HTTPException(status_code=404, detail=f"this profile has no mode {mode_id}")


@router.get("/vocabulary/resolve")
async def vocabulary(
    state: State,
    text: Annotated[str, Query(min_length=2, max_length=120)],
    kinds: Annotated[list[Kind] | None, Query()] = None,
) -> resolve_vocabulary.ResolveVocabularyResult:
    """What a phrase could mean here, with the film count each reading would touch."""
    args = resolve_vocabulary.ResolveVocabularyArgs(text=text, kinds=kinds or [])
    return await anyio.to_thread.run_sync(partial(resolve_vocabulary.resolve, state.vocab, args))


@router.get("/health")
async def health(state: State) -> Health:
    """What doctor checks, as json, so the page can say why it is empty."""
    chat = await state.chat.health()
    embed = await state.embedder.health()
    record = verify.active(state.db)
    conn = state.db.read()
    fitted = state.profile
    return Health(
        ok=chat.ok and embed.ok,
        chat={
            "provider": state.chat.name,
            "model": state.chat.model,
            "ok": chat.ok,
            "detail": chat.detail,
        },
        embeddings={"provider": state.embedder.provider, "ok": embed.ok, "detail": embed.detail},
        index={
            "index_id": record.index_id,
            "model": record.fingerprint.model_id,
            "dim": record.fingerprint.dim,
            "vectors": record.n_vectors,
        },
        profile={}
        if fitted is None
        else {
            "profile_id": fitted.profile_id,
            "tier": fitted.tier,
            "n_rated": fitted.n_rated,
            "stale": fitted.stale,
        },
        corpus={name: int(conn.execute(sql).fetchone()[0]) for name, sql in _COUNTS.items()},
    )
