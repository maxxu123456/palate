"""The recommender facade: async at the edge because the compute is not."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, Protocol

import anyio
import numpy as np

from palate.config import RetrievalSettings
from palate.db.connect import Database
from palate.errors import ThinHistoryError
from palate.index import verify
from palate.index.vecstore import VecStore
from palate.providers.base import EmbeddingProvider
from palate.retrieval.candidates import (
    NO_FILTERS,
    CandidatePool,
    CandidateStore,
    ChannelBudget,
    HardFilters,
    from_preferences,
    generate_candidates,
)
from palate.retrieval.diversity import DiversityConfig, diversify, query_mode_posterior
from palate.retrieval.evidence import Evidence, RecommendedFilm, build_evidence, load_cards
from palate.retrieval.features import (
    FeatureInputs,
    FilmFacets,
    bm25_scores,
    build_matrix,
    load_facets,
)
from palate.retrieval.fusion import Condition, FusionWeights, load_weights, prior_weights
from palate.retrieval.score import ScoredPool, score_pool
from palate.retrieval.vocab import Vocabulary, normalise
from palate.taste import profile as taste
from palate.taste.memory import PreferenceFilter, PreferenceStore
from palate.taste.modes import mode_argmax
from palate.taste.profile import TasteProfile

COLD_HINT = "too few ratings for an unconditioned list, describe what you are in the mood for"

NO_PREFERENCES = PreferenceFilter(exclude={}, require={})

DEFAULT_RETRIEVAL = RetrievalSettings()


@dataclass(frozen=True, slots=True)
class RecommendRequest:
    """One ask, with every knob the caller is allowed to turn."""

    query_text: str | None = None
    n: int = 10
    filters: HardFilters = NO_FILTERS
    similar_to: tuple[int, ...] = ()
    rerank: Literal["auto", "none", "cross_encoder", "llm"] = "auto"
    rerank_depth: int = 100
    diversity: Literal["auto", "on", "off"] = "auto"
    apply_preferences: bool = True
    include_watched: bool = False
    explain: bool = True
    offset: int = 0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    """What the filters and the scaler did, in numbers the caller can quote."""

    considered: int
    after_filter: int
    excluded_watched: int
    excluded_by_preference: int
    removed_by_clause: Mapping[str, int]
    removed_top10_by_clause: Mapping[str, int]
    reranked: int
    stages: tuple[str, ...]
    prefilter_path: str
    overfetch_factor: float
    scaled_by: Mapping[str, str]
    most_restrictive_clause: str | None
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class RecommendResponse:
    """The answer plus everything needed to argue with it."""

    films: tuple[RecommendedFilm, ...]
    diagnostics: SearchDiagnostics
    pool_size: int
    channels_used: Mapping[str, int]
    stage_latency_ms: Mapping[str, float]
    cost_usd: float
    profile_id: str
    condition: str
    degraded: tuple[str, ...]


class Recommender(Protocol):
    """What the CLI and, later, the agent tools call."""

    async def recommend(self, req: RecommendRequest) -> RecommendResponse: ...

    async def similar_to(
        self, tmdb_id: int, *, n: int = 10, filters: HardFilters = NO_FILTERS
    ) -> RecommendResponse: ...

    def vocabulary(self) -> Vocabulary: ...


def build_store(db: Database, retrieval: RetrievalSettings = DEFAULT_RETRIEVAL) -> CandidateStore:
    """A candidate store over the active index, on this thread's connection."""
    record = verify.active(db)
    conn = db.read()
    vecs = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
    return CandidateStore(
        conn,
        vecs=vecs,
        id_cap=retrieval.prefilter_id_cap,
        overfetch_floor=retrieval.overfetch_floor,
        overfetch_cap=retrieval.overfetch_cap,
    )


def budget_from(retrieval: RetrievalSettings) -> ChannelBudget:
    """Channel depths as configured."""
    return ChannelBudget(
        dense_per_mode=retrieval.dense_per_mode,
        dense_query=retrieval.dense_query,
        bm25=retrieval.bm25,
        people=retrieval.people,
        keyword=retrieval.keyword,
        popular=retrieval.popular,
        pool_max=retrieval.pool_max,
    )


def diversity_from(retrieval: RetrievalSettings) -> DiversityConfig:
    """Slot and cap settings as configured."""
    return DiversityConfig(
        max_per_director=retrieval.max_per_director,
        max_per_decade=retrieval.max_per_decade,
        max_per_collection=retrieval.max_per_collection,
        max_known_directors=retrieval.max_known_directors,
    )


@dataclass(frozen=True, slots=True)
class _Ranked:
    """One scoring pass, kept whole so a second pass can reuse the same weights."""

    pool: CandidatePool
    scored: ScoredPool
    weights: FusionWeights
    mode_of: Mapping[int, int]
    scaled_by: Mapping[str, str]


def _split_penalties(
    penalties: Mapping[str, float],
) -> tuple[frozenset[str], dict[str, float]]:
    """Country dislikes own their own column, so they never also land in the generic penalty."""
    countries = frozenset(
        key.split(":", 1)[1]
        for key, weight in penalties.items()
        if key.startswith("country:") and weight < 0.0
    )
    rest = {k: v for k, v in penalties.items() if not k.startswith("country:")}
    return countries, rest


def _mode_assignment(
    profile: TasteProfile, ids: Sequence[int], vectors: Mapping[int, Sequence[float]]
) -> tuple[dict[int, int], list[float]]:
    confidences = {m.mode_id: m.confidence for m in profile.modes}
    floor = min(confidences.values()) if confidences else 1.0
    if not profile.modes:
        return {}, [1.0] * len(ids)
    dim = len(profile.modes[0].centroid)
    rows = np.zeros((len(ids), dim))
    for i, tmdb_id in enumerate(ids):
        vector = vectors.get(tmdb_id)
        if vector is not None:
            rows[i] = vector
    assigned = mode_argmax(rows, profile.modes)
    mode_of = {
        tmdb_id: int(mode)
        for tmdb_id, mode in zip(ids, assigned, strict=True)
        if tmdb_id in vectors
    }
    return mode_of, [confidences.get(mode_of.get(i, -1), floor) for i in ids]


def _rank(
    store: CandidateStore,
    profile: TasteProfile,
    req: RecommendRequest,
    filters: HardFilters,
    *,
    query_embedding: Sequence[float] | None,
    penalties: Mapping[str, float],
    retrieval: RetrievalSettings,
    condition: Condition,
    weights: FusionWeights | None = None,
) -> _Ranked:
    pool = generate_candidates(
        store,
        profile,
        query_embedding=query_embedding,
        query_text=req.query_text,
        filters=filters,
        budget=budget_from(retrieval),
        rrf_k=retrieval.rrf_k,
    )
    vectors = store.vecs.vectors(pool.ids)
    countries, soft = _split_penalties(penalties)
    matrix = build_matrix(
        store.conn,
        profile,
        pool.ids,
        FeatureInputs(
            vectors=vectors,
            bm25=bm25_scores(pool),
            query_embedding=query_embedding,
            soft_countries=countries,
            penalties=soft,
        ),
    )
    if weights is None:
        stored = load_weights(store.conn, profile.profile_id).get(condition)
        weights = stored or prior_weights(matrix.active, condition=condition)
    mode_of, confidence = _mode_assignment(profile, matrix.ids, vectors)
    scored = score_pool(matrix, weights, tier=profile.tier, mode_confidence=confidence)
    return _Ranked(
        pool=pool,
        scored=scored,
        weights=weights,
        mode_of=mode_of,
        scaled_by=dict(zip(matrix.names, matrix.scaled_by, strict=True)),
    )


def _only(filters: HardFilters, clause: str) -> HardFilters:
    """One clause on its own, so a removal can be attributed to it."""
    if clause == "watched":
        return HardFilters(exclude_watched=True)
    return HardFilters(exclude_watched=False, **{clause: getattr(filters, clause)})


def _removed_top10(
    store: CandidateStore,
    profile: TasteProfile,
    req: RecommendRequest,
    filters: HardFilters,
    loose: HardFilters,
    kept: frozenset[int],
    *,
    query_embedding: Sequence[float] | None,
    penalties: Mapping[str, float],
    retrieval: RetrievalSettings,
    condition: Condition,
    weights: FusionWeights,
) -> dict[str, int]:
    """Score the list the optional clauses never got to see, and say what each one cost."""
    wider = _rank(
        store,
        profile,
        req,
        loose,
        query_embedding=query_embedding,
        penalties=penalties,
        retrieval=retrieval,
        condition=condition,
        weights=weights,
    )
    head = [i for i in wider.scored.order()[:10] if i not in kept]
    if not head:
        return {}
    out: dict[str, int] = {}
    for clause in _optional_clauses(filters):
        allowed = store.allow(_only(filters, clause)).ids
        lost = sum(1 for tmdb_id in head if tmdb_id not in allowed)
        if lost:
            out[clause] = lost
    return out


def _optional_clauses(filters: HardFilters) -> list[str]:
    """The clauses the caller chose, which are the ones worth arguing about."""
    names = [
        "year_min",
        "year_max",
        "runtime_min",
        "runtime_max",
        "include_languages",
        "exclude_languages",
        "include_genres",
        "exclude_genres",
        "include_keywords",
        "exclude_keywords",
        "include_countries",
        "exclude_countries",
        "include_people",
        "exclude_people",
        "include_decades",
        "exclude_decades",
        "exclude_ids",
    ]
    live = [n for n in names if getattr(filters, n)]
    if filters.min_vote_count:
        live.append("min_vote_count")
    return live


def _select(
    ranked: _Ranked,
    profile: TasteProfile,
    req: RecommendRequest,
    *,
    query_embedding: Sequence[float] | None,
    vectors: Mapping[int, Sequence[float]],
    facets: Mapping[int, FilmFacets],
    cfg: DiversityConfig,
) -> list[int]:
    head = ranked.scored.order()[: max(req.rerank_depth, req.n + req.offset)]
    wanted = req.n + req.offset
    if req.diversity == "off":
        return head[req.offset : wanted]
    posterior = (
        query_mode_posterior(query_embedding, profile.modes)
        if query_embedding is not None and profile.modes
        else None
    )
    known = frozenset(int(a.entity_id) for a in profile.affinities.get("director", ()))
    selected, _ = diversify(
        head,
        ranked.scored.as_map(),
        vectors,
        facets,
        ranked.mode_of,
        known,
        wanted,
        cfg,
        posterior=posterior,
        modes=profile.modes,
    )
    return selected[req.offset :]


def run_recommend(
    db: Database,
    profile: TasteProfile,
    req: RecommendRequest,
    *,
    query_embedding: Sequence[float] | None = None,
    session_id: str | None = None,
    retrieval: RetrievalSettings = DEFAULT_RETRIEVAL,
) -> RecommendResponse:
    """The whole pipeline, synchronous, on this thread's connection."""
    started = time.perf_counter()
    taste.assert_fresh(db, profile)
    condition: Condition = "query" if query_embedding is not None else "unconditioned"
    if profile.tier == "cold" and condition == "unconditioned":
        raise ThinHistoryError(COLD_HINT)
    degraded = [
        n
        for n, on in (
            ("cold_start", profile.tier == "cold"),
            ("thin_history", profile.tier == "thin"),
        )
        if on
    ]
    degraded.append("no_reranker")
    store = build_store(db, retrieval)
    memory = PreferenceStore(db)
    stated = memory.as_filter(session_id) if req.apply_preferences else NO_PREFERENCES
    penalties = memory.as_penalties(session_id) if req.apply_preferences else {}
    asked = req.filters if not req.include_watched else replace(req.filters, exclude_watched=False)
    compiled = from_preferences(store.conn, stated)
    filters = asked.merge(compiled)
    timing: dict[str, float] = {}
    mark = time.perf_counter()
    ranked = _rank(
        store,
        profile,
        req,
        filters,
        query_embedding=query_embedding,
        penalties=penalties,
        retrieval=retrieval,
        condition=condition,
    )
    timing["rank"] = _since(mark)
    mark = time.perf_counter()
    vectors = store.vecs.vectors(ranked.scored.ids)
    facets = load_facets(store.conn, ranked.scored.ids)
    selected = _select(
        ranked,
        profile,
        req,
        query_embedding=query_embedding,
        vectors=vectors,
        facets=facets,
        cfg=diversity_from(retrieval),
    )
    timing["diversity"] = _since(mark)
    mark = time.perf_counter()
    films = _films(store, profile, req, ranked, selected)
    timing["evidence"] = _since(mark)
    loose = HardFilters(exclude_watched=filters.exclude_watched).merge(compiled)
    removed = (
        _removed_top10(
            store,
            profile,
            req,
            filters,
            loose,
            ranked.pool.allow.ids,
            query_embedding=query_embedding,
            penalties=penalties,
            retrieval=retrieval,
            condition=condition,
            weights=ranked.weights,
        )
        if req.explain and _optional_clauses(asked)
        else {}
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    return RecommendResponse(
        films=films,
        diagnostics=SearchDiagnostics(
            considered=ranked.pool.allow.n_corpus,
            after_filter=len(ranked.pool.allow.ids),
            excluded_watched=ranked.pool.allow.excluded_watched,
            excluded_by_preference=_preference_cost(store, asked, filters, compiled),
            removed_by_clause=ranked.pool.allow.removed_by_clause,
            removed_top10_by_clause=removed,
            reranked=0,
            stages=("candidates", "features", "fusion", "diversity", "evidence"),
            prefilter_path=ranked.pool.prefilter_path,
            overfetch_factor=ranked.pool.overfetch_factor,
            scaled_by=ranked.scaled_by,
            most_restrictive_clause=filters.most_restrictive(ranked.pool.allow.removed_by_clause),
            elapsed_ms=elapsed,
        ),
        pool_size=len(ranked.pool.ids),
        channels_used={c.value: n for c, n in ranked.pool.per_channel_counts.items()},
        stage_latency_ms=timing,
        cost_usd=0.0,
        profile_id=profile.profile_id,
        condition=condition,
        degraded=tuple(degraded),
    )


def _since(mark: float) -> float:
    return (time.perf_counter() - mark) * 1000.0


def _preference_cost(
    store: CandidateStore, asked: HardFilters, filters: HardFilters, compiled: HardFilters
) -> int:
    """How many films stated memory removed on top of what the caller asked for."""
    if compiled == HardFilters():
        return 0
    return len(store.allow(asked).ids) - len(store.allow(filters).ids)


def _films(
    store: CandidateStore,
    profile: TasteProfile,
    req: RecommendRequest,
    ranked: _Ranked,
    selected: Sequence[int],
) -> tuple[RecommendedFilm, ...]:
    cards = load_cards(store.conn, selected)
    terms = normalise(req.query_text or "").split()
    evidence = (
        build_evidence(
            store.conn,
            profile,
            selected,
            mode_of=ranked.mode_of,
            query_terms=terms,
            cards=cards,
        )
        if req.explain
        else {}
    )
    out: list[RecommendedFilm] = []
    for tmdb_id in selected:
        card = cards.get(tmdb_id)
        if card is None:
            continue
        rows: tuple[Evidence, ...] = evidence.get(tmdb_id, ())
        out.append(
            RecommendedFilm(
                tmdb_id=tmdb_id,
                title=card.title,
                year=card.year,
                directors=card.directors,
                countries=card.countries,
                original_language=card.original_language,
                runtime=card.runtime,
                score=float(ranked.scored.as_map()[tmdb_id]),
                confidence=ranked.scored.confidence_of(tmdb_id),
                mode_id=ranked.mode_of.get(tmdb_id),
                mode_label=None,
                in_watchlist=card.in_watchlist,
                feature_contributions=ranked.scored.shares(tmdb_id),
                evidence=rows,
            )
        )
    return tuple(out)


class LocalRecommender:
    """The in process recommender. Every method crosses into a worker thread."""

    def __init__(
        self,
        db: Database,
        *,
        embedder: EmbeddingProvider | None = None,
        session_id: str | None = None,
        retrieval: RetrievalSettings = DEFAULT_RETRIEVAL,
        profile: TasteProfile | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.session_id = session_id
        self.retrieval = retrieval
        self._profile = profile

    def profile(self) -> TasteProfile:
        """The newest fresh profile, refused when the index moved under it."""
        if self._profile is None:
            found = taste.latest(self.db)
            if found is None:
                raise ThinHistoryError("no taste profile yet, run: palate profile build")
            self._profile = found
        return self._profile

    def vocabulary(self) -> Vocabulary:
        """The corpus vocabulary, for resolving what the user said into ids."""
        return Vocabulary(self.db.read())

    async def recommend(self, req: RecommendRequest) -> RecommendResponse:
        """Embed the query if there is one, then rank off the event loop."""
        query = await self._embed(req.query_text)
        return await anyio.to_thread.run_sync(
            partial(
                run_recommend,
                self.db,
                self.profile(),
                req,
                query_embedding=query,
                session_id=self.session_id,
                retrieval=self.retrieval,
            )
        )

    async def similar_to(
        self, tmdb_id: int, *, n: int = 10, filters: HardFilters = NO_FILTERS
    ) -> RecommendResponse:
        """Films near one the user named, which needs no embedding call at all."""
        req = RecommendRequest(
            n=n,
            filters=filters.merge(HardFilters(exclude_ids=frozenset({tmdb_id}))),
            similar_to=(tmdb_id,),
        )
        vector = await anyio.to_thread.run_sync(partial(self._vector_of, tmdb_id))
        if vector is None:
            raise ThinHistoryError(f"film {tmdb_id} has no vector in the active index")
        return await anyio.to_thread.run_sync(
            partial(
                run_recommend,
                self.db,
                self.profile(),
                req,
                query_embedding=vector,
                session_id=self.session_id,
                retrieval=self.retrieval,
            )
        )

    def _vector_of(self, tmdb_id: int) -> Sequence[float] | None:
        return build_store(self.db, self.retrieval).vecs.vectors([tmdb_id]).get(tmdb_id)

    async def _embed(self, text: str | None) -> Sequence[float] | None:
        if not text:
            return None
        if self.embedder is None:
            raise ThinHistoryError("a query needs an embedding provider, run: palate doctor")
        verify.require_active_match(self.db, self.embedder)
        return await self.embedder.embed_query(text)
