"""Running the matrix. Same seed, same code, same numbers, and a row per arm per fold."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal

import anyio
import numpy as np
import orjson

from palate.clock import now_iso
from palate.config import RetrievalSettings
from palate.db.connect import Database
from palate.errors import EvalError
from palate.eval.baselines import BASELINES, FoldInputs
from palate.eval.labels import graded_relevance
from palate.eval.metrics import (
    MetricCI,
    bootstrap_ci,
    intra_list_distance,
    mean_ci,
    mrr_at_k,
    ndcg_at_k,
    novelty,
    paired_bootstrap,
    recall_at_k,
    spearman_conditional,
    spearman_pessimistic,
    unknown_director_rate,
)
from palate.eval.queries import QueryCase
from palate.eval.split import Fold, Split, load_corpus_ids
from palate.eval.systems import SystemConfig, blocked_reason
from palate.ids import new_id
from palate.index import verify
from palate.index.vecstore import VecStore
from palate.providers.base import EmbeddingProvider
from palate.retrieval.candidates import HardFilters, generate_candidates
from palate.retrieval.diversity import DiversityConfig, diversify, query_mode_posterior
from palate.retrieval.features import FeatureInputs, bm25_scores, build_matrix, load_facets
from palate.retrieval.fusion import (
    Condition,
    FusionFold,
    FusionQuery,
    FusionWeights,
    combine,
    fit_fusion_weights,
    load_weights,
    prior_weights,
    save_weights,
)
from palate.retrieval.recommend import budget_from, build_store, diversity_from
from palate.retrieval.score import score_pool
from palate.taste import profile as taste
from palate.taste.modes import mode_argmax
from palate.taste.profile import TasteProfile

type EvalCondition = Literal["unconditioned", "query_review", "query_synth", "query_cold", "agent"]

QUERY_CONDITIONS: tuple[EvalCondition, ...] = ("query_review", "query_synth", "query_cold")

NDCG_K = 10
RECALL_K = 50
LIST_K = 10

# Features that only change the score, so they are switched off by zeroing their weight.
METADATA_FEATURES = ("decade_aff", "runtime_aff", "lang_aff", "country_aff")
EXPOSURE_FEATURES = ("decade_exposure",)

# Kinds that seed a retrieval channel, so they are switched off by shaping the profile.
PEOPLE_KINDS = ("director", "writer", "actor")


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Everything a run may read, plus the per fold profile cache that makes a sweep bearable."""

    db: Database
    split: Split
    seed: int = 0
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    embedder: EmbeddingProvider | None = None
    code_sha: str = ""
    resamples: int = 200
    restarts: int = 8
    max_rounds: int = 40
    profiles: dict[tuple[int, float, int | None], TasteProfile] = field(default_factory=dict)
    _popularity: dict[int, float] = field(default_factory=dict)

    def popularity(self) -> Mapping[int, float]:
        """Corpus wide vote counts, read once because novelty needs the whole distribution."""
        if not self._popularity:
            rows = self.db.read().execute("select tmdb_id, vote_count from films")
            self._popularity.update({int(r["tmdb_id"]): float(r["vote_count"]) for r in rows})
        return self._popularity

    def profile(self, fold: Fold, cfg: SystemConfig) -> TasteProfile:
        """The profile for this fold, fitted on its train slice and nothing later."""
        key = (fold.fold, cfg.alpha, cfg.history_cap)
        if key not in self.profiles:
            self.profiles[key] = taste.build(
                self.db,
                only=_capped(fold, cfg.history_cap),
                alpha=cfg.alpha,
                seed=self.seed,
                split_name=self.split.name,
                fold=fold.fold,
            )
        return self.profiles[key]


def ratings_of(db: Database, ids: Sequence[int]) -> dict[int, int]:
    """Half star ratings for an id set, which is what a gain vector is built from."""
    rows = db.read().execute(
        "select tmdb_id, rating_half from user_films where rating_half is not null "
        "and tmdb_id in (select value from json_each(?))",
        (orjson.dumps(list(ids)).decode(),),
    )
    return {int(r["tmdb_id"]): int(r["rating_half"]) for r in rows}


@dataclass(frozen=True, slots=True)
class RunResult:
    """One arm on one fold under one condition, with everything the table needs."""

    run_id: str
    system: str
    config_sha: str
    fold: int
    condition: EvalCondition
    ranked: tuple[int, ...]
    scores: Mapping[int, float]
    metrics: Mapping[str, MetricCI]
    pool_size: int
    pool_recall: float
    prefilter_path: str
    stage_latency_ms: Mapping[str, float]
    cost_usd: float
    profile_id: str
    elapsed_ms: float
    degraded: tuple[str, ...] = ()
    n_cases: int = 0


def _capped(fold: Fold, history_cap: int | None) -> tuple[int, ...]:
    """The data-efficiency arm keeps the most recent ratings, which is the honest direction."""
    if history_cap is None or history_cap >= len(fold.train):
        return fold.train
    return fold.train[-history_cap:]


def candidates_for(db: Database, fold: Fold) -> tuple[int, ...]:
    """The eligible corpus minus what the user had already seen at the cut."""
    return tuple(sorted(load_corpus_ids(db.read()) - set(fold.train)))


def shape_profile(profile: TasteProfile, cfg: SystemConfig) -> TasteProfile:
    """Turn off what an arm says is off, at the level that also changes what is retrieved."""
    changes: dict[str, Any] = {}
    if not cfg.use_modes:
        changes["modes"] = ()
    elif cfg.n_modes is not None:
        changes["modes"] = profile.modes[: cfg.n_modes]
    if not cfg.use_repulsion:
        changes["anti_modes"] = ()
    if not cfg.use_ridge:
        changes["direction"] = None
    drop = set()
    if not cfg.use_people_prior:
        drop |= set(PEOPLE_KINDS)
    if not cfg.use_keyword_prior:
        drop.add("keyword")
    if drop:
        changes["affinities"] = {k: v for k, v in profile.affinities.items() if k not in drop}
    return replace(profile, **changes) if changes else profile


def _off(cfg: SystemConfig) -> tuple[str, ...]:
    names: list[str] = []
    if not cfg.use_metadata_prior:
        names.extend(METADATA_FEATURES)
    if not cfg.use_exposure_features:
        names.extend(EXPOSURE_FEATURES)
    return tuple(names)


def weights_for(
    cfg: SystemConfig, active: Sequence[str], condition: Condition, stored: FusionWeights | None
) -> FusionWeights:
    """Published weights when the arm asks for them, the signed prior otherwise."""
    base = stored if cfg.fusion == "weighted" and stored is not None else None
    if base is None:
        base = prior_weights(active, condition=condition)
    silenced = {name: 0.0 for name in _off(cfg)}
    return replace(base, beta={**dict(base.beta), **silenced})


def _validation_query(ctx: EvalContext, cfg: SystemConfig, fold: Fold) -> FusionQuery | None:
    """One pool from a profile fitted on inner only, graded by the validation ratings."""
    inner = taste.build(
        ctx.db,
        only=fold.inner,
        alpha=cfg.alpha,
        seed=ctx.seed,
        split_name=ctx.split.name,
        fold=fold.fold,
    )
    store = build_store(ctx.db, ctx.retrieval)
    shaped = shape_profile(inner, cfg)
    pool = generate_candidates(
        store,
        shaped,
        filters=HardFilters(exclude_watched=False, exclude_ids=frozenset(fold.inner)),
        budget=budget_from(ctx.retrieval),
        channels=sorted(cfg.channels, key=lambda c: c.value),
        rrf_k=ctx.retrieval.rrf_k,
    )
    if not pool.ids:
        return None
    vectors = store.vecs.vectors(pool.ids)
    matrix = build_matrix(
        store.conn,
        shaped,
        pool.ids,
        FeatureInputs(vectors=vectors, bm25=bm25_scores(pool)),
    )
    ratings = ratings_of(ctx.db, fold.val)
    gain = np.array(
        [2.0 ** graded_relevance(ratings.get(i, 0)) - 1.0 for i in matrix.ids], dtype=np.float64
    )
    if gain.max(initial=0.0) <= 0.0:
        return None
    return FusionQuery(X=matrix.X, gain=gain, ids=matrix.ids, names=matrix.names)


def fit_weights(
    ctx: EvalContext, cfg: SystemConfig, *, condition: Condition = "unconditioned"
) -> tuple[FusionWeights, list[FusionWeights]] | None:
    """Leave one fold out over the validation slices. Test labels are never touched here."""
    queries = {}
    for fold in ctx.split.folds:
        query = _validation_query(ctx, cfg, fold)
        if query is not None:
            queries[fold.fold] = query
    if len(queries) < 2:
        return None
    fits = [
        fit_fusion_weights(
            [
                FusionFold(
                    name=f"fold{held}",
                    inner=tuple(q for number, q in queries.items() if number != held),
                    val=(queries[held],),
                )
            ],
            condition=condition,
            seed=ctx.seed + held,
            restarts=ctx.restarts,
            max_rounds=ctx.max_rounds,
        )
        for held in sorted(queries)
    ]
    return combine(fits), fits


def publish_weights(
    ctx: EvalContext,
    cfg: SystemConfig,
    *,
    condition: Condition = "unconditioned",
) -> FusionWeights | None:
    """Fit, write the per feature stability, and hang the vector on every fold profile."""
    fitted = fit_weights(ctx, cfg, condition=condition)
    if fitted is None:
        return None
    combined, fits = fitted
    for fold in ctx.split.folds:
        save_weights(ctx.db, ctx.profile(fold, cfg).profile_id, {condition: combined})
    _store_stability(ctx, fits, condition=condition)
    return combined


def _store_stability(ctx: EvalContext, fits: Sequence[FusionWeights], *, condition: str) -> None:
    names = sorted({name for fit in fits for name in fit.beta})
    rows = []
    for name in names:
        values = np.array([fit.beta.get(name, 0.0) for fit in fits], dtype=np.float64)
        middle = float(np.median(values))
        agree = int(sum(1 for v in values if (v > 0) == (middle > 0) and (v != 0) == (middle != 0)))
        rows.append(
            (
                ctx.split.name,
                condition,
                name,
                float(values.mean()),
                float(values.std()),
                agree,
                int((values != 0.0).sum()),
            )
        )
    with ctx.db.write() as conn:
        conn.executemany(
            "insert or replace into eval_weight_stability (split_name, condition, feature, "
            "mean_beta, sd_beta, sign_agree, n_active) values (?,?,?,?,?,?,?)",
            rows,
        )


@dataclass(frozen=True, slots=True)
class _Ranking:
    """One scoring pass over one pool."""

    order: tuple[int, ...]
    scores: dict[int, float]
    pool_size: int
    prefilter_path: str
    mode_of: Mapping[int, int]
    stages: dict[str, float]


def _mode_assignment(
    profile: TasteProfile, ids: Sequence[int], vectors: Mapping[int, Sequence[float]]
) -> tuple[dict[int, int], list[float]]:
    if not profile.modes:
        return {}, [1.0] * len(ids)
    confidences = {m.mode_id: m.confidence for m in profile.modes}
    floor = min(confidences.values())
    rows = np.zeros((len(ids), len(profile.modes[0].centroid)))
    for i, tmdb_id in enumerate(ids):
        vector = vectors.get(tmdb_id)
        if vector is not None:
            rows[i] = vector
    assigned = mode_argmax(rows, profile.modes)
    mode_of = {i: int(m) for i, m in zip(ids, assigned, strict=True) if i in vectors}
    return mode_of, [confidences.get(mode_of.get(i, -1), floor) for i in ids]


def rank_system(
    ctx: EvalContext,
    cfg: SystemConfig,
    fold: Fold,
    profile: TasteProfile,
    *,
    query_embedding: Sequence[float] | None = None,
    query_text: str | None = None,
) -> _Ranking:
    """The real pipeline with this arm's knobs, over the corpus minus the fold's train set."""
    stages: dict[str, float] = {}
    mark = time.perf_counter()
    store = build_store(ctx.db, ctx.retrieval)
    shaped = shape_profile(profile, cfg)
    filters = HardFilters(exclude_watched=False, exclude_ids=frozenset(fold.train))
    pool = generate_candidates(
        store,
        shaped,
        query_embedding=query_embedding,
        query_text=query_text,
        filters=filters,
        budget=budget_from(ctx.retrieval),
        channels=sorted(cfg.channels, key=lambda c: c.value),
        rrf_k=ctx.retrieval.rrf_k,
    )
    stages["candidates"] = _since(mark)
    mark = time.perf_counter()
    vectors = store.vecs.vectors(pool.ids)
    matrix = build_matrix(
        store.conn,
        shaped,
        pool.ids,
        FeatureInputs(vectors=vectors, bm25=bm25_scores(pool), query_embedding=query_embedding),
    )
    condition: Condition = "query" if query_embedding is not None else "unconditioned"
    stored = load_weights(store.conn, profile.profile_id).get(condition)
    weights = weights_for(cfg, matrix.active, condition, stored)
    mode_of, confidence = _mode_assignment(shaped, matrix.ids, vectors)
    scored = score_pool(
        matrix,
        weights,
        tier=profile.tier if cfg.confidence_shrinkage else "full",
        mode_confidence=confidence,
    )
    stages["score"] = _since(mark)
    mark = time.perf_counter()
    if cfg.fusion == "rrf":
        scores = {i: float(pool.rrf.get(i, 0.0)) for i in matrix.ids}
        order = sorted(scores, key=lambda i: (-scores[i], i))
    else:
        scores = scored.as_map()
        order = scored.order()
    order = _diversified(ctx, cfg, shaped, order, scores, vectors, store.conn, query_embedding)
    stages["diversity"] = _since(mark)
    return _Ranking(
        order=tuple(order),
        scores=scores,
        pool_size=len(pool.ids),
        prefilter_path=pool.prefilter_path,
        mode_of=mode_of,
        stages=stages,
    )


def _diversified(
    ctx: EvalContext,
    cfg: SystemConfig,
    profile: TasteProfile,
    order: Sequence[int],
    scores: Mapping[int, float],
    vectors: Mapping[int, Sequence[float]],
    conn: sqlite3.Connection,
    query_embedding: Sequence[float] | None,
) -> list[int]:
    """Diversity reshuffles the head only, so the tail keeps carrying recall at 50."""
    if cfg.diversity == "off" or not order:
        return list(order)
    head = list(order[: max(cfg.rerank_depth, LIST_K)])
    facets = load_facets(conn, head)
    posterior = (
        query_mode_posterior(query_embedding, profile.modes)
        if query_embedding is not None and profile.modes
        else None
    )
    known = (
        frozenset(int(a.entity_id) for a in profile.affinities.get("director", ()))
        if cfg.novelty_cap
        else frozenset()
    )
    mode_of, _ = _mode_assignment(profile, head, vectors)
    selected, _ = diversify(
        head,
        {i: scores[i] for i in head},
        vectors,
        facets,
        mode_of,
        known,
        LIST_K,
        _diversity_cfg(ctx, cfg),
        posterior=posterior,
        modes=profile.modes,
    )
    chosen = set(selected)
    return [*selected, *(i for i in order if i not in chosen)]


def _diversity_cfg(ctx: EvalContext, cfg: SystemConfig) -> DiversityConfig:
    base = diversity_from(ctx.retrieval)
    return base if cfg.novelty_cap else replace(base, max_known_directors=LIST_K)


def _since(mark: float) -> float:
    return (time.perf_counter() - mark) * 1000.0


def _inputs(ctx: EvalContext, fold: Fold, profile: TasteProfile) -> FoldInputs:
    conn = ctx.db.read()
    candidates = candidates_for(ctx.db, fold)
    record = verify.active(ctx.db)
    store = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
    return FoldInputs(
        conn=conn,
        profile=profile,
        candidates=candidates,
        train=fold.train,
        vectors=store.vectors([*candidates, *fold.train]),
    )


def _baseline_ranking(
    ctx: EvalContext, cfg: SystemConfig, fold: Fold, profile: TasteProfile
) -> _Ranking:
    inputs = _inputs(ctx, fold, profile)
    mark = time.perf_counter()
    ranked = BASELINES[cfg.name](inputs, fold)
    places = {i: float(-rank) for rank, i in enumerate(ranked)}
    return _Ranking(
        order=tuple(ranked),
        scores=places,
        pool_size=len(inputs.candidates),
        prefilter_path="none",
        mode_of={},
        stages={"baseline": _since(mark)},
    )


def _list_metrics(
    ctx: EvalContext, ranked: Sequence[int], profile: TasteProfile
) -> dict[str, float]:
    head = list(ranked[:LIST_K])
    if not head:
        return {}
    conn = ctx.db.read()
    record = verify.active(ctx.db)
    vectors = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim).vectors(head)
    facets = load_facets(conn, head)
    known = {int(a.entity_id) for a in profile.affinities.get("director", ())}
    return {
        "ild@10": intra_list_distance(head, vectors),
        "novelty@10": novelty(head, ctx.popularity()),
        "unknown_director@10": unknown_director_rate(
            head, known, directors={i: f.directors for i, f in facets.items()}
        ),
    }


def _metrics(
    ctx: EvalContext, fold: Fold, ranking: _Ranking, profile: TasteProfile
) -> dict[str, MetricCI]:
    rel = fold.rel
    watch = fold.rel_watch
    ranked = ranking.order
    graded = partial(ndcg_at_k, k=NDCG_K)
    out = {
        "ndcg@10": bootstrap_ci(
            ranked, rel, graded, n_resamples=ctx.resamples, seed=ctx.seed + fold.fold
        ),
        "ndcg@10_watch": bootstrap_ci(
            ranked, watch, graded, n_resamples=ctx.resamples, seed=ctx.seed + fold.fold
        ),
        "recall@50": bootstrap_ci(
            ranked,
            rel,
            partial(recall_at_k, k=RECALL_K),
            n_resamples=ctx.resamples,
            seed=ctx.seed + fold.fold,
        ),
    }
    actual = {i: float(r) for i, r in rel.items()}
    floor = min(ranking.scores.values(), default=0.0)
    rho, n, coverage = spearman_conditional(ranked, ranking.scores, actual)
    out["spearman_cond"] = MetricCI(rho, rho, rho, n)
    out["spearman_coverage"] = MetricCI(coverage, coverage, coverage, n)
    pessimistic, total = spearman_pessimistic(ranking.scores, actual, floor=floor)
    out["spearman_pess"] = MetricCI(pessimistic, pessimistic, pessimistic, total)
    for name, value in _list_metrics(ctx, ranked, profile).items():
        out[name] = MetricCI(value, value, value, LIST_K)
    return out


def _run_unconditioned(ctx: EvalContext, cfg: SystemConfig, fold: Fold) -> RunResult:
    # Warm the profile before the clock starts, or the first arm of a fold wears the fitting.
    profile = ctx.profile(fold, cfg)
    started = time.perf_counter()
    ranking = (
        _baseline_ranking(ctx, cfg, fold, profile)
        if cfg.is_baseline
        else rank_system(ctx, cfg, fold, profile)
    )
    reached = set(ranking.order)
    return RunResult(
        run_id=new_id("eval_"),
        system=cfg.name,
        config_sha=cfg.config_sha,
        fold=fold.fold,
        condition="unconditioned",
        ranked=ranking.order[:RECALL_K],
        scores={i: ranking.scores[i] for i in ranking.order[:RECALL_K]},
        metrics=_metrics(ctx, fold, ranking, profile),
        pool_size=ranking.pool_size,
        pool_recall=len(reached & set(fold.test)) / max(1, len(fold.test)),
        prefilter_path=ranking.prefilter_path,
        stage_latency_ms=ranking.stages,
        cost_usd=0.0,
        profile_id=profile.profile_id,
        elapsed_ms=_since(started),
        degraded=_degraded(ctx, cfg, fold, profile, "unconditioned"),
    )


def _degraded(
    ctx: EvalContext, cfg: SystemConfig, fold: Fold, profile: TasteProfile, condition: Condition
) -> tuple[str, ...]:
    """Every reason this row is weaker than its name suggests, carried into the report."""
    out = []
    if cfg.reranker != "none":
        out.append("no_reranker")
    if profile.tier != "full":
        out.append(f"tier_{profile.tier}")
    if fold.underpowered:
        out.append("underpowered_fold")
    if cfg.fusion == "weighted" and not cfg.is_baseline:
        stored = load_weights(ctx.db.read(), profile.profile_id).get(condition)
        if stored is None or stored.fallback:
            out.append("fusion_fallback")
    return tuple(out)


async def _rankings_for(
    ctx: EvalContext,
    cfg: SystemConfig,
    fold: Fold,
    profile: TasteProfile,
    cases: Sequence[QueryCase],
) -> list[_Ranking]:
    """One ranking per query. A baseline ignores the query, so it is ranked once and reused."""
    if cfg.is_baseline:
        return [_baseline_ranking(ctx, cfg, fold, profile)] * len(cases)
    embedder = ctx.embedder
    if embedder is None:
        raise EvalError("query mode needs an embedding provider in the context")
    out: list[_Ranking] = []
    for case in cases:
        embedding = await embedder.embed_query(case.text)
        out.append(
            await anyio.to_thread.run_sync(
                partial(
                    rank_system,
                    ctx,
                    cfg,
                    fold,
                    profile,
                    query_embedding=embedding,
                    query_text=case.text,
                )
            )
        )
    return out


async def _run_queries(
    ctx: EvalContext,
    cfg: SystemConfig,
    fold: Fold,
    cases: Sequence[QueryCase],
    *,
    condition: EvalCondition,
) -> RunResult:
    """One run over a query set: the target film is the only right answer, out of the corpus."""
    profile = ctx.profile(fold, cfg)
    started = time.perf_counter()
    if condition == "query_cold":
        profile = replace(profile, tier="cold", modes=(), anti_modes=(), direction=None)
    rankings = await _rankings_for(ctx, cfg, fold, profile, cases)
    reciprocal: list[float] = []
    found: list[float] = []
    pools: list[int] = []
    path = "none"
    for case, ranking in zip(cases, rankings, strict=True):
        rel = {case.target_tmdb_id: 3}
        reciprocal.append(mrr_at_k(ranking.order, rel, k=RECALL_K))
        found.append(recall_at_k(ranking.order, rel, k=RECALL_K, min_rel=1))
        pools.append(ranking.pool_size)
        path = ranking.prefilter_path
    seed = ctx.seed + fold.fold
    return RunResult(
        run_id=new_id("eval_"),
        system=cfg.name,
        config_sha=cfg.config_sha,
        fold=fold.fold,
        condition=condition,
        ranked=(),
        scores={},
        metrics={
            "mrr@50": mean_ci(reciprocal, n_resamples=ctx.resamples, seed=seed),
            "recall@50": mean_ci(found, n_resamples=ctx.resamples, seed=seed),
        },
        pool_size=int(np.mean(pools)) if pools else 0,
        pool_recall=float(np.mean(found)) if found else 0.0,
        prefilter_path=path,
        stage_latency_ms={},
        cost_usd=0.0,
        profile_id=profile.profile_id,
        elapsed_ms=_since(started),
        degraded=_degraded(ctx, cfg, fold, profile, "query"),
        n_cases=len(cases),
    )


async def run_system(
    cfg: SystemConfig,
    fold: Fold,
    ctx: EvalContext,
    *,
    condition: EvalCondition = "unconditioned",
    cases: Sequence[QueryCase] = (),
) -> RunResult:
    """One arm on one fold. The compute is synchronous, so it crosses into a worker thread."""
    if condition in QUERY_CONDITIONS:
        return await _run_queries(ctx, cfg, fold, cases, condition=condition)
    return await anyio.to_thread.run_sync(partial(_run_unconditioned, ctx, cfg, fold))


async def run_matrix(
    cfgs: Sequence[SystemConfig],
    ctx: EvalContext,
    *,
    conditions: Sequence[EvalCondition] = ("unconditioned",),
    cases: Mapping[int, Sequence[QueryCase]] | None = None,
    force: bool = False,
) -> list[RunResult]:
    """Every arm on every fold, skipping what is already stored and what cannot run yet."""
    done = frozenset() if force else _already(ctx)
    out: list[RunResult] = []
    for cfg in cfgs:
        blocked = blocked_reason(cfg)
        if blocked is not None:
            continue
        for fold in ctx.split.folds:
            for condition in conditions:
                if (cfg.name, ctx.split.name, fold.fold, condition, cfg.config_sha) in done:
                    continue
                out.append(
                    await run_system(
                        cfg,
                        fold,
                        ctx,
                        condition=condition,
                        cases=(cases or {}).get(fold.fold, ()),
                    )
                )
    return out


def _already(ctx: EvalContext) -> frozenset[tuple[str, str, int, str, str]]:
    rows = ctx.db.read().execute(
        "select system, split_name, fold, condition, config_sha from eval_run where split_name = ?",
        (ctx.split.name,),
    )
    return frozenset(
        (
            str(r["system"]),
            str(r["split_name"]),
            int(r["fold"]),
            str(r["condition"]),
            str(r["config_sha"]),
        )
        for r in rows
    )


def persist(
    ctx: EvalContext, cfgs: Mapping[str, SystemConfig], results: Sequence[RunResult]
) -> None:
    """Write the runs, their top fifty and every metric, so the report reads the database."""
    with ctx.db.write() as conn:
        for run in results:
            cfg = cfgs[run.system]
            conn.execute(
                "insert into eval_run (run_id, system, config_json, config_sha, split_name, fold, "
                "condition, profile_id, code_sha, seed, started_at, elapsed_ms, cost_usd, "
                "pool_size, pool_recall, prefilter_path) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.system,
                    orjson.dumps(cfg.to_row()).decode(),
                    run.config_sha,
                    ctx.split.name,
                    run.fold,
                    run.condition,
                    run.profile_id,
                    ctx.code_sha,
                    ctx.seed,
                    now_iso(),
                    run.elapsed_ms,
                    run.cost_usd,
                    run.pool_size,
                    run.pool_recall,
                    run.prefilter_path,
                ),
            )
            conn.executemany(
                "insert into eval_ranking (run_id, rank, tmdb_id, score) values (?,?,?,?)",
                [
                    (run.run_id, rank, tmdb_id, run.scores.get(tmdb_id, 0.0))
                    for rank, tmdb_id in enumerate(run.ranked, start=1)
                ],
            )
            conn.executemany(
                "insert into eval_metric (run_id, metric, value, ci_lo, ci_hi, n) "
                "values (?,?,?,?,?,?)",
                [
                    (run.run_id, name, ci.point, ci.lo, ci.hi, ci.n)
                    for name, ci in run.metrics.items()
                ],
            )


def deltas(
    ctx: EvalContext,
    results: Sequence[RunResult],
    *,
    reference: str = "full",
    metric: str = "ndcg@10",
    condition: EvalCondition = "unconditioned",
) -> list[tuple[str, MetricCI, int, int]]:
    """Paired bootstrap of every arm against one reference, fold by fold, on the same draw."""
    by_system: dict[str, dict[int, RunResult]] = {}
    for run in results:
        if run.condition == condition:
            by_system.setdefault(run.system, {})[run.fold] = run
    # The per fold deltas are averaged. That is a summary of five intervals, not a pooled one.
    base = by_system.get(reference, {})
    out: list[tuple[str, MetricCI, int, int]] = []
    graded = partial(ndcg_at_k, k=NDCG_K)
    for system, runs in sorted(by_system.items()):
        if system == reference:
            continue
        points: list[float] = []
        los: list[float] = []
        his: list[float] = []
        positive = 0
        for fold in ctx.split.folds:
            here = runs.get(fold.fold)
            there = base.get(fold.fold)
            if here is None or there is None:
                continue
            ci, _ = paired_bootstrap(
                here.ranked,
                there.ranked,
                fold.rel,
                graded,
                n_resamples=ctx.resamples,
                seed=ctx.seed + fold.fold,
            )
            points.append(ci.point)
            los.append(ci.lo)
            his.append(ci.hi)
            positive += int(ci.point > 0.0)
        if not points:
            continue
        summary = MetricCI(
            float(np.mean(points)), float(np.mean(los)), float(np.mean(his)), len(points)
        )
        out.append((system, summary, positive, len(points)))
    _store_deltas(ctx, out, reference=reference, metric=metric, condition=condition)
    return out


def _store_deltas(
    ctx: EvalContext,
    rows: Sequence[tuple[str, MetricCI, int, int]],
    *,
    reference: str,
    metric: str,
    condition: str,
) -> None:
    with ctx.db.write() as conn:
        conn.executemany(
            "insert or replace into eval_delta (split_name, system, reference, condition, metric, "
            "delta, ci_lo, ci_hi, folds_positive, folds_total) values (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    ctx.split.name,
                    system,
                    reference,
                    condition,
                    metric,
                    ci.point,
                    ci.lo,
                    ci.hi,
                    positive,
                    total,
                )
                for system, ci, positive, total in rows
            ],
        )
