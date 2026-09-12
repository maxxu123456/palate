"""Fusion weights, fitted per condition directly on the metric they are judged by."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import orjson

from palate.db.connect import Database
from palate.retrieval.features import FEATURES, PENALTY_FEATURES

type Condition = Literal["unconditioned", "query"]

PRIOR_BETA: dict[str, float] = {
    "mode_affinity": 1.0,
    "mode_margin": 1.0,
    "anti_affinity": -1.0,
    "ridge_pref": 1.0,
    "ridge_leverage": -0.5,
    "director_aff": 1.0,
    "writer_aff": 0.5,
    "actor_aff": 0.5,
    "keyword_aff": 0.75,
    "decade_aff": 0.5,
    "decade_exposure": 0.25,
    "runtime_aff": 0.25,
    "lang_aff": 0.5,
    "country_aff": 0.25,
    "country_penalty": -1.0,
    "soft_pref_penalty": -1.0,
    "popularity": 0.0,
    "vote_quality": 0.25,
    "bm25": 0.5,
    "query_sim": 1.0,
    "ce_score": 1.0,
}

WEIGHTS_VERSION = "1"

# Above this the fit is reading its own fold rather than the user, and the prior takes over.
OVERFIT_GAP_MAX = 0.05

DELTAS = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)

NDCG_K = 10


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """One weight vector, the condition it was fitted for, and how well it travelled."""

    condition: Condition
    beta: Mapping[str, float]
    # Rank metrics ignore an additive constant, so the bias is carried rather than fitted.
    bias: float = 0.0
    fitted_on: str = "prior"
    ndcg_inner: float = 0.0
    ndcg_val: float = 0.0
    overfit_gap: float = 0.0
    n_active: int = 0
    fallback: bool = False
    version: str = WEIGHTS_VERSION

    def vector(self, names: Sequence[str]) -> np.ndarray:
        """The weights in one feature matrix's column order."""
        return np.array([self.beta.get(name, 0.0) for name in names], dtype=np.float64)

    def to_row(self) -> dict[str, Any]:
        """The json shape stored on the profile."""
        return {
            "condition": self.condition,
            "beta": dict(self.beta),
            "bias": self.bias,
            "fitted_on": self.fitted_on,
            "ndcg_inner": self.ndcg_inner,
            "ndcg_val": self.ndcg_val,
            "overfit_gap": self.overfit_gap,
            "n_active": self.n_active,
            "fallback": self.fallback,
            "version": self.version,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> FusionWeights:
        """Rebuild from the stored json."""
        return cls(
            condition="query" if row["condition"] == "query" else "unconditioned",
            beta={str(k): float(v) for k, v in row["beta"].items()},
            bias=float(row.get("bias", 0.0)),
            fitted_on=str(row.get("fitted_on", "prior")),
            ndcg_inner=float(row.get("ndcg_inner", 0.0)),
            ndcg_val=float(row.get("ndcg_val", 0.0)),
            overfit_gap=float(row.get("overfit_gap", 0.0)),
            n_active=int(row.get("n_active", 0)),
            fallback=bool(row.get("fallback", False)),
            version=str(row.get("version", WEIGHTS_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class FusionQuery:
    """One scored pool with the graded relevance the metric is computed against."""

    X: np.ndarray
    gain: np.ndarray
    ids: tuple[int, ...]
    names: tuple[str, ...] = FEATURES


@dataclass(frozen=True, slots=True)
class FusionFold:
    """One split of the queries: fit on inner, publish the gap against val."""

    name: str
    inner: tuple[FusionQuery, ...]
    val: tuple[FusionQuery, ...]


def ndcg_at_k(
    scores: np.ndarray, gain: np.ndarray, ids: Sequence[int], *, k: int = NDCG_K
) -> float:
    """NDCG at k with ties broken on tmdb_id, so one ordering exists per score vector."""
    if gain.size == 0 or float(gain.max()) <= 0.0:
        return 0.0
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    got = np.array([gain[i] for i in order[:k]], dtype=np.float64)
    best = np.sort(gain)[::-1][:k]
    ideal = float(best @ discount[: best.size])
    return float(got @ discount[: got.size]) / ideal if ideal > 0.0 else 0.0


def mean_ndcg(queries: Sequence[FusionQuery], beta: np.ndarray, *, k: int = NDCG_K) -> float:
    """Mean NDCG over a set of pools for one weight vector."""
    if not queries:
        return 0.0
    return float(np.mean([ndcg_at_k(q.X @ beta, q.gain, q.ids, k=k) for q in queries]))


def check_signs(beta: Mapping[str, float], *, condition: str, reranked: bool = False) -> str:
    """Empty when every sign is defensible, the reason to reject the fit when it is not."""
    for name in sorted(PENALTY_FEATURES):
        if beta.get(name, 0.0) > 0.0:
            return f"{name} came out positive, which recommends what the user dislikes"
    if condition == "query" and beta.get("query_sim", 0.0) <= 0.0:
        return "query_sim is not positive, so the user's own words would not order the list"
    if reranked and beta.get("ce_score", 0.0) < 0.0:
        return "ce_score came out negative with a reranker active"
    return ""


def _bounds(
    names: Sequence[str], condition: str, *, reranked: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Half lines the search may not leave, so a sign violation is impossible by construction."""
    low = np.full(len(names), -4.0)
    high = np.full(len(names), 4.0)
    for i, name in enumerate(names):
        if name in PENALTY_FEATURES:
            high[i] = 0.0
        if name == "query_sim" and condition == "query":
            low[i] = 0.05
        if name == "ce_score" and reranked:
            low[i] = 0.0
    return low, high


def unit_l1(beta: Mapping[str, float]) -> dict[str, float]:
    """One scale for every vector, which a rank metric does not notice but a median does."""
    scale = sum(abs(v) for v in beta.values()) or 1.0
    return {name: value / scale for name, value in beta.items() if value}


def prior_share(name: str) -> float:
    """One feature's share of the signed prior, which is what an unfitted column is worth."""
    return unit_l1(PRIOR_BETA).get(name, 0.0)


def active_features(queries: Sequence[FusionQuery], names: Sequence[str]) -> tuple[str, ...]:
    """Columns some query actually produced. An all zero column has no identifiable weight."""
    if not queries:
        return ()
    live = np.zeros(len(names), dtype=bool)
    for query in queries:
        live |= np.any(query.X != 0.0, axis=0)
    return tuple(name for name, keep in zip(names, live, strict=True) if keep)


def prior_weights(
    active: Sequence[str],
    *,
    condition: Condition,
    fitted_on: str = "prior",
    ndcg_val: float = 0.0,
) -> FusionWeights:
    """The signed prior restricted to the live features and normalised to unit L1."""
    normalised = unit_l1({name: PRIOR_BETA.get(name, 0.0) for name in active})
    return FusionWeights(
        condition=condition,
        beta=normalised,
        fitted_on=fitted_on,
        ndcg_val=ndcg_val,
        n_active=len(normalised),
        fallback=True,
    )


def _ascend(
    queries: Sequence[FusionQuery],
    start: np.ndarray,
    columns: Sequence[int],
    low: np.ndarray,
    high: np.ndarray,
    *,
    deltas: Sequence[float],
    max_rounds: int,
    l1: float,
) -> tuple[np.ndarray, float]:
    beta = np.clip(start, low, high)

    def objective(vector: np.ndarray) -> float:
        return mean_ndcg(queries, vector) - l1 * float(np.abs(vector).sum())

    best = objective(beta)
    for _ in range(max_rounds):
        improved = False
        for column in columns:
            for delta in deltas:
                trial = beta.copy()
                trial[column] = float(np.clip(beta[column] + delta, low[column], high[column]))
                if trial[column] == beta[column]:
                    continue
                score = objective(trial)
                if score > best + 1e-12:
                    beta, best, improved = trial, score, True
        if not improved:
            break
    return beta, best


def _prune(
    beta: np.ndarray, columns: Sequence[int], max_active: int, low: np.ndarray
) -> np.ndarray:
    live = [c for c in columns if abs(beta[c]) > 1e-9]
    if len(live) <= max_active:
        return beta
    # A feature the condition requires to be positive is never the one L1 deletes.
    required = {c for c in live if low[c] > 0.0}
    rest = sorted((c for c in live if c not in required), key=lambda c: (-abs(beta[c]), c))
    keep = required | set(rest[: max(max_active - len(required), 0)])
    out = beta.copy()
    for column in live:
        if column not in keep:
            out[column] = 0.0
    return out


def fit_fusion_weights(
    folds: Sequence[FusionFold],
    *,
    condition: Condition,
    seed: int = 0,
    restarts: int = 8,
    deltas: Sequence[float] = DELTAS,
    max_rounds: int = 40,
    l1: float = 0.01,
    max_active: int = 10,
    reranked: bool = False,
) -> FusionWeights:
    """Coordinate ascent directly on validation NDCG@10. L1 drives dead features to zero."""
    if not folds:
        raise ValueError("no folds to fit on")
    inner = tuple(q for fold in folds for q in fold.inner)
    val = tuple(q for fold in folds for q in fold.val)
    if not inner:
        raise ValueError("no inner queries to fit on")
    names = inner[0].names
    active = active_features(inner, names)
    columns = [i for i, name in enumerate(names) if name in active]
    low, high = _bounds(names, condition, reranked=reranked)
    prior = np.array([PRIOR_BETA.get(name, 0.0) for name in names], dtype=np.float64)
    seeded = np.zeros(len(names))
    seeded[columns] = prior[columns]
    rng = np.random.default_rng(seed)
    best_beta, best_score = _ascend(
        inner, seeded, columns, low, high, deltas=deltas, max_rounds=max_rounds, l1=l1
    )
    for _ in range(max(restarts - 1, 0)):
        jittered = seeded.copy()
        jittered[columns] += rng.normal(0.0, 0.5, size=len(columns))
        beta, score = _ascend(
            inner, jittered, columns, low, high, deltas=deltas, max_rounds=max_rounds, l1=l1
        )
        if score > best_score:
            best_beta, best_score = beta, score
    best_beta = _prune(best_beta, columns, max_active, low)
    return _publish(best_beta, names, active, inner, val, condition, folds, reranked=reranked)


def _publish(
    beta: np.ndarray,
    names: Sequence[str],
    active: Sequence[str],
    inner: Sequence[FusionQuery],
    val: Sequence[FusionQuery],
    condition: Condition,
    folds: Sequence[FusionFold],
    *,
    reranked: bool = False,
) -> FusionWeights:
    fitted_on = ", ".join(fold.name for fold in folds)
    weights = {name: float(v) for name, v in zip(names, beta, strict=True) if v}
    ndcg_inner = mean_ndcg(inner, beta)
    ndcg_val = mean_ndcg(val, beta) if val else ndcg_inner
    gap = ndcg_inner - ndcg_val
    if gap > OVERFIT_GAP_MAX or check_signs(weights, condition=condition, reranked=reranked):
        return prior_weights(active, condition=condition, fitted_on=fitted_on, ndcg_val=ndcg_val)
    return FusionWeights(
        condition=condition,
        beta=unit_l1(weights),
        fitted_on=fitted_on,
        ndcg_inner=ndcg_inner,
        ndcg_val=ndcg_val,
        overfit_gap=gap,
        n_active=len(weights),
    )


def combine(fits: Sequence[FusionWeights]) -> FusionWeights:
    """The shipping vector is the per feature median across folds, not one fold's guess."""
    if not fits:
        raise ValueError("nothing to combine")
    # One vote each. A longer vector would otherwise carry the median on size alone.
    votes = [unit_l1(fit.beta) for fit in fits]
    names = sorted({name for vote in votes for name in vote})
    median = {name: float(np.median([vote.get(name, 0.0) for vote in votes])) for name in names}
    beta = unit_l1(median)
    return FusionWeights(
        condition=fits[0].condition,
        beta=beta,
        fitted_on=f"median of {len(fits)} folds",
        ndcg_inner=float(np.mean([f.ndcg_inner for f in fits])),
        ndcg_val=float(np.mean([f.ndcg_val for f in fits])),
        overfit_gap=float(np.mean([f.overfit_gap for f in fits])),
        n_active=len(beta),
        fallback=any(f.fallback for f in fits),
    )


def load_weights(conn: sqlite3.Connection, profile_id: str) -> dict[str, FusionWeights]:
    """Whatever the harness last published for this profile, keyed by condition."""
    row = conn.execute(
        "select fusion_weights_json from taste_profiles where profile_id = ?", (profile_id,)
    ).fetchone()
    if row is None or row["fusion_weights_json"] is None:
        return {}
    stored = orjson.loads(str(row["fusion_weights_json"]))
    return {str(k): FusionWeights.from_row(v) for k, v in stored.items()}


def save_weights(db: Database, profile_id: str, weights: Mapping[str, FusionWeights]) -> None:
    """Publish one weight vector per condition against a profile."""
    payload = orjson.dumps({k: v.to_row() for k, v in weights.items()}).decode()
    with db.write() as conn:
        conn.execute(
            "update taste_profiles set fusion_weights_json = ? where profile_id = ?",
            (payload, profile_id),
        )
