"""Ranking metrics and their intervals. One user is one ranking, so the interval matters most."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

type Ranked = Sequence[Any]
type Judgements = Mapping[Any, int]
type MetricFn = Callable[[Ranked, Judgements], float]

DEFAULT_K = 10
RECALL_K = 50
POOL_K = 200
RESAMPLES = 1000


@dataclass(frozen=True, slots=True)
class MetricCI:
    """One metric with the interval that says whether it is worth reading."""

    point: float
    lo: float
    hi: float
    n: int

    @property
    def spans_zero(self) -> bool:
        """A delta whose interval covers zero is no measurable effect, never a small win."""
        return self.lo <= 0.0 <= self.hi


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain over an already ordered list of gains."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ideal_dcg(rel: Iterable[int], k: int = DEFAULT_K) -> float:
    """The best DCG these labels allow, truncated at k."""
    best = sorted(rel, reverse=True)[:k]
    return dcg([2.0**r - 1.0 for r in best])


def ndcg_at_k(
    ranked: Ranked, rel: Judgements, k: int = DEFAULT_K, *, idcg: float | None = None
) -> float:
    """NDCG at k. Anything outside the judged set scores zero, which biases low on purpose."""
    ideal = ideal_dcg(rel.values(), k) if idcg is None else idcg
    if ideal <= 0.0:
        return 0.0
    return dcg([2.0 ** rel.get(i, 0) - 1.0 for i in list(ranked)[:k]]) / ideal


def recall_at_k(ranked: Ranked, rel: Judgements, k: int = RECALL_K, min_rel: int = 2) -> float:
    """Share of the films the user actually liked that appear in the first k."""
    wanted = {i for i, r in rel.items() if r >= min_rel}
    if not wanted:
        return 0.0
    return len(wanted & set(list(ranked)[:k])) / len(wanted)


def mrr_at_k(ranked: Ranked, rel: Judgements, k: int = RECALL_K, min_rel: int = 1) -> float:
    """Reciprocal rank of the first relevant film, which is the query-mode headline."""
    for place, tmdb_id in enumerate(list(ranked)[:k], start=1):
        if rel.get(tmdb_id, 0) >= min_rel:
            return 1.0 / place
    return 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with averaged ties, zero when either side is constant."""
    if x.size < 2:
        return 0.0
    a = _average_ranks(x)
    b = _average_ranks(y)
    sa = float(a.std())
    sb = float(b.std())
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(((a - a.mean()) @ (b - b.mean())) / (a.size * sa * sb))


def spearman_conditional(
    ranked: Ranked,
    scores: Mapping[Any, float],
    actual: Mapping[Any, float],
    *,
    pool_k: int = POOL_K,
) -> tuple[float, int, float]:
    """Rho over the test films the system surfaced, which reads better than it deserves to."""
    kept = [i for i in list(ranked)[:pool_k] if i in actual]
    coverage = len(kept) / len(actual) if actual else 0.0
    if len(kept) < 2:
        return 0.0, len(kept), coverage
    x = np.array([scores.get(i, 0.0) for i in kept], dtype=np.float64)
    y = np.array([actual[i] for i in kept], dtype=np.float64)
    return spearman(x, y), len(kept), coverage


def spearman_pessimistic(
    scores: Mapping[Any, float], actual: Mapping[Any, float], *, floor: float
) -> tuple[float, int]:
    """Rho over every corpus-present test film, with the floor standing in for unranked ones."""
    items = sorted(actual)
    if len(items) < 2:
        return 0.0, len(items)
    x = np.array([scores.get(i, floor) for i in items], dtype=np.float64)
    y = np.array([actual[i] for i in items], dtype=np.float64)
    return spearman(x, y), len(items)


def intra_list_distance(ids: Sequence[Any], emb: Mapping[Any, Sequence[float]]) -> float:
    """Mean pairwise cosine distance inside one list, which is what diversity buys."""
    rows = [np.asarray(emb[i], dtype=np.float64) for i in ids if i in emb]
    if len(rows) < 2:
        return 0.0
    matrix = np.vstack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    similarity = unit @ unit.T
    upper = similarity[np.triu_indices(len(rows), k=1)]
    return float(1.0 - upper.mean())


def mode_coverage(ids: Sequence[Any], mode_of: Mapping[Any, int], n_modes: int) -> float:
    """How many taste modes a list touches, against the most it could have touched."""
    if not ids or n_modes <= 0:
        return 0.0
    reachable = min(n_modes, len(ids))
    return len({mode_of[i] for i in ids if i in mode_of}) / reachable


def novelty(ids: Sequence[Any], popularity: Mapping[Any, float]) -> float:
    """Mean popularity-rank surprisal, 0 for the corpus's most popular film and 1 for its least."""
    if not ids or len(popularity) < 2:
        return 0.0
    order = sorted(popularity, key=lambda i: (-popularity[i], i))
    place = {i: r for r, i in enumerate(order, start=1)}
    ceiling = math.log2(len(order))
    seen = [math.log2(place.get(i, len(order))) / ceiling for i in ids]
    return float(sum(seen) / len(seen))


def unknown_director_rate(
    ids: Sequence[Any], known_directors: Collection[int], *, directors: Mapping[Any, Sequence[int]]
) -> float:
    """Share of the list by directors the user has never rated, which novelty is paid for in."""
    if not ids:
        return 0.0
    known = set(known_directors)
    unknown = sum(1 for i in ids if not known & set(directors.get(i, ())))
    return unknown / len(ids)


def prefilter_recall_at_k(exact_ids: Sequence[Any], approx_ids: Sequence[Any], k: int) -> float:
    """What the over-fetch path lost against a brute force scan of the same allow set."""
    head = list(exact_ids)[:k]
    if not head:
        return 1.0
    return len(set(head) & set(list(approx_ids)[:k])) / len(head)


def _draw(rng: np.random.Generator, items: Sequence[Any]) -> set[Any]:
    picked = rng.integers(0, len(items), size=len(items))
    return {items[int(i)] for i in picked}


def bootstrap_ci(
    ranked: Ranked,
    rel: Judgements,
    metric_fn: MetricFn,
    *,
    n_resamples: int = RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> MetricCI:
    """Resample the judged item set, repeats collapsing to a slightly wider interval."""
    point = metric_fn(ranked, rel)
    items = sorted(rel)
    if len(items) < 2 or n_resamples <= 0:
        return MetricCI(point, point, point, len(items))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [metric_fn(ranked, {i: rel[i] for i in _draw(rng, items)}) for _ in range(n_resamples)]
    )
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return MetricCI(point, float(lo), float(hi), len(items))


def mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> MetricCI:
    """Mean over a query set, with an interval over which queries happened to land in it."""
    if not values:
        return MetricCI(0.0, 0.0, 0.0, 0)
    arr = np.asarray(values, dtype=np.float64)
    point = float(arr.mean())
    if arr.size < 2 or n_resamples <= 0:
        return MetricCI(point, point, point, int(arr.size))
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return MetricCI(point, float(lo), float(hi), int(arr.size))


def paired_bootstrap(
    ranked_a: Ranked,
    ranked_b: Ranked,
    rel: Judgements,
    metric_fn: MetricFn,
    *,
    n_resamples: int = RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[MetricCI, float]:
    """Delta interval and the share of resamples where a beat b. The same draw feeds both."""
    point = metric_fn(ranked_a, rel) - metric_fn(ranked_b, rel)
    items = sorted(rel)
    if len(items) < 2 or n_resamples <= 0:
        return MetricCI(point, point, point, len(items)), float(point > 0.0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for t in range(n_resamples):
        sample = {i: rel[i] for i in _draw(rng, items)}
        deltas[t] = metric_fn(ranked_a, sample) - metric_fn(ranked_b, sample)
    lo, hi = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return MetricCI(point, float(lo), float(hi), len(items)), float((deltas > 0.0).mean())
