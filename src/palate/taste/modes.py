"""Multi centroid taste, because one centroid averages a person into someone who never existed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from palate.taste.signals import PreferenceSignal

KAPPA_MODE = 3.0
MIN_MODE_MEMBERS = 3
MIN_MODE_COHERENCE = 0.35

# Low enough to act as a max without hinging on one outlier centroid.
AFFINITY_T = 0.05

# A mild negative is not a dislike, it is a shrug, and shrugs do not deserve a centroid.
DISLIKE_CUT = -0.5

type Polarity = Literal["like", "dislike"]


@dataclass(frozen=True, slots=True)
class ModeMember:
    """One film's place in a mode."""

    tmdb_id: int
    cosine: float
    signal: float


@dataclass(frozen=True, slots=True)
class TasteMode:
    """One cluster of the history, with the confidence its own mass earns it."""

    mode_id: int
    polarity: Polarity
    centroid: np.ndarray
    mass: float
    n_members: int
    mean_signal: float
    coherence: float
    confidence: float
    exemplars: tuple[int, ...]
    members: tuple[ModeMember, ...] = ()
    label: str | None = None


def k_max_for(n_liked: int) -> int:
    """Scale the search range with history. 12 clusters over 40 films is reading noise."""
    return max(2, min(12, n_liked // 40))


def spherical_kmeans(
    X: np.ndarray,
    k: int,
    weights: np.ndarray,
    *,
    seed: int,
    n_init: int = 8,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means on the unit sphere. Centroids renormalised every iteration."""
    n = X.shape[0]
    if n == 0 or k < 1:
        raise ValueError(f"cannot cluster {n} points into {k} modes")
    k = min(k, n)
    best_score = -np.inf
    best: tuple[np.ndarray, np.ndarray] | None = None
    for attempt in range(n_init):
        rng = np.random.default_rng(seed + attempt)
        centroids = _plus_plus(X, k, weights, rng)
        previous = -np.inf
        for _ in range(max_iter):
            labels = np.argmax(X @ centroids.T, axis=1)
            centroids = _recentre(X, labels, weights, centroids)
            score = _objective(X, centroids, weights)
            if score - previous < tol:
                break
            previous = score
        labels = np.argmax(X @ centroids.T, axis=1)
        score = _objective(X, centroids, weights)
        if score > best_score:
            best_score, best = score, (centroids, labels)
    assert best is not None
    return best


def choose_k(
    X: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    k_min: int = 2,
    k_max: int | None = None,
) -> tuple[int, dict[int, float]]:
    """Weighted silhouette on cosine distance. Ties break toward smaller k."""
    n = X.shape[0]
    top = k_max_for(n) if k_max is None else k_max
    top = max(k_min, min(top, n - 1))
    distance = 1.0 - X @ X.T
    np.fill_diagonal(distance, 0.0)
    scores: dict[int, float] = {}
    best_k = k_min
    best_score = -np.inf
    for k in range(k_min, top + 1):
        _, labels = spherical_kmeans(X, k, weights, seed=seed)
        scores[k] = silhouette(distance, labels, weights)
        if scores[k] > best_score + 1e-9:
            best_k, best_score = k, scores[k]
    return best_k, scores


def silhouette(distance: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean silhouette. A point alone in its cluster scores zero, not one."""
    clusters = np.unique(labels)
    if clusters.size < 2:
        return 0.0
    n = labels.size
    totals = np.zeros((n, clusters.size))
    masses = np.zeros((n, clusters.size))
    for column, cluster in enumerate(clusters):
        member = weights * (labels == cluster)
        totals[:, column] = distance @ member
        masses[:, column] = member.sum()
    own = np.searchsorted(clusters, labels)
    rows = np.arange(n)
    masses[rows, own] -= weights
    inside = np.where(
        masses[rows, own] > 0, totals[rows, own] / np.maximum(masses[rows, own], 1e-12), 0.0
    )
    outside = totals / np.maximum(masses, 1e-12)
    outside[rows, own] = np.inf
    nearest = outside.min(axis=1)
    widest = np.maximum(inside, nearest)
    score = np.where(masses[rows, own] > 0, (nearest - inside) / np.maximum(widest, 1e-12), 0.0)
    return float(weights @ score / weights.sum())


def build_modes(
    signals: Sequence[PreferenceSignal],
    embeddings: Mapping[int, Sequence[float]],
    *,
    polarity: Polarity,
    seed: int,
    k: int | None = None,
) -> list[TasteMode]:
    """Like modes from s > 0, dislike modes from s < -0.5, thin or incoherent ones dropped."""
    picked = [s for s in signals if _in_polarity(s, polarity) and s.tmdb_id in embeddings]
    if len(picked) < MIN_MODE_MEMBERS:
        return []
    X = _unit_rows(np.array([embeddings[s.tmdb_id] for s in picked], dtype=np.float64))
    weights = np.maximum(np.array([s.mass for s in picked]), 1e-6)
    if k is None:
        k, _ = choose_k(X, weights, seed=seed, k_max=k_max_for(len(picked)))
    centroids, labels = spherical_kmeans(X, k, weights, seed=seed)
    modes: list[TasteMode] = []
    for cluster in range(centroids.shape[0]):
        rows = np.flatnonzero(labels == cluster)
        if rows.size < MIN_MODE_MEMBERS:
            continue
        cosines = X[rows] @ centroids[cluster]
        coherence = float(cosines.mean())
        if coherence < MIN_MODE_COHERENCE:
            continue
        members = tuple(
            ModeMember(picked[i].tmdb_id, float(c), picked[i].s)
            for i, c in zip(rows, cosines, strict=True)
        )
        mass = float(sum(picked[i].mass for i in rows))
        modes.append(
            TasteMode(
                mode_id=len(modes),
                polarity=polarity,
                centroid=centroids[cluster].astype(np.float32),
                mass=mass,
                n_members=len(members),
                mean_signal=float(np.mean([m.signal for m in members])),
                coherence=coherence,
                confidence=mass / (mass + KAPPA_MODE),
                exemplars=_exemplars(members, polarity),
                members=members,
            )
        )
    return modes


def mode_affinity(
    candidates: np.ndarray, modes: Sequence[TasteMode], *, temperature: float = AFFINITY_T
) -> np.ndarray:
    """Smooth max of cosine to each mode, with confidence as an additive log prior."""
    rows = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    if not modes:
        return np.zeros(rows.shape[0])
    centroids = np.array([m.centroid for m in modes], dtype=np.float64)
    prior = np.log(np.maximum([m.confidence for m in modes], 1e-9))
    terms = rows @ centroids.T / temperature + prior
    peak = terms.max(axis=1)
    return temperature * (peak + np.log(np.exp(terms - peak[:, None]).sum(axis=1)))


def mode_margin(
    candidates: np.ndarray,
    modes: Sequence[TasteMode],
    anti_modes: Sequence[TasteMode],
    *,
    temperature: float = AFFINITY_T,
) -> np.ndarray:
    """Liked affinity minus anti affinity, which is what a raw cosine cannot see."""
    liked = mode_affinity(candidates, modes, temperature=temperature)
    hated = mode_affinity(candidates, anti_modes, temperature=temperature)
    return np.asarray(liked - hated)


def mode_argmax(candidates: np.ndarray, modes: Sequence[TasteMode]) -> np.ndarray:
    """Nearest mode by raw cosine, no prior, because it drives slots and the explanation."""
    rows = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    if not modes:
        return np.full(rows.shape[0], -1)
    centroids = np.array([m.centroid for m in modes], dtype=np.float64)
    ids = np.array([m.mode_id for m in modes])
    return ids[np.argmax(rows @ centroids.T, axis=1)]


def _in_polarity(signal: PreferenceSignal, polarity: Polarity) -> bool:
    return signal.s > 0.0 if polarity == "like" else signal.s < DISLIKE_CUT


def _exemplars(members: Sequence[ModeMember], polarity: Polarity) -> tuple[int, ...]:
    sign = 1.0 if polarity == "like" else -1.0
    ordered = sorted(members, key=lambda m: sign * m.signal, reverse=True)
    return tuple(m.tmdb_id for m in ordered[:5])


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0.0, norms, 1.0)


def _objective(X: np.ndarray, centroids: np.ndarray, weights: np.ndarray) -> float:
    return float(weights @ np.max(X @ centroids.T, axis=1))


def _plus_plus(X: np.ndarray, k: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    picked = [int(rng.choice(X.shape[0], p=weights / weights.sum()))]
    closest = 1.0 - X @ X[picked[0]]
    for _ in range(1, k):
        chance = np.clip(closest, 0.0, None) * weights
        total = float(chance.sum())
        nxt = (
            int(rng.choice(X.shape[0], p=chance / total))
            if total > 0
            else int(rng.integers(X.shape[0]))
        )
        picked.append(nxt)
        closest = np.minimum(closest, 1.0 - X @ X[nxt])
    return X[picked].copy()


def _recentre(
    X: np.ndarray, labels: np.ndarray, weights: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    # Renormalising after every assignment is the whole difference from plain k-means.
    out = np.empty_like(centroids)
    for cluster in range(centroids.shape[0]):
        rows = np.flatnonzero(labels == cluster)
        if rows.size == 0:
            out[cluster] = X[int(np.argmin(np.max(X @ centroids.T, axis=1)))]
            continue
        out[cluster] = weights[rows] @ X[rows]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 0.0, norms, 1.0)
