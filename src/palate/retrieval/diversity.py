"""Mode slots, a composite redundancy, and the two repair passes over the caps."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from palate.retrieval.features import FilmFacets
from palate.taste.modes import MIN_MODE_COHERENCE, MIN_MODE_MEMBERS, TasteMode


@dataclass(frozen=True, slots=True)
class DiversityConfig:
    """Every knob the final ten are shaped by."""

    lambda_mmr: float = 0.7
    w_embed: float = 0.6
    w_director: float = 0.25
    w_collection: float = 0.15
    max_per_director: int = 2
    max_per_decade: int = 3
    max_per_collection: int = 1
    max_known_directors: int = 4
    query_posterior_concentrated: float = 0.6


@dataclass(frozen=True, slots=True)
class DiversityReport:
    """What the slots and the caps actually did."""

    slots: Mapping[int, int]
    quotas_applied: bool
    posterior_max: float
    dropped_by_cap: Mapping[str, int]
    known_directors: int


def query_mode_posterior(
    query_emb: Sequence[float], modes: Sequence[TasteMode], *, temperature: float = 0.07
) -> dict[int, float]:
    """softmax over cos(query, centroid) / temperature."""
    if not modes:
        return {}
    query = np.asarray(query_emb, dtype=np.float64)
    scale = float(np.linalg.norm(query)) or 1.0
    centroids = np.array([m.centroid for m in modes], dtype=np.float64)
    cosines = centroids @ query / scale
    terms = cosines / temperature
    weights = np.exp(terms - terms.max())
    weights /= weights.sum()
    return {m.mode_id: float(w) for m, w in zip(modes, weights, strict=True)}


def allocate_slots(
    modes: Sequence[TasteMode],
    n: int,
    cfg: DiversityConfig,
    *,
    posterior: Mapping[int, float] | None = None,
) -> dict[int, int]:
    """Profile mass without a query, the posterior with one, nothing when it is concentrated."""
    if n <= 0 or not modes:
        return {}
    if posterior is not None:
        if posterior and max(posterior.values()) > cfg.query_posterior_concentrated:
            return {}
        weights = {m.mode_id: float(posterior.get(m.mode_id, 0.0)) for m in modes}
    else:
        # Square root, so a dominant mode does not take nine slots of ten.
        weights = {m.mode_id: m.confidence * math.sqrt(m.mass) for m in modes}
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    exact = {mode_id: n * w / total for mode_id, w in weights.items()}
    slots = {mode_id: int(value) for mode_id, value in exact.items()}
    for mode in modes:
        # The floor is size and shape, never confidence, or a sharp new interest never shows.
        if mode.n_members >= MIN_MODE_MEMBERS and mode.coherence >= MIN_MODE_COHERENCE:
            slots[mode.mode_id] = max(slots[mode.mode_id], 1)
    _balance(slots, exact, n)
    return {mode_id: value for mode_id, value in slots.items() if value > 0}


def _balance(slots: dict[int, int], exact: Mapping[int, float], n: int) -> None:
    while sum(slots.values()) < n:
        pick = max(exact, key=lambda k: (exact[k] - slots[k], -k))
        slots[pick] += 1
    while sum(slots.values()) > n:
        pick = max(slots, key=lambda k: (slots[k] - exact[k], -k))
        if slots[pick] <= 0:
            return
        slots[pick] -= 1


def _normalised(ranked: Sequence[int], scores: Mapping[int, float]) -> dict[int, float]:
    if not ranked:
        return {}
    values = [scores[i] for i in ranked]
    low, high = min(values), max(values)
    spread = high - low
    if spread <= 0.0:
        return dict.fromkeys(ranked, 1.0)
    return {i: (scores[i] - low) / spread for i in ranked}


def _redundancy(
    candidate: int,
    selected: Sequence[int],
    embeddings: Mapping[int, Sequence[float]],
    meta: Mapping[int, FilmFacets],
    cfg: DiversityConfig,
) -> float:
    """Two films by one director are near duplicates here whatever their embeddings say."""
    if not selected:
        return 0.0
    mine = embeddings.get(candidate)
    facets = meta.get(candidate)
    worst = 0.0
    for other in selected:
        theirs = embeddings.get(other)
        cosine = (
            float(np.dot(mine, theirs) / (np.linalg.norm(mine) * np.linalg.norm(theirs)))
            if mine is not None and theirs is not None
            else 0.0
        )
        share = meta.get(other)
        same_director = (
            1.0
            if facets is not None
            and share is not None
            and set(facets.directors) & set(share.directors)
            else 0.0
        )
        same_collection = (
            1.0
            if facets is not None
            and share is not None
            and facets.collection_id is not None
            and facets.collection_id == share.collection_id
            else 0.0
        )
        worst = max(
            worst,
            cfg.w_embed * cosine
            + cfg.w_director * same_director
            + cfg.w_collection * same_collection,
        )
    return worst


def _pick(
    pool: Sequence[int],
    selected: list[int],
    normalised: Mapping[int, float],
    embeddings: Mapping[int, Sequence[float]],
    meta: Mapping[int, FilmFacets],
    cfg: DiversityConfig,
) -> int | None:
    best: tuple[float, int] | None = None
    for candidate in pool:
        value = cfg.lambda_mmr * normalised[candidate] - (1.0 - cfg.lambda_mmr) * _redundancy(
            candidate, selected, embeddings, meta, cfg
        )
        if best is None or (value, -candidate) > (best[0], -best[1]):
            best = (value, candidate)
    return None if best is None else best[1]


def _caps_ok(
    candidate: int, selected: Sequence[int], meta: Mapping[int, FilmFacets], cfg: DiversityConfig
) -> str:
    facets = meta.get(candidate)
    if facets is None:
        return ""
    chosen = [meta[i] for i in selected if i in meta]
    directors = set(facets.directors)
    if directors and sum(1 for f in chosen if directors & set(f.directors)) >= cfg.max_per_director:
        return "director"
    if sum(1 for f in chosen if f.decade == facets.decade) >= cfg.max_per_decade:
        return "decade"
    if (
        facets.collection_id is not None
        and sum(1 for f in chosen if f.collection_id == facets.collection_id)
        >= cfg.max_per_collection
    ):
        return "collection"
    return ""


def diversify(
    ranked: Sequence[int],
    scores: Mapping[int, float],
    embeddings: Mapping[int, Sequence[float]],
    meta: Mapping[int, FilmFacets],
    mode_of: Mapping[int, int],
    known_directors: frozenset[int],
    n: int,
    cfg: DiversityConfig,
    *,
    posterior: Mapping[int, float] | None = None,
    modes: Sequence[TasteMode] = (),
) -> tuple[list[int], DiversityReport]:
    """Quotas, then greedy MMR inside each quota, then the caps, then the novelty cap."""
    normalised = _normalised(ranked, scores)
    slots = allocate_slots(modes, n, cfg, posterior=posterior)
    dropped: dict[str, int] = {}
    selected: list[int] = []
    remaining = list(ranked)
    for mode_id in sorted(slots, key=lambda m: (-slots[m], m)):
        for _ in range(slots[mode_id]):
            if len(selected) >= n:
                break
            pool = [i for i in remaining if mode_of.get(i) == mode_id]
            pool = _without_cap_breaks(pool, selected, meta, cfg, dropped)
            pick = _pick(pool, selected, normalised, embeddings, meta, cfg)
            if pick is None:
                break
            selected.append(pick)
            remaining.remove(pick)
    while len(selected) < n:
        pool = _without_cap_breaks(remaining, selected, meta, cfg, dropped)
        pick = _pick(pool, selected, normalised, embeddings, meta, cfg)
        if pick is None:
            break
        selected.append(pick)
        remaining.remove(pick)
    selected = _novelty(selected, remaining, normalised, embeddings, meta, cfg, known_directors)
    selected.sort(key=lambda i: (-scores[i], i))
    known = sum(1 for i in selected if i in meta and set(meta[i].directors) & known_directors)
    return selected, DiversityReport(
        slots=slots,
        quotas_applied=bool(slots),
        posterior_max=max(posterior.values()) if posterior else 0.0,
        dropped_by_cap=dropped,
        known_directors=known,
    )


def _without_cap_breaks(
    pool: Sequence[int],
    selected: Sequence[int],
    meta: Mapping[int, FilmFacets],
    cfg: DiversityConfig,
    dropped: dict[str, int],
) -> list[int]:
    out: list[int] = []
    for candidate in pool:
        reason = _caps_ok(candidate, selected, meta, cfg)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        out.append(candidate)
    return out


def _novelty(
    selected: list[int],
    remaining: Sequence[int],
    normalised: Mapping[int, float],
    embeddings: Mapping[int, Sequence[float]],
    meta: Mapping[int, FilmFacets],
    cfg: DiversityConfig,
    known_directors: frozenset[int],
) -> list[int]:
    """A correct ranking of ten directors the user already knows is a useless recommendation."""
    if not known_directors:
        return selected

    def is_known(tmdb_id: int) -> bool:
        facets = meta.get(tmdb_id)
        return facets is not None and bool(set(facets.directors) & known_directors)

    pool = [i for i in remaining if not is_known(i)]
    while sum(1 for i in selected if is_known(i)) > cfg.max_known_directors:
        worst = min((i for i in selected if is_known(i)), key=lambda i: (normalised[i], -i))
        selected.remove(worst)
        fresh = _without_cap_breaks(pool, selected, meta, cfg, {})
        pick = _pick(fresh, selected, normalised, embeddings, meta, cfg)
        if pick is None:
            selected.append(worst)
            return selected
        selected.append(pick)
        pool.remove(pick)
    return selected
