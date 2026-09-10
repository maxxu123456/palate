"""The stage one score, and how much of it a thin history is allowed to claim."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from palate.retrieval.features import FeatureMatrix
from palate.retrieval.fusion import FusionWeights

TIER_CONFIDENCE: Mapping[str, float] = {"cold": 0.3, "thin": 0.65, "full": 1.0}

# How hard leverage above the pool median pulls a prediction's confidence down.
TAU = 1.0


@dataclass(frozen=True, slots=True)
class ScoredPool:
    """One pool with its score, its per feature share, and its confidence."""

    ids: tuple[int, ...]
    scores: np.ndarray
    contributions: np.ndarray
    names: tuple[str, ...]
    confidence: np.ndarray
    at: Mapping[int, int]

    def order(self) -> list[int]:
        """Best first, ties on tmdb_id so two runs never disagree."""
        return sorted(self.ids, key=lambda i: (-float(self.scores[self.at[i]]), i))

    def as_map(self) -> dict[int, float]:
        """Score per film."""
        return dict(zip(self.ids, (float(s) for s in self.scores), strict=True))

    def confidence_of(self, tmdb_id: int) -> float:
        """Confidence for one film."""
        return float(self.confidence[self.at[tmdb_id]])

    def shares(self, tmdb_id: int, *, limit: int = 6) -> dict[str, float]:
        """The features that moved this film, largest absolute contribution first."""
        row = self.contributions[self.at[tmdb_id]]
        live = [(n, float(v)) for n, v in zip(self.names, row, strict=True) if v]
        live.sort(key=lambda item: -abs(item[1]))
        return dict(live[:limit])


def leverage_ratio(matrix: FeatureMatrix) -> np.ndarray:
    """Each candidate's ridge leverage against the pool median, one where nothing was fitted."""
    raw = matrix.raw_column("ridge_leverage")
    mask = matrix.supported("ridge_leverage")
    if not mask.any():
        return np.ones(len(matrix.ids))
    middle = float(np.median(raw[mask])) or 1.0
    return np.where(mask, raw / middle, 1.0)


def score_pool(
    matrix: FeatureMatrix,
    weights: FusionWeights,
    *,
    tier: str,
    mode_confidence: Sequence[float] | None = None,
    tau: float = TAU,
) -> ScoredPool:
    """Weighted sum of the scaled features, with the confidence each row earns."""
    beta = weights.vector(matrix.names)
    contributions = matrix.X * beta
    scores = contributions.sum(axis=1) + weights.bias
    modes = (
        np.ones(len(matrix.ids))
        if mode_confidence is None
        else np.asarray(mode_confidence, dtype=np.float64)
    )
    confidence = (
        TIER_CONFIDENCE.get(tier, 1.0)
        * modes
        / (1.0 + tau * np.clip(leverage_ratio(matrix), 0.0, None))
    )
    at = {tmdb_id: i for i, tmdb_id in enumerate(matrix.ids)}
    return ScoredPool(matrix.ids, scores, contributions, matrix.names, confidence, at)
