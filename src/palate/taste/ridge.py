"""The discriminative direction, with exact leave one out so lambda costs one SVD."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# At 150 examples in 768 dimensions the fit is noise whatever the LOOCV says.
MIN_RIDGE_N = 150

LAMBDAS = tuple(float(10.0**power) for power in np.arange(-1.0, 4.0, 0.25))


@dataclass(frozen=True, slots=True)
class PreferenceDirection:
    """One fitted direction through the embedding plus metadata space."""

    w: np.ndarray
    b: float
    lam: float
    leverage_diag: np.ndarray
    x_mean: np.ndarray
    sigma2: float
    n_fit: int
    loocv_r2: float
    feature_names: tuple[str, ...] = ()
    # Kept in memory during a fit so explain() can use the exact quadratic form. Never persisted.
    svd_u: np.ndarray | None = None
    svd_s: np.ndarray | None = None
    svd_vt: np.ndarray | None = None


def fit_preference_direction(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lambdas: Sequence[float] = LAMBDAS,
    feature_names: Sequence[str] = (),
) -> PreferenceDirection:
    """Ridge with exact leave one out over the whole lambda grid from a single SVD."""
    design = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError(f"design {design.shape} does not match {target.size} targets")
    if not lambdas:
        raise ValueError("the lambda grid is empty")
    n = design.shape[0]
    x_mean = design.mean(axis=0)
    y_mean = float(target.mean())
    centered = design - x_mean
    residual = target - y_mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    projected = u.T @ residual
    squared = s**2

    best_lam = float(lambdas[0])
    best_press = np.inf
    for candidate in lambdas:
        press = float((_loo(u, squared, projected, residual, float(candidate)) ** 2).sum())
        if press < best_press:
            best_lam, best_press = float(candidate), press

    shrink = squared / (squared + best_lam)
    w = vt.T @ (s / (squared + best_lam) * projected)
    fitted = centered @ w
    rss = float(((residual - fitted) ** 2).sum())
    total = float(residual @ residual)
    return PreferenceDirection(
        w=w,
        b=y_mean - float(x_mean @ w),
        lam=best_lam,
        leverage_diag=_inverse_diagonal(vt, squared, best_lam),
        x_mean=x_mean,
        sigma2=rss / max(n - float(shrink.sum()), 1.0),
        n_fit=n,
        loocv_r2=1.0 - best_press / total if total > 0.0 else 0.0,
        feature_names=tuple(feature_names),
        svd_u=u,
        svd_s=s,
        svd_vt=vt,
    )


def loocv_residuals(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Leave one out residuals for one lambda, without refitting anything."""
    design = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    centered = design - design.mean(axis=0)
    residual = target - float(target.mean())
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return _loo(u, s**2, u.T @ residual, residual, lam)


def predict_with_leverage(
    direction: PreferenceDirection, Xc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted signal, and how far outside the sampled directions each row sits."""
    rows = np.atleast_2d(np.asarray(Xc, dtype=np.float64))
    prediction = rows @ direction.w + direction.b
    centered = rows - direction.x_mean
    if direction.svd_vt is None or direction.svd_s is None:
        # Only the diagonal survives persistence, so a loaded profile gets its approximation.
        return prediction, (centered**2) @ direction.leverage_diag
    projected = (centered @ direction.svd_vt.T) ** 2
    squared = direction.svd_s**2
    inside = projected @ (1.0 / (squared + direction.lam))
    outside = np.clip((centered**2).sum(axis=1) - projected.sum(axis=1), 0.0, None)
    return prediction, inside + outside / direction.lam


def _loo(
    u: np.ndarray,
    squared: np.ndarray,
    projected: np.ndarray,
    residual: np.ndarray,
    lam: float,
) -> np.ndarray:
    """(y_i - yhat_i) / (1 - H_ii), which is the leave one out residual exactly."""
    shrink = squared / (squared + lam)
    hat = (u**2) @ shrink
    return np.asarray((residual - u @ (shrink * projected)) / np.clip(1.0 - hat, 1e-12, None))


def _inverse_diagonal(vt: np.ndarray, squared: np.ndarray, lam: float) -> np.ndarray:
    """Diagonal of (X^T X + lam I) inverse, including the directions the fit never saw."""
    loadings = vt**2
    inside = loadings.T @ (1.0 / (squared + lam))
    return np.asarray(inside + (1.0 - loadings.sum(axis=0)) / lam)
