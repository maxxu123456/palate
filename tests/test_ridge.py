"""The closed form leave one out is the whole reason lambda is affordable, so it is checked."""

from __future__ import annotations

import numpy as np
import pytest

from palate.taste.ridge import (
    LAMBDAS,
    fit_preference_direction,
    loocv_residuals,
    predict_with_leverage,
)


def sample(*, n: int = 40, p: int = 6, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    truth = rng.standard_normal(p)
    return X, X @ truth + rng.normal(0.0, 0.3, n) + 1.5


def brute_force(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Refit on every leave one out split of the same centred problem."""
    centered = X - X.mean(axis=0)
    residual = y - y.mean()
    out = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        kept = centered[keep]
        weights = np.linalg.solve(kept.T @ kept + lam * np.eye(X.shape[1]), kept.T @ residual[keep])
        out[i] = residual[i] - centered[i] @ weights
    return out


@pytest.mark.parametrize("lam", [0.1, 1.0, 17.5, 1000.0])
def test_the_closed_form_equals_brute_force_leave_one_out(lam: float) -> None:
    X, y = sample()
    assert loocv_residuals(X, y, lam) == pytest.approx(brute_force(X, y, lam), rel=1e-9)


def test_it_holds_when_there_are_more_features_than_examples() -> None:
    X, y = sample(n=12, p=30)
    assert loocv_residuals(X, y, 2.5) == pytest.approx(brute_force(X, y, 2.5), rel=1e-7)


def test_the_chosen_lambda_is_the_one_that_minimises_press() -> None:
    X, y = sample(n=60, p=8)
    direction = fit_preference_direction(X, y)
    press = {lam: float((loocv_residuals(X, y, lam) ** 2).sum()) for lam in LAMBDAS}
    assert direction.lam == min(press, key=lambda lam: press[lam])
    total = float(((y - y.mean()) ** 2).sum())
    assert direction.loocv_r2 == pytest.approx(1.0 - press[direction.lam] / total)


def test_a_fit_on_noise_admits_it_learned_nothing() -> None:
    rng = np.random.default_rng(11)
    direction = fit_preference_direction(rng.standard_normal((40, 25)), rng.standard_normal(40))
    assert direction.loocv_r2 < 0.2


def test_prediction_recovers_the_signal_it_was_fitted_on() -> None:
    X, y = sample(n=200, p=5)
    direction = fit_preference_direction(X, y)
    predicted, _ = predict_with_leverage(direction, X)
    assert float(np.corrcoef(predicted, y)[0, 1]) > 0.95


def test_leverage_is_higher_in_a_direction_the_history_never_sampled() -> None:
    rng = np.random.default_rng(5)
    flat = np.zeros((80, 4))
    flat[:, :2] = rng.standard_normal((80, 2))
    direction = fit_preference_direction(flat, rng.standard_normal(80), lambdas=(1.0,))
    seen = np.array([[1.0, 0.0, 0.0, 0.0]])
    unseen = np.array([[0.0, 0.0, 1.0, 0.0]])
    _, sampled = predict_with_leverage(direction, seen)
    _, extrapolated = predict_with_leverage(direction, unseen)
    assert extrapolated[0] > sampled[0] * 5


def test_the_stored_diagonal_still_ranks_extrapolation_above_interpolation() -> None:
    rng = np.random.default_rng(5)
    flat = np.zeros((80, 4))
    flat[:, :2] = rng.standard_normal((80, 2))
    fitted = fit_preference_direction(flat, rng.standard_normal(80), lambdas=(1.0,))
    stripped = type(fitted)(
        w=fitted.w,
        b=fitted.b,
        lam=fitted.lam,
        leverage_diag=fitted.leverage_diag,
        x_mean=fitted.x_mean,
        sigma2=fitted.sigma2,
        n_fit=fitted.n_fit,
        loocv_r2=fitted.loocv_r2,
    )
    _, sampled = predict_with_leverage(stripped, np.array([[1.0, 0.0, 0.0, 0.0]]))
    _, extrapolated = predict_with_leverage(stripped, np.array([[0.0, 0.0, 1.0, 0.0]]))
    assert extrapolated[0] > sampled[0]


def test_a_design_that_does_not_match_its_targets_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match"):
        fit_preference_direction(np.zeros((4, 3)), np.zeros(5))
