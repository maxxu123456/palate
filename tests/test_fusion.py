"""Fitting the weights on the metric itself, and the guards that reject a bad fit."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from fixtures.synth import pipeline

from palate.retrieval.features import FEATURES, PENALTY_FEATURES
from palate.retrieval.fusion import (
    PRIOR_BETA,
    Condition,
    FusionFold,
    FusionQuery,
    FusionWeights,
    active_features,
    check_signs,
    combine,
    fit_fusion_weights,
    load_weights,
    mean_ndcg,
    ndcg_at_k,
    prior_weights,
    save_weights,
    unit_l1,
)

# Small enough to fit inside a test, large enough that the ascent has something to find.
RESTARTS = 2
ROUNDS = 8

TRUTH = {
    "mode_affinity": 1.5,
    "ridge_pref": 1.0,
    "director_aff": 0.8,
    "anti_affinity": -1.2,
    "query_sim": 1.4,
    "bm25": 0.4,
    "popularity": -0.3,
}

UNCONDITIONED = ("query_sim", "bm25", "ce_score")


def truth_vector() -> np.ndarray:
    return np.array([TRUTH.get(name, 0.0) for name in FEATURES], dtype=np.float64)


def make_query(rng: np.random.Generator, start: int, *, conditioned: bool) -> FusionQuery:
    """A pool whose graded relevance really does follow one weight vector."""
    n = 60
    X = rng.normal(size=(n, len(FEATURES)))
    if not conditioned:
        for name in UNCONDITIONED:
            X[:, FEATURES.index(name)] = 0.0
    X[:, FEATURES.index("ce_score")] = 0.0
    utility = X @ truth_vector() + rng.normal(0.0, 0.1, size=n)
    order = np.argsort(-utility)
    gain = np.zeros(n)
    gain[order[:3]] = 3.0
    gain[order[3:9]] = 2.0
    gain[order[9:20]] = 1.0
    return FusionQuery(X=X, gain=gain, ids=tuple(range(start, start + n)))


def make_folds(seed: int, *, conditioned: bool, folds: int = 2) -> list[FusionFold]:
    rng = np.random.default_rng(seed)
    out: list[FusionFold] = []
    for index in range(folds):
        inner = tuple(
            make_query(rng, 1000 * index + 100 * i, conditioned=conditioned) for i in range(6)
        )
        val = tuple(
            make_query(rng, 5000 + 1000 * index + 10 * i, conditioned=conditioned) for i in range(2)
        )
        out.append(FusionFold(f"fold{index}", inner, val))
    return out


def prior_vector(active: Sequence[str]) -> np.ndarray:
    weights = prior_weights(active, condition="query")
    return weights.vector(FEATURES)


def test_a_the_fit_beats_the_signed_prior() -> None:
    folds = make_folds(11, conditioned=True)
    fitted = fit_fusion_weights(
        folds, condition="query", seed=1, restarts=RESTARTS, max_rounds=ROUNDS
    )
    val = [q for fold in folds for q in fold.val]
    active = active_features([q for fold in folds for q in fold.inner], FEATURES)
    assert not fitted.fallback
    assert mean_ndcg(val, fitted.vector(FEATURES)) >= mean_ndcg(val, prior_vector(active))


@pytest.mark.parametrize("condition", ["query", "unconditioned"])
def test_b_no_penalty_feature_comes_out_positive(condition: Condition) -> None:
    folds = make_folds(3, conditioned=condition == "query")
    fitted = fit_fusion_weights(
        folds, condition=condition, seed=0, restarts=RESTARTS, max_rounds=ROUNDS
    )
    assert fitted.condition == condition
    for name in PENALTY_FEATURES:
        assert fitted.beta.get(name, 0.0) <= 0.0, name
    assert check_signs(fitted.beta, condition=condition) == ""


def test_b2_the_sign_guard_names_what_went_wrong() -> None:
    assert "anti_affinity" in check_signs({"anti_affinity": 0.4}, condition="query")
    assert "query_sim" in check_signs({"query_sim": 0.0}, condition="query")
    assert check_signs({"query_sim": 0.0}, condition="unconditioned") == ""
    reranked = {"query_sim": 1.0, "ce_score": -1.0}
    assert "ce_score" in check_signs(reranked, condition="query", reranked=True)
    assert check_signs(reranked, condition="query") == ""


def test_c_the_query_fit_keeps_the_words_in_the_ordering() -> None:
    fitted = fit_fusion_weights(
        make_folds(5, conditioned=True),
        condition="query",
        seed=2,
        restarts=RESTARTS,
        max_rounds=ROUNDS,
    )
    assert fitted.beta["query_sim"] > 0.0


def test_d_an_unconditioned_fit_leaves_the_query_columns_unidentified() -> None:
    folds = make_folds(7, conditioned=False)
    fitted = fit_fusion_weights(
        folds, condition="unconditioned", seed=3, restarts=RESTARTS, max_rounds=ROUNDS
    )
    assert "query_sim" not in fitted.beta
    assert "bm25" not in fitted.beta
    assert "ce_score" not in fitted.beta


def test_e_the_gap_guard_falls_back_to_the_signed_prior() -> None:
    rng = np.random.default_rng(21)
    folds = make_folds(9, conditioned=True, folds=1)
    noisy = tuple(FusionQuery(X=q.X, gain=rng.permutation(q.gain), ids=q.ids) for q in folds[0].val)
    guarded = fit_fusion_weights(
        [FusionFold("fold0", folds[0].inner, noisy)],
        condition="query",
        seed=4,
        restarts=RESTARTS,
        max_rounds=ROUNDS,
    )
    assert guarded.fallback
    assert guarded.overfit_gap == 0.0
    assert guarded.beta["anti_affinity"] < 0.0
    assert len(set(guarded.beta.values())) > 1


def test_f_the_fallback_is_the_prior_and_not_uniform() -> None:
    active = [name for name in FEATURES if name != "ce_score"]
    weights = prior_weights(active, condition="unconditioned")
    assert weights.fallback
    assert sum(abs(v) for v in weights.beta.values()) == pytest.approx(1.0)
    scale = weights.beta["mode_affinity"] / PRIOR_BETA["mode_affinity"]
    for name, value in weights.beta.items():
        assert value == pytest.approx(PRIOR_BETA[name] * scale)
    assert "popularity" not in weights.beta


def test_g_ndcg_breaks_ties_on_film_id() -> None:
    gain = np.array([0.0, 3.0])
    tied = np.array([1.0, 1.0])
    assert ndcg_at_k(tied, gain, (1, 2)) == pytest.approx(3.0 / np.log2(3) / 3.0)
    assert ndcg_at_k(tied, gain, (9, 2)) == pytest.approx(1.0)
    assert ndcg_at_k(np.zeros(2), np.zeros(2), (1, 2)) == 0.0


def test_h_combine_takes_the_per_feature_median() -> None:
    fits = [
        FusionWeights(condition="query", beta={"mode_affinity": v, "anti_affinity": -1.0})
        for v in (1.0, 2.0, 9.0)
    ]
    merged = combine(fits)
    assert merged.beta["mode_affinity"] == pytest.approx(2.0 / 3.0)
    assert merged.beta["anti_affinity"] == pytest.approx(-1.0 / 3.0)
    assert merged.fitted_on == "median of 3 folds"


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("fusion"), docs=False)
    yield built
    built.close()


def test_i_weights_round_trip_through_the_profile(fitted: pipeline.Fitted) -> None:
    assert load_weights(fitted.db.read(), fitted.profile.profile_id) == {}
    published = fit_fusion_weights(
        make_folds(13, conditioned=True), condition="query", restarts=RESTARTS, max_rounds=ROUNDS
    )
    save_weights(fitted.db, fitted.profile.profile_id, {"query": published})
    reloaded = load_weights(fitted.db.read(), fitted.profile.profile_id)
    assert reloaded["query"] == published


def hostile_reranker(rng: np.random.Generator, start: int) -> FusionQuery:
    """A pool where the reranker column points backwards, which an unpinned fit will follow."""
    n = 60
    X = rng.normal(size=(n, len(FEATURES)))
    utility = X @ truth_vector() + rng.normal(0.0, 0.1, size=n)
    X[:, FEATURES.index("ce_score")] = -utility
    order = np.argsort(-utility)
    gain = np.zeros(n)
    gain[order[:3]] = 3.0
    gain[order[3:9]] = 2.0
    gain[order[9:20]] = 1.0
    return FusionQuery(X=X, gain=gain, ids=tuple(range(start, start + n)))


def reranked_folds(seed: int) -> list[FusionFold]:
    rng = np.random.default_rng(seed)
    inner = tuple(hostile_reranker(rng, 100 * i) for i in range(6))
    val = tuple(hostile_reranker(rng, 5000 + 10 * i) for i in range(2))
    return [FusionFold("fold0", inner, val)]


def test_j_a_fit_leaves_the_fitter_on_one_scale() -> None:
    fitted = fit_fusion_weights(
        make_folds(11, conditioned=True),
        condition="query",
        seed=1,
        restarts=RESTARTS,
        max_rounds=ROUNDS,
    )
    assert not fitted.fallback
    assert sum(abs(v) for v in fitted.beta.values()) == pytest.approx(1.0)


def test_k_a_longer_vector_does_not_own_the_median() -> None:
    small = FusionWeights(condition="query", beta={"mode_affinity": 0.5, "popularity": 0.5})
    large = FusionWeights(condition="query", beta={"mode_affinity": -0.6, "popularity": 2.4})
    # The raw mean of these two puts mode_affinity at -0.05, which no fold voted for.
    assert combine([small, large]).beta["mode_affinity"] > 0.0
    assert unit_l1(large.beta)["mode_affinity"] == pytest.approx(-0.2)


def test_l_the_reranker_column_is_pinned_when_a_reranker_is_active() -> None:
    unpinned = fit_fusion_weights(
        reranked_folds(17), condition="query", seed=5, restarts=RESTARTS, max_rounds=ROUNDS
    )
    assert unpinned.beta["ce_score"] < 0.0
    pinned = fit_fusion_weights(
        reranked_folds(17),
        condition="query",
        seed=5,
        restarts=RESTARTS,
        max_rounds=ROUNDS,
        reranked=True,
    )
    assert pinned.beta.get("ce_score", 0.0) >= 0.0
    assert check_signs(pinned.beta, condition="query", reranked=True) == ""
