"""The pool feature matrix, and the support mask that stops a sparse channel dying."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fixtures.synth import pipeline
from fixtures.synth.world import CLUSTERS, HATED, LOVED

from palate.retrieval.candidates import ChannelBudget, generate_candidates
from palate.retrieval.features import (
    FEATURES,
    MIN_SUPPORT,
    PENALTY_FEATURES,
    FeatureInputs,
    FeatureMatrix,
    bm25_scores,
    build_matrix,
    load_facets,
    scale_feature,
)

QUERY_TEXT = "a film about what happens"

# One dislike and one like, so the signed column has to come out both ways.
PENALTIES = {
    f"genre:{CLUSTERS[HATED[0]].genre}": -1.0,
    f"language:{CLUSTERS[LOVED[0]].language}": 0.5,
}

BUDGET = ChannelBudget(
    dense_per_mode=40, dense_query=40, bm25=40, people=60, keyword=40, popular=40
)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("features"))
    yield built
    built.close()


@pytest.fixture(scope="module")
def matrix(fitted: pipeline.Fitted) -> FeatureMatrix:
    store = fitted.store()
    centre = fitted.world.centres[LOVED[0]].tolist()
    pool = generate_candidates(
        store,
        fitted.profile,
        query_embedding=centre,
        query_text=QUERY_TEXT,
        budget=BUDGET,
    )
    inputs = FeatureInputs(
        vectors=store.vecs.vectors(pool.ids),
        bm25=bm25_scores(pool),
        query_embedding=centre,
        soft_countries=frozenset({CLUSTERS[HATED[0]].country}),
        penalties=PENALTIES,
    )
    return build_matrix(fitted.db.read(), fitted.profile, pool.ids, inputs)


def test_a_no_dead_columns(matrix: FeatureMatrix) -> None:
    for name in FEATURES:
        if int(matrix.supported(name).sum()) < MIN_SUPPORT:
            continue
        assert np.any(matrix.column(name) != 0.0), name


def test_b_absent_entries_are_exactly_zero(matrix: FeatureMatrix) -> None:
    for name in FEATURES:
        absent = ~matrix.supported(name)
        assert np.all(matrix.column(name)[absent] == 0.0), name


def test_c_a_partly_covered_channel_keeps_its_scale(matrix: FeatureMatrix) -> None:
    partial = [n for n in FEATURES if 0 < int(matrix.supported(n).sum()) < len(matrix.ids)]
    assert "bm25" in partial
    for name in partial:
        if int(matrix.supported(name).sum()) >= MIN_SUPPORT:
            assert matrix.scaled_by[FEATURES.index(name)] != "zeroed", name


def test_d_ignoring_the_mask_is_what_collapses_a_column() -> None:
    rng = np.random.default_rng(3)
    values = np.zeros(1200)
    support = np.zeros(1200, dtype=bool)
    support[:300] = True
    values[:300] = rng.normal(0.4, 0.2, size=300)
    masked, how = scale_feature(values, support)
    assert how == "mad"
    assert float(np.std(masked[support])) > 0.5
    # Over the whole pool three quarters of the rows are absent, so the spread reads as zero.
    _, naive = scale_feature(values, np.ones(1200, dtype=bool))
    assert naive == "rank"


def test_e_a_genuine_zero_is_not_an_absence() -> None:
    values = np.array([0.0] * 9 + [1.0])
    support = np.ones(10, dtype=bool)
    scaled, how = scale_feature(values, support)
    assert how in {"mad", "rank"}
    assert scaled[-1] != scaled[0]


def test_f_thin_support_zeroes_the_column() -> None:
    values = np.arange(20.0)
    support = np.zeros(20, dtype=bool)
    support[: MIN_SUPPORT - 1] = True
    scaled, how = scale_feature(values, support)
    assert how == "zeroed"
    assert not scaled.any()


def test_g_a_constant_column_is_zeroed_after_ranking() -> None:
    values = np.full(20, 2.5)
    scaled, how = scale_feature(values, np.ones(20, dtype=bool))
    assert how == "zeroed"
    assert not scaled.any()


def test_h_everything_stays_inside_the_clip(matrix: FeatureMatrix) -> None:
    assert np.isfinite(matrix.X).all()
    assert float(np.abs(matrix.X).max()) <= 3.0


def test_i_the_ridge_columns_line_up_with_the_fitted_names(
    fitted: pipeline.Fitted, matrix: FeatureMatrix
) -> None:
    direction = fitted.profile.direction
    assert direction is not None
    assert int(matrix.supported("ridge_pref").sum()) == len(matrix.ids)
    assert float(np.abs(matrix.raw_column("ridge_leverage")).max()) > 0.0
    assert set(direction.feature_names) > {"emb:0", "meta:log_runtime"}


def test_j_the_signed_penalty_column_comes_out_both_ways(
    fitted: pipeline.Fitted, matrix: FeatureMatrix
) -> None:
    hated = CLUSTERS[HATED[0]]
    loved = CLUSTERS[LOVED[0]]
    facets = load_facets(fitted.db.read(), matrix.ids)
    raw = matrix.raw_column("soft_pref_penalty")
    disliked = 0
    for position, tmdb_id in enumerate(matrix.ids):
        film = facets[tmdb_id]
        expected = 0.0
        if hated.genre in film.genres:
            expected += 1.0
            disliked += 1
        if film.language == loved.language:
            expected -= 0.5
        assert raw[position] == pytest.approx(expected)
    assert disliked
    assert set(FEATURES) >= PENALTY_FEATURES


def test_k_query_similarity_favours_the_queried_cluster(
    fitted: pipeline.Fitted, matrix: FeatureMatrix
) -> None:
    similarity = matrix.raw_column("query_sim")
    inside = [
        similarity[i]
        for i, tmdb_id in enumerate(matrix.ids)
        if fitted.world.cluster_of(tmdb_id) == LOVED[0]
    ]
    outside = [
        similarity[i]
        for i, tmdb_id in enumerate(matrix.ids)
        if fitted.world.cluster_of(tmdb_id) != LOVED[0]
    ]
    assert inside and outside
    assert float(np.mean(inside)) > float(np.mean(outside)) + 0.3
