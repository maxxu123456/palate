"""The planted clusters are the answer key, so the fitter either finds them or it does not."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from fixtures.synth import HATED, LOVED, build_world

from palate.taste.modes import (
    MIN_MODE_COHERENCE,
    MIN_MODE_MEMBERS,
    TasteMode,
    build_modes,
    choose_k,
    k_max_for,
    mode_affinity,
    mode_argmax,
    mode_margin,
    spherical_kmeans,
)
from palate.taste.signals import PreferenceSignal, ResidualCalibrator


@pytest.fixture(scope="module")
def world() -> Any:
    return build_world()


@pytest.fixture(scope="module")
def signals(world: Any) -> list[PreferenceSignal]:
    return ResidualCalibrator().fit(world.rated, world.store()).transform(world.rated)


@pytest.fixture(scope="module")
def liked(world: Any, signals: list[PreferenceSignal]) -> list[TasteMode]:
    return build_modes(signals, world.vectors, polarity="like", seed=0)


@pytest.fixture(scope="module")
def disliked(world: Any, signals: list[PreferenceSignal]) -> list[TasteMode]:
    return build_modes(signals, world.vectors, polarity="dislike", seed=0)


def purity(world: Any, mode: TasteMode) -> tuple[int, float]:
    clusters = [world.cluster_of(m.tmdb_id) for m in mode.members]
    top = max(set(clusters), key=clusters.count)
    return top, clusters.count(top) / len(clusters)


def test_every_loved_cluster_comes_back_as_its_own_mode(world: Any, liked: list[TasteMode]) -> None:
    for planted in LOVED:
        centre = world.centres[planted]
        matched = [m for m in liked if float(m.centroid @ centre) > 0.85]
        assert matched, f"cluster {planted} was not recovered"
        best = max(matched, key=lambda m: float(m.centroid @ centre))
        found, share = purity(world, best)
        assert found == planted
        assert share > 0.7


def test_the_hated_cluster_comes_back_as_an_anti_mode(
    world: Any, disliked: list[TasteMode]
) -> None:
    for planted in HATED:
        centre = world.centres[planted]
        matched = [m for m in disliked if float(m.centroid @ centre) > 0.85]
        assert matched, f"cluster {planted} was not recovered"
        assert purity(world, matched[0])[0] == planted


def test_a_thin_or_incoherent_cluster_is_dropped(liked: list[TasteMode]) -> None:
    assert liked
    assert all(m.n_members >= MIN_MODE_MEMBERS for m in liked)
    assert all(m.coherence >= MIN_MODE_COHERENCE for m in liked)
    assert [m.mode_id for m in liked] == list(range(len(liked)))


def test_confidence_comes_from_mass_not_from_count(liked: list[TasteMode]) -> None:
    for mode in liked:
        assert mode.confidence == pytest.approx(mode.mass / (mode.mass + 3.0))
        assert 0.0 < mode.confidence < 1.0


def test_the_search_range_grows_with_the_history() -> None:
    assert k_max_for(0) == 2
    assert k_max_for(40) == 2
    assert k_max_for(120) == 3
    assert k_max_for(400) == 10
    assert k_max_for(5000) == 12


def test_centroids_stay_on_the_unit_sphere() -> None:
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((60, 12))
    X = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    centroids, labels = spherical_kmeans(X, 3, np.ones(60), seed=0)
    assert np.linalg.norm(centroids, axis=1) == pytest.approx(np.ones(3))
    assert set(labels.tolist()) <= {0, 1, 2}


def test_the_silhouette_prefers_the_planted_number_of_clusters(world: Any) -> None:
    ids = [i for planted in (0, 1, 2) for i in world.members(planted)[:60]]
    X = world.matrix(ids)
    best, scores = choose_k(X, np.ones(len(ids)), seed=0, k_min=2, k_max=6)
    assert best == 3
    assert scores[3] > scores[2]


def test_affinity_is_a_smooth_max_and_not_a_mean(world: Any, liked: list[TasteMode]) -> None:
    near = world.centres[LOVED[0]][None, :]
    closest = max(liked, key=lambda m: float(near[0] @ m.centroid))
    alone = float(mode_affinity(near, [closest])[0])
    crowded = float(mode_affinity(near, liked)[0])
    assert crowded == pytest.approx(alone, abs=0.02)
    assert crowded > float(np.mean([near[0] @ m.centroid for m in liked]))
    assert float(mode_affinity(near, [])[0]) == 0.0


def test_confidence_tilts_affinity_instead_of_suppressing_it(liked: list[TasteMode]) -> None:
    mode = liked[0]
    shy = replace(mode, mass=1.0, confidence=1.0 / 4.0)
    point = np.asarray(mode.centroid, dtype=np.float64)[None, :]
    strong = float(mode_affinity(point, [mode])[0])
    weak = float(mode_affinity(point, [shy])[0])
    assert strong - weak < 0.08


def test_the_margin_separates_a_loved_cluster_from_a_hated_one(
    world: Any, liked: list[TasteMode], disliked: list[TasteMode]
) -> None:
    loved = world.centres[LOVED[0]][None, :]
    hated = world.centres[HATED[0]][None, :]
    assert float(mode_margin(loved, liked, disliked)[0]) > 0.0
    assert float(mode_margin(hated, liked, disliked)[0]) < 0.0


def test_argmax_names_the_mode_the_film_actually_resembles(
    world: Any, liked: list[TasteMode]
) -> None:
    centres = np.array([world.centres[c] for c in LOVED])
    picked = mode_argmax(centres, liked)
    assert len(set(picked.tolist())) == len(LOVED)
    assert mode_argmax(centres, []).tolist() == [-1, -1]


def test_a_label_never_reaches_the_scoring_path(world: Any, liked: list[TasteMode]) -> None:
    point = world.centres[LOVED[0]][None, :]
    before = float(mode_affinity(point, liked)[0])
    labelled = [replace(m, label="the one about trains") for m in liked]
    assert float(mode_affinity(point, labelled)[0]) == before
    assert mode_argmax(point, labelled).tolist() == mode_argmax(point, liked).tolist()
    for scorer in (mode_affinity, mode_margin, mode_argmax):
        assert "label" not in inspect.getsource(scorer)


def test_a_history_too_thin_to_cluster_gives_no_modes(world: Any) -> None:
    two = [PreferenceSignal(i, 1.0, 1.0, 1.0, 1.0) for i in world.members(0)[:2]]
    assert build_modes(two, world.vectors, polarity="like", seed=0) == []
