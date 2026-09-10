"""Mode slots, composite redundancy, the hard caps and the novelty cap."""

from __future__ import annotations

import numpy as np
import pytest

from palate.retrieval.diversity import (
    DiversityConfig,
    allocate_slots,
    diversify,
    query_mode_posterior,
)
from palate.retrieval.features import FilmFacets
from palate.taste.modes import TasteMode

CFG = DiversityConfig()

DIM = 8


def mode(
    mode_id: int, *, axis: int, mass: float, members: int = 10, coherence: float = 0.8
) -> TasteMode:
    centroid = np.zeros(DIM, dtype=np.float32)
    centroid[axis] = 1.0
    return TasteMode(
        mode_id=mode_id,
        polarity="like",
        centroid=centroid,
        mass=mass,
        n_members=members,
        mean_signal=1.0,
        coherence=coherence,
        confidence=mass / (mass + 3.0),
        exemplars=(),
    )


def film(
    tmdb_id: int,
    *,
    director: int = 0,
    decade: int = 2000,
    collection: int | None = None,
) -> FilmFacets:
    return FilmFacets(
        tmdb_id=tmdb_id,
        decade=decade,
        runtime_bucket=2,
        language="en",
        popularity=1.0,
        vote_average=7.0,
        vote_count=100,
        collection_id=collection,
        directors=(director,),
    )


def axis_vector(axis: int) -> list[float]:
    return [1.0 if i == axis else 0.0 for i in range(DIM)]


def test_a_the_posterior_is_a_softmax_over_the_centroids() -> None:
    modes = [mode(0, axis=0, mass=9.0), mode(1, axis=1, mass=9.0)]
    posterior = query_mode_posterior(axis_vector(0), modes)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior[0] > posterior[1]
    mixed = [(a + b) / np.sqrt(2) for a, b in zip(axis_vector(0), axis_vector(1), strict=True)]
    even = query_mode_posterior(mixed, modes)
    assert even[0] == pytest.approx(even[1])


def test_b_a_concentrated_posterior_turns_the_quotas_off() -> None:
    modes = [mode(0, axis=0, mass=9.0), mode(1, axis=1, mass=9.0)]
    sharp = query_mode_posterior(axis_vector(0), modes)
    assert max(sharp.values()) > CFG.query_posterior_concentrated
    assert allocate_slots(modes, 10, CFG, posterior=sharp) == {}
    flat = {0: 0.5, 1: 0.5}
    assert allocate_slots(modes, 10, CFG, posterior=flat) == {0: 5, 1: 5}


def test_c_slots_follow_the_square_root_of_mass() -> None:
    modes = [mode(0, axis=0, mass=90.0), mode(1, axis=1, mass=10.0)]
    slots = allocate_slots(modes, 10, CFG)
    assert sum(slots.values()) == 10
    # Raw mass would be nine slots of ten. The square root pulls that back.
    assert slots[0] <= 8
    assert slots[1] >= 2


def test_d_a_small_sharp_mode_still_earns_a_slot() -> None:
    big = mode(0, axis=0, mass=200.0, members=200)
    sharp = mode(1, axis=1, mass=0.4, members=4, coherence=0.4)
    assert sharp.confidence < 0.5
    slots = allocate_slots([big, sharp], 10, CFG)
    assert slots[1] >= 1


def test_e_one_director_twice_is_redundant_whatever_the_embeddings_say() -> None:
    ranked = [1, 2, 3]
    scores = {1: 3.0, 2: 2.0, 3: 1.9}
    embeddings = {1: axis_vector(0), 2: axis_vector(1), 3: axis_vector(2)}
    meta = {1: film(1, director=7), 2: film(2, director=7), 3: film(3, director=9)}
    loose = DiversityConfig(max_per_director=2, max_per_decade=9)
    selected, _ = diversify(ranked, scores, embeddings, meta, {}, frozenset(), 2, loose, modes=())
    assert selected == [1, 3]
    blind = DiversityConfig(max_per_director=2, max_per_decade=9, w_director=0.0)
    on_cosine, _ = diversify(ranked, scores, embeddings, meta, {}, frozenset(), 2, blind, modes=())
    assert on_cosine == [1, 2]


def test_f_the_hard_caps_hold() -> None:
    ranked = list(range(1, 9))
    scores = {i: 10.0 - i for i in ranked}
    embeddings = {i: axis_vector(i % DIM) for i in ranked}
    meta = {
        1: film(1, director=1, decade=1990),
        2: film(2, director=1, decade=1990),
        3: film(3, director=1, decade=1990),
        4: film(4, director=2, decade=1990),
        5: film(5, director=3, decade=2000, collection=5),
        6: film(6, director=4, decade=2000, collection=5),
        7: film(7, director=5, decade=2010),
        8: film(8, director=6, decade=2010),
    }
    selected, report = diversify(
        ranked, scores, embeddings, meta, {}, frozenset(), 6, CFG, modes=()
    )
    assert 3 not in selected
    assert len([i for i in selected if meta[i].decade == 1990]) <= CFG.max_per_decade
    assert len([i for i in selected if meta[i].collection_id == 5]) <= CFG.max_per_collection
    assert report.dropped_by_cap


def test_g_the_novelty_cap_limits_directors_the_user_already_knows() -> None:
    ranked = list(range(1, 13))
    scores = {i: 20.0 - i for i in ranked}
    embeddings = {i: axis_vector(i % DIM) for i in ranked}
    meta = {i: film(i, director=100 + i, decade=1960 + 10 * i) for i in ranked}
    known = frozenset(100 + i for i in range(1, 9))
    selected, report = diversify(ranked, scores, embeddings, meta, {}, known, 6, CFG, modes=())
    assert len(selected) == 6
    assert report.known_directors <= CFG.max_known_directors
    assert any(meta[i].directors[0] not in known for i in selected)


def test_h_quotas_spread_the_list_across_the_modes() -> None:
    modes = [mode(0, axis=0, mass=30.0), mode(1, axis=1, mass=30.0)]
    ranked = list(range(1, 11))
    scores = {i: 20.0 - i for i in ranked}
    embeddings = {i: axis_vector(0 if i <= 6 else 1) for i in ranked}
    meta = {i: film(i, director=i, decade=1950 + 10 * i) for i in ranked}
    mode_of = {i: (0 if i <= 6 else 1) for i in ranked}
    selected, report = diversify(
        ranked, scores, embeddings, meta, mode_of, frozenset(), 4, CFG, modes=modes
    )
    assert report.quotas_applied
    assert report.slots == {0: 2, 1: 2}
    assert sorted(mode_of[i] for i in selected) == [0, 0, 1, 1]


def test_i_an_empty_pool_returns_nothing() -> None:
    selected, report = diversify([], {}, {}, {}, {}, frozenset(), 5, CFG, modes=())
    assert selected == []
    assert report.slots == {}
