"""Metrics against hand-computed numbers, and intervals that reproduce from a seed."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from palate.eval.labels import (
    graded_relevance,
    histogram,
    is_hard_negative,
    labels_of,
    watch_relevance,
)
from palate.eval.metrics import (
    bootstrap_ci,
    dcg,
    ideal_dcg,
    intra_list_distance,
    mode_coverage,
    mrr_at_k,
    ndcg_at_k,
    novelty,
    paired_bootstrap,
    prefilter_recall_at_k,
    recall_at_k,
    spearman,
    spearman_conditional,
    spearman_pessimistic,
    unknown_director_rate,
)
from palate.taste.signals import RatedFilm

REL = {1: 3, 2: 0, 3: 2}


def test_graded_cutpoints_are_the_published_ones() -> None:
    assert [graded_relevance(h) for h in range(1, 11)] == [0, 0, 0, 0, 0, 0, 1, 2, 3, 3]
    assert [is_hard_negative(h) for h in (1, 6, 7, 10)] == [True, True, False, False]
    assert watch_relevance(1) == 1
    assert labels_of(RatedFilm(tmdb_id=1, rating_half=9)) == (3, 1)


def test_histogram_counts_every_half_star_in_order() -> None:
    rated = [RatedFilm(tmdb_id=i, rating_half=1 + i % 3) for i in range(9)]
    assert histogram(rated) == {"1": 3, "2": 3, "3": 3}


def test_dcg_and_idcg_are_the_textbook_sums() -> None:
    assert dcg([7.0, 0.0, 3.0]) == pytest.approx(8.5)
    assert ideal_dcg([3, 0, 2], 3) == pytest.approx(8.892789260714373)


def test_ndcg_matches_the_hand_computed_value() -> None:
    assert ndcg_at_k([1, 2, 3], REL, 3) == pytest.approx(0.95583058934618)
    assert ndcg_at_k([1, 3, 2], REL, 3) == pytest.approx(1.0)
    assert ndcg_at_k([2, 3, 1], REL, 3) < ndcg_at_k([1, 2, 3], REL, 3)


def test_unjudged_films_cost_the_ranking_their_slot() -> None:
    assert ndcg_at_k([99, 1, 3], REL, 3) < ndcg_at_k([1, 3, 99], REL, 3)
    assert ndcg_at_k([], REL, 3) == 0.0
    assert ndcg_at_k([1, 2, 3], {}, 3) == 0.0


def test_a_frozen_idcg_overrides_the_recomputed_one() -> None:
    assert ndcg_at_k([1], REL, 3, idcg=7.0) == pytest.approx(1.0)
    assert ndcg_at_k([1], REL, 3, idcg=0.0) == 0.0


def test_recall_counts_only_the_films_the_user_liked() -> None:
    rel = {1: 3, 2: 2, 3: 1, 4: 0}
    assert recall_at_k([1, 9, 9], rel, k=3) == pytest.approx(0.5)
    assert recall_at_k([1, 2], rel, k=2) == pytest.approx(1.0)
    assert recall_at_k([3, 4], rel, k=2) == 0.0
    assert recall_at_k([1], {4: 0}, k=2) == 0.0


def test_mrr_finds_the_first_relevant_film() -> None:
    assert mrr_at_k([9, 9, 1], REL, k=50) == pytest.approx(1 / 3)
    assert mrr_at_k([1], REL, k=50) == pytest.approx(1.0)
    assert mrr_at_k([2, 9], REL, k=50) == 0.0


def test_spearman_handles_direction_ties_and_constants() -> None:
    rising = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(rising, rising) == pytest.approx(1.0)
    assert spearman(rising, rising[::-1]) == pytest.approx(-1.0)
    assert spearman(rising, np.ones(4)) == 0.0
    tied = np.array([1.0, 1.0, 2.0, 2.0])
    assert spearman(tied, np.array([1.0, 1.0, 5.0, 5.0])) == pytest.approx(1.0)


def test_conditional_spearman_reports_its_own_coverage() -> None:
    actual = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0}
    scores = {1: 0.9, 2: 0.8, 3: 0.7}
    rho, n, coverage = spearman_conditional([1, 2, 3, 99], scores, actual, pool_k=10)
    assert rho == pytest.approx(1.0)
    assert n == 3
    assert coverage == pytest.approx(0.75)


def test_pessimistic_spearman_floors_the_films_that_never_ranked() -> None:
    actual = {1: 5.0, 2: 4.0, 3: 1.0}
    rho, n = spearman_pessimistic({1: 0.9, 2: 0.8}, actual, floor=-5.0)
    assert n == 3
    assert rho == pytest.approx(1.0)
    worse, _ = spearman_pessimistic({3: 0.9}, actual, floor=-5.0)
    assert worse < rho


def test_intra_list_distance_spans_identical_to_orthogonal() -> None:
    emb = {1: [1.0, 0.0], 2: [1.0, 0.0], 3: [0.0, 1.0]}
    assert intra_list_distance([1, 2], emb) == pytest.approx(0.0)
    assert intra_list_distance([1, 3], emb) == pytest.approx(1.0)
    assert intra_list_distance([1], emb) == 0.0


def test_mode_coverage_is_capped_by_the_shorter_side() -> None:
    assert mode_coverage([1, 2, 3], {1: 0, 2: 1, 3: 2}, 3) == pytest.approx(1.0)
    assert mode_coverage([1, 2, 3], {1: 0, 2: 0, 3: 0}, 3) == pytest.approx(1 / 3)
    assert mode_coverage([1], {1: 0}, 4) == pytest.approx(1.0)
    assert mode_coverage([], {}, 4) == 0.0


def test_novelty_runs_from_the_most_popular_to_the_least() -> None:
    popularity = {1: 100.0, 2: 50.0, 3: 10.0, 4: 1.0}
    assert novelty([1], popularity) == pytest.approx(0.0)
    assert novelty([4], popularity) == pytest.approx(1.0)
    assert novelty([2], popularity) < novelty([3], popularity)


def test_unknown_director_rate_reads_the_credit_join() -> None:
    directors = {1: [10], 2: [11, 12], 3: []}
    assert unknown_director_rate([1, 2, 3], {10}, directors=directors) == pytest.approx(2 / 3)
    assert unknown_director_rate([1], {10, 11}, directors=directors) == 0.0
    assert unknown_director_rate([], {10}, directors=directors) == 0.0


def test_prefilter_recall_compares_the_two_paths() -> None:
    assert prefilter_recall_at_k([1, 2, 3], [1, 2, 3], 3) == pytest.approx(1.0)
    assert prefilter_recall_at_k([1, 2, 3], [1, 9, 3], 3) == pytest.approx(2 / 3)
    assert prefilter_recall_at_k([], [1], 3) == 1.0


def test_a_bootstrap_interval_reproduces_from_its_seed() -> None:
    rel = {i: (3 if i % 3 == 0 else 0) for i in range(30)}
    ranked = list(range(30))
    metric = partial(ndcg_at_k, k=10)
    first = bootstrap_ci(ranked, rel, metric, n_resamples=200, seed=7)
    again = bootstrap_ci(ranked, rel, metric, n_resamples=200, seed=7)
    other = bootstrap_ci(ranked, rel, metric, n_resamples=200, seed=8)
    assert first == again
    assert first != other
    assert first.lo <= first.point <= first.hi
    assert first.n == 30


def test_a_single_judged_film_has_no_interval() -> None:
    narrow = bootstrap_ci([1], {1: 3}, partial(ndcg_at_k, k=10), n_resamples=50)
    assert narrow.lo == narrow.point == narrow.hi
    assert narrow.n == 1


def test_a_paired_delta_of_a_ranking_against_itself_is_exactly_zero() -> None:
    rel = {i: (2 if i % 2 == 0 else 0) for i in range(20)}
    ranked = list(range(20))
    ci, wins = paired_bootstrap(ranked, ranked, rel, partial(ndcg_at_k, k=10), n_resamples=100)
    assert ci.point == 0.0
    assert ci.lo == ci.hi == 0.0
    assert wins == 0.0
    assert ci.spans_zero


def test_a_better_ranking_wins_the_paired_bootstrap() -> None:
    rel = {i: (3 if i < 5 else 0) for i in range(20)}
    good = list(range(20))
    bad = list(reversed(range(20)))
    ci, wins = paired_bootstrap(good, bad, rel, partial(ndcg_at_k, k=10), n_resamples=200, seed=3)
    assert ci.point > 0.0
    assert wins > 0.9
    assert not ci.spans_zero
