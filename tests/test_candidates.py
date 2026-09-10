"""Hard filters, the six channels, and the pool their ranks agree on."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import orjson
import pytest
from fixtures.synth import pipeline
from fixtures.synth.world import CLUSTERS, LOVED

from palate.retrieval.candidates import (
    IMPOSSIBLE,
    CandidatePool,
    Channel,
    ChannelBudget,
    ChannelHit,
    HardFilters,
    from_preferences,
    generate_candidates,
    reciprocal_rank_fusion,
)
from palate.taste.memory import PreferenceFilter

QUERY_TEXT = "a film about what happens"

TIGHT = ChannelBudget(dense_per_mode=20, dense_query=20, bm25=20, people=40, keyword=20, popular=20)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("candidates"))
    yield built
    built.close()


def pool(fitted: pipeline.Fitted, **overrides: Any) -> CandidatePool:
    """The pool a loved cluster centre plus a matching phrase produces."""
    centre = fitted.world.centres[LOVED[0]].tolist()
    overrides.setdefault("query_embedding", centre)
    overrides.setdefault("query_text", QUERY_TEXT)
    return generate_candidates(fitted.store(), fitted.profile, **overrides)


def column(fitted: pipeline.Fitted, sql: str, ids: Sequence[int]) -> list[Any]:
    rows = fitted.db.read().execute(sql, (orjson.dumps(list(ids)).decode(),))
    return [r[0] for r in rows]


def test_a_watched_films_never_enter_the_pool(fitted: pipeline.Fitted) -> None:
    watched = {r.tmdb_id for r in fitted.world.rated}
    found = pool(fitted)
    assert watched
    assert not watched & set(found.ids)
    assert found.allow.excluded_watched == len(watched)


def test_b_every_channel_contributes_and_some_ids_are_its_own(fitted: pipeline.Fitted) -> None:
    found = pool(fitted, budget=TIGHT)
    for channel in Channel:
        assert found.per_channel_counts.get(channel, 0) > 0, channel
    assert found.per_channel_unique.get(Channel.PEOPLE, 0) > 0


def test_c_the_same_profile_gives_the_same_pool_twice(fitted: pipeline.Fitted) -> None:
    first = pool(fitted)
    second = pool(fitted)
    assert first.ids == second.ids
    assert first.ids == tuple(sorted(first.ids, key=lambda i: (-first.rrf[i], i)))


def test_d_a_year_floor_removes_every_older_film(fitted: pipeline.Fitted) -> None:
    found = pool(fitted, filters=HardFilters(year_min=2000))
    years = column(
        fitted,
        "select year from films where tmdb_id in (select value from json_each(?))",
        found.ids,
    )
    assert years
    assert min(years) >= 2000
    assert found.allow.removed_by_clause["year_min"] > 0


def test_e_disjoint_includes_merge_to_nothing(fitted: pipeline.Fitted) -> None:
    left = HardFilters(include_languages=frozenset({"ru"}))
    right = HardFilters(include_languages=frozenset({"fr"}))
    merged = left.merge(right)
    assert merged is IMPOSSIBLE
    allow = fitted.store().allow(merged)
    assert allow.ids == frozenset()
    assert merged.most_restrictive(allow.removed_by_clause) == "unsatisfiable"


def test_f_reciprocal_rank_fusion_sums_over_channels() -> None:
    hits = (
        ChannelHit(7, Channel.BM25, 1, 9.0),
        ChannelHit(7, Channel.POPULAR, 3, 1.0),
        ChannelHit(9, Channel.BM25, 2, 8.0),
    )
    scores = reciprocal_rank_fusion(hits, k=10)
    assert scores[7] == pytest.approx(1 / 11 + 1 / 13)
    assert scores[9] == pytest.approx(1 / 12)
    weighted = reciprocal_rank_fusion(hits, k=10, weights={Channel.POPULAR: 0.0})
    assert weighted[7] == pytest.approx(1 / 11)


def test_g_the_overfetch_path_still_honours_the_filters(fitted: pipeline.Fitted) -> None:
    found = generate_candidates(
        fitted.store(id_cap=10),
        fitted.profile,
        query_embedding=fitted.world.centres[LOVED[0]].tolist(),
        filters=HardFilters(year_min=1990),
        budget=ChannelBudget(pool_max=200),
    )
    assert found.prefilter_path == "metadata_overfetch"
    assert found.overfetch_factor >= 2.0
    assert not {r.tmdb_id for r in fitted.world.rated} & set(found.ids)
    assert len(found.ids) <= 200


def test_h_a_hard_preference_compiles_to_an_exclusion(fitted: pipeline.Fitted) -> None:
    hated = CLUSTERS[2].genre
    stated = PreferenceFilter(exclude={"genre": frozenset({str(hated)})}, require={})
    compiled = from_preferences(fitted.db.read(), stated)
    assert compiled.exclude_genres == frozenset({hated})
    allow = fitted.store().allow(compiled)
    assert allow.removed_by_clause["exclude_genres"] > 0
    survivors = column(
        fitted,
        f"select count(*) from film_genres where genre_id = {hated} "
        "and tmdb_id in (select value from json_each(?))",
        sorted(allow.ids),
    )
    assert survivors == [0]


def test_i_most_restrictive_names_the_worst_clause() -> None:
    filters = HardFilters(year_min=2000, min_vote_count=100)
    assert filters.most_restrictive({"year_min": 10, "min_vote_count": 400}) == "min_vote_count"
    assert filters.most_restrictive({"year_min": 0}) is None


def test_j_a_runtime_preference_caps_the_pool(fitted: pipeline.Fitted) -> None:
    stated = PreferenceFilter(exclude={}, require={}, max_runtime=110)
    compiled = from_preferences(fitted.db.read(), stated)
    assert compiled.runtime_max == 110
    found = pool(fitted, filters=compiled)
    runtimes = column(
        fitted,
        "select runtime from films where tmdb_id in (select value from json_each(?))",
        found.ids,
    )
    assert max(runtimes) <= 110


def test_k_the_unconditioned_path_still_fills_the_pool(fitted: pipeline.Fitted) -> None:
    found = generate_candidates(fitted.store(), fitted.profile)
    assert len(found.ids) > 100
    assert Channel.DENSE_QUERY not in found.per_channel_counts
    assert Channel.BM25 not in found.per_channel_counts


def test_l_the_path_matches_the_allow_set_size(tmp_path: Path) -> None:
    built = pipeline.fit(tmp_path, docs=False)
    try:
        store = built.store()
        allow = store.allow(HardFilters())
        assert store.plan(len(allow.ids)).path == "allow_json"
        narrow = built.store(id_cap=10)
        assert narrow.plan(narrow.vecs.count() - 5).path == "exclude_json"
        assert narrow.plan(len(allow.ids)).path == "metadata_overfetch"
    finally:
        built.close()


def test_m_merge_tightens_every_clause() -> None:
    left = HardFilters(
        year_min=1960, year_max=2010, exclude_genres=frozenset({1}), min_vote_count=10
    )
    right = HardFilters(
        year_min=1980, year_max=1999, exclude_genres=frozenset({2}), min_vote_count=50
    )
    merged = left.merge(right)
    assert (merged.year_min, merged.year_max) == (1980, 1999)
    assert merged.exclude_genres == frozenset({1, 2})
    assert merged.min_vote_count == 50
    assert not merged.unsatisfiable


def test_n_an_empty_include_set_means_unconstrained() -> None:
    one = HardFilters(include_languages=frozenset({"ru"}), exclude_watched=False)
    assert one.merge(HardFilters()).include_languages == frozenset({"ru"})
    assert one.merge(HardFilters()).exclude_watched is True
    assert one.merge(one) == one
    overlapping = HardFilters(include_languages=frozenset({"ru", "fr"}))
    assert one.merge(overlapping).include_languages == frozenset({"ru"})
