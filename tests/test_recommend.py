"""The whole pipeline, from the corpus to a ranked list with evidence under it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from fixtures.synth import pipeline
from fixtures.synth import sqlite as synth_sqlite
from fixtures.synth.world import LOVED
from typer.testing import CliRunner

from palate.cli import app
from palate.errors import StaleArtifact, ThinHistoryError
from palate.index import fts
from palate.retrieval.candidates import HardFilters
from palate.retrieval.fusion import FusionWeights, save_weights
from palate.retrieval.recommend import (
    LocalRecommender,
    RecommendRequest,
    RecommendResponse,
    run_recommend,
)
from palate.taste import profile as taste

QUERY = "a film about what happens"


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("recommend"))
    yield built
    built.close()


def ask(fitted: pipeline.Fitted, **overrides: Any) -> RecommendResponse:
    """One request against the loved cluster's centre when a query is asked for."""
    req = RecommendRequest(**overrides)
    embedding = fitted.world.centres[LOVED[0]].tolist() if req.query_text else None
    return run_recommend(fitted.db, fitted.profile, req, query_embedding=embedding)


def test_a_an_unconditioned_list_comes_back_with_evidence(fitted: pipeline.Fitted) -> None:
    answer = ask(fitted, n=10)
    assert len(answer.films) == 10
    assert answer.condition == "unconditioned"
    assert all(film.evidence for film in answer.films)
    assert any(row.kind == "shared_person" for film in answer.films for row in film.evidence)
    assert [f.score for f in answer.films] == sorted((f.score for f in answer.films), reverse=True)


def test_b_nothing_watched_ever_reaches_the_answer(fitted: pipeline.Fitted) -> None:
    watched = {r.tmdb_id for r in fitted.world.rated}
    answer = ask(fitted, n=10, query_text=QUERY)
    assert not watched & {f.tmdb_id for f in answer.films}
    assert answer.diagnostics.excluded_watched == len(watched)


def test_c_a_query_pulls_the_list_towards_that_cluster(fitted: pipeline.Fitted) -> None:
    conditioned = ask(fitted, n=10, query_text=QUERY)
    assert conditioned.condition == "query"
    hits = sum(1 for f in conditioned.films if fitted.world.cluster_of(f.tmdb_id) == LOVED[0])
    plain = ask(fitted, n=10)
    misses = sum(1 for f in plain.films if fitted.world.cluster_of(f.tmdb_id) == LOVED[0])
    assert hits > misses


def test_d_the_same_request_gives_the_same_answer(fitted: pipeline.Fitted) -> None:
    first = ask(fitted, n=8, query_text=QUERY)
    second = ask(fitted, n=8, query_text=QUERY)
    assert [f.tmdb_id for f in first.films] == [f.tmdb_id for f in second.films]


def test_e_a_filter_reports_what_it_cost_the_top_ten(fitted: pipeline.Fitted) -> None:
    answer = ask(fitted, n=10, filters=HardFilters(year_min=2000))
    assert all(f.year is None or f.year >= 2000 for f in answer.films)
    assert answer.diagnostics.removed_by_clause["year_min"] > 0
    assert answer.diagnostics.most_restrictive_clause in {"year_min", "watched"}
    assert sum(answer.diagnostics.removed_top10_by_clause.values()) > 0


def test_f_paging_slices_one_consistent_list(fitted: pipeline.Fitted) -> None:
    whole = ask(fitted, n=10)
    tail = ask(fitted, n=5, offset=5)
    assert [f.tmdb_id for f in tail.films] == [f.tmdb_id for f in whole.films[5:]]


def test_g_diversity_off_takes_the_plain_top_n(fitted: pipeline.Fitted) -> None:
    shaped = ask(fitted, n=10)
    plain = ask(fitted, n=10, diversity="off")
    assert [f.tmdb_id for f in shaped.films] != [f.tmdb_id for f in plain.films]
    directors = [d for f in plain.films for d in f.directors]
    assert len(set(directors)) <= len(directors)


def test_h_the_caps_hold_in_a_real_answer(fitted: pipeline.Fitted) -> None:
    answer = ask(fitted, n=10)
    seen: dict[str, int] = {}
    for film in answer.films:
        for name in film.directors:
            seen[name] = seen.get(name, 0) + 1
    assert seen and max(seen.values()) <= 2


def test_i_published_weights_are_the_ones_that_run(fitted: pipeline.Fitted) -> None:
    baseline = ask(fitted, n=10)
    published = FusionWeights(
        condition="unconditioned",
        beta={"popularity": 1.0},
        fitted_on="a test",
        n_active=1,
    )
    save_weights(fitted.db, fitted.profile.profile_id, {"unconditioned": published})
    try:
        loaded = ask(fitted, n=10)
        assert [f.tmdb_id for f in loaded.films] != [f.tmdb_id for f in baseline.films]
        assert set(loaded.films[0].feature_contributions) == {"popularity"}
    finally:
        save_weights(fitted.db, fitted.profile.profile_id, {})


def test_j_a_cold_profile_refuses_to_guess(tmp_path: Path) -> None:
    thin = pipeline.small_world(rated_per_cluster=4, rated_background=4)
    built = pipeline.fit(tmp_path, thin, docs=False)
    try:
        assert built.profile.tier == "cold"
        with pytest.raises(ThinHistoryError):
            run_recommend(built.db, built.profile, RecommendRequest(n=5))
        answered = run_recommend(
            built.db,
            built.profile,
            RecommendRequest(n=5),
            query_embedding=built.world.centres[LOVED[0]].tolist(),
        )
        assert answered.films
        assert "cold_start" in answered.degraded
    finally:
        built.close()


def test_k_similar_to_needs_no_embedder(fitted: pipeline.Fitted) -> None:
    seed = next(
        f.tmdb_id
        for f in fitted.world.films
        if f.cluster == LOVED[0] and f.tmdb_id not in {r.tmdb_id for r in fitted.world.rated}
    )

    async def scenario() -> RecommendResponse:
        return await LocalRecommender(fitted.db).similar_to(seed, n=5)

    answer = anyio.run(scenario)
    assert seed not in {f.tmdb_id for f in answer.films}
    assert sum(1 for f in answer.films if fitted.world.cluster_of(f.tmdb_id) == LOVED[0]) >= 3


def test_l_a_hated_genre_filter_empties_the_list_with_a_hint(fitted: pipeline.Fitted) -> None:
    every_genre = frozenset(range(1, 11000))
    answer = ask(fitted, n=5, filters=HardFilters(include_genres=every_genre, year_min=3000))
    assert answer.films == ()
    assert answer.diagnostics.most_restrictive_clause == "year_min"


def test_m_the_cli_prints_a_ranked_list(offline_env: Path) -> None:
    built = pipeline.fit(offline_env, pipeline.small_world())
    fts.rebuild(built.db)
    built.db.close()
    result = CliRunner().invoke(app, ["recommend", "-n", "5"])
    assert result.exit_code == 0, result.output
    assert "films passed the filters" in result.output
    assert "unconditioned" in result.output


def test_n_the_cli_says_so_when_there_is_no_profile(offline_env: Path) -> None:
    result = CliRunner().invoke(app, ["recommend", "-n", "5"])
    assert result.exit_code == 2
    assert "profile" in result.output


def test_o_the_diagnostics_say_which_columns_were_live(fitted: pipeline.Fitted) -> None:
    answer = ask(fitted, n=10)
    scaled = answer.diagnostics.scaled_by
    # Nothing states a country preference here and nothing reranks, so both sit out.
    assert scaled["country_penalty"] == "zeroed"
    assert scaled["ce_score"] == "zeroed"
    assert scaled["mode_affinity"] != "zeroed"
    assert answer.degraded == ("no_reranker",)


def test_p_the_profile_must_match_the_active_index(fitted: pipeline.Fitted) -> None:
    synth_sqlite.install_index(fitted.db, fitted.world, revision="2")
    stale = taste.load(fitted.db, fitted.profile.profile_id)
    assert stale is not None and stale.stale
    with pytest.raises(StaleArtifact, match="was built for"):
        run_recommend(fitted.db, stale, RecommendRequest(n=5))
