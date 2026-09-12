"""The matrix end to end: same seed same ranking, the knobs bite, and the table prints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fixtures.synth import pipeline
from typer.testing import CliRunner

from palate.cli import app
from palate.eval import report as reporting
from palate.eval.harness import (
    EvalContext,
    candidates_for,
    deltas,
    persist,
    publish_weights,
    rank_system,
    run_matrix,
    run_system,
    shape_profile,
    weights_for,
)
from palate.eval.labels import labels_of
from palate.eval.queries import QueryCase
from palate.eval.split import Split, SplitSpec, build_split, freeze_split, load_corpus_ids
from palate.eval.systems import BY_NAME, FULL, by_name
from palate.providers.embed.fake import FakeEmbedder
from palate.retrieval.fusion import prior_weights
from palate.taste import profile as taste

SPEC = SplitSpec(name="rolling", min_reliable=50, cuts=(0.6, 0.8))

ARMS = ("popularity", "director_affinity", "dense_only", "full")

runner = CliRunner()


def world(tmp_path: Path) -> tuple[pipeline.Fitted, Split]:
    """A fitted world and a frozen two fold split over it."""
    fitted = pipeline.fit(tmp_path)
    rated = taste.load_rated(fitted.db.read())
    split = build_split(rated, load_corpus_ids(fitted.db.read()), SPEC, relevance=labels_of)
    freeze_split(split, fitted.db)
    return fitted, split


def context(fitted: pipeline.Fitted, split: Split) -> EvalContext:
    return EvalContext(db=fitted.db, split=split, resamples=40, restarts=2, max_rounds=4)


async def test_a_baseline_and_a_system_both_produce_metrics(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    fold = split.folds[0]
    for name in ("director_affinity", "full"):
        run = await run_system(by_name(name), fold, ctx)
        assert run.system == name
        assert run.metrics["ndcg@10"].lo <= run.metrics["ndcg@10"].point
        assert run.metrics["ndcg@10_watch"].n > 0
        assert 0.0 <= run.pool_recall <= 1.0
        assert run.ranked
        assert not set(run.ranked) & set(fold.train)
    fitted.close()


async def test_the_same_seed_reproduces_the_ranking_exactly(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    first = await run_system(by_name("full"), split.folds[0], ctx)
    second = await run_system(by_name("full"), split.folds[0], ctx)
    assert first.ranked == second.ranked
    assert first.metrics["ndcg@10"] == second.metrics["ndcg@10"]
    assert first.config_sha == second.config_sha
    assert first.run_id != second.run_id
    fitted.close()


async def test_the_pool_is_built_without_the_folds_own_training_films(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    fold = split.folds[0]
    ranking = rank_system(ctx, FULL, fold, ctx.profile(fold, FULL))
    assert not set(ranking.order) & set(fold.train)
    assert set(ranking.order) <= set(candidates_for(fitted.db, fold))
    assert ranking.pool_size == len(ranking.order)
    fitted.close()


def test_switching_a_component_off_changes_the_profile_it_retrieves_from(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    profile = ctx.profile(split.folds[0], FULL)
    assert profile.modes
    bare = shape_profile(profile, by_name("dense_only"))
    assert bare.direction is None
    assert bare.anti_modes == ()
    assert "director" not in bare.affinities
    assert "keyword" not in bare.affinities
    assert bare.modes == profile.modes
    assert shape_profile(profile, FULL) is profile
    fitted.close()


def test_a_silenced_feature_gets_a_zero_weight() -> None:
    active = ("mode_affinity", "decade_aff", "decade_exposure", "lang_aff")
    prior = prior_weights(active, condition="unconditioned")
    plain = weights_for(FULL, active, "unconditioned", None)
    assert plain.beta == prior.beta
    off = weights_for(
        replace(FULL, use_metadata_prior=False, use_exposure_features=False),
        active,
        "unconditioned",
        None,
    )
    assert off.beta["decade_aff"] == 0.0
    assert off.beta["decade_exposure"] == 0.0
    assert off.beta["mode_affinity"] == prior.beta["mode_affinity"]


async def test_two_arms_that_differ_rank_differently(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    fold = split.folds[0]
    dense = await run_system(by_name("dense_only"), fold, ctx)
    whole = await run_system(by_name("full"), fold, ctx)
    assert dense.ranked != whole.ranked
    assert dense.pool_size <= whole.pool_size
    fitted.close()


async def test_the_matrix_skips_what_is_already_stored(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    arms = [by_name(name) for name in ARMS]
    first = await run_matrix(arms, ctx)
    assert len(first) == len(arms) * len(split.folds)
    persist(ctx, BY_NAME, first)
    assert await run_matrix(arms, ctx) == []
    again = await run_matrix(arms, ctx, force=True)
    assert len(again) == len(first)
    fitted.close()


async def test_an_arm_that_needs_a_reranker_never_runs(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    blocked = [by_name("+cross_encoder(minilm)"), by_name("doc_no_credits")]
    assert await run_matrix(blocked, ctx) == []
    fitted.close()


async def test_deltas_are_paired_against_the_reference(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    runs = await run_matrix([by_name(n) for n in ARMS], ctx)
    persist(ctx, BY_NAME, runs)
    rows = deltas(ctx, runs, reference="full")
    assert {system for system, _, _, _ in rows} == {"popularity", "director_affinity", "dense_only"}
    for _, ci, positive, total in rows:
        assert total == len(split.folds)
        assert 0 <= positive <= total
        assert ci.lo <= ci.point <= ci.hi
    stored = reporting.load_deltas(fitted.db, split.name)
    assert ("popularity", "full") in stored
    fitted.close()


def test_fusion_weights_are_fitted_without_touching_a_test_label(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    published = publish_weights(ctx, FULL)
    assert published is not None
    assert published.fitted_on.startswith("median of")
    rows = (
        fitted.db.read()
        .execute(
            "select feature, n_active from eval_weight_stability where split_name = ?",
            (split.name,),
        )
        .fetchall()
    )
    assert rows
    assert all(int(r["n_active"]) <= len(split.folds) for r in rows)
    fitted.close()


async def test_query_mode_scores_one_target_out_of_the_whole_corpus(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = replace(context(fitted, split), embedder=FakeEmbedder(dim=48))
    fold = split.folds[0]
    cases = [
        QueryCase(
            query_id=f"review:{i}",
            text="slow and cold and very wet",
            target_tmdb_id=i,
            source="review",
            n_chars=26,
        )
        for i in fold.test[:3]
    ]
    run = await run_system(by_name("full"), fold, ctx, condition="query_review", cases=cases)
    assert run.condition == "query_review"
    assert run.n_cases == 3
    assert 0.0 <= run.metrics["mrr@50"].point <= 1.0
    assert 0.0 <= run.metrics["recall@50"].point <= 1.0
    baseline = await run_system(
        by_name("popularity"), fold, ctx, condition="query_review", cases=cases
    )
    assert baseline.n_cases == 3
    fitted.close()


async def test_the_report_prints_the_header_the_table_and_the_caveats(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    ctx = context(fitted, split)
    runs = await run_matrix([by_name(n) for n in ARMS], ctx)
    persist(ctx, BY_NAME, runs)
    deltas(ctx, runs, reference="full")
    deltas(ctx, runs, reference="director_affinity")
    text = reporting.render(fitted.db, split, surviving=(0, 0))
    assert "reliable dated" in text
    assert "catalogue days" in text
    assert "| arm | ndcg@10 |" in text
    for name in ARMS:
        assert f"| {name} |" in text
    assert "d vs director_affinity" in text
    assert "Honest notes" in text
    assert "lower bound" in text
    assert "no item-item collaborative filtering" in text.casefold()
    assert "+cross_encoder(minilm)" in text
    fitted.close()


def test_an_empty_split_reports_that_rather_than_a_table(tmp_path: Path) -> None:
    fitted, split = world(tmp_path)
    text = reporting.render(fitted.db, split)
    assert reporting.EMPTY in text
    assert "signed prior" in text
    fitted.close()


def test_the_table_pastes_between_the_markers(tmp_path: Path) -> None:
    page = tmp_path / "README.md"
    page.write_text(
        f"before\n\n{reporting.START_MARKER}\nold\n{reporting.END_MARKER}\n\nafter\n",
        encoding="utf-8",
    )
    assert reporting.paste_into(page, "| arm |\n|---|")
    text = page.read_text(encoding="utf-8")
    assert "old" not in text
    assert "| arm |" in text
    assert text.startswith("before")
    assert text.rstrip().endswith("after")
    plain = tmp_path / "plain.md"
    plain.write_text("no markers here\n", encoding="utf-8")
    assert not reporting.paste_into(plain, "| arm |")


def test_the_cli_builds_a_split_runs_a_smoke_matrix_and_reports(tmp_path: Path) -> None:
    fitted = pipeline.fit(tmp_path)
    fitted.close()
    env = {"PALATE_HOME": str(tmp_path), "PALATE_EVAL__MIN_RELIABLE": "50"}
    built = runner.invoke(app, ["eval", "split", "build"], env=env)
    assert built.exit_code == 0, built.output
    assert "rolling" in built.output
    ran = runner.invoke(app, ["eval", "run", "--smoke", "--no-fit", "--resamples", "20"], env=env)
    assert ran.exit_code == 0, ran.output
    assert "runs stored" in ran.output
    page = tmp_path / "README.md"
    page.write_text(f"{reporting.START_MARKER}\n{reporting.END_MARKER}\n", encoding="utf-8")
    shown = runner.invoke(app, ["eval", "report", "--into", str(page)], env=env)
    assert shown.exit_code == 0, shown.output
    pasted = page.read_text(encoding="utf-8")
    assert "| arm | ndcg@10 |" in pasted
    # The readme takes the short table. The whole report would not fit in it.
    assert "### Honest notes" not in pasted


def test_an_unknown_arm_name_is_refused() -> None:
    with pytest.raises(KeyError):
        by_name("nope")
