"""Rerankers, offline. A stub checkpoint stands in for the Hub so no torch is needed."""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest
from fixtures.eval import record

from palate.agent.prompts import load
from palate.db.connect import open_database
from palate.errors import ConfigError, MissingExtra
from palate.eval import report as reporting
from palate.eval.harness import (
    EvalContext,
    publish_weights,
    rerank_metrics,
    run_matrix,
    run_system,
    weights_for,
)
from palate.eval.metrics import MetricCI
from palate.eval.systems import blocked_reason, by_name, rerank_key
from palate.hf.device import check_precision
from palate.hf.models import ModelPin
from palate.paths import migrations_dir
from palate.providers.base import (
    RerankCandidate,
    Reranker,
    RerankReport,
    RerankResult,
)
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.providers.rerank.cache import ScoreCache
from palate.providers.rerank.identity import IdentityReranker
from palate.providers.rerank.llm import LLMReranker, borda, parse_permutation, windows
from palate.retrieval.fusion import FusionWeights, prior_share
from palate.retrieval.rerank import COLD_QUERY, load_texts, mode_query
from palate.retrieval.rerank import candidates as rerank_candidates

PIN = ModelPin("minilm", "cross-encoder/ms-marco-MiniLM-L6-v2", "b" * 40)

POOL = (
    RerankCandidate(1, "a long quiet film about a journey", prior_score=0.1),
    RerankCandidate(2, "a loud comedy about a wedding", prior_score=0.9),
    RerankCandidate(3, "a quiet journey through wet country", prior_score=0.5),
)


class StubCrossEncoder:
    """The v6 contract: everything but the repo id is keyword only, and predict takes a callable."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        num_labels: int | None = None,
        max_length: int | None = None,
        activation_fn: Any = None,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.repo_id = model_name_or_path
        self.max_length = max_length
        self.device = device
        self.revision = revision
        self.seen: list[dict[str, Any]] = []

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int = 32,
        activation_fn: Any = None,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> list[float]:
        """Word overlap, which is deterministic and reorders the pool the way a model would."""
        self.seen.append({"batch_size": batch_size, "activation_fn": activation_fn})
        return [float(len(set(q.split()) & set(d.split()))) for q, d in inputs]


@pytest.fixture
def stub_st(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def build(**overrides: Any) -> Any:
    from palate.providers.rerank.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(pin=PIN, device="cpu", **overrides)


def test_the_cross_encoder_names_the_local_extra_when_it_is_absent() -> None:
    from palate.providers.rerank.cross_encoder import CrossEncoderReranker

    with pytest.raises(MissingExtra) as exc:
        CrossEncoderReranker(pin=PIN, device="cpu")
    assert exc.value.extra == "local"
    assert "uv sync --extra local" in str(exc.value)


def test_half_precision_on_mps_is_refused() -> None:
    with pytest.raises(ConfigError) as exc:
        check_precision("float16", "mps")
    assert "float32" in str(exc.value)
    assert check_precision("float32", "mps") == "float32"
    assert check_precision("float16", "cpu") == "float16"


def test_the_pin_is_passed_with_keyword_arguments_only(stub_st: types.ModuleType) -> None:
    ranker = build(max_length=256)
    assert ranker.model.repo_id == PIN.repo_id
    assert ranker.model.revision == PIN.revision
    assert ranker.model.max_length == 256
    assert ranker.model_key == f"{PIN.repo_id}@{'b' * 12}"


def test_scores_are_raw_logits_not_the_checkpoint_sigmoid(stub_st: types.ModuleType) -> None:
    ranker = build(batch_size=8)
    anyio.run(lambda: ranker.rerank("a quiet journey", POOL, top_k=3, doc_version="v1"))
    call = ranker.model.seen[0]
    assert call["batch_size"] == 8
    assert call["activation_fn"](7.5) == 7.5


def test_the_pool_is_reordered_by_the_model_not_by_the_prior(stub_st: types.ModuleType) -> None:
    ranker = build()
    report = anyio.run(lambda: ranker.rerank("a quiet journey", POOL, top_k=3, doc_version="v1"))
    # The prior put 2 first. Ties fall to the lower film id, so two runs never disagree.
    assert report.order() == (1, 3, 2)
    assert [r.rank for r in report.results] == [1, 2, 3]
    assert report.n_pairs == 3
    assert report.scores()[3] > report.scores()[2]


def test_top_k_truncates_the_report(stub_st: types.ModuleType) -> None:
    ranker = build()
    report = anyio.run(lambda: ranker.rerank("a quiet journey", POOL, top_k=2, doc_version="v1"))
    assert report.order() == (1, 3)
    assert report.n_pairs == 3


def test_the_first_call_is_cold_and_a_warm_up_pays_that_cost_up_front(
    stub_st: types.ModuleType,
) -> None:
    ranker = build()

    async def two_calls() -> tuple[bool, bool]:
        first = await ranker.rerank("a quiet journey", POOL, top_k=3, doc_version="v1")
        second = await ranker.rerank("a loud wedding", POOL, top_k=3, doc_version="v1")
        return first.cold_start, second.cold_start

    assert anyio.run(two_calls) == (True, False)
    warmed = build()

    async def warm_first() -> bool:
        await warmed.warmup()
        report = await warmed.rerank("a quiet journey", POOL, top_k=3, doc_version="v1")
        return bool(report.cold_start)

    assert anyio.run(warm_first) is False


def test_an_empty_pool_still_answers(stub_st: types.ModuleType) -> None:
    ranker = build()
    report = anyio.run(lambda: ranker.rerank("anything", (), top_k=10, doc_version="v1"))
    assert report.results == ()
    assert report.n_pairs == 0


def test_closing_drops_the_checkpoint(stub_st: types.ModuleType) -> None:
    ranker = build()
    anyio.run(ranker.aclose)
    assert ranker.model is None


def test_identity_keeps_the_prior_order_and_the_prior_scores() -> None:
    ranker = IdentityReranker()
    report = anyio.run(lambda: ranker.rerank("ignored", POOL, top_k=3, doc_version="v1"))
    assert report.order() == (2, 3, 1)
    assert report.scores() == {1: 0.1, 2: 0.9, 3: 0.5}
    assert report.cold_start is False
    assert ranker.calls == 1


def test_identity_honours_top_k() -> None:
    ranker = IdentityReranker()
    report = anyio.run(lambda: ranker.rerank("ignored", POOL, top_k=1, doc_version="v1"))
    assert report.order() == (2,)
    assert report.n_pairs == 3


def test_both_rerankers_satisfy_the_protocol(stub_st: types.ModuleType) -> None:
    assert isinstance(IdentityReranker(), Reranker)
    assert isinstance(build(), Reranker)


class Reversing:
    """A reranker that inverts the stage one order, so its effect on a ranking is unmistakable."""

    name = "reversing"

    def __init__(self) -> None:
        self.model_key = "reversing"
        self.queries: list[str] = []
        self.warm = False

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
        doc_version: str,
        span: Any = None,
    ) -> RerankReport:
        """Highest ce_score to whatever stage one liked least."""
        self.queries.append(query)
        cold, self.warm = not self.warm, True
        worst_first = sorted(candidates, key=lambda c: (c.prior_score, -c.film_id))
        return RerankReport(
            results=tuple(
                RerankResult(c.film_id, float(place), place + 1)
                for place, c in enumerate(worst_first[:top_k])
            ),
            model_key=self.model_key,
            n_pairs=len(candidates),
            cold_start=cold,
        )

    async def aclose(self) -> None:
        return None


def test_windows_cover_every_candidate_and_overlap() -> None:
    assert windows(0, 20, 10) == []
    assert windows(5, 20, 10) == [(0, 5)]
    assert windows(20, 20, 10) == [(0, 20)]
    assert windows(25, 20, 10) == [(0, 20), (5, 25)]
    covered = {i for start, end in windows(47, 20, 10) for i in range(start, end)}
    assert covered == set(range(47))


def test_a_permutation_survives_a_messy_reply() -> None:
    assert parse_permutation("2, 0, 1", range(3)) == [2, 0, 1]
    assert parse_permutation("Ranking: 2 then 0.", range(3)) == [2, 0, 1]
    # A dropped label keeps its arrival order rather than vanishing from the pool.
    assert parse_permutation("1", range(3)) == [1, 0, 2]
    assert parse_permutation("9, 1, 1, 0", range(3)) == [1, 0, 2]


def test_borda_points_are_comparable_across_window_sizes() -> None:
    assert borda([0, 1, 2], 3) == {0: 1.0, 1: 0.5, 2: 0.0}
    assert borda([0], 1) == {0: 1.0}
    assert max(borda([3, 2, 1, 0], 4).values()) == 1.0


def llm_ranker(replies: Sequence[str], **overrides: Any) -> tuple[LLMReranker, FakeChatProvider]:
    chat = FakeChatProvider([ScriptedTurn(text=r) for r in replies], model="fake-model")
    return LLMReranker(chat=chat, model="fake-model", **overrides), chat


def test_the_listwise_reranker_follows_the_permutation_it_was_given() -> None:
    ranker, chat = llm_ranker(["2, 0, 1"], window=20, stride=10)
    report = anyio.run(lambda: ranker.rerank("slow and cold", POOL, top_k=3, doc_version="v1"))
    # Stage one order is 2, 3, 1 by prior score, so window places are 0:2, 1:3, 2:1.
    assert report.order() == (1, 2, 3)
    assert report.model_key == "fake-model@rerank_listwise.1"
    sent = chat.calls[0][0][0].content
    assert "slow and cold" in sent
    assert "0. a loud comedy about a wedding" in sent


def test_the_prompt_is_a_file_and_its_hash_is_available() -> None:
    ranker, _ = llm_ranker(["0, 1, 2"])
    assert "best match first" in load("rerank_listwise")
    assert len(ranker.prompt_key) == 64


def test_overlapping_windows_are_averaged_rather_than_last_one_wins() -> None:
    pool = tuple(RerankCandidate(i, f"film {i}", prior_score=1.0 - i / 10.0) for i in range(1, 5))
    ranker, chat = llm_ranker(["0, 1, 2", "2, 1, 0"], window=3, stride=1)
    report = anyio.run(lambda: ranker.rerank("anything", pool, top_k=4, doc_version="v1"))
    assert len(chat.calls) == 2
    # Film 3 is last in one window (0.0) and middle in the other (0.5), so neither window wins.
    assert report.scores()[3] == pytest.approx(0.25)
    assert report.scores()[2] == pytest.approx(0.25)
    assert report.scores()[1] == pytest.approx(1.0)


def test_a_pointwise_looking_reply_still_orders_the_pool() -> None:
    ranker, _ = llm_ranker(["I would say 1, then 2, and 0 last."])
    report = anyio.run(lambda: ranker.rerank("anything", POOL, top_k=3, doc_version="v1"))
    assert report.order() == (3, 1, 2)


def cached(tmp_path: Path) -> ScoreCache:
    db = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)
    return ScoreCache(db)


def test_a_score_survives_a_round_trip_and_the_doc_version_is_in_the_key(tmp_path: Path) -> None:
    store = cached(tmp_path)
    store.put("m", "slow", "v1", {1: 0.5, 2: -0.25})
    assert store.get("m", "slow", "v1", [1, 2, 3]) == {1: 0.5, 2: -0.25}
    assert store.get("m", "slow", "v2", [1, 2]) == {}
    assert store.get("other", "slow", "v1", [1, 2]) == {}
    assert store.get("m", "fast", "v1", [1, 2]) == {}
    store.db.close()


def test_a_disabled_cache_writes_nothing_and_reads_nothing(tmp_path: Path) -> None:
    store = cached(tmp_path)
    store.put("m", "slow", "v1", {1: 0.5})
    off = ScoreCache(store.db, enabled=False)
    off.put("m", "slow", "v1", {2: 0.5})
    assert off.get("m", "slow", "v1", [1, 2]) == {}
    assert store.get("m", "slow", "v1", [1, 2]) == {1: 0.5}
    store.db.close()


def test_the_second_sweep_over_one_pool_runs_no_forward_pass(
    tmp_path: Path, stub_st: types.ModuleType
) -> None:
    store = cached(tmp_path)
    ranker = build(cache=store)

    async def twice() -> tuple[RerankReport, RerankReport]:
        first = await ranker.rerank("a quiet journey", POOL, top_k=3, doc_version="v1")
        second = await ranker.rerank("a quiet journey", POOL, top_k=3, doc_version="v1")
        return first, second

    first, second = anyio.run(twice)
    assert (first.n_pairs, first.cache_hits) == (3, 0)
    assert (second.n_pairs, second.cache_hits) == (0, 3)
    assert second.order() == first.order()
    assert len(ranker.model.seen) == 1
    store.db.close()


def planted(tmp_path: Path) -> tuple[Any, Any]:
    from fixtures.synth import pipeline

    from palate.eval.labels import labels_of
    from palate.eval.split import SplitSpec, build_split, freeze_split, load_corpus_ids
    from palate.taste import profile as taste

    fitted = pipeline.fit(tmp_path)
    rated = taste.load_rated(fitted.db.read())
    spec = SplitSpec(name="rolling", min_reliable=50, cuts=(0.6, 0.8))
    split = build_split(rated, load_corpus_ids(fitted.db.read()), spec, relevance=labels_of)
    freeze_split(split, fitted.db)
    return fitted, split


def test_an_arm_is_blocked_until_its_reranker_is_in_the_context() -> None:
    arm = by_name("+cross_encoder(minilm)")
    assert rerank_key(arm) == "minilm"
    assert blocked_reason(arm) == "needs a minilm reranker in the eval context"
    assert blocked_reason(arm, rerankers={"minilm"}) is None
    assert rerank_key(by_name("full")) is None
    assert blocked_reason(by_name("doc_no_credits"), rerankers={"minilm"}) is not None


async def test_a_reranker_moves_the_ranking_and_full_stays_where_it_was(tmp_path: Path) -> None:
    fitted, split = planted(tmp_path)
    ranker = Reversing()
    ctx = EvalContext(db=fitted.db, split=split, resamples=20, restarts=2, max_rounds=4)
    reranked = replace(ctx, rerankers={"minilm": ranker})
    fold = split.folds[0]
    plain = await run_system(by_name("full"), fold, ctx)
    armed = await run_system(by_name("+cross_encoder(minilm)"), fold, reranked)
    assert armed.ranked != plain.ranked
    assert armed.stage_latency_ms["rerank"] > 0.0
    assert "no_reranker" not in armed.degraded
    assert (
        "no_reranker" in (await run_system(by_name("+cross_encoder(minilm)"), fold, ctx)).degraded
    )
    # An unconditioned run has no user text, so the reranker is handed the synthesised query.
    assert ranker.queries and all(q for q in ranker.queries)
    fitted.close()


async def test_the_matrix_runs_a_rerank_arm_only_when_the_context_can(tmp_path: Path) -> None:
    fitted, split = planted(tmp_path)
    arms = [by_name(n) for n in ("full", "+cross_encoder(minilm)", "doc_no_credits")]
    ctx = EvalContext(db=fitted.db, split=split, resamples=20, restarts=2, max_rounds=4)
    bare = await run_matrix(arms, ctx)
    assert {r.system for r in bare} == {"full"}
    armed = await run_matrix(arms, replace(ctx, rerankers={"minilm": Reversing()}), force=True)
    assert {r.system for r in armed} == {"full", "+cross_encoder(minilm)"}
    fitted.close()


def test_the_synthesised_query_is_built_from_the_modes(tmp_path: Path) -> None:
    from fixtures.synth import pipeline

    fitted = pipeline.fit(tmp_path)
    query = mode_query(fitted.profile, fitted.db.read())
    assert query and query != COLD_QUERY
    assert query == mode_query(fitted.profile, fitted.db.read())
    fitted.close()


def test_a_film_with_no_rendered_document_keeps_its_stage_one_place(tmp_path: Path) -> None:
    from fixtures.synth import pipeline

    fitted = pipeline.fit(tmp_path)
    conn = fitted.db.read()
    ids = [int(r["tmdb_id"]) for r in conn.execute("select tmdb_id from film_docs limit 3")]
    texts = load_texts(conn, [*ids, -1])
    pool = rerank_candidates([*ids, -1], {i: 1.0 for i in ids}, texts)
    assert [c.film_id for c in pool] == ids
    assert all(c.text for c in pool)
    fitted.close()


def test_a_reranked_arm_gets_a_ce_score_weight_the_fit_never_produced() -> None:
    fitted = FusionWeights(condition="unconditioned", beta={"mode_affinity": 1.0}, fitted_on="f")
    plain = weights_for(by_name("full"), ("mode_affinity",), "unconditioned", fitted)
    armed = weights_for(
        by_name("+cross_encoder(minilm)"), ("mode_affinity",), "unconditioned", fitted
    )
    assert plain.beta.get("ce_score", 0.0) == 0.0
    assert armed.beta["ce_score"] == pytest.approx(prior_share("ce_score"))
    assert armed.beta["ce_score"] > 0.0


async def test_published_weights_do_not_silence_the_reranker(tmp_path: Path) -> None:
    fitted, split = planted(tmp_path)
    ctx = EvalContext(db=fitted.db, split=split, resamples=20, restarts=2, max_rounds=4)
    publish_weights(ctx, by_name("full"))
    fold = split.folds[0]
    plain = await run_system(by_name("full"), fold, ctx)
    armed = await run_system(
        by_name("+cross_encoder(minilm)"), fold, replace(ctx, rerankers={"minilm": Reversing()})
    )
    # Every reranked film is scored differently. Whether that reorders the head is the measurement.
    shared = set(plain.scores) & set(armed.scores)
    assert shared
    assert all(plain.scores[i] != armed.scores[i] for i in shared)
    fitted.close()


def test_the_report_says_which_reranker_an_arm_is_still_waiting_for() -> None:
    rows, found = record.load()
    assert reporting.rerank_table(rows, found) == reporting.NO_RERANK
    assert "needs a minilm reranker" in reporting.not_run(rows)
    assert "needs a minilm reranker" not in reporting.not_run(rows, rerankers={"minilm"})
    assert "doc_no_credits" in reporting.not_run(rows, rerankers={"minilm"})
    # An arm nobody asked to run is not a blocked arm, so it is not listed as one.
    assert "no_diversity" not in reporting.not_run([])


def test_the_rerank_table_carries_the_delta_and_the_dollars() -> None:
    rows = [
        reporting.Row("full", {"ndcg@10": MetricCI(0.30, 0.2, 0.4, 10)}, 2),
        reporting.Row("+cross_encoder(minilm)", {"ndcg@10": MetricCI(0.35, 0.25, 0.45, 10)}, 2),
        reporting.Row(
            "+llm_rerank", {"ndcg@10": MetricCI(0.28, 0.18, 0.38, 10)}, 2, cost_usd=0.0021
        ),
    ]
    found = {
        ("+cross_encoder(minilm)", "full"): reporting.Delta(0.05, 0.01, 0.09, 2, 2),
        ("+llm_rerank", "full"): reporting.Delta(-0.02, -0.06, 0.03, 0, 2),
    }
    table = reporting.rerank_table(rows, found)
    assert "| +0.050 |" in table
    # A local reranker costs zero dollars and real seconds, and the table shows both.
    assert "| -0.020 ns |  |  | 0.0021 |" in table
    assert "| full | 0.300 |" in table


def timed(ms: float, reranked: int, *, cold: bool) -> Any:
    from palate.eval.harness import _Ranking

    return _Ranking(
        order=(),
        scores={},
        pool_size=0,
        prefilter_path="none",
        mode_of={},
        stages={"rerank": ms},
        reranked=reranked,
        cold_start=cold,
    )


def test_the_fold_that_paid_the_load_is_not_averaged_into_the_warm_number() -> None:
    assert rerank_metrics([timed(5.0, 0, cold=True)]) == {}
    both = rerank_metrics([timed(900.0, 100, cold=True), timed(40.0, 100, cold=False)])
    assert both["rerank_ms_cold"].point == pytest.approx(900.0)
    assert both["rerank_ms_warm"].point == pytest.approx(40.0)
    assert both["rerank_ms_warm"].n == 100
    warm_only = rerank_metrics([timed(40.0, 100, cold=False), timed(60.0, 100, cold=False)])
    assert set(warm_only) == {"rerank_ms_warm"}
    assert warm_only["rerank_ms_warm"].point == pytest.approx(50.0)


async def test_a_run_carries_its_rerank_latency_into_the_stored_metrics(tmp_path: Path) -> None:
    fitted, split = planted(tmp_path)
    ctx = EvalContext(db=fitted.db, split=split, resamples=20, restarts=2, max_rounds=4)
    armed = replace(ctx, rerankers={"minilm": Reversing()})
    first = await run_system(by_name("+cross_encoder(minilm)"), split.folds[0], armed)
    second = await run_system(by_name("+cross_encoder(minilm)"), split.folds[1], armed)
    assert "rerank_ms_cold" in first.metrics
    assert "rerank_ms_warm" not in first.metrics
    assert "rerank_ms_warm" in second.metrics
    assert first.metrics["rerank_ms_cold"].n == 100
    plain = await run_system(by_name("full"), split.folds[0], ctx)
    assert not [name for name in plain.metrics if name.startswith("rerank_ms")]
    fitted.close()


def test_the_rerank_table_reads_cold_and_warm_from_the_stored_metrics() -> None:
    rows = [
        reporting.Row("full", {"ndcg@10": MetricCI(0.30, 0.2, 0.4, 10)}, 2),
        reporting.Row(
            "+cross_encoder(bge-m3)",
            {
                "ndcg@10": MetricCI(0.35, 0.25, 0.45, 10),
                "rerank_ms_cold": MetricCI(2400.0, 2400.0, 2400.0, 100),
                "rerank_ms_warm": MetricCI(810.0, 810.0, 810.0, 100),
            },
            2,
        ),
    ]
    table = reporting.rerank_table(rows, {})
    assert "| cold ms | warm ms | usd |" in table
    assert "| 2400 | 810 | 0.0000 |" in table
    # The control reranks nothing, so it has no rerank latency to print at all.
    assert "| full | 0.300 | 0.20-0.40 |  |  |  |  | 0.0000 |" in table
