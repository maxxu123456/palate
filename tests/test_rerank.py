"""Rerankers, offline. A stub checkpoint stands in for the Hub so no torch is needed."""

from __future__ import annotations

import sys
import types
from typing import Any

import anyio
import pytest

from palate.errors import ConfigError, MissingExtra
from palate.hf.device import check_precision
from palate.hf.models import ModelPin
from palate.providers.base import RerankCandidate, Reranker
from palate.providers.rerank.identity import IdentityReranker

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
