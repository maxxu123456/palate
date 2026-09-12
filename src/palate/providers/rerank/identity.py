"""The ablation control and the offline test double: stage one order, unchanged."""

from __future__ import annotations

import time
from collections.abc import Sequence

from palate.providers.base import (
    RerankCandidate,
    RerankReport,
    RerankResult,
    SpanLike,
)


class IdentityReranker:
    """Hands back the prior order and the prior scores, so ce_score adds no information."""

    name = "identity"

    def __init__(self, *, model_key: str = "identity") -> None:
        self.model_key = model_key
        self.calls = 0

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
        doc_version: str,
        span: SpanLike | None = None,
    ) -> RerankReport:
        """The pool as it arrived, so an arm can pay the plumbing without paying a model."""
        started = time.perf_counter()
        self.calls += 1
        ranked = sorted(candidates, key=lambda c: (-c.prior_score, c.film_id))
        results = tuple(
            RerankResult(c.film_id, c.prior_score, rank)
            for rank, c in enumerate(ranked[:top_k], start=1)
        )
        return RerankReport(
            results=results,
            model_key=self.model_key,
            n_pairs=len(candidates),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def aclose(self) -> None:
        """Nothing was ever opened."""
        return None
