"""A Hub cross-encoder on MPS, behind the optional extra, scoring raw logits."""

from __future__ import annotations

import time
from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio

from palate.extras import require
from palate.hf.device import check_precision, mps_limiter, resolve_device
from palate.hf.models import ModelPin
from palate.providers.base import (
    RerankCandidate,
    RerankReport,
    RerankResult,
    SpanLike,
)
from palate.providers.rerank.cache import ScoreCache

# A fixed pair, so the warm-up costs the same on every machine and never touches user text.
WARMUP_PAIR = ("slow and cold", "A long quiet film about a journey through wet country.")


def _identity(scores: Any) -> Any:
    # The checkpoint's own sigmoid squashes the tail into float ties and the eval wants the spread.
    return scores


class CrossEncoderReranker:
    """Float32 always, and the revision rides in the model key so a moved pin is a new cache."""

    name = "cross_encoder"

    def __init__(
        self,
        *,
        pin: ModelPin,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        dtype: str = "float32",
        limiter: anyio.CapacityLimiter | None = None,
        cache: ScoreCache | None = None,
    ) -> None:
        st = require("local", "sentence_transformers")
        self.pin = pin
        self.device = resolve_device(device)
        self.dtype = check_precision(dtype, self.device)
        self.batch_size = batch_size
        self.model_key = f"{pin.repo_id}@{pin.revision[:12]}"
        self.cache = cache
        self._limiter = limiter
        self._warm = False
        # num_labels, max_length, activation_fn and device are keyword only from v6 on.
        self.model: Any = st.CrossEncoder(
            pin.repo_id,
            revision=pin.revision,
            device=self.device,
            max_length=max_length,
        )

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
        doc_version: str,
        span: SpanLike | None = None,
    ) -> RerankReport:
        """Score every (query, document) pair and return them best first."""
        started = time.perf_counter()
        known = (
            self.cache.get(self.model_key, query, doc_version, [c.film_id for c in candidates])
            if self.cache is not None
            else {}
        )
        fresh = [c for c in candidates if c.film_id not in known]
        cold = not self._warm and bool(fresh)
        scored = await self._predict([(query, c.text) for c in fresh])
        if fresh:
            self._warm = True
        computed = dict(zip((c.film_id for c in fresh), scored, strict=True))
        if self.cache is not None:
            self.cache.put(self.model_key, query, doc_version, computed)
        scores = {**known, **computed}
        ranked = sorted(candidates, key=lambda c: (-scores[c.film_id], c.film_id))
        results = tuple(
            RerankResult(candidate.film_id, scores[candidate.film_id], rank)
            for rank, candidate in enumerate(ranked[:top_k], start=1)
        )
        report = RerankReport(
            results=results,
            model_key=self.model_key,
            n_pairs=len(fresh),
            cache_hits=len(known),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            cold_start=cold,
        )
        if span is not None:
            span.event("rerank", model_key=self.model_key, n_pairs=len(fresh), cold_start=cold)
        return report

    async def warmup(self) -> float:
        """One forward pass on a fixed pair, so the first real query does not pay the load."""
        started = time.perf_counter()
        await self._predict([WARMUP_PAIR])
        self._warm = True
        return (time.perf_counter() - started) * 1000.0

    async def aclose(self) -> None:
        """Drop the checkpoint, because torch does not hand unified memory back on its own."""
        self.model = None

    async def _predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        # A blocking forward pass on the event loop stalls every other stream on the worker.
        call = partial(
            self.model.predict,
            list(pairs),
            batch_size=self.batch_size,
            activation_fn=_identity,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = await anyio.to_thread.run_sync(call, limiter=self._limiter or mps_limiter())
        return [float(x) for x in out]
