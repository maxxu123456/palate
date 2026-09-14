"""The paraphrase arm, which reuses the reranker checkpoint rather than loading a second one."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio
from sentence_transformers import CrossEncoder

from palate.hf.device import check_precision, mps_limiter, resolve_device
from palate.hf.models import ModelPin


def _sigmoid(scores: Any) -> Any:
    # Entailment is reported as a probability, so the threshold sweep has one scale.
    import math

    return [1.0 / (1.0 + math.exp(-float(x))) for x in scores]


class CrossEncoderNLI:
    """The reranker checkpoint, reused through the same gate rather than loaded twice."""

    def __init__(
        self,
        *,
        pin: ModelPin,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        dtype: str = "float32",
        limiter: anyio.CapacityLimiter | None = None,
    ) -> None:
        self.pin = pin
        self.device = resolve_device(device)
        self.dtype = check_precision(dtype, self.device)
        self.batch_size = batch_size
        self.model_key = f"{pin.repo_id}@{pin.revision[:12]}"
        self._limiter = limiter
        self.model: Any = CrossEncoder(
            pin.repo_id,
            revision=pin.revision,
            device=self.device,
            max_length=max_length,
        )

    async def entails(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One score per (premise, hypothesis), off the event loop and one batch at a time."""
        if not pairs:
            return []
        call = partial(
            self.model.predict,
            list(pairs),
            batch_size=self.batch_size,
            activation_fn=_sigmoid,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        out = await anyio.to_thread.run_sync(call, limiter=self._limiter or mps_limiter())
        return [float(x) for x in out]

    async def aclose(self) -> None:
        """Drop the checkpoint, because torch does not hand unified memory back on its own."""
        self.model = None
