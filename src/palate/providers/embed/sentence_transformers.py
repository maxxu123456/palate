"""embeddinggemma on MPS, through sentence-transformers, behind the optional extra."""

from __future__ import annotations

import time
from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio

from palate.extras import require
from palate.hf.device import mps_limiter, resolve_device
from palate.hf.models import ModelPin
from palate.providers.base import EmbeddingBatch, ProviderHealth, SpanLike, Vector
from palate.providers.fingerprint import CANARY_TEXT, EmbeddingFingerprint


class SentenceTransformersEmbedder:
    """Float32 throughout. Half precision on MPS buys nothing and risks quiet wrongness."""

    provider = "sentence_transformers"

    def __init__(
        self,
        *,
        pin: ModelPin,
        device: str | None = None,
        batch_size: int = 64,
        truncate_dim: int | None = None,
        limiter: anyio.CapacityLimiter | None = None,
    ) -> None:
        st = require("local", "sentence_transformers")
        self.pin = pin
        self.device = resolve_device(device)
        self.max_batch = batch_size
        self.truncate_dim = truncate_dim
        self._limiter = limiter
        self.model: Any = st.SentenceTransformer(
            pin.repo_id,
            revision=pin.revision,
            device=self.device,
            truncate_dim=truncate_dim,
        )

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        """The revision is a real commit sha, which is the only honest local pin."""
        return EmbeddingFingerprint(
            provider=self.provider,
            model_id=self.pin.repo_id,
            revision=self.pin.revision,
            dim=self.truncate_dim
            or self.pin.dim
            or int(self.model.get_sentence_embedding_dimension()),
            normalized=True,
            query_prompt=self.pin.query_prompt,
            document_prompt=self.pin.document_prompt,
            pooling=self.pin.pooling,
        )

    async def ready(self) -> EmbeddingFingerprint:
        """The model is already loaded, so there is nothing to probe."""
        return self.fingerprint

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch:
        """The document side of the asymmetry, run off the event loop."""
        started = time.perf_counter()
        vectors = await self._encode([self.pin.document_prompt + t for t in texts])
        return EmbeddingBatch(
            vectors=vectors,
            fingerprint=self.fingerprint,
            input_tokens=None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def embed_query(self, text: str, *, span: SpanLike | None = None) -> Vector:
        """The query side. Using the document prefix here degrades everything, silently."""
        vectors = await self._encode([self.pin.query_prompt + text])
        return vectors[0]

    async def health(self) -> ProviderHealth:
        """Encode the canary once, which also warms the model."""
        started = time.perf_counter()
        await self._encode([CANARY_TEXT])
        elapsed = (time.perf_counter() - started) * 1000
        return ProviderHealth(True, f"{self.pin.repo_id} on {self.device}", elapsed)

    async def aclose(self) -> None:
        """Drop the model so the next process is not fighting it for memory."""
        self.model = None

    async def _encode(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        # A blocking forward pass on the event loop stalls every other stream on the worker.
        call = partial(
            self.model.encode,
            list(texts),
            batch_size=self.max_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = await anyio.to_thread.run_sync(call, limiter=self._limiter or mps_limiter())
        return tuple(tuple(float(x) for x in row) for row in result)
