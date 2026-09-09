"""Hash seeded unit vectors: deterministic on every platform, which index fixtures need."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

import numpy as np

from palate.providers.base import EmbeddingBatch, ProviderHealth, SpanLike, Vector
from palate.providers.fingerprint import EmbeddingFingerprint


class FakeEmbedder:
    """The same text always gives the same vector, on any machine and any Python."""

    provider = "fake"

    def __init__(self, *, dim: int = 64, seed: int = 0, max_batch: int = 128) -> None:
        self.dim = dim
        self.seed = seed
        self.max_batch = max_batch

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        """A real fingerprint, so the index identity path is exercised offline."""
        return EmbeddingFingerprint(
            provider=self.provider,
            model_id=f"fake-{self.dim}",
            revision=str(self.seed),
            dim=self.dim,
            normalized=True,
            pooling="none",
        )

    async def ready(self) -> EmbeddingFingerprint:
        """Nothing to probe."""
        return self.fingerprint

    def vector(self, text: str) -> Vector:
        """One unit vector derived from the text, through a seeded generator."""
        digest = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        raw = rng.standard_normal(self.dim)
        norm = float(np.linalg.norm(raw))
        unit = raw / norm if norm else raw
        return tuple(float(x) for x in unit)

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch:
        """Every document, in order."""
        started = time.perf_counter()
        vectors = tuple(self.vector(t) for t in texts)
        return EmbeddingBatch(
            vectors=vectors,
            fingerprint=self.fingerprint,
            input_tokens=sum(len(t) for t in texts) // 4,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def embed_query(self, text: str, *, span: SpanLike | None = None) -> Vector:
        """A query vector in the same space, since the fake has no asymmetry."""
        return self.vector(text)

    async def health(self) -> ProviderHealth:
        """Always up."""
        return ProviderHealth(True, "fake embedder", 0.0)

    async def aclose(self) -> None:
        """Nothing to close."""
        return None
