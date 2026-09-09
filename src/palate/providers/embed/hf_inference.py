"""feature_extraction through the HF router, where the array shape varies by model."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from pydantic import SecretStr

from palate.extras import require
from palate.providers.base import EmbeddingBatch, ProviderHealth, SpanLike, Vector
from palate.providers.embed.openai_compat import l2_normalize
from palate.providers.fingerprint import CANARY_TEXT, EmbeddingFingerprint


def pool(raw: Any) -> Vector:
    """One text's output, whatever nesting the model returned it in."""
    values = _as_lists(raw)
    while values and isinstance(values[0], list):
        columns = list(zip(*values, strict=True))
        values = [sum(col) / len(col) for col in columns]
    return tuple(float(x) for x in values)


def _as_lists(raw: Any) -> list[Any]:
    if hasattr(raw, "tolist"):
        listed: list[Any] = raw.tolist()
        return listed
    return list(raw)


class HFInferenceEmbedder:
    """The router pools nothing, so mean pooling over the token axis happens here."""

    provider = "hf_inference"

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | None = None,
        provider_route: str = "auto",
        client: Any | None = None,
        max_batch: int = 32,
        dim: int | None = None,
        query_prompt: str = "",
        document_prompt: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.provider_route = provider_route
        self.max_batch = max_batch
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.timeout_s = timeout_s
        self._dim = dim
        if client is None:
            hub = require("hf", "huggingface_hub")
            client = hub.AsyncInferenceClient(
                model=model,
                provider=provider_route,
                api_key=api_key.get_secret_value() if api_key else None,
                timeout=timeout_s,
            )
        self.client = client

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        """No revision from the router, so pooling and prompts carry the identity."""
        if self._dim is None:
            raise RuntimeError("await ready() before reading the fingerprint")
        return EmbeddingFingerprint(
            provider=self.provider,
            model_id=self.model,
            revision=None,
            dim=self._dim,
            normalized=True,
            query_prompt=self.query_prompt,
            document_prompt=self.document_prompt,
            pooling="mean",
        )

    async def ready(self) -> EmbeddingFingerprint:
        """One probe call when the dimension was not configured."""
        if self._dim is None:
            self._dim = len(await self.embed_query(CANARY_TEXT))
        return self.fingerprint

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch:
        """Documents, batched and pooled."""
        started = time.perf_counter()
        vectors: list[Vector] = []
        for start in range(0, len(texts), self.max_batch):
            chunk = [self.document_prompt + t for t in texts[start : start + self.max_batch]]
            raw = await self.client.feature_extraction(text=chunk)
            vectors.extend(l2_normalize(pool(row)) for row in _as_lists(raw))
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return EmbeddingBatch(
            vectors=tuple(vectors),
            fingerprint=self.fingerprint,
            input_tokens=None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def embed_query(self, text: str, *, span: SpanLike | None = None) -> Vector:
        """The query side, pooled the same way."""
        raw = await self.client.feature_extraction(text=self.query_prompt + text)
        return l2_normalize(pool(raw))

    async def health(self) -> ProviderHealth:
        """One embedding of the canary sentence."""
        started = time.perf_counter()
        try:
            await self.embed_query(CANARY_TEXT)
        except Exception as exc:
            return ProviderHealth(False, f"hf router refused ({exc})")
        return ProviderHealth(
            True, f"hf router embedded with {self.model}", (time.perf_counter() - started) * 1000
        )

    async def aclose(self) -> None:
        """Close the hub client, which owns its own session."""
        closer = getattr(self.client, "close", None)
        if closer is not None:
            await closer()
