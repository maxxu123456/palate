"""Any host serving POST {base_url}/embeddings."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import httpx
import orjson
from pydantic import SecretStr

from palate.errors import ProviderTimeout, ProviderUnavailable
from palate.providers.base import EmbeddingBatch, ProviderHealth, SpanLike, Vector, json_int
from palate.providers.fingerprint import CANARY_TEXT, EmbeddingFingerprint
from palate.providers.retry import DEFAULT_RETRY, RetryPolicy, classify, with_retry


def l2_normalize(vector: Sequence[float]) -> Vector:
    """Unit length, so every index in the system stores the same kind of vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    return tuple(x / norm for x in vector) if norm else tuple(vector)


class OpenAICompatEmbedder:
    """Normalises unconditionally, because cosine then reduces to a dot product."""

    provider = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        client: httpx.AsyncClient,
        api_key: SecretStr | None = None,
        max_batch: int = 64,
        dim: int | None = None,
        truncate_dim: int | None = None,
        query_prompt: str = "",
        document_prompt: str = "",
        timeout_s: float = 120.0,
        retry: RetryPolicy = DEFAULT_RETRY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client
        self.api_key = api_key
        self.max_batch = max_batch
        self.truncate_dim = truncate_dim
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.timeout_s = timeout_s
        self.retry = retry
        self._dim = dim

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        """The host reports no revision, so the canary is the only drift check."""
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
            pooling="server",
        )

    async def ready(self) -> EmbeddingFingerprint:
        """One probe request when the dimension was not configured."""
        if self._dim is None:
            probe = await self._embed([CANARY_TEXT])
            self._dim = len(probe[0])
        return self.fingerprint

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch:
        """Documents, prefixed and batched."""
        await self.ready()
        started = time.perf_counter()
        vectors: list[Vector] = []
        tokens = 0
        for start in range(0, len(texts), self.max_batch):
            chunk = [self.document_prompt + t for t in texts[start : start + self.max_batch]]
            batch, counted = await self._embed_counted(chunk)
            vectors.extend(batch)
            tokens += counted
        return EmbeddingBatch(
            vectors=tuple(vectors),
            fingerprint=self.fingerprint,
            input_tokens=tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def embed_query(self, text: str, *, span: SpanLike | None = None) -> Vector:
        """The query side of an asymmetric model."""
        vectors = await self._embed([self.query_prompt + text])
        return vectors[0]

    async def health(self) -> ProviderHealth:
        """One probe embedding, since /models says nothing about the embedding route."""
        started = time.perf_counter()
        try:
            await self._embed([CANARY_TEXT])
        except (ProviderTimeout, ProviderUnavailable) as exc:
            return ProviderHealth(False, str(exc))
        elapsed = (time.perf_counter() - started) * 1000
        return ProviderHealth(True, f"{self.base_url} embedded", elapsed)

    async def aclose(self) -> None:
        """The client is shared, so closing it is the caller's job."""
        return None

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

    async def _embed(self, inputs: Sequence[str]) -> tuple[Vector, ...]:
        vectors, _ = await self._embed_counted(inputs)
        return vectors

    async def _embed_counted(self, inputs: Sequence[str]) -> tuple[tuple[Vector, ...], int]:
        body: dict[str, object] = {
            "model": self.model,
            "input": list(inputs),
            "encoding_format": "float",
        }
        if self.truncate_dim is not None:
            body["dimensions"] = self.truncate_dim

        async def once() -> tuple[tuple[Vector, ...], int]:
            request = self.client.build_request(
                "POST",
                f"{self.base_url}/embeddings",
                json=body,
                headers=self._headers(),
                timeout=self.timeout_s,
            )
            try:
                response = await self.client.send(request)
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(
                    "embeddings timed out",
                    provider=self.provider,
                    model=self.model,
                    retryable=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(
                    f"{self.base_url} unreachable ({exc})",
                    provider=self.provider,
                    model=self.model,
                    retryable=True,
                ) from exc
            error = classify(response, provider=self.provider, model=self.model)
            if error is not None:
                raise error
            payload = orjson.loads(response.content)
            rows = sorted(payload.get("data") or [], key=lambda r: r.get("index", 0))
            usage = payload.get("usage") or {}
            vectors = tuple(l2_normalize(row["embedding"]) for row in rows)
            return vectors, json_int(usage.get("prompt_tokens"))

        return await with_retry(once, policy=self.retry)
