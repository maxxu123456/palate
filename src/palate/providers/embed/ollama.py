"""Ollama /api/embed, which already returns L2 normalised vectors."""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx
import orjson

from palate.errors import ProviderTimeout, ProviderUnavailable
from palate.providers.base import EmbeddingBatch, ProviderHealth, SpanLike, Vector, json_int
from palate.providers.fingerprint import CANARY_TEXT, EmbeddingFingerprint
from palate.providers.retry import LOCAL_RETRY, RetryPolicy, classify, with_retry

BASE_URL = "http://127.0.0.1:11434"


class OllamaEmbedder:
    """Ollama templates the input itself, so its vectors are not the library's vectors."""

    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        client: httpx.AsyncClient,
        base_url: str = BASE_URL,
        max_batch: int = 64,
        dim: int | None = None,
        revision: str | None = None,
        query_prompt: str = "",
        document_prompt: str = "",
        timeout_s: float = 120.0,
        retry: RetryPolicy = LOCAL_RETRY,
    ) -> None:
        self.model = model
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.max_batch = max_batch
        self.timeout_s = timeout_s
        self.retry = retry
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self._dim = dim
        self._revision = revision

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        """Known only after ready(), because the digest and the dim both come from the host."""
        if self._dim is None:
            raise RuntimeError("await ready() before reading the fingerprint")
        return EmbeddingFingerprint(
            provider=self.provider,
            model_id=self.model,
            revision=self._revision,
            dim=self._dim,
            normalized=True,
            query_prompt=self.query_prompt,
            document_prompt=self.document_prompt,
            pooling="ollama",
        )

    async def ready(self) -> EmbeddingFingerprint:
        """Resolve the model digest and the dimension, one request each at most."""
        if self._revision is None:
            self._revision = await self._digest()
        if self._dim is None:
            vectors = await self._embed([CANARY_TEXT])
            self._dim = len(vectors[0])
        return self.fingerprint

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch:
        """Documents, prefixed and batched to max_batch."""
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
        """The query side, which carries a different prefix on an asymmetric model."""
        vectors = await self._embed([self.query_prompt + text])
        return vectors[0]

    async def health(self) -> ProviderHealth:
        """Ask the daemon what it serves."""
        started = time.perf_counter()
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
        except httpx.HTTPError:
            return ProviderHealth(
                False, f"no ollama at {self.base_url}, try: ollama pull {self.model}"
            )
        elapsed = (time.perf_counter() - started) * 1000
        if self._names(orjson.loads(response.content)).get(self.model.split(":")[0]) is None:
            return ProviderHealth(False, f"ollama pull {self.model}", elapsed)
        return ProviderHealth(True, f"ollama serving {self.model}", elapsed)

    async def aclose(self) -> None:
        """The client is shared, so closing it is the caller's job."""
        return None

    @staticmethod
    def _names(payload: dict[str, object]) -> dict[str, str]:
        models = payload.get("models")
        if not isinstance(models, list):
            return {}
        return {
            str(m.get("model", "")).split(":")[0]: str(m.get("digest", ""))
            for m in models
            if isinstance(m, dict)
        }

    async def _digest(self) -> str | None:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=10.0)
        except httpx.HTTPError:
            return None
        return self._names(orjson.loads(response.content)).get(self.model.split(":")[0])

    async def _embed(self, inputs: Sequence[str]) -> tuple[Vector, ...]:
        vectors, _ = await self._embed_counted(inputs)
        return vectors

    async def _embed_counted(self, inputs: Sequence[str]) -> tuple[tuple[Vector, ...], int]:
        body = {"model": self.model, "input": list(inputs), "truncate": True}

        async def once() -> tuple[tuple[Vector, ...], int]:
            request = self.client.build_request(
                "POST", f"{self.base_url}/api/embed", json=body, timeout=self.timeout_s
            )
            try:
                response = await self.client.send(request)
            except httpx.TimeoutException as exc:
                raise ProviderTimeout(
                    "ollama embed timed out",
                    provider=self.provider,
                    model=self.model,
                    retryable=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(
                    f"no ollama at {self.base_url} ({exc})",
                    provider=self.provider,
                    model=self.model,
                    retryable=True,
                ) from exc
            error = classify(response, provider=self.provider, model=self.model)
            if error is not None:
                raise error
            payload = orjson.loads(response.content)
            rows = payload.get("embeddings") or []
            return tuple(tuple(float(x) for x in row) for row in rows), json_int(
                payload.get("prompt_eval_count")
            )

        return await with_retry(once, policy=self.retry)
