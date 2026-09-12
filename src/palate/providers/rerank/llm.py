"""Listwise rerank through a chat model: permutations over a sliding window, Borda averaged."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence

from palate.agent.prompts import load, prompt_sha
from palate.providers.base import (
    ChatProvider,
    Message,
    RerankCandidate,
    RerankReport,
    RerankResult,
    SpanLike,
)
from palate.providers.rerank.cache import ScoreCache

# One line of the candidate block is one film, and the label is what the model echoes back.
MAX_CHARS = 400

_LABEL = re.compile(r"\d+")


def windows(n: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Half open spans covering every candidate, overlapping by size minus stride."""
    if n <= 0:
        return []
    step = max(stride, 1)
    starts = list(range(0, max(n - size, 0) + 1, step))
    if starts[-1] + size < n:
        starts.append(n - size)
    return [(start, min(start + size, n)) for start in starts]


def parse_permutation(reply: str, labels: Sequence[int]) -> list[int]:
    """Labels in the order the model gave them, missing ones appended as they arrived."""
    allowed = list(labels)
    seen: list[int] = []
    for found in _LABEL.findall(reply):
        label = int(found)
        if label in allowed and label not in seen:
            seen.append(label)
    seen.extend(label for label in allowed if label not in seen)
    return seen


def borda(order: Sequence[int], size: int) -> dict[int, float]:
    """Points in [0, 1] by place, so windows of different lengths average on one scale."""
    if size <= 1:
        return dict.fromkeys(order, 1.0)
    return {label: (size - 1 - place) / (size - 1) for place, label in enumerate(order)}


class LLMReranker:
    """Never pointwise. A one to ten score from a chat model clusters on 7 and carries no order."""

    name = "llm"

    def __init__(
        self,
        *,
        chat: ChatProvider,
        model: str,
        window: int = 20,
        stride: int = 10,
        passes: int = 1,
        prompt_name: str = "rerank_listwise",
        prompt_version: str = "1",
        cache: ScoreCache | None = None,
    ) -> None:
        self.chat = chat
        self.model = model
        self.window = window
        self.stride = stride
        self.passes = passes
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.cache = cache
        self.model_key = f"{model}@{prompt_name}.{prompt_version}"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
        doc_version: str,
        span: SpanLike | None = None,
    ) -> RerankReport:
        """Sweep the pool in overlapping windows and average the places each film was given."""
        started = time.perf_counter()
        ordered = sorted(candidates, key=lambda c: (-c.prior_score, c.film_id))
        known = (
            self.cache.get(self.model_key, query, doc_version, [c.film_id for c in ordered])
            if self.cache is not None
            else {}
        )
        if len(known) == len(ordered) and ordered:
            return self._report(ordered, known, top_k, started, calls=0, hits=len(known), cost=0.0)
        points: dict[int, list[float]] = {c.film_id: [] for c in ordered}
        calls = 0
        cost = 0.0
        for _ in range(max(self.passes, 1)):
            for start, end in windows(len(ordered), self.window, self.stride):
                chunk = ordered[start:end]
                spent, order = await self._one_window(query, chunk)
                calls += 1
                cost += spent
                for label, point in borda(order, len(chunk)).items():
                    points[chunk[label].film_id].append(point)
        scores = {
            film_id: (sum(seen) / len(seen) if seen else 0.0) for film_id, seen in points.items()
        }
        if self.cache is not None:
            self.cache.put(self.model_key, query, doc_version, scores)
        if span is not None:
            span.event("rerank", model_key=self.model_key, n_pairs=len(ordered), calls=calls)
        return self._report(ordered, scores, top_k, started, calls=calls, hits=0, cost=cost)

    async def aclose(self) -> None:
        """The chat provider is shared, so closing it is the caller's job."""
        return None

    def _report(
        self,
        ordered: Sequence[RerankCandidate],
        scores: Mapping[int, float],
        top_k: int,
        started: float,
        *,
        calls: int,
        hits: int,
        cost: float,
    ) -> RerankReport:
        ranked = sorted(ordered, key=lambda c: (-scores[c.film_id], c.film_id))
        return RerankReport(
            results=tuple(
                RerankResult(c.film_id, scores[c.film_id], rank)
                for rank, c in enumerate(ranked[:top_k], start=1)
            ),
            model_key=self.model_key,
            n_pairs=len(ordered) if calls else 0,
            cache_hits=hits,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            cost_usd=cost,
            cold_start=False,
        )

    async def _one_window(
        self, query: str, chunk: Sequence[RerankCandidate]
    ) -> tuple[float, list[int]]:
        lines = "\n".join(f"{i}. {c.text[:MAX_CHARS]}" for i, c in enumerate(chunk))
        text = load(self.prompt_name).format(query=query, candidates=lines)
        done = await self.chat.complete(
            [Message(role="user", content=text)], temperature=0.0, seed=0
        )
        return float(done.cost_usd or 0.0), parse_permutation(done.content, range(len(chunk)))

    @property
    def prompt_key(self) -> str:
        """Hash of the prompt text, so an edit to it is a different cache."""
        return prompt_sha(self.prompt_name)
