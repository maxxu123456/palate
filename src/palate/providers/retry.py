"""Retry policy, the backoff loop, and HTTP status to exception."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import anyio
import httpx

from palate.errors import (
    ModelNotFound,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderContextOverflow,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from palate.providers.base import SpanLike

RETRYABLE: frozenset[type[ProviderError]] = frozenset(
    (ProviderRateLimited, ProviderUnavailable, ProviderTimeout)
)

# Phrases hosts use for a prompt that did not fit. Only ever checked on a 4xx body.
_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "prompt is too long",
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times, how long between, and how long in total."""

    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    max_total_s: float = 120.0
    jitter: Literal["full", "none"] = "full"
    retry_on: frozenset[type[ProviderError]] = field(default=RETRYABLE)


DEFAULT_RETRY = RetryPolicy()
# A local box that is down stays down, so waiting two minutes for it is wasted time.
LOCAL_RETRY = RetryPolicy(max_attempts=2, max_total_s=30.0)


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header. A date form is treated as absent."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def looks_like_overflow(body: str) -> bool:
    """True when a 4xx body is really a context window complaint."""
    lowered = body.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


def classify(
    response: httpx.Response, *, provider: str, model: str | None = None
) -> ProviderError | None:
    """Map one HTTP response onto the error tree, or None when it is fine."""
    status = response.status_code
    if status < 400:
        return None
    body = _body(response)
    where = f"{provider} returned {status}"
    if status == 429:
        return ProviderRateLimited(
            f"{where}",
            provider=provider,
            model=model,
            retry_after=parse_retry_after(response.headers.get("retry-after")),
        )
    if status == 413 or (status == 400 and looks_like_overflow(body)):
        return ProviderContextOverflow(f"{where}: {body[:200]}", provider=provider, model=model)
    if status == 400:
        return ProviderBadRequest(f"{where}: {body[:200]}", provider=provider, model=model)
    if status in (401, 403):
        return ProviderAuthError(f"{where}, check the api key", provider=provider, model=model)
    if status == 404:
        return ModelNotFound(f"{where}: {model} is not served here", provider=provider, model=model)
    if status in (408, 409) or status >= 500:
        return ProviderUnavailable(f"{where}", provider=provider, model=model, retryable=True)
    return ProviderBadRequest(f"{where}: {body[:200]}", provider=provider, model=model)


def _body(response: httpx.Response) -> str:
    try:
        return response.text
    except (httpx.ResponseNotRead, UnicodeDecodeError):
        return ""


def _delay(policy: RetryPolicy, attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, policy.max_delay_s)
    backoff = min(policy.base_delay_s * (2 ** (attempt - 1)), policy.max_delay_s)
    return random.uniform(0.0, backoff) if policy.jitter == "full" else backoff


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = DEFAULT_RETRY,
    deadline: float | None = None,
    span: SpanLike | None = None,
) -> T:
    """Call fn until it succeeds, the policy runs out, or the run deadline passes."""
    started = anyio.current_time()
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except ProviderError as exc:
            if type(exc) not in policy.retry_on or attempt >= policy.max_attempts:
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = _delay(policy, attempt, retry_after)
            elapsed = anyio.current_time() - started
            # A retry must never push a run past the wall clock it was given.
            if elapsed + delay > policy.max_total_s:
                raise
            if deadline is not None and anyio.current_time() + delay > deadline:
                raise
            if span is not None:
                span.event("retry", attempt=attempt, delay_s=delay, error=type(exc).__name__)
            await anyio.sleep(delay)
