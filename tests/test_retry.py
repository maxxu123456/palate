"""A retry must never widen a failure into a hang, and a 400 must never be retried."""

from __future__ import annotations

import httpx
import pytest

from palate.errors import (
    ModelNotFound,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderContextOverflow,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from palate.providers.retry import (
    DEFAULT_RETRY,
    RetryPolicy,
    classify,
    looks_like_overflow,
    parse_retry_after,
    with_retry,
)

FAST = RetryPolicy(max_attempts=4, base_delay_s=0.001, max_delay_s=0.01, jitter="none")


def response(
    status: int, *, text: str = "", headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status, text=text, headers=headers or {})


class Recorder:
    """Collects the retry events a span would have been given."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, name: str, **fields: object) -> None:
        self.events.append((name, fields))

    def set(self, **fields: object) -> None:
        self.events.append(("set", fields))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, type(None)),
        (400, ProviderBadRequest),
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (404, ModelNotFound),
        (408, ProviderUnavailable),
        (409, ProviderUnavailable),
        (413, ProviderContextOverflow),
        (429, ProviderRateLimited),
        (500, ProviderUnavailable),
        (503, ProviderUnavailable),
    ],
)
def test_the_status_table(status: int, expected: type) -> None:
    error = classify(response(status), provider="openai_compat", model="m")
    assert isinstance(error, expected) if error is not None else expected is type(None)


def test_a_400_that_is_really_a_context_complaint_is_not_a_bad_request() -> None:
    body = '{"error": {"message": "This model\'s maximum context length is 8192 tokens"}}'
    error = classify(response(400, text=body), provider="ollama", model="qwen3:8b")
    assert isinstance(error, ProviderContextOverflow)
    assert looks_like_overflow(body)


def test_retry_after_is_read_and_a_date_form_is_ignored() -> None:
    error = classify(response(429, headers={"Retry-After": "7"}), provider="openrouter", model="m")
    assert isinstance(error, ProviderRateLimited)
    assert error.retry_after == 7.0
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


async def test_a_rate_limit_is_retried_until_it_succeeds() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderRateLimited("429", provider="p", retry_after=0.001)
        return "ok"

    span = Recorder()
    assert await with_retry(flaky, policy=FAST, span=span) == "ok"
    assert attempts == 3
    assert [name for name, _ in span.events] == ["retry", "retry"]
    assert span.events[0][1]["attempt"] == 1


async def test_a_bad_request_is_never_retried() -> None:
    attempts = 0

    async def broken() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderBadRequest("400", provider="p")

    with pytest.raises(ProviderBadRequest):
        await with_retry(broken, policy=FAST)
    assert attempts == 1


async def test_the_attempt_budget_is_the_last_word() -> None:
    attempts = 0

    async def always_down() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderUnavailable("503", provider="p", retryable=True)

    with pytest.raises(ProviderUnavailable):
        await with_retry(always_down, policy=FAST)
    assert attempts == FAST.max_attempts


async def test_a_retry_never_pushes_a_run_past_its_total() -> None:
    policy = RetryPolicy(max_attempts=10, base_delay_s=5.0, max_total_s=1.0, jitter="none")
    attempts = 0

    async def slow_to_recover() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderTimeout("timeout", provider="p", retryable=True)

    with pytest.raises(ProviderTimeout):
        await with_retry(slow_to_recover, policy=policy)
    # The first delay alone would blow the budget, so there is no second attempt.
    assert attempts == 1


async def test_a_context_overflow_is_not_retried_because_the_prompt_will_not_shrink() -> None:
    attempts = 0

    async def too_big() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderContextOverflow("413", provider="p")

    with pytest.raises(ProviderContextOverflow):
        await with_retry(too_big, policy=FAST)
    assert attempts == 1


def test_the_default_policy_retries_only_the_three_transient_kinds() -> None:
    assert ProviderRateLimited in DEFAULT_RETRY.retry_on
    assert ProviderUnavailable in DEFAULT_RETRY.retry_on
    assert ProviderTimeout in DEFAULT_RETRY.retry_on
    assert ProviderBadRequest not in DEFAULT_RETRY.retry_on
    assert ProviderAuthError not in DEFAULT_RETRY.retry_on
