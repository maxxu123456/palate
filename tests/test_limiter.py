"""The limiter never exceeds its rate, backs off on 429 and climbs back."""

from __future__ import annotations

from typing import Any

import anyio
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from palate.tmdb.limiter import AIMDLimiter

TOLERANCE = 1e-6


class FakeTime:
    """A clock that only moves when something sleeps, so tests are instant."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def make(rate: float = 20.0, burst: int = 20, **kwargs: Any) -> tuple[AIMDLimiter, FakeTime]:
    fake = FakeTime()
    limiter = AIMDLimiter(
        rate_per_s=rate, burst=burst, clock=fake.clock, sleep=fake.sleep, **kwargs
    )
    return limiter, fake


async def stamps(limiter: AIMDLimiter, fake: FakeTime, n: int) -> list[float]:
    out: list[float] = []
    for _ in range(n):
        await limiter.acquire()
        out.append(fake.now)
    return out


def test_a_non_positive_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        AIMDLimiter(rate_per_s=0.0, burst=5)


def test_the_burst_goes_out_immediately_and_the_rest_is_paced() -> None:
    async def scenario() -> list[float]:
        limiter, fake = make(rate=10.0, burst=5)
        return await stamps(limiter, fake, 8)

    times = anyio.run(scenario)
    assert times[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert times[5] == pytest.approx(0.1)
    assert times[7] == pytest.approx(0.3)


@settings(max_examples=50, deadline=None)
@given(
    rate=st.floats(min_value=1.0, max_value=50.0),
    burst=st.integers(min_value=1, max_value=20),
    n=st.integers(min_value=1, max_value=60),
)
def test_no_window_ever_exceeds_the_configured_rate(rate: float, burst: int, n: int) -> None:
    async def scenario() -> list[float]:
        limiter, fake = make(rate=rate, burst=burst)
        return await stamps(limiter, fake, n)

    times = anyio.run(scenario)
    assert times == sorted(times)
    # A token bucket that started full can never issue more than burst + rate * window.
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            assert (j - i) <= burst + rate * (times[j] - times[i]) + TOLERANCE


def test_a_429_halves_the_rate_and_serves_retry_after() -> None:
    async def scenario() -> tuple[float, float]:
        limiter, fake = make(rate=20.0, burst=20)
        await limiter.acquire()
        limiter.penalise(retry_after=5.0)
        halved = limiter.current_rate
        await limiter.acquire()
        return halved, fake.now

    halved, resumed_at = anyio.run(scenario)
    assert halved == 10.0
    assert resumed_at == pytest.approx(5.0)


def test_the_rate_floors_instead_of_halving_to_nothing() -> None:
    limiter, _ = make(rate=20.0, burst=20, floor_per_s=2.0)
    for _ in range(10):
        limiter.penalise(retry_after=None)
    assert limiter.current_rate == 2.0


def test_successes_climb_back_to_the_configured_rate() -> None:
    limiter, _ = make(rate=20.0, burst=20, recover_per_s=0.5, recover_every=4)
    limiter.penalise(retry_after=None)
    assert limiter.current_rate == 10.0
    for _ in range(4):
        limiter.succeed()
    assert limiter.current_rate == 10.5
    for _ in range(4 * 100):
        limiter.succeed()
    assert limiter.current_rate == 20.0


def test_recovery_does_not_overshoot_after_a_half_step() -> None:
    limiter, _ = make(rate=3.0, burst=3, floor_per_s=0.5, recover_per_s=2.0, recover_every=1)
    limiter.penalise(retry_after=None)
    assert limiter.current_rate == pytest.approx(1.5)
    limiter.succeed()
    assert limiter.current_rate == pytest.approx(3.0)
    limiter.succeed()
    assert limiter.current_rate == pytest.approx(3.0)


def test_eight_workers_share_one_bucket() -> None:
    async def scenario() -> list[float]:
        limiter, fake = make(rate=10.0, burst=2)
        seen: list[float] = []

        async def worker() -> None:
            for _ in range(3):
                await limiter.acquire()
                seen.append(fake.now)

        async with anyio.create_task_group() as group:
            for _ in range(8):
                group.start_soon(worker)
        return seen

    times = anyio.run(scenario)
    assert len(times) == 24
    # Two free, then one every tenth of a second, whichever worker asked.
    assert max(times) == pytest.approx(2.2)
