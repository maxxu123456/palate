"""Token bucket that halves its rate on 429 and recovers additively."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import anyio

# Float drift leaves the bucket a few ulps short of a whole token, which would
# otherwise spin forever waiting for an amount of time too small to measure.
_ALMOST = 1e-9


# TMDB documents no limit any more, so the configured rate only needs headroom.
class AIMDLimiter:
    """Paces outbound requests, halving on a 429 and climbing back on success."""

    def __init__(
        self,
        *,
        rate_per_s: float,
        burst: int,
        floor_per_s: float = 2.0,
        recover_per_s: float = 0.5,
        recover_every: int = 200,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self._configured = float(rate_per_s)
        self._rate = float(rate_per_s)
        self._burst = float(max(1, burst))
        self._floor = min(float(floor_per_s), self._configured)
        self._recover_per_s = float(recover_per_s)
        self._recover_every = max(1, recover_every)
        self._clock = clock
        self._sleep = sleep
        self._tokens = self._burst
        self._updated = clock()
        self._held_until = 0.0
        self._successes = 0
        self._lock = anyio.Lock()

    @property
    def current_rate(self) -> float:
        """Requests per second the bucket is refilling at right now."""
        return self._rate

    @property
    def configured_rate(self) -> float:
        """The rate recovery climbs back toward."""
        return self._configured

    async def acquire(self) -> None:
        """Wait until one request may go out."""
        # The lock is held across the sleep, so waiters leave in the order they arrived.
        async with self._lock:
            while True:
                now = self._clock()
                if now < self._held_until - _ALMOST:
                    # Nothing accrues while a Retry-After is being served.
                    self._updated = now
                    self._tokens = 0.0
                    await self._sleep(self._held_until - now)
                    continue
                self._refill(now)
                if self._tokens >= 1.0 - _ALMOST:
                    self._tokens -= 1.0
                    return
                await self._sleep((1.0 - self._tokens) / self._rate)

    def penalise(self, *, retry_after: float | None) -> None:
        """Halve the rate after a 429 and hold off for Retry-After if it was sent."""
        now = self._clock()
        self._refill(now)
        self._rate = max(self._floor, self._rate / 2.0)
        self._successes = 0
        self._tokens = 0.0
        if retry_after is not None and retry_after > 0:
            self._held_until = max(self._held_until, now + retry_after)

    def succeed(self) -> None:
        """Count a good response and climb back toward the configured rate."""
        if self._rate >= self._configured:
            return
        self._successes += 1
        if self._successes >= self._recover_every:
            self._successes = 0
            self._rate = min(self._configured, self._rate + self._recover_per_s)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
