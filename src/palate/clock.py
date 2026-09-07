"""Time behind one indirection so tests can freeze it."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

_source: Callable[[], float] | None = None


def now() -> datetime:
    """Current UTC time."""
    if _source is not None:
        return datetime.fromtimestamp(_source(), tz=UTC)
    return datetime.now(UTC)


def now_ns() -> int:
    """Monotonic-ish wall clock in nanoseconds, for span timing."""
    if _source is not None:
        return int(_source() * 1_000_000_000)
    return time.time_ns()


def now_iso() -> str:
    """UTC timestamp as stored in every text date column."""
    return now().isoformat(timespec="microseconds")


@contextmanager
def frozen(at: datetime) -> Iterator[None]:
    """Pin the clock for the duration of the block."""
    global _source
    previous = _source
    stamp = at.timestamp()
    _source = lambda: stamp  # noqa: E731
    try:
        yield
    finally:
        _source = previous
