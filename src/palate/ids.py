"""Sortable time-ordered identifiers, so a trace listing needs no ORDER BY on a date."""

from __future__ import annotations

import os
import threading

from palate.clock import now_ns

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_lock = threading.Lock()
_last_ms = 0
_counter = 0


def _base36(value: int, width: int) -> str:
    out: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        out.append(_ALPHABET[rem])
    return "".join(reversed(out)).rjust(width, "0")


def _monotonic_ms() -> tuple[int, int]:
    """Milliseconds plus a per-millisecond counter, so two ids in the same ms still sort."""
    global _last_ms, _counter
    with _lock:
        ms = now_ns() // 1_000_000
        if ms == _last_ms:
            _counter += 1
        else:
            _last_ms, _counter = ms, 0
        return ms, _counter


def new_id(prefix: str = "") -> str:
    """Time-ordered id: base36 milliseconds, a sequence, then randomness."""
    ms, seq = _monotonic_ms()
    body = f"{_base36(ms, 9)}{_base36(seq, 3)}{os.urandom(5).hex()}"
    return f"{prefix}{body}" if prefix else body


def new_run_id() -> str:
    """Id for one agent run."""
    return new_id("run_")


def new_span_id() -> str:
    """Id for one traced span."""
    return new_id("spn_")


def new_session_id() -> str:
    """Id for one chat session."""
    return new_id("ses_")
