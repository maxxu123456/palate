"""What this SQLite build can actually do, probed rather than assumed."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from palate.errors import SqliteExtensionUnavailable
from palate.providers.fingerprint import pack_f32

# vec0 rejects the seventeenth, and the probe below finds the real number.
METADATA_CEILING = 24

_FIX = "fix it with: uv python install 3.13"


@dataclass(frozen=True, slots=True)
class VecCapability:
    """One connection's vector and full text support."""

    vec_version: str
    max_metadata_columns: int
    fts5: bool


def serialize_f32(vector: Sequence[float]) -> bytes:
    """Little endian float32, the only layout vec0 accepts."""
    return pack_f32(vector)


def probe(conn: sqlite3.Connection) -> VecCapability:
    """Ask the connection what it supports, raising if sqlite-vec is out of reach."""
    try:
        version = str(conn.execute("select vec_version()").fetchone()[0])
    except sqlite3.Error as exc:
        raise SqliteExtensionUnavailable(f"sqlite-vec is not loaded ({exc}), {_FIX}") from exc
    return VecCapability(
        vec_version=version,
        max_metadata_columns=_metadata_limit(conn),
        fts5=_has_fts5(conn),
    )


def _metadata_limit(conn: sqlite3.Connection) -> int:
    limit = 0
    for n in range(1, METADATA_CEILING + 1):
        columns = ", ".join(f"probe{i} integer" for i in range(n))
        name = f"vec_probe_{n}"
        try:
            conn.execute(
                f"create virtual table temp.{name} using vec0("
                f"id integer primary key, embedding float[2], {columns})"
            )
        except sqlite3.Error:
            break
        conn.execute(f"drop table temp.{name}")
        limit = n
    return limit


def _has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("create virtual table temp.fts_probe using fts5(a)")
    except sqlite3.Error:
        return False
    conn.execute("drop table temp.fts_probe")
    return True
