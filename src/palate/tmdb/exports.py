"""The daily id dump at files.tmdb.org, used as a backstop enumerator."""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import orjson

from palate.errors import TMDBError

EXPORT_BASE = "https://files.tmdb.org/p/exports"

# The dump is published around 08:00 UTC, so before that the newest one is yesterday's.
PUBLISHED_HOUR_UTC = 8


@dataclass(frozen=True, slots=True)
class ExportEntry:
    """One line of the dump: an id and enough to sort it."""

    tmdb_id: int
    original_title: str | None
    popularity: float
    adult: bool
    video: bool


def latest_day(at: datetime) -> str:
    """Date of the newest published dump, as TMDB spells it in the filename."""
    moment = at.astimezone(UTC)
    day = (
        moment.date() if moment.hour >= PUBLISHED_HOUR_UTC else (moment - timedelta(days=1)).date()
    )
    return day.isoformat()


def export_url(day: str) -> str:
    """URL of the movie id dump for an ISO date."""
    year, month, date = day.split("-")
    return f"{EXPORT_BASE}/movie_ids_{month}_{date}_{year}.json.gz"


def iter_entries(blob: bytes) -> Iterator[ExportEntry]:
    """Parse the gzipped newline-delimited dump."""
    try:
        text = gzip.decompress(blob)
    except (OSError, EOFError) as exc:
        raise TMDBError(f"the id export is not gzip ({exc})") from exc
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise TMDBError(f"the id export has a line that is not json ({exc})") from exc
        yield ExportEntry(
            tmdb_id=int(row["id"]),
            original_title=row.get("original_title"),
            popularity=float(row.get("popularity") or 0.0),
            adult=bool(row.get("adult", False)),
            video=bool(row.get("video", False)),
        )


def read_export(path: Path) -> Iterator[ExportEntry]:
    """Parse a dump already on disk."""
    return iter_entries(path.read_bytes())


async def fetch_export(client: httpx.AsyncClient, day: str) -> bytes:
    """Download one dump. It needs no key, which is why it works as a backstop."""
    response = await client.get(export_url(day))
    if response.status_code != 200:
        raise TMDBError(f"the id export for {day} returned {response.status_code}")
    return response.content
