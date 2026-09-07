"""Read a Letterboxd export zip and reconcile its five CSVs into one row per film."""

from __future__ import annotations

import csv
import io
import re
import sqlite3
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from palate.clock import now_iso
from palate.db.connect import Database
from palate.hashing import sha256_file
from palate.ingest.resolve import (
    NullTitleSearch,
    Resolution,
    TitleSearch,
    load_manual,
    normalize_title,
    resolve,
)

CSV_FILES = ("ratings.csv", "diary.csv", "watched.csv", "watchlist.csv", "reviews.csv")

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ExportRow:
    """One film, after the five CSVs have been reconciled on its Letterboxd URI."""

    source_csv: str
    uri: str
    title: str
    year: int | None
    rating_half: int | None
    watched_date: date | None
    logged_date: date | None
    is_rewatch: bool
    review: str | None
    date_source: str = "none"
    rewatch_count: int = 0
    in_watchlist: bool = False
    watchlist_added_on: date | None = None
    liked: bool = False


@dataclass(frozen=True, slots=True)
class DroppedRow:
    """An export line that could not even be identified, kept so the match rate stays honest."""

    source_csv: str
    title: str | None
    year: int | None
    uri: str | None
    rating: float | None
    reason: str


@dataclass(slots=True)
class Export:
    """Everything one export zip contained."""

    path: Path
    sha256: str
    rows: list[ExportRow] = field(default_factory=list)
    dropped: list[DroppedRow] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImportReport:
    """What one `palate ingest` run wrote."""

    import_id: int
    n_rows: int
    n_resolved: int
    n_unresolved: int
    n_needs_review: int
    counts: dict[str, int]

    @property
    def match_rate(self) -> float:
        return self.n_resolved / self.n_rows if self.n_rows else 0.0


def parse_rating(text: str | None) -> int | None:
    """Half stars as an integer 1 to 10. Nothing downstream ever compares a rating float."""
    if not text:
        return None
    try:
        stars = float(text)
    except ValueError:
        return None
    half = round(stars * 2)
    if half < 1:
        return None
    return min(half, 10)


def parse_date(text: str | None) -> date | None:
    """Letterboxd writes plain YYYY-MM-DD, sometimes with a time glued on."""
    if not text:
        return None
    match = _DATE.match(text.strip())
    if match is None:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def parse_year(text: str | None) -> int | None:
    """Release year as an integer, or None when the export left it blank."""
    if not text:
        return None
    try:
        year = int(text.strip())
    except ValueError:
        return None
    return year if 1870 < year < 2200 else None


def leaks_identity(review: str | None, terms: Iterable[str]) -> bool:
    """True when a review names the film it is about, which would leak into a query set."""
    if not review:
        return False
    haystack = normalize_title(review)
    words = {w for w in _WORD.findall(haystack)}
    for term in terms:
        needle = normalize_title(term)
        if not needle:
            continue
        if " " in needle:
            if needle in haystack:
                return True
        elif needle in words:
            return True
    return False


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    try:
        blob = zf.read(name)
    except KeyError:
        return []
    text = blob.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return [{(k or "").strip(): (v or "") for k, v in row.items()} for row in reader]


def _identity(row: dict[str, str], source: str) -> tuple[str, str, int | None] | DroppedRow:
    uri = row.get("Letterboxd URI", "").strip()
    title = row.get("Name", "").strip()
    year = parse_year(row.get("Year"))
    if not uri:
        return DroppedRow(source, title or None, year, None, None, "no_letterboxd_uri")
    return uri, title, year


@dataclass(slots=True)
class _Accumulator:
    """Mutable per-URI state while the five CSVs are folded together."""

    source_csv: str
    uri: str
    title: str
    year: int | None
    ratings_rating: int | None = None
    ratings_date: date | None = None
    diary_rating: int | None = None
    diary_rating_on: date | None = None
    first_watched: date | None = None
    diary_logged: date | None = None
    diary_rows: int = 0
    marked_rewatch: bool = False
    watched_date_col: date | None = None
    in_watchlist: bool = False
    watchlist_added_on: date | None = None
    review: str | None = None

    def finish(self) -> ExportRow:
        watched = self.first_watched
        if watched is not None:
            date_source = "diary"
        elif self.ratings_date is not None or self.diary_logged is not None:
            date_source = "ratings"
        else:
            date_source = "none"
        logged = self.diary_logged or self.ratings_date or self.watched_date_col
        rating = self.diary_rating if self.diary_rating is not None else self.ratings_rating
        return ExportRow(
            source_csv=self.source_csv,
            uri=self.uri,
            title=self.title,
            year=self.year,
            rating_half=rating,
            watched_date=watched,
            logged_date=logged,
            is_rewatch=self.diary_rows > 1 or self.marked_rewatch,
            review=self.review,
            date_source=date_source,
            rewatch_count=self.diary_rows,
            in_watchlist=self.in_watchlist,
            watchlist_added_on=self.watchlist_added_on,
        )


def read_export(path: Path) -> Export:
    """Open an export zip and fold its five CSVs into one row per Letterboxd URI."""
    export = Export(path=path, sha256=sha256_file(path))
    acc: dict[str, _Accumulator] = {}

    with zipfile.ZipFile(path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        for name in CSV_FILES:
            rows = _read_csv(zf, members[name]) if name in members else []
            export.counts[name] = len(rows)
            for row in rows:
                ident = _identity(row, name)
                if isinstance(ident, DroppedRow):
                    export.dropped.append(ident)
                    continue
                uri, title, year = ident
                state = acc.get(uri)
                if state is None:
                    state = acc[uri] = _Accumulator(name, uri, title, year)
                elif title and not state.title:
                    state.title = title
                if state.year is None:
                    state.year = year
                _fold(state, name, row)

    export.rows = [state.finish() for state in acc.values()]
    return export


def _fold(state: _Accumulator, source: str, row: dict[str, str]) -> None:
    entered = parse_date(row.get("Date"))
    rating = parse_rating(row.get("Rating"))
    if source == "ratings.csv":
        # Latest entered rating wins, because Letterboxd keeps only the current one.
        if rating is not None and (
            state.ratings_date is None or _at_least(entered, state.ratings_date)
        ):
            state.ratings_rating = rating
        state.ratings_date = _earliest(state.ratings_date, entered)
    elif source == "diary.csv":
        state.diary_rows += 1
        watched = parse_date(row.get("Watched Date"))
        state.first_watched = _earliest(state.first_watched, watched)
        state.diary_logged = _earliest(state.diary_logged, entered)
        if row.get("Rewatch", "").strip().lower() in {"yes", "true", "1"}:
            state.marked_rewatch = True
        # The latest diary entry supplies the rating, it is the most recent judgement.
        if rating is not None and (
            state.diary_rating_on is None or _at_least(watched or entered, state.diary_rating_on)
        ):
            state.diary_rating = rating
            state.diary_rating_on = watched or entered
    elif source == "watched.csv":
        state.watched_date_col = _earliest(state.watched_date_col, entered)
    elif source == "watchlist.csv":
        state.in_watchlist = True
        state.watchlist_added_on = _earliest(state.watchlist_added_on, entered)
    elif source == "reviews.csv":
        text = row.get("Review", "").strip()
        if text:
            state.review = text
        watched = parse_date(row.get("Watched Date"))
        state.first_watched = _earliest(state.first_watched, watched)


def _earliest(current: date | None, candidate: date | None) -> date | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


def _at_least(candidate: date | None, current: date) -> bool:
    return candidate is not None and candidate >= current


def import_export(
    db: Database,
    path: Path,
    *,
    search: TitleSearch | None = None,
) -> ImportReport:
    """Read an export zip, resolve its titles and write the user history."""
    export = read_export(path)
    finder = search or NullTitleSearch()
    manual = load_manual(db)
    resolutions = [resolve(row, finder, manual=manual.get(row.uri)) for row in export.rows]
    return _write(db, export, resolutions)


def _write(db: Database, export: Export, resolutions: Sequence[Resolution]) -> ImportReport:
    stamp = now_iso()
    resolved = [r for r in resolutions if r.tmdb_id is not None]
    needs_review = [r for r in resolutions if r.needs_review]
    with db.write() as conn:
        cursor = conn.execute(
            "insert into letterboxd_imports (imported_at, zip_sha256, n_ratings, n_diary, "
            "n_watched, n_watchlist, n_reviews, n_resolved, n_unresolved) "
            "values (?,?,?,?,?,?,?,?,?)",
            (
                stamp,
                export.sha256,
                export.counts.get("ratings.csv", 0),
                export.counts.get("diary.csv", 0),
                export.counts.get("watched.csv", 0),
                export.counts.get("watchlist.csv", 0),
                export.counts.get("reviews.csv", 0),
                len(resolved),
                len(export.rows) - len(resolved),
            ),
        )
        import_id = int(cursor.lastrowid or 0)

        by_uri = {row.uri: row for row in export.rows}
        for resolution in resolutions:
            row = by_uri[resolution.uri]
            # The stub film goes first, both later rows point at it.
            if resolution.tmdb_id is not None:
                _upsert_stub_film(conn, resolution.tmdb_id, row, stamp)
            conn.execute(
                "insert into title_resolutions (letterboxd_uri, title, year, tmdb_id, method, "
                "confidence, candidates_json, needs_review, resolved_at) values (?,?,?,?,?,?,?,?,?) "
                "on conflict(letterboxd_uri) do update set title = excluded.title, "
                "year = excluded.year, tmdb_id = excluded.tmdb_id, method = excluded.method, "
                "confidence = excluded.confidence, candidates_json = excluded.candidates_json, "
                "needs_review = excluded.needs_review, resolved_at = excluded.resolved_at",
                (
                    resolution.uri,
                    row.title,
                    row.year,
                    resolution.tmdb_id,
                    resolution.method,
                    resolution.confidence,
                    resolution.candidates_json(),
                    int(resolution.needs_review),
                    stamp,
                ),
            )
            if resolution.tmdb_id is None:
                conn.execute(
                    "insert into unmatched_export_row (source_csv, title, year, uri, rating, "
                    "reason, seen_at) values (?,?,?,?,?,?,?)",
                    (
                        row.source_csv,
                        row.title,
                        row.year,
                        row.uri,
                        row.rating_half / 2 if row.rating_half is not None else None,
                        resolution.reason or "unresolved",
                        stamp,
                    ),
                )
                continue
            _upsert_user_film(conn, resolution.tmdb_id, row, import_id)

        for dropped in export.dropped:
            conn.execute(
                "insert into unmatched_export_row (source_csv, title, year, uri, rating, reason, "
                "seen_at) values (?,?,?,?,?,?,?)",
                (
                    dropped.source_csv,
                    dropped.title,
                    dropped.year,
                    dropped.uri,
                    dropped.rating,
                    dropped.reason,
                    stamp,
                ),
            )

    return ImportReport(
        import_id=import_id,
        n_rows=len(export.rows),
        n_resolved=len(resolved),
        n_unresolved=len(export.rows) - len(resolved),
        n_needs_review=len(needs_review),
        counts=dict(export.counts),
    )


def _upsert_stub_film(conn: sqlite3.Connection, tmdb_id: int, row: ExportRow, stamp: str) -> None:
    """Placeholder film row so the history has something to point at before the crawl."""
    conn.execute(
        "insert into films (tmdb_id, title, year, fetched_at, detail_version) "
        "values (?,?,?,?,0) on conflict(tmdb_id) do nothing",
        (tmdb_id, row.title or str(tmdb_id), row.year, stamp),
    )


def _upsert_user_film(
    conn: sqlite3.Connection, tmdb_id: int, row: ExportRow, import_id: int
) -> None:
    conn.execute(
        "insert into user_films (tmdb_id, rating_half, watched_date, logged_date, date_source, "
        "is_rewatch, rewatch_count, in_watchlist, watchlist_added_on, liked, review_text, "
        "review_chars, review_leaks_identity, letterboxd_uri, import_id) "
        "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "on conflict(tmdb_id) do update set rating_half = excluded.rating_half, "
        "watched_date = excluded.watched_date, logged_date = excluded.logged_date, "
        "date_source = excluded.date_source, is_rewatch = excluded.is_rewatch, "
        "rewatch_count = excluded.rewatch_count, in_watchlist = excluded.in_watchlist, "
        "watchlist_added_on = excluded.watchlist_added_on, review_text = excluded.review_text, "
        "review_chars = excluded.review_chars, "
        "review_leaks_identity = excluded.review_leaks_identity, "
        "letterboxd_uri = excluded.letterboxd_uri, import_id = excluded.import_id",
        (
            tmdb_id,
            row.rating_half,
            row.watched_date.isoformat() if row.watched_date else None,
            row.logged_date.isoformat() if row.logged_date else None,
            row.date_source,
            int(row.is_rewatch),
            row.rewatch_count,
            int(row.in_watchlist),
            row.watchlist_added_on.isoformat() if row.watchlist_added_on else None,
            int(row.liked),
            row.review,
            len(row.review or ""),
            int(leaks_identity(row.review, [row.title])),
            row.uri,
            import_id,
        ),
    )


def iter_rows(export: Export) -> Iterator[ExportRow]:
    """Rows in a stable order, newest watch first, so a listing reads sensibly."""
    return iter(
        sorted(export.rows, key=lambda r: (r.watched_date or date.min, r.title), reverse=True)
    )
