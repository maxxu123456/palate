"""Which crawled films are recommendable, and the vote floor that decides it."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from palate.db.connect import Database

# A flat floor would delete the pre-2000 tail of every market that is not one of these.
MAJOR_MARKETS = frozenset({"US", "GB"})

# Under this a title is a short, and shorts are a different recommendation problem.
MIN_FEATURE_RUNTIME = 40


def vote_floor(year: int | None, region: str | None) -> int:
    """Era and region aware minimum vote count."""
    base = 25
    if year is not None:
        if year < 1980:
            base = 5
        elif year < 2000:
            base = 10
    if region is not None and region not in MAJOR_MARKETS:
        base = max(3, base // 2)
    return base


@dataclass(frozen=True, slots=True)
class FilmRow:
    """The film columns eligibility actually reads."""

    tmdb_id: int
    year: int | None = None
    runtime: int | None = None
    vote_count: int = 0
    adult: bool = False
    status: str | None = None


@dataclass(frozen=True, slots=True)
class FilmStats:
    """One film_stats row. A film with no detail crawl yet has the defaults."""

    tmdb_id: int
    n_directors: int = 0
    n_cast: int = 0
    n_keywords: int = 0
    log_vote_count: float | None = None
    has_overview: bool = False
    is_animation: bool = False
    is_documentary: bool = False
    primary_region: str | None = None


def eligibility(film: FilmRow, stats: FilmStats) -> tuple[bool, str | None, int]:
    """Returns (eligible, reason, vote_floor_used). The reason is recorded, never deleted."""
    floor = vote_floor(film.year, stats.primary_region)
    if film.status != "Released":
        return False, "not_released", floor
    if film.runtime is not None and film.runtime < MIN_FEATURE_RUNTIME:
        return False, "short", floor
    if film.adult:
        return False, "adult", floor
    if film.vote_count < floor:
        return False, f"votes_below_{floor}", floor
    # A missing overview gets a different document template, it is not a deletion.
    return True, None, floor


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """What one pass over corpus_members decided."""

    n_members: int
    n_eligible: int
    reasons: dict[str, int]

    @property
    def n_ineligible(self) -> int:
        return self.n_members - self.n_eligible


@dataclass(frozen=True, slots=True)
class RegionCoverage:
    """Members and eligible members for one production country."""

    region: str | None
    n_members: int
    n_eligible: int

    @property
    def share(self) -> float:
        """Fraction of this region's crawled films that survived the floor."""
        return self.n_eligible / self.n_members if self.n_members else 0.0


_MEMBERS = (
    "select f.tmdb_id, f.year, f.runtime, f.vote_count, f.adult, f.status, "
    "s.primary_region from corpus_members m join films f on f.tmdb_id = m.tmdb_id "
    "left join film_stats s on s.tmdb_id = m.tmdb_id"
)


def _decide(row: sqlite3.Row) -> tuple[bool, str | None, int]:
    film = FilmRow(
        tmdb_id=int(row["tmdb_id"]),
        year=None if row["year"] is None else int(row["year"]),
        runtime=None if row["runtime"] is None else int(row["runtime"]),
        vote_count=int(row["vote_count"]),
        adult=bool(row["adult"]),
        status=None if row["status"] is None else str(row["status"]),
    )
    region = row["primary_region"]
    stats = FilmStats(film.tmdb_id, primary_region=None if region is None else str(region))
    return eligibility(film, stats)


def apply_eligibility(db: Database) -> EligibilityReport:
    """Score every corpus member against the floor and record the verdict."""
    rows = db.read().execute(_MEMBERS).fetchall()
    updates: list[tuple[int, str | None, int, int]] = []
    reasons: dict[str, int] = {}
    n_eligible = 0
    for row in rows:
        ok, reason, floor = _decide(row)
        n_eligible += int(ok)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
        updates.append((int(ok), reason, floor, int(row["tmdb_id"])))
    if updates:
        with db.write() as conn:
            conn.executemany(
                "update corpus_members set eligible = ?, ineligible_reason = ?, "
                "vote_floor_used = ? where tmdb_id = ?",
                updates,
            )
    return EligibilityReport(len(rows), n_eligible, reasons)


def coverage(db: Database) -> tuple[RegionCoverage, ...]:
    """Eligible share per production country. A global number hides a bad region."""
    rows = db.read().execute(
        "select s.primary_region as region, count(*) as n_members, "
        "coalesce(sum(m.eligible), 0) as n_eligible from corpus_members m "
        "left join film_stats s on s.tmdb_id = m.tmdb_id "
        "group by s.primary_region order by n_members desc, region"
    )
    return tuple(
        RegionCoverage(
            region=None if r["region"] is None else str(r["region"]),
            n_members=int(r["n_members"]),
            n_eligible=int(r["n_eligible"]),
        )
        for r in rows
    )
