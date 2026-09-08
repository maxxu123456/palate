"""Turn an export row into a TMDB id, and say how sure it is."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from palate.db.connect import Database

if TYPE_CHECKING:
    from palate.ingest.letterboxd import ExportRow

Method = Literal["uri", "exact", "fuzzy", "manual", "failed"]

REVIEW_BELOW = 0.75
_TMDB_SLUG = re.compile(r"/film/tmdb-(\d+)")
_ARTICLES = ("the ", "a ", "an ", "le ", "la ", "les ", "el ", "il ", "lo ", "der ", "die ", "das ")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_VOTE_CEILING = math.log1p(200_000)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One TMDB search hit, with only the fields the scorer reads."""

    tmdb_id: int
    title: str
    original_title: str | None = None
    year: int | None = None
    vote_count: int = 0
    release_years: tuple[int, ...] = ()

    @property
    def years(self) -> tuple[int, ...]:
        """Primary year first, then every other territory release."""
        primary = [self.year] if self.year is not None else []
        return tuple(primary + [y for y in self.release_years if y != self.year])


@dataclass(frozen=True, slots=True)
class Resolution:
    """The decision for one export row, plus the runners up."""

    uri: str
    tmdb_id: int | None
    method: Method
    confidence: float
    candidates: tuple[tuple[int, str, int | None, float], ...] = ()
    reason: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.tmdb_id is None or self.confidence < REVIEW_BELOW

    def candidates_json(self) -> str | None:
        return json.dumps(self.candidates[:5]) if self.candidates else None


class TitleSearch(Protocol):
    """Whatever can turn a title and year into TMDB candidates."""

    def search(self, title: str, year: int | None) -> Sequence[Candidate]: ...


class NullTitleSearch:
    """No backend configured, so only URIs that already carry a TMDB id resolve."""

    def search(self, title: str, year: int | None) -> Sequence[Candidate]:
        return ()


def normalize_title(title: str) -> str:
    """Case-folded, diacritic-stripped, article-stripped, punctuation-free form."""
    text = unicodedata.normalize("NFKD", title).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    for article in _ARTICLES:
        if text.startswith(article):
            return text[len(article) :]
    return text


def levenshtein(a: str, b: str) -> int:
    """Edit distance, two rows at a time."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def title_similarity(query: str, candidate: Candidate) -> float:
    """Best normalised Levenshtein ratio against the title and the original title."""
    left = normalize_title(query)
    best = 0.0
    for text in (candidate.title, candidate.original_title):
        if not text:
            continue
        right = normalize_title(text)
        longest = max(len(left), len(right))
        if longest == 0:
            continue
        best = max(best, 1.0 - levenshtein(left, right) / longest)
    return best


def same_title(query: str, candidate: Candidate) -> bool:
    """Titles match once case, diacritics, punctuation and a leading article are gone."""
    wanted = normalize_title(query)
    return wanted == normalize_title(candidate.title) or (
        bool(candidate.original_title) and wanted == normalize_title(candidate.original_title or "")
    )


def year_drift(year: int | None, candidate: Candidate) -> int | None:
    """Years to the nearest release. A restoration is a release, not a different film."""
    if year is None or not candidate.years:
        return None
    return min(abs(year - known) for known in candidate.years)


def year_proximity(year: int | None, candidate: Candidate) -> float:
    """1.0 on a year the film was released in, falling to 0 four years out."""
    drift = year_drift(year, candidate)
    if drift is None:
        return 0.5
    return max(0.0, 1.0 - drift / 4.0)


def vote_weight(candidate: Candidate) -> float:
    """Log vote count squashed into 0 to 1, only ever a tiebreak."""
    return min(1.0, math.log1p(max(candidate.vote_count, 0)) / _VOTE_CEILING)


def score(row_title: str, row_year: int | None, candidate: Candidate) -> float:
    """Ranking score. Confidence comes from which rule matched, not from this number."""
    return (
        0.6 * title_similarity(row_title, candidate)
        + 0.25 * year_proximity(row_year, candidate)
        + 0.15 * vote_weight(candidate)
    )


def tmdb_id_in_uri(uri: str) -> int | None:
    """Letterboxd slugs sometimes encode the TMDB id outright."""
    match = _TMDB_SLUG.search(uri)
    return int(match.group(1)) if match else None


def resolve(
    row: ExportRow,
    search: TitleSearch,
    *,
    manual: Resolution | None = None,
) -> Resolution:
    """Decide which TMDB film an export row means."""
    if manual is not None and manual.tmdb_id is not None:
        return Resolution(row.uri, manual.tmdb_id, "manual", 1.0)

    direct = tmdb_id_in_uri(row.uri)
    if direct is not None:
        return Resolution(row.uri, direct, "uri", 1.0)

    candidates = list(search.search(row.title, row.year))
    if not candidates:
        return Resolution(row.uri, None, "failed", 0.0, reason="no_candidates")

    ranked = sorted(candidates, key=lambda c: score(row.title, row.year, c), reverse=True)
    top = tuple(
        (c.tmdb_id, c.title, c.year, round(score(row.title, row.year, c), 4)) for c in ranked[:5]
    )
    return _classify(row, ranked, top)


def _classify(
    row: ExportRow,
    ranked: list[Candidate],
    top: tuple[tuple[int, str, int | None, float], ...],
) -> Resolution:
    exact = [c for c in ranked if same_title(row.title, c) and row.year in c.years]
    # Two real films share a title and a year often enough that picking the popular one is wrong.
    if len(exact) > 1:
        return Resolution(row.uri, None, "failed", 0.4, top, reason="ambiguous_title_year")
    if exact and row.year is not None:
        return Resolution(row.uri, exact[0].tmdb_id, "exact", 0.95, top)

    best = ranked[0]
    similarity = title_similarity(row.title, best)
    drift = year_drift(row.year, best)

    # Reissues and international releases land a year out, which is common, not exceptional.
    if same_title(row.title, best) and drift == 1:
        return Resolution(row.uri, best.tmdb_id, "fuzzy", 0.85, top)
    if similarity >= 0.85 and drift is not None and drift <= 2:
        return Resolution(
            row.uri, best.tmdb_id, "fuzzy", _band(similarity, 0.85, 1.0, 0.70, 0.80), top
        )
    if similarity >= 0.60 and drift is not None and drift <= 3:
        return Resolution(
            row.uri, best.tmdb_id, "fuzzy", _band(similarity, 0.60, 1.0, 0.55, 0.70), top
        )
    return Resolution(row.uri, None, "failed", 0.0, top, reason="no_confident_match")


def _band(value: float, low: float, high: float, floor: float, ceiling: float) -> float:
    span = (value - low) / (high - low) if high > low else 0.0
    return round(floor + max(0.0, min(1.0, span)) * (ceiling - floor), 4)


def load_manual(db: Database) -> dict[str, Resolution]:
    """Previous manual decisions, which survive a re-import."""
    rows = db.read().execute(
        "select letterboxd_uri, tmdb_id from title_resolutions where method = 'manual'"
    )
    return {
        str(r["letterboxd_uri"]): Resolution(
            str(r["letterboxd_uri"]), int(r["tmdb_id"]), "manual", 1.0
        )
        for r in rows
        if r["tmdb_id"] is not None
    }


def record_manual(db: Database, uri: str, tmdb_id: int, *, resolved_at: str) -> None:
    """Write a human decision, which every later import keeps."""
    with db.write() as conn:
        # The picked film may be one the crawl has never seen, so give it something to point at.
        conn.execute(
            "insert into films (tmdb_id, title, year, fetched_at, detail_version) "
            "select ?, title, year, ?, 0 from title_resolutions where letterboxd_uri = ? "
            "on conflict(tmdb_id) do nothing",
            (tmdb_id, resolved_at, uri),
        )
        conn.execute(
            "update title_resolutions set tmdb_id = ?, method = 'manual', confidence = 1.0, "
            "needs_review = 0, resolved_at = ? where letterboxd_uri = ?",
            (tmdb_id, resolved_at, uri),
        )
