"""Structured evidence and the film shape a caller renders. Never prose."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import orjson

from palate.retrieval.features import FilmFacets, load_facets
from palate.taste.affinity import EntityAffinity
from palate.taste.profile import TasteProfile

type EvidenceKind = Literal[
    "rated_film", "shared_person", "shared_keyword", "metadata", "mode", "overview_span"
]

# Ordered by how much a reader trusts them, which is also the order they are emitted in.
PERSON_KINDS = ("director", "writer", "actor")
METADATA_KINDS = ("decade", "language", "runtime_bucket")

_WORD = re.compile(r"[^\W_]{3,}", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One checkable fact behind a recommendation."""

    kind: EvidenceKind
    text: str
    source_table: str
    source_id: str
    value: float | str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class RecommendedFilm:
    """One answer row, with the evidence a groundedness check can run against."""

    tmdb_id: int
    title: str
    year: int | None
    directors: tuple[str, ...]
    countries: tuple[str, ...]
    original_language: str | None
    runtime: int | None
    score: float
    confidence: float
    mode_id: int | None
    mode_label: str | None
    in_watchlist: bool
    feature_contributions: Mapping[str, float]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class FilmCard:
    """The display columns one recommendation needs."""

    tmdb_id: int
    title: str
    year: int | None
    directors: tuple[str, ...]
    countries: tuple[str, ...]
    original_language: str | None
    runtime: int | None
    overview: str
    in_watchlist: bool


_CARDS = (
    "select f.tmdb_id, f.title, f.year, f.runtime, f.original_language, "
    "coalesce(f.overview, '') as overview, coalesce(u.in_watchlist, 0) as in_watchlist "
    "from films f left join user_films u on u.tmdb_id = f.tmdb_id "
    "where f.tmdb_id in (select value from json_each(?))"
)

_DIRECTORS = (
    "select c.tmdb_id, p.name from credits c join people p on p.person_id = c.person_id "
    "where c.job = 'Director' and c.tmdb_id in (select value from json_each(?)) "
    "order by c.tmdb_id, c.ord, p.name"
)

_COUNTRIES = (
    "select fc.tmdb_id, co.name from film_countries fc "
    "join countries co on co.iso_3166_1 = fc.iso_3166_1 "
    "where fc.tmdb_id in (select value from json_each(?)) order by fc.tmdb_id, co.name"
)

_RATED = (
    "select u.tmdb_id, f.title, u.rating_half from user_films u "
    "join films f on f.tmdb_id = u.tmdb_id "
    "where u.rating_half is not null and u.tmdb_id in (select value from json_each(?))"
)


def _grouped(conn: sqlite3.Connection, sql: str, ids: Sequence[int]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for row in conn.execute(sql, (orjson.dumps(list(ids)).decode(),)):
        out.setdefault(int(row["tmdb_id"]), []).append(str(row["name"]))
    return out


def load_cards(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, FilmCard]:
    """Titles, credits and watchlist membership for a set of films."""
    directors = _grouped(conn, _DIRECTORS, ids)
    countries = _grouped(conn, _COUNTRIES, ids)
    out: dict[int, FilmCard] = {}
    for row in conn.execute(_CARDS, (orjson.dumps(list(ids)).decode(),)):
        tmdb_id = int(row["tmdb_id"])
        out[tmdb_id] = FilmCard(
            tmdb_id=tmdb_id,
            title=str(row["title"]),
            year=None if row["year"] is None else int(row["year"]),
            directors=tuple(directors.get(tmdb_id, ())),
            countries=tuple(countries.get(tmdb_id, ())),
            original_language=None
            if row["original_language"] is None
            else str(row["original_language"]),
            runtime=None if row["runtime"] is None else int(row["runtime"]),
            overview=str(row["overview"]),
            in_watchlist=bool(row["in_watchlist"]),
        )
    return out


def _tables(profile: TasteProfile) -> dict[str, dict[str, EntityAffinity]]:
    kinds = (*PERSON_KINDS, "keyword", *METADATA_KINDS)
    return {kind: {a.entity_id: a for a in profile.affinities.get(kind, ())} for kind in kinds}


def _support_titles(
    conn: sqlite3.Connection, tables: Mapping[str, Mapping[str, EntityAffinity]]
) -> dict[int, tuple[str, float]]:
    wanted = {
        film
        for kind in PERSON_KINDS
        for affinity in tables[kind].values()
        for film in affinity.support_films
    }
    if not wanted:
        return {}
    rows = conn.execute(_RATED, (orjson.dumps(sorted(wanted)).decode(),))
    return {int(r["tmdb_id"]): (str(r["title"]), int(r["rating_half"]) / 2.0) for r in rows}


def _best_person(
    film: FilmFacets, tables: Mapping[str, Mapping[str, EntityAffinity]]
) -> tuple[str, EntityAffinity] | None:
    best: tuple[str, EntityAffinity] | None = None
    credits = (("director", film.directors), ("writer", film.writers), ("actor", film.actors))
    for kind, people in credits:
        table = tables[kind]
        for person in people:
            found = table.get(str(person))
            if found is not None and (best is None or found.affinity > best[1].affinity):
                best = (kind, found)
    return best


def overview_spans(overview: str, terms: Sequence[str], *, limit: int = 2) -> list[Evidence]:
    """Where the user's own words land in the film's overview, as char offsets."""
    if not overview or not terms:
        return []
    wanted = {t.casefold() for t in terms if len(t) >= 3}
    out: list[Evidence] = []
    seen: set[str] = set()
    for match in _WORD.finditer(overview):
        word = match.group(0).casefold()
        if word not in wanted or word in seen:
            continue
        seen.add(word)
        out.append(
            Evidence(
                kind="overview_span",
                text=overview[match.start() : match.end()],
                source_table="films",
                source_id="overview",
                value=word,
                span=(match.start(), match.end()),
            )
        )
        if len(out) >= limit:
            break
    return out


def build_evidence(
    conn: sqlite3.Connection,
    profile: TasteProfile,
    ids: Sequence[int],
    *,
    mode_of: Mapping[int, int] | None = None,
    query_terms: Sequence[str] = (),
    cards: Mapping[int, FilmCard] | None = None,
    limit: int = 5,
) -> dict[int, tuple[Evidence, ...]]:
    """Why each film is here, as rows another component can check against the database."""
    facets = load_facets(conn, ids)
    tables = _tables(profile)
    rated = _support_titles(conn, tables)
    known = cards if cards is not None else load_cards(conn, ids)
    modes = {m.mode_id: m for m in profile.modes}
    out: dict[int, tuple[Evidence, ...]] = {}
    for tmdb_id in ids:
        film = facets.get(tmdb_id)
        if film is None:
            continue
        rows: list[Evidence] = []
        mode = modes.get((mode_of or {}).get(tmdb_id, -1))
        if mode is not None:
            rows.append(
                Evidence(
                    kind="mode",
                    text=f"sits with {mode.n_members} films you rated in one corner of your taste",
                    source_table="taste_modes",
                    source_id=f"{profile.profile_id}:{mode.mode_id}",
                    value=mode.coherence,
                )
            )
        rows.extend(_person_rows(film, tables, rated))
        rows.extend(_keyword_rows(film, tables))
        rows.extend(_metadata_rows(film, tables))
        card = known.get(tmdb_id)
        if card is not None:
            rows.extend(overview_spans(card.overview, query_terms))
        out[tmdb_id] = tuple(rows[:limit])
    return out


def _person_rows(
    film: FilmFacets,
    tables: Mapping[str, Mapping[str, EntityAffinity]],
    rated: Mapping[int, tuple[str, float]],
) -> list[Evidence]:
    found = _best_person(film, tables)
    if found is None:
        return []
    kind, affinity = found
    rows = [
        Evidence(
            kind="shared_person",
            text=f"{affinity.name} is a {kind} you rate {affinity.affinity:+.2f} over "
            f"{affinity.n} films",
            source_table="credits",
            source_id=affinity.entity_id,
            value=affinity.affinity,
        )
    ]
    for support in affinity.support_films:
        seen = rated.get(support)
        if seen is not None:
            rows.append(
                Evidence(
                    kind="rated_film",
                    text=f"you rated {seen[0]} {seen[1]:g}",
                    source_table="user_films",
                    source_id=str(support),
                    value=seen[1],
                )
            )
            break
    return rows


def _keyword_rows(
    film: FilmFacets, tables: Mapping[str, Mapping[str, EntityAffinity]]
) -> list[Evidence]:
    table = tables["keyword"]
    found = [table[str(k)] for k in film.keywords if str(k) in table]
    if not found:
        return []
    best = max(found, key=lambda a: a.affinity)
    return [
        Evidence(
            kind="shared_keyword",
            text=f"tagged {best.name}, which you rate {best.affinity:+.2f} over {best.n} films",
            source_table="film_keywords",
            source_id=best.entity_id,
            value=best.affinity,
        )
    ]


def _metadata_rows(
    film: FilmFacets, tables: Mapping[str, Mapping[str, EntityAffinity]]
) -> list[Evidence]:
    handles = {
        "decade": str(film.decade),
        "language": film.language,
        "runtime_bucket": str(film.runtime_bucket),
    }
    found = [
        tables[kind][handles[kind]] for kind in METADATA_KINDS if handles[kind] in tables[kind]
    ]
    if not found:
        return []
    best = max(found, key=lambda a: abs(a.affinity))
    return [
        Evidence(
            kind="metadata",
            text=f"{best.name}, which you rate {best.affinity:+.2f} over {best.n} films",
            source_table="taste_affinities",
            source_id=f"{best.kind}:{best.entity_id}",
            value=best.affinity,
        )
    ]
