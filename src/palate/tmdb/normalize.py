"""Turn a stored TMDB payload into the rows the corpus tables hold."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from palate.tmdb.models import MovieDetail

# Bumped whenever this module changes what it extracts, so `tmdb renormalize`
# can tell an old row from a new one without re-crawling.
DETAIL_VERSION = 2

ANIMATION_GENRE = 16
DOCUMENTARY_GENRE = 99

# Blockbusters credit hundreds of people and the tail is noise for every channel
# that reads credits. The raw payload keeps the rest if that judgement changes.
CAST_LIMIT = 30
CREW_DEPARTMENTS = frozenset(
    {"Directing", "Writing", "Camera", "Editing", "Sound", "Production", "Art"}
)


@dataclass(frozen=True, slots=True)
class PersonRow:
    """One person, deduplicated across the credits of a single film."""

    person_id: int
    name: str
    gender: int | None
    known_for_department: str | None
    popularity: float | None


@dataclass(frozen=True, slots=True)
class CreditRow:
    """One cast or crew credit, keyed by TMDB's own stable credit id."""

    credit_id: str
    tmdb_id: int
    person_id: int
    credit_kind: str
    department: str | None
    job: str | None
    character: str | None
    ord: int


@dataclass(frozen=True, slots=True)
class NormalizedFilm:
    """Everything one movie payload contributes, one attribute per table."""

    tmdb_id: int
    imdb_id: str | None
    title: str
    original_title: str | None
    release_date: str | None
    year: int | None
    runtime: int | None
    original_language: str | None
    overview: str | None
    tagline: str | None
    popularity: float | None
    vote_average: float | None
    vote_count: int
    budget: int | None
    revenue: int | None
    adult: bool
    status: str | None
    poster_path: str | None
    collection_id: int | None
    collection_name: str | None
    primary_region: str | None
    genres: tuple[tuple[int, str], ...]
    keywords: tuple[tuple[int, str], ...]
    countries: tuple[tuple[str, str], ...]
    languages: tuple[tuple[str, str], ...]
    companies: tuple[tuple[int, str, str | None], ...]
    people: tuple[PersonRow, ...]
    credits: tuple[CreditRow, ...]

    @property
    def director_ids(self) -> tuple[int, ...]:
        """Every director, because co-directed films are common."""
        return tuple(c.person_id for c in self.credits if c.job == "Director")


def normalize_movie(payload: Mapping[str, Any]) -> NormalizedFilm:
    """Parse one /movie/{id} payload into rows."""
    detail = MovieDetail.model_validate(payload)
    people: dict[int, PersonRow] = {}
    credits: list[CreditRow] = []

    for member in sorted(detail.credits.cast, key=lambda c: c.order)[:CAST_LIMIT]:
        people.setdefault(
            member.id,
            PersonRow(
                member.id,
                member.name,
                member.gender,
                member.known_for_department,
                member.popularity,
            ),
        )
        credits.append(
            CreditRow(
                credit_id=member.credit_id,
                tmdb_id=detail.id,
                person_id=member.id,
                credit_kind="cast",
                department="Acting",
                job=None,
                character=member.character,
                ord=member.order,
            )
        )

    for index, worker in enumerate(detail.credits.crew):
        if worker.department not in CREW_DEPARTMENTS:
            continue
        people.setdefault(
            worker.id,
            PersonRow(
                worker.id,
                worker.name,
                worker.gender,
                worker.known_for_department,
                worker.popularity,
            ),
        )
        credits.append(
            CreditRow(
                credit_id=worker.credit_id,
                tmdb_id=detail.id,
                person_id=worker.id,
                credit_kind="crew",
                department=worker.department,
                job=worker.job,
                character=None,
                ord=index,
            )
        )

    collection = detail.belongs_to_collection
    return NormalizedFilm(
        tmdb_id=detail.id,
        imdb_id=detail.imdb_id or detail.external_ids.imdb_id,
        title=detail.title or detail.original_title or str(detail.id),
        original_title=detail.original_title,
        release_date=detail.release_date or None,
        year=detail.year,
        runtime=detail.runtime,
        original_language=detail.original_language,
        overview=detail.overview or None,
        tagline=detail.tagline or None,
        popularity=detail.popularity,
        vote_average=detail.vote_average,
        vote_count=detail.vote_count,
        budget=detail.budget,
        revenue=detail.revenue,
        adult=detail.adult,
        status=detail.status,
        poster_path=detail.poster_path,
        collection_id=collection.id if collection else None,
        collection_name=collection.name if collection else None,
        primary_region=detail.primary_region,
        genres=tuple((g.id, g.name) for g in detail.genres),
        keywords=tuple((k.id, k.name) for k in detail.keywords.keywords),
        countries=tuple(
            (c.iso_3166_1, c.name or c.iso_3166_1) for c in detail.production_countries
        ),
        languages=tuple(
            (s.iso_639_1, s.english_name or s.name or s.iso_639_1) for s in detail.spoken_languages
        ),
        companies=tuple((c.id, c.name, c.origin_country) for c in detail.production_companies),
        people=tuple(people.values()),
        credits=tuple(credits),
    )


def write_film(
    conn: sqlite3.Connection,
    film: NormalizedFilm,
    *,
    fetched_at: str,
    etag: str | None = None,
) -> None:
    """Write every derived row for one film, replacing whatever was there."""
    _write_films(conn, film, fetched_at=fetched_at, etag=etag)
    _write_people(conn, film)
    _write_credits(conn, film)
    _write_genres(conn, film)
    _write_keywords(conn, film)
    _write_countries(conn, film)
    _write_languages(conn, film)
    _write_companies(conn, film)
    _write_stats(conn, film, computed_at=fetched_at)


def _write_films(
    conn: sqlite3.Connection, film: NormalizedFilm, *, fetched_at: str, etag: str | None
) -> None:
    conn.execute(
        "insert into films (tmdb_id, imdb_id, title, original_title, release_date, year, runtime, "
        "original_language, overview, tagline, popularity, popularity_at_crawl, vote_average, "
        "vote_count, budget, revenue, adult, status, poster_path, collection_id, collection_name, "
        "detail_version, etag, fetched_at) "
        "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "on conflict(tmdb_id) do update set "
        "imdb_id = excluded.imdb_id, title = excluded.title, "
        "original_title = excluded.original_title, release_date = excluded.release_date, "
        "year = excluded.year, runtime = excluded.runtime, "
        "original_language = excluded.original_language, overview = excluded.overview, "
        "tagline = excluded.tagline, popularity = excluded.popularity, "
        # Frozen at the first crawl, because today's popularity leaks forward.
        "popularity_at_crawl = coalesce(films.popularity_at_crawl, excluded.popularity), "
        "vote_average = excluded.vote_average, vote_count = excluded.vote_count, "
        "budget = excluded.budget, revenue = excluded.revenue, adult = excluded.adult, "
        "status = excluded.status, poster_path = excluded.poster_path, "
        "collection_id = excluded.collection_id, collection_name = excluded.collection_name, "
        "detail_version = excluded.detail_version, etag = excluded.etag, "
        "fetched_at = excluded.fetched_at",
        (
            film.tmdb_id,
            film.imdb_id,
            film.title,
            film.original_title,
            film.release_date,
            film.year,
            film.runtime,
            film.original_language,
            film.overview,
            film.tagline,
            film.popularity,
            film.popularity,
            film.vote_average,
            film.vote_count,
            film.budget,
            film.revenue,
            int(film.adult),
            film.status,
            film.poster_path,
            film.collection_id,
            film.collection_name,
            DETAIL_VERSION,
            etag,
            fetched_at,
        ),
    )


def _write_stats(conn: sqlite3.Connection, film: NormalizedFilm, *, computed_at: str) -> None:
    # primary_region lives only here, because film_countries is a set and loses TMDB's order.
    genre_ids = {genre_id for genre_id, _ in film.genres}
    conn.execute(
        "insert into film_stats (tmdb_id, n_directors, n_cast, n_keywords, log_vote_count, "
        "has_overview, is_animation, is_documentary, primary_region, computed_at) "
        "values (?,?,?,?,?,?,?,?,?,?) on conflict(tmdb_id) do update set "
        "n_directors = excluded.n_directors, n_cast = excluded.n_cast, "
        "n_keywords = excluded.n_keywords, log_vote_count = excluded.log_vote_count, "
        "has_overview = excluded.has_overview, is_animation = excluded.is_animation, "
        "is_documentary = excluded.is_documentary, primary_region = excluded.primary_region, "
        "computed_at = excluded.computed_at",
        (
            film.tmdb_id,
            len(film.director_ids),
            sum(1 for c in film.credits if c.credit_kind == "cast"),
            len(film.keywords),
            math.log1p(film.vote_count),
            int(bool(film.overview)),
            int(ANIMATION_GENRE in genre_ids),
            int(DOCUMENTARY_GENRE in genre_ids),
            film.primary_region,
            computed_at,
        ),
    )


def _write_people(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    conn.executemany(
        "insert into people (person_id, name, gender, known_for_department, popularity) "
        "values (?,?,?,?,?) on conflict(person_id) do update set name = excluded.name, "
        "gender = excluded.gender, known_for_department = excluded.known_for_department, "
        "popularity = excluded.popularity",
        [
            (p.person_id, p.name, p.gender, p.known_for_department, p.popularity)
            for p in film.people
        ],
    )


def _write_credits(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    # Replaced wholesale, so a renormalize cannot leave a dropped credit behind.
    conn.execute("delete from credits where tmdb_id = ?", (film.tmdb_id,))
    conn.executemany(
        "insert into credits (credit_id, tmdb_id, person_id, credit_kind, department, job, "
        "character, ord) values (?,?,?,?,?,?,?,?) on conflict(credit_id) do update set "
        "tmdb_id = excluded.tmdb_id, person_id = excluded.person_id, "
        "credit_kind = excluded.credit_kind, department = excluded.department, "
        "job = excluded.job, character = excluded.character, ord = excluded.ord",
        [
            (
                c.credit_id,
                c.tmdb_id,
                c.person_id,
                c.credit_kind,
                c.department,
                c.job,
                c.character,
                c.ord,
            )
            for c in film.credits
        ],
    )


def _write_genres(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    # Any conflict is a no-op: a reference row that is already there is already right.
    conn.executemany(
        "insert into genres (genre_id, name) values (?,?) on conflict do nothing",
        film.genres,
    )
    conn.execute("delete from film_genres where tmdb_id = ?", (film.tmdb_id,))
    conn.executemany(
        "insert into film_genres (tmdb_id, genre_id) values (?,?)",
        [(film.tmdb_id, genre_id) for genre_id, _ in film.genres],
    )


def _write_keywords(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    conn.executemany(
        "insert into keywords (keyword_id, name) values (?,?) on conflict do nothing",
        film.keywords,
    )
    conn.execute("delete from film_keywords where tmdb_id = ?", (film.tmdb_id,))
    conn.executemany(
        "insert into film_keywords (tmdb_id, keyword_id) values (?,?)",
        [(film.tmdb_id, keyword_id) for keyword_id, _ in film.keywords],
    )


def _write_countries(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    conn.executemany(
        "insert into countries (iso_3166_1, name) values (?,?) on conflict do nothing",
        film.countries,
    )
    conn.execute("delete from film_countries where tmdb_id = ?", (film.tmdb_id,))
    conn.executemany(
        "insert into film_countries (tmdb_id, iso_3166_1) values (?,?)",
        [(film.tmdb_id, code) for code, _ in film.countries],
    )


def _write_languages(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    conn.executemany(
        "insert into languages (iso_639_1, name) values (?,?) on conflict do nothing",
        film.languages,
    )
    conn.execute("delete from film_languages where tmdb_id = ?", (film.tmdb_id,))
    conn.executemany(
        "insert into film_languages (tmdb_id, iso_639_1) values (?,?)",
        [(film.tmdb_id, code) for code, _ in film.languages],
    )


def _write_companies(conn: sqlite3.Connection, film: NormalizedFilm) -> None:
    conn.executemany(
        "insert into companies (company_id, name, origin_country) values (?,?,?) "
        "on conflict do nothing",
        film.companies,
    )
    conn.execute("delete from film_companies where tmdb_id = ?", (film.tmdb_id,))
    conn.executemany(
        "insert into film_companies (tmdb_id, company_id) values (?,?)",
        [(film.tmdb_id, company_id) for company_id, _, _ in film.companies],
    )
