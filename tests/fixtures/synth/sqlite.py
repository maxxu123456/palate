"""Loads a synthetic world into a migrated database, vectors and all."""

from __future__ import annotations

import sqlite3

from fixtures.synth.world import DIM, SynthFilm, SynthWorld
from palate.db.connect import Database
from palate.index.vecstore import VecRow, VecStore
from palate.index.verify import point_at, table_name
from palate.providers.fingerprint import EmbeddingFingerprint, pack_f32
from palate.taste.signals import RatedFilm

STAMP = "2026-09-09T09:00:00.000000+00:00"

GENRE_NAMES = {
    12: "Adventure",
    14: "Fantasy",
    16: "Animation",
    18: "Drama",
    27: "Horror",
    28: "Action",
    35: "Comedy",
    36: "History",
    53: "Thriller",
    80: "Crime",
    99: "Documentary",
    878: "Science Fiction",
    10402: "Music",
    10749: "Romance",
    10752: "War",
}
LANGUAGE_NAMES = {
    "ru": "Russian",
    "cn": "Cantonese",
    "en": "English",
    "fr": "French",
    "sv": "Swedish",
}
COUNTRY_NAMES = {
    "SU": "Soviet Union",
    "HK": "Hong Kong",
    "US": "United States",
    "FR": "France",
    "SE": "Sweden",
}


def fingerprint(*, dim: int = DIM, revision: str = "1") -> EmbeddingFingerprint:
    """A real fingerprint, so the index identity and staleness paths are exercised."""
    return EmbeddingFingerprint(
        provider="synth",
        model_id=f"synth-{dim}",
        revision=revision,
        dim=dim,
        normalized=True,
        pooling="none",
    )


def install(db: Database, world: SynthWorld, *, revision: str = "1", activate: bool = True) -> str:
    """Write the films, the history and the vectors, then point the index at them."""
    _films(db, world)
    _history(db, world.rated)
    return install_index(db, world, revision=revision, activate=activate)


def install_index(
    db: Database, world: SynthWorld, *, revision: str = "1", activate: bool = True
) -> str:
    """Create one vec0 table for this fingerprint and fill it with the planted vectors."""
    dim = len(next(iter(world.vectors.values())))
    mark = fingerprint(dim=dim, revision=revision)
    table = table_name(mark.key)
    watched = _watched(db)
    with db.write() as conn:
        row = mark.to_row()
        conn.execute(
            "insert into embedding_indexes (index_id, provider, model_id, revision, dim, "
            "normalized, query_prompt, document_prompt, pooling, doc_template_version, "
            "table_name, canary_text, canary_vec, n_vectors, status, created_at, completed_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ready',?,?) on conflict(index_id) do nothing",
            (
                mark.key,
                row["provider"],
                row["model_id"],
                row["revision"],
                row["dim"],
                int(row["normalized"]),
                row["query_prompt"],
                row["document_prompt"],
                row["pooling"],
                row["doc_template_version"],
                table,
                "canary",
                pack_f32([0.0] * dim),
                len(world.vectors),
                STAMP,
                STAMP,
            ),
        )
        store = VecStore(conn, table=table, dim=dim)
        store.create()
        store.upsert(
            [
                VecRow(
                    film_id=film.tmdb_id,
                    embedding=world.vectors[film.tmdb_id].tolist(),
                    decade=film.decade,
                    year=film.year,
                    runtime=film.runtime,
                    vote_count=film.vote_count,
                    original_language=film.language,
                    is_watched=int(film.tmdb_id in watched),
                )
                for film in world.films
            ]
        )
        if activate:
            point_at(conn, mark.key)
    return mark.key


def _watched(db: Database) -> set[int]:
    """Whatever history is already loaded, so the vec0 column is not a lie."""
    rows = db.read().execute(
        "select tmdb_id from user_films where rating_half is not null "
        "or watched_date is not null or logged_date is not null"
    )
    return {int(r["tmdb_id"]) for r in rows}


def _films(db: Database, world: SynthWorld) -> None:
    with db.write() as conn:
        conn.executemany(
            "insert or ignore into genres (genre_id, name) values (?,?)", GENRE_NAMES.items()
        )
        conn.executemany(
            "insert or ignore into languages (iso_639_1, name) values (?,?)",
            LANGUAGE_NAMES.items(),
        )
        conn.executemany(
            "insert or ignore into countries (iso_3166_1, name) values (?,?)",
            COUNTRY_NAMES.items(),
        )
        for film in world.films:
            _film(conn, film)
        conn.executemany(
            "insert or ignore into corpus_members (tmdb_id, source, added_at) values (?,'discover',?)",
            [(f.tmdb_id, STAMP) for f in world.films],
        )


def _film(conn: sqlite3.Connection, film: SynthFilm) -> None:
    conn.execute(
        "insert into films (tmdb_id, title, original_title, release_date, year, runtime, "
        "original_language, overview, popularity, popularity_at_crawl, vote_average, vote_count, "
        "adult, status, fetched_at) values (?,?,?,?,?,?,?,?,?,?,?,?,0,'Released',?)",
        (
            film.tmdb_id,
            film.title,
            film.title,
            f"{film.year}-06-01",
            film.year,
            film.runtime,
            film.language,
            film.overview,
            film.popularity,
            film.popularity,
            film.vote_average,
            film.vote_count,
            STAMP,
        ),
    )
    people: list[tuple[int, str, str]] = [(film.director_id, film.director_name, "Directing")]
    credits: list[tuple[str, int, str, str, str | None, int]] = [
        (f"c{film.tmdb_id}dir", film.director_id, "crew", "Directing", "Director", 0)
    ]
    writer_id = film.director_id + 100
    people.append((writer_id, f"Writer {writer_id}", "Writing"))
    credits.append((f"c{film.tmdb_id}wri", writer_id, "crew", "Writing", "Screenplay", 0))
    for slot in range(3):
        actor_id = 800 + (film.tmdb_id + slot * 7) % 40
        people.append((actor_id, f"Actor {actor_id}", "Acting"))
        credits.append((f"c{film.tmdb_id}a{slot}", actor_id, "cast", "Acting", None, slot))
    conn.executemany(
        "insert or ignore into people (person_id, name, known_for_department) values (?,?,?)",
        people,
    )
    conn.executemany(
        "insert or ignore into credits (credit_id, tmdb_id, person_id, credit_kind, department, "
        "job, ord) values (?,?,?,?,?,?,?)",
        [
            (cid, film.tmdb_id, pid, kind, dept, job, ord_)
            for cid, pid, kind, dept, job, ord_ in credits
        ],
    )
    conn.executemany(
        "insert or ignore into film_genres (tmdb_id, genre_id) values (?,?)",
        [(film.tmdb_id, g) for g in film.genres],
    )
    conn.executemany(
        "insert or ignore into keywords (keyword_id, name) values (?,?)",
        [(k, f"keyword {k}") for k in film.keywords],
    )
    conn.executemany(
        "insert or ignore into film_keywords (tmdb_id, keyword_id) values (?,?)",
        [(film.tmdb_id, k) for k in film.keywords],
    )
    conn.executemany(
        "insert or ignore into film_countries (tmdb_id, iso_3166_1) values (?,?)",
        [(film.tmdb_id, film.country)],
    )
    conn.executemany(
        "insert or ignore into film_languages (tmdb_id, iso_639_1) values (?,?)",
        [(film.tmdb_id, film.language)],
    )


def _history(db: Database, rated: tuple[RatedFilm, ...]) -> None:
    with db.write() as conn:
        conn.executemany(
            "insert into user_films (tmdb_id, rating_half, watched_date, logged_date, "
            "date_source, date_reliable, is_rewatch) values (?,?,?,?,?,?,?)",
            [
                (
                    film.tmdb_id,
                    film.rating_half,
                    film.watched_at.isoformat() if film.watched_at else None,
                    film.watched_at.isoformat() if film.watched_at else None,
                    film.date_source,
                    int(film.date_reliable),
                    int(film.is_rewatch),
                )
                for film in rated
            ],
        )
