"""The two translations a naive vocabulary lookup gets badly wrong."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from palate.db.connect import Database, open_database
from palate.paths import migrations_dir
from palate.retrieval.vocab import Vocabulary, normalise

STAMP = "2026-09-10T09:00:00.000000+00:00"

GENRES = {10402: "Music", 18: "Drama", 35: "Comedy"}
KEYWORDS = {
    1: "musical",
    2: "stage musical",
    3: "broadway",
    4: "song and dance",
    5: "memory",
    6: "orchestra",
}
LANGUAGES = {"ru": "Russian", "hy": "Armenian", "en": "English", "de": "German"}
COUNTRIES = {"SU": "Soviet Union", "US": "United States", "DE": "Germany"}

FILMS = (
    (1, "Cabaret", "en", "US", (10402,), (1, 3)),
    (2, "Amadeus", "en", "US", (10402, 18), (6,)),
    (3, "Singin in the Rain", "en", "US", (35,), (2, 4)),
    (4, "Stalker", "ru", "SU", (18,), (5,)),
    (5, "The Color of Pomegranates", "hy", "SU", (18,), (5,)),
    (6, "Wings of Desire", "de", "DE", (18,), (5,)),
)


def _write(conn: sqlite3.Connection) -> None:
    conn.executemany("insert into genres (genre_id, name) values (?,?)", GENRES.items())
    conn.executemany("insert into keywords (keyword_id, name) values (?,?)", KEYWORDS.items())
    conn.executemany("insert into languages (iso_639_1, name) values (?,?)", LANGUAGES.items())
    conn.executemany("insert into countries (iso_3166_1, name) values (?,?)", COUNTRIES.items())
    for tmdb_id, title, language, country, genres, keywords in FILMS:
        conn.execute(
            "insert into films (tmdb_id, title, year, runtime, original_language, "
            "vote_count, adult, fetched_at) values (?,?,1979,120,?,500,0,?)",
            (tmdb_id, title, language, STAMP),
        )
        conn.execute(
            "insert into corpus_members (tmdb_id, source, added_at) values (?,'discover',?)",
            (tmdb_id, STAMP),
        )
        conn.execute(
            "insert into film_countries (tmdb_id, iso_3166_1) values (?,?)", (tmdb_id, country)
        )
        conn.executemany(
            "insert into film_genres (tmdb_id, genre_id) values (?,?)",
            [(tmdb_id, g) for g in genres],
        )
        conn.executemany(
            "insert into film_keywords (tmdb_id, keyword_id) values (?,?)",
            [(tmdb_id, k) for k in keywords],
        )


@pytest.fixture
def vocab(tmp_path: Path) -> Iterator[Vocabulary]:
    db: Database = open_database(tmp_path / "palate.db", migrations=migrations_dir())
    with db.write() as conn:
        _write(conn)
    yield Vocabulary(db.read())
    db.close()


def test_a_musical_resolves_to_a_keyword_set_not_the_music_genre(vocab: Vocabulary) -> None:
    matches = vocab.resolve("musical")
    assert matches[0].kind == "keyword"
    assert set(matches[0].ids) == {"1", "2", "3", "4"}
    assert matches[0].affected_films == 2
    genres = [m for m in matches if m.kind == "genre"]
    assert genres and set(genres[0].ids) == {"10402"}
    assert genres[0].affected_films == 2


def test_b_the_music_genre_alone_would_have_caught_amadeus(vocab: Vocabulary) -> None:
    by_genre = vocab.affected("genre", ("10402",))
    by_keyword = vocab.affected("keyword", ("1", "2", "3", "4"))
    assert by_genre == by_keyword == 2
    both = set(vocab.resolve("musical")[0].ids)
    assert "6" not in both


def test_c_russian_resolves_language_first(vocab: Vocabulary) -> None:
    matches = vocab.resolve("russian")
    assert matches[0].kind == "language"
    assert matches[0].ids == ("ru",)
    assert matches[0].affected_films == 1
    countries = [m for m in matches if m.kind == "country"]
    assert countries and countries[0].ids == ("SU",)
    assert countries[0].affected_films == 2


def test_d_the_country_arm_keeps_the_films_the_language_arm_misses(vocab: Vocabulary) -> None:
    country = next(m for m in vocab.resolve("russian") if m.kind == "country")
    assert country.affected_films > vocab.resolve("russian")[0].affected_films


def test_e_a_negation_and_a_plural_resolve_the_same_way(vocab: Vocabulary) -> None:
    assert normalise("not Musicals!") == "musicals"
    assert vocab.resolve("no musicals") == vocab.resolve("musical")


def test_f_prefer_reorders_without_changing_the_sets(vocab: Vocabulary) -> None:
    plain = vocab.resolve("russian")
    preferred = vocab.resolve("russian", prefer=("country",))
    assert preferred[0].kind == "country"
    assert {m.kind: m.ids for m in plain} == {m.kind: m.ids for m in preferred}


def test_g_counts_and_suggestions_come_from_the_corpus(vocab: Vocabulary) -> None:
    assert vocab.counts("genre")["18"] == 4
    assert vocab.counts("language")["en"] == 3
    assert vocab.suggest("keyword", "musicl", n=1) == [("1", "musical", 1)]
    assert vocab.suggest("genre", "zzzz") == []


def test_h_an_unknown_word_resolves_to_nothing(vocab: Vocabulary) -> None:
    assert vocab.resolve("kjhgfd") == []
    assert vocab.resolve("   ") == []


def test_i_a_partial_name_still_finds_the_biggest_match(vocab: Vocabulary) -> None:
    matches = vocab.resolve("desire")
    assert matches == []
    keyword = vocab.resolve("song")
    assert keyword and keyword[0].kind == "keyword"
    assert keyword[0].ids == ("4",)
