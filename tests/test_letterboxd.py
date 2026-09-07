"""The export reader, the reconciler and the title resolver, on a fixture export."""

from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from palate.db.connect import Database, open_database
from palate.ingest.letterboxd import (
    ExportRow,
    import_export,
    leaks_identity,
    parse_rating,
    read_export,
)
from palate.ingest.resolve import (
    Candidate,
    NullTitleSearch,
    Resolution,
    normalize_title,
    record_manual,
    resolve,
    title_similarity,
    tmdb_id_in_uri,
)
from palate.paths import migrations_dir

FIXTURE = Path(__file__).parent / "fixtures" / "letterboxd" / "export.zip"

# What a TMDB search would have returned for the fixture, including the awkward cases.
CATALOGUE: dict[str, list[Candidate]] = {
    "Stalker": [
        Candidate(1398, "Stalker", "Сталкер", 1979, 2100),
        Candidate(9999, "Stalker", None, 2010, 40),
    ],
    "The Turin Horse": [Candidate(52847, "The Turin Horse", "A torinoi lo", 2011, 900)],
    "Sátántangó": [Candidate(30640, "Satantango", "Sátántangó", 1994, 700)],
    # A restoration whose TMDB year sits one off the Letterboxd year.
    "Mirror": [Candidate(11916, "Mirror", "Зеркало", 1974, 800)],
    # Two real films, same title, same year. Popularity must not decide this.
    "Hamlet": [
        Candidate(10275, "Hamlet", None, 1964, 120),
        Candidate(41874, "Hamlet", "Gamlet", 1964, 95),
    ],
    "Dr. Strangelove, or: How I Learned to Stop Worrying": [
        Candidate(935, "Dr. Strangelove, or: How I Learned to Stop Worrying", None, 1964, 5200)
    ],
    "Hard to Be a God": [Candidate(238636, "Hard to Be a God", "Trudno byt bogom", 2013, 300)],
    "Werckmeister Harmonies": [Candidate(24238, "Werckmeister Harmonies", None, 2000, 400)],
}


class StaticTitleSearch:
    """Stands in for TMDB search, so the ingest path is exercised with no network."""

    def __init__(self, catalogue: dict[str, list[Candidate]]) -> None:
        self.catalogue = catalogue
        self.calls: list[tuple[str, int | None]] = []

    def search(self, title: str, year: int | None) -> Sequence[Candidate]:
        self.calls.append((title, year))
        return self.catalogue.get(title, [])


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)


def row_for(rows: list[ExportRow], title: str) -> ExportRow:
    return next(r for r in rows if r.title == title)


def test_half_stars_always_land_on_one_to_ten() -> None:
    assert parse_rating("0.5") == 1
    assert parse_rating("2.5") == 5
    assert parse_rating("5.0") == 10
    assert parse_rating("") is None
    assert parse_rating(None) is None
    assert parse_rating("not a number") is None


def test_reads_the_five_csvs() -> None:
    export = read_export(FIXTURE)
    assert export.counts == {
        "ratings.csv": 8,
        "diary.csv": 4,
        "watched.csv": 10,
        "watchlist.csv": 2,
        "reviews.csv": 2,
    }
    assert len(export.sha256) == 64
    # One row per Letterboxd URI, not one per CSV line.
    assert len(export.rows) == 10
    assert len({r.uri for r in export.rows}) == 10


def test_a_row_with_no_uri_is_kept_as_dropped() -> None:
    export = read_export(FIXTURE)
    assert [d.reason for d in export.dropped] == ["no_letterboxd_uri"]


def test_diary_supplies_the_watch_date_and_the_rating() -> None:
    stalker = row_for(read_export(FIXTURE).rows, "Stalker")
    assert stalker.watched_date == date(2024, 6, 15)
    assert stalker.date_source == "diary"
    assert stalker.rating_half == 9
    assert stalker.rewatch_count == 2
    assert stalker.is_rewatch


def test_ratings_only_film_has_no_watch_date() -> None:
    turin = row_for(read_export(FIXTURE).rows, "The Turin Horse")
    assert turin.watched_date is None
    assert turin.date_source == "ratings"
    assert turin.logged_date == date(2026, 1, 11)
    assert turin.rating_half == 8
    # Watchlisted and already seen is normal, and is not a contradiction.
    assert turin.in_watchlist


def test_watched_but_unrated_keeps_a_null_rating() -> None:
    hard = row_for(read_export(FIXTURE).rows, "Hard to Be a God")
    assert hard.rating_half is None
    # watched.csv alone carries no diary or ratings date, so the date is not trustworthy.
    assert hard.date_source == "none"
    assert hard.logged_date == date(2025, 5, 5)


def test_watchlist_only_film_is_not_watched() -> None:
    werck = row_for(read_export(FIXTURE).rows, "Werckmeister Harmonies")
    assert werck.in_watchlist
    assert werck.watchlist_added_on == date(2026, 2, 20)
    assert werck.rating_half is None
    assert werck.watched_date is None


def test_quoted_title_survives_the_csv() -> None:
    strangelove = row_for(
        read_export(FIXTURE).rows, "Dr. Strangelove, or: How I Learned to Stop Worrying"
    )
    assert strangelove.rating_half == 1


def test_review_naming_the_film_is_flagged() -> None:
    rows = read_export(FIXTURE).rows
    satantango = row_for(rows, "Sátántangó")
    turin = row_for(rows, "The Turin Horse")
    assert satantango.review is not None
    assert leaks_identity(satantango.review, [satantango.title])
    assert turin.review is not None
    assert not leaks_identity(turin.review, [turin.title])


def test_normalize_strips_articles_and_diacritics() -> None:
    assert normalize_title("The Turin Horse") == "turin horse"
    assert normalize_title("Sátántangó") == "satantango"
    assert (
        normalize_title("Dr. Strangelove, or: How I Learned") == "dr strangelove or how i learned"
    )


def test_tmdb_id_in_uri() -> None:
    assert tmdb_id_in_uri("https://boxd.it/film/tmdb-11104/") == 11104
    assert tmdb_id_in_uri("https://boxd.it/film/stalker/") is None


def test_title_similarity_sees_the_original_title() -> None:
    candidate = Candidate(30640, "Satantango", "Sátántangó", 1994, 700)
    assert title_similarity("Sátántangó", candidate) == 1.0
    assert title_similarity("Werckmeister Harmonies", candidate) < 0.5


def test_uri_id_beats_a_search(db: Database) -> None:
    search = StaticTitleSearch(CATALOGUE)
    rows = read_export(FIXTURE).rows
    chungking = row_for(rows, "Chungking Express")
    resolution = resolve(chungking, search)
    assert resolution.tmdb_id == 11104
    assert resolution.method == "uri"
    assert resolution.confidence == 1.0
    assert search.calls == []


def test_exact_title_and_year_is_confident() -> None:
    resolution = resolve(
        row_for(read_export(FIXTURE).rows, "Stalker"), StaticTitleSearch(CATALOGUE)
    )
    assert resolution.tmdb_id == 1398
    assert resolution.method == "exact"
    assert resolution.confidence == 0.95
    assert not resolution.needs_review


def test_a_year_off_by_one_is_a_reissue_not_a_failure() -> None:
    resolution = resolve(row_for(read_export(FIXTURE).rows, "Mirror"), StaticTitleSearch(CATALOGUE))
    assert resolution.tmdb_id == 11916
    assert resolution.method == "fuzzy"
    assert resolution.confidence == 0.85
    assert not resolution.needs_review


def test_two_films_sharing_a_title_and_year_are_never_auto_picked() -> None:
    resolution = resolve(row_for(read_export(FIXTURE).rows, "Hamlet"), StaticTitleSearch(CATALOGUE))
    assert resolution.tmdb_id is None
    assert resolution.reason == "ambiguous_title_year"
    assert resolution.needs_review
    assert {c[0] for c in resolution.candidates} == {10275, 41874}


def test_nothing_found_is_recorded_not_guessed() -> None:
    row = row_for(read_export(FIXTURE).rows, "A Home Movie Nobody Filed")
    resolution = resolve(row, StaticTitleSearch(CATALOGUE))
    assert resolution.tmdb_id is None
    assert resolution.method == "failed"
    assert resolution.reason == "no_candidates"


def test_manual_resolution_wins() -> None:
    row = row_for(read_export(FIXTURE).rows, "Hamlet")
    manual = Resolution(row.uri, 41874, "manual", 1.0)
    resolution = resolve(row, StaticTitleSearch(CATALOGUE), manual=manual)
    assert resolution.tmdb_id == 41874
    assert resolution.method == "manual"


def test_import_writes_the_three_tables(db: Database) -> None:
    report = import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    conn = db.read()
    assert report.n_rows == 10
    assert report.n_resolved == 8
    assert report.n_unresolved == 2
    assert 0.79 < report.match_rate < 0.81

    resolutions = conn.execute("select count(*) from title_resolutions").fetchone()[0]
    assert resolutions == 10
    user_films = conn.execute("select count(*) from user_films").fetchone()[0]
    assert user_films == 8
    # Two unresolved rows plus the line with no URI at all.
    unmatched = conn.execute(
        "select reason, count(*) as n from unmatched_export_row group by reason"
    ).fetchall()
    assert {r["reason"]: r["n"] for r in unmatched} == {
        "ambiguous_title_year": 1,
        "no_candidates": 1,
        "no_letterboxd_uri": 1,
    }


def test_import_records_the_run(db: Database) -> None:
    report = import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    row = db.read().execute("select * from letterboxd_imports").fetchone()
    assert row["import_id"] == report.import_id
    assert row["n_ratings"] == 8
    assert row["n_diary"] == 4
    assert row["n_resolved"] == 8
    assert len(row["zip_sha256"]) == 64


def test_imported_history_keeps_half_stars_as_integers(db: Database) -> None:
    import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    rows = (
        db.read()
        .execute(
            "select u.rating_half, f.title from user_films u join films f using (tmdb_id) "
            "where u.rating_half is not null"
        )
        .fetchall()
    )
    ratings = {str(r["title"]): r["rating_half"] for r in rows}
    assert ratings["Stalker"] == 9
    assert ratings["Sátántangó"] == 10
    assert all(isinstance(v, int) for v in ratings.values())


def test_imported_history_carries_the_review_flag(db: Database) -> None:
    import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    row = (
        db.read()
        .execute("select review_chars, review_leaks_identity from user_films where tmdb_id = 30640")
        .fetchone()
    )
    assert row["review_chars"] > 0
    assert row["review_leaks_identity"] == 1


def test_reimport_is_idempotent(db: Database) -> None:
    first = import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    second = import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    conn = db.read()
    assert second.import_id != first.import_id
    assert conn.execute("select count(*) from user_films").fetchone()[0] == 8
    assert conn.execute("select count(*) from title_resolutions").fetchone()[0] == 10
    assert conn.execute("select count(*) from films").fetchone()[0] == 8


def test_a_manual_decision_survives_a_reimport(db: Database) -> None:
    import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    record_manual(db, "https://boxd.it/film/hamlet-1964/", 41874, resolved_at="2026-09-14T00:00:00")
    import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    row = (
        db.read()
        .execute(
            "select tmdb_id, method, needs_review from title_resolutions "
            "where letterboxd_uri = 'https://boxd.it/film/hamlet-1964/'"
        )
        .fetchone()
    )
    assert row["tmdb_id"] == 41874
    assert row["method"] == "manual"
    assert row["needs_review"] == 0


def test_without_a_search_backend_only_uri_ids_resolve(db: Database) -> None:
    report = import_export(db, FIXTURE, search=NullTitleSearch())
    assert report.n_resolved == 1
    row = (
        db.read()
        .execute(
            "select u.tmdb_id as tmdb_id, t.method as method from user_films u "
            "join title_resolutions t on t.letterboxd_uri = u.letterboxd_uri"
        )
        .fetchone()
    )
    assert row["tmdb_id"] == 11104
    assert row["method"] == "uri"


def test_stub_film_rows_are_marked_unenriched(db: Database) -> None:
    import_export(db, FIXTURE, search=StaticTitleSearch(CATALOGUE))
    versions = {r["detail_version"] for r in db.read().execute("select detail_version from films")}
    assert versions == {0}


def write_export(path: Path, rows: list[list[str]]) -> Path:
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n").writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ratings.csv", buf.getvalue().encode("utf-8"))
    return path


@settings(max_examples=60, deadline=None)
@given(
    title=st.text(min_size=1, max_size=40),
    stars=st.sampled_from(["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]),
    year=st.integers(min_value=1900, max_value=2030),
)
def test_arbitrary_titles_never_crash_the_parser(title: str, stars: str, year: int) -> None:
    with tempfile.TemporaryDirectory() as work:
        target = write_export(
            Path(work) / "fuzz.zip",
            [
                ["Date", "Name", "Year", "Letterboxd URI", "Rating"],
                ["2026-01-01", title, str(year), "https://boxd.it/film/fuzz/", stars],
            ],
        )
        export = read_export(target)
    assert len(export.rows) == 1
    rating = export.rows[0].rating_half
    assert rating is not None
    assert 1 <= rating <= 10


def test_sqlite_refuses_a_float_rating(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), db.write() as conn:
        conn.execute("insert into films (tmdb_id, title, fetched_at) values (1, 'x', 't')")
        conn.execute(
            "insert into user_films (tmdb_id, rating_half, date_source) values (1, 4.5, 'none')"
        )
