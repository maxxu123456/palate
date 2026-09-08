"""The vote floor sets the recall ceiling, so it is tested before any ranking exists."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from palate.db.connect import Database, open_database
from palate.ingest.corpus import (
    FilmRow,
    FilmStats,
    apply_eligibility,
    coverage,
    eligibility,
    vote_floor,
)
from palate.paths import migrations_dir
from palate.tmdb.crawl import record_member
from palate.tmdb.normalize import normalize_movie, write_film

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"
STAMP = "2026-09-08T12:00:00.000000+00:00"


def payload(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)
    yield database
    database.close()


def load(db: Database, name: str, *, source: str = "discover") -> int:
    film = normalize_movie(payload(name))
    with db.write() as conn:
        write_film(conn, film, fetched_at=STAMP)
        record_member(conn, film.tmdb_id, source, STAMP)
    return film.tmdb_id


@pytest.mark.parametrize(
    ("year", "region", "expected"),
    [
        (2015, "US", 25),
        (2015, "HK", 12),
        (1994, "US", 10),
        (1994, "HK", 5),
        (1979, "GB", 5),
        (1979, "SU", 3),
        (None, None, 25),
        (None, "HK", 12),
    ],
)
def test_vote_floor_bends_for_era_and_region(
    year: int | None, region: str | None, expected: int
) -> None:
    assert vote_floor(year, region) == expected


def test_the_floor_never_drops_below_three() -> None:
    assert vote_floor(1975, "PL") == 3


def test_a_mid_nineties_hong_kong_title_survives_a_flat_floor_would_delete() -> None:
    film = FilmRow(1, year=1994, runtime=97, vote_count=7, status="Released")
    ok, reason, floor = eligibility(film, FilmStats(1, primary_region="HK"))
    assert (ok, reason, floor) == (True, None, 5)
    # The same film read as a US release would have been cut.
    assert eligibility(film, FilmStats(1, primary_region="US"))[:2] == (False, "votes_below_10")


def test_the_reasons_are_ordered_so_the_first_true_one_is_reported() -> None:
    unreleased = FilmRow(1, year=2027, vote_count=0, status="Post Production")
    assert eligibility(unreleased, FilmStats(1))[1] == "not_released"
    short = FilmRow(2, year=2015, runtime=12, vote_count=900, status="Released")
    assert eligibility(short, FilmStats(2))[1] == "short"
    adult = FilmRow(3, year=2015, runtime=90, vote_count=900, adult=True, status="Released")
    assert eligibility(adult, FilmStats(3))[1] == "adult"


def test_a_missing_overview_is_not_a_deletion() -> None:
    film = FilmRow(4, year=2015, runtime=90, vote_count=900, status="Released")
    assert eligibility(film, FilmStats(4, has_overview=False))[0] is True


def test_an_unknown_runtime_is_not_treated_as_a_short() -> None:
    film = FilmRow(5, year=2015, runtime=None, vote_count=900, status="Released")
    assert eligibility(film, FilmStats(5))[0] is True


def test_normalize_records_the_region_the_floor_keys_off(db: Database) -> None:
    load(db, "movie_11104_chungking_express.json")
    row = db.read().execute("select * from film_stats where tmdb_id = 11104").fetchone()
    assert row["primary_region"] == "HK"
    assert row["n_directors"] == 1
    assert row["n_keywords"] == 3
    assert row["has_overview"] == 1
    assert row["is_animation"] == 0


def test_apply_eligibility_writes_the_verdict_and_the_floor(db: Database) -> None:
    load(db, "movie_11104_chungking_express.json")
    load(db, "movie_802_no_overview.json")
    report = apply_eligibility(db)
    assert report.n_members == 2
    assert report.n_eligible == 2
    rows = {
        int(r["tmdb_id"]): r
        for r in db.read().execute("select tmdb_id, eligible, vote_floor_used from corpus_members")
    }
    assert rows[11104]["vote_floor_used"] == 5
    # 1970, Poland, 14 votes. A flat 25 would have deleted it.
    assert rows[802]["vote_floor_used"] == 3
    assert rows[802]["eligible"] == 1


def test_an_ineligible_film_keeps_its_row_and_its_reason(db: Database) -> None:
    tmdb_id = load(db, "movie_802_no_overview.json")
    with db.write() as conn:
        conn.execute("update films set vote_count = 1 where tmdb_id = ?", (tmdb_id,))
    report = apply_eligibility(db)
    assert report.n_eligible == 0
    assert report.reasons == {"votes_below_3": 1}
    row = (
        db.read()
        .execute(
            "select eligible, ineligible_reason from corpus_members where tmdb_id = ?", (tmdb_id,)
        )
        .fetchone()
    )
    assert row["eligible"] == 0
    assert row["ineligible_reason"] == "votes_below_3"
    assert db.read().execute("select count(*) from films").fetchone()[0] == 1


def test_a_film_with_no_detail_crawl_yet_is_held_back(db: Database) -> None:
    with db.write() as conn:
        conn.execute(
            "insert into films (tmdb_id, title, year, fetched_at, detail_version) "
            "values (?,?,?,?,0)",
            (999, "Unknown", 2001, STAMP),
        )
        record_member(conn, 999, "export", STAMP)
    report = apply_eligibility(db)
    assert report.reasons == {"not_released": 1}


def test_eligibility_reruns_cleanly_after_the_votes_move(db: Database) -> None:
    tmdb_id = load(db, "movie_802_no_overview.json")
    with db.write() as conn:
        conn.execute("update films set vote_count = 1 where tmdb_id = ?", (tmdb_id,))
    apply_eligibility(db)
    with db.write() as conn:
        conn.execute("update films set vote_count = 40 where tmdb_id = ?", (tmdb_id,))
    apply_eligibility(db)
    row = (
        db.read()
        .execute(
            "select eligible, ineligible_reason from corpus_members where tmdb_id = ?", (tmdb_id,)
        )
        .fetchone()
    )
    assert row["eligible"] == 1
    assert row["ineligible_reason"] is None


def test_coverage_breaks_the_number_down_by_region(db: Database) -> None:
    load(db, "movie_603_matrix.json")
    load(db, "movie_11104_chungking_express.json")
    with db.write() as conn:
        conn.execute("update films set vote_count = 1 where tmdb_id = 11104")
    apply_eligibility(db)
    by_region = {c.region: c for c in coverage(db)}
    assert by_region["US"].n_eligible == 1
    assert by_region["HK"].n_eligible == 0
    assert by_region["HK"].share == 0.0
