"""The queue is the crawl's state: killing the workers loses nothing."""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import httpx
import orjson
import pytest
from pydantic import SecretStr

from palate import clock
from palate.db.connect import Database, open_database
from palate.errors import DiscoverWindowTooLarge
from palate.paths import migrations_dir
from palate.providers.http import build_client
from palate.tmdb.client import TMDBClient
from palate.tmdb.crawl import (
    Crawler,
    CrawlQueue,
    DiscoverWindow,
    compact,
    enqueue_row,
    load_raw,
    plan_windows,
    probe_plan,
    probe_window,
    record_member,
    renormalize,
    seed_export,
    seed_history,
    seed_onehop,
    seed_windows,
    status,
    stub_film,
)
from palate.tmdb.exports import export_url, iter_entries, latest_day, read_export
from palate.tmdb.normalize import DETAIL_VERSION, normalize_movie

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"
TOKEN = SecretStr("read-token-value")
START = datetime(2026, 9, 7, 18, 0, 0, tzinfo=UTC)

MOVIES = {
    603: "movie_603_matrix.json",
    1398: "movie_1398_stalker.json",
    11104: "movie_11104_chungking_express.json",
    802: "movie_802_no_overview.json",
}


def payload(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


def make_db(tmp_path: Path) -> Database:
    return open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)


class FakeTMDB:
    """Serves the committed payloads, with failures the test asks for."""

    def __init__(self, *, fail_on: Callable[[httpx.Request], httpx.Response | None] | None = None):
        self.requests: list[httpx.Request] = []
        self.fail_on = fail_on

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_on is not None:
            forced = self.fail_on(request)
            if forced is not None:
                return forced
        path = request.url.path
        if path.startswith("/3/movie/") and path.count("/") == 3:
            tmdb_id = int(path.rsplit("/", 1)[-1])
            name = MOVIES.get(tmdb_id)
            if name is None:
                return httpx.Response(404, json={"status_code": 34})
            return httpx.Response(200, json=payload(name), headers={"ETag": f'"v-{tmdb_id}"'})
        if path.endswith("/recommendations"):
            return httpx.Response(200, json=payload("recommendations_603.json"))
        if path.endswith("/similar"):
            return httpx.Response(200, json=payload("similar_603.json"))
        if path == "/3/discover/movie":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=payload(f"discover_page{page}.json"))
        if path.endswith(".json.gz"):
            lines = [
                {"id": 603, "original_title": "The Matrix", "popularity": 80.4},
                {"id": 1398, "original_title": "Stalker", "popularity": 12.5},
                {"id": 4242, "original_title": "Adult Feature", "popularity": 99.0, "adult": True},
            ]
            body = b"\n".join(orjson.dumps(line) for line in lines)
            return httpx.Response(200, content=gzip.compress(body))
        return httpx.Response(404, json={"status_code": 34})


def run_crawl(db: Database, fake: FakeTMDB, **kwargs: Any) -> Any:
    """Drain the queue once against the fake, returning the report."""

    async def scenario() -> Any:
        async with build_client(transport=fake.transport()) as http:
            client = TMDBClient(token=TOKEN, client=http)
            crawler = Crawler(db, client, concurrency=kwargs.pop("concurrency", 2), **kwargs)
            return await crawler.run()

    return anyio.run(scenario)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = make_db(tmp_path)
    yield database
    database.close()


def queue_rows(db: Database) -> list[dict[str, Any]]:
    rows = db.read().execute("select * from crawl_queue order by id").fetchall()
    return [dict(row) for row in rows]


def seed_film(db: Database, tmdb_id: int, title: str, rating: int | None = None) -> None:
    """A history row the way ingest leaves it, stub film included."""
    with db.write() as conn:
        stub_film(conn, tmdb_id, title, None, "2026-09-01T00:00:00")
        conn.execute(
            "insert into user_films (tmdb_id, rating_half, date_source) values (?,?,'ratings') "
            "on conflict(tmdb_id) do update set rating_half = excluded.rating_half",
            (tmdb_id, rating),
        )


def test_migration_0004_creates_the_queue(db: Database) -> None:
    names = {
        str(r["name"])
        for r in db.read().execute("select name from sqlite_master where type = 'table'")
    }
    assert {"corpus_members", "crawl_runs", "crawl_queue"} <= names


def test_the_same_work_is_only_queued_once(db: Database) -> None:
    queue = CrawlQueue(db)
    assert queue.enqueue("detail", tmdb_id=603) == 1
    assert queue.enqueue("detail", tmdb_id=603) == 0
    assert queue.enqueue("onehop", tmdb_id=603, params={"endpoint": "similar"}) == 1
    assert queue.enqueue("onehop", tmdb_id=603, params={"endpoint": "similar"}) == 0
    assert len(queue_rows(db)) == 2


def test_a_lease_is_exclusive_until_it_expires(db: Database) -> None:
    queue = CrawlQueue(db, lease_s=120)
    queue.enqueue("detail", tmdb_id=603)
    assert queue.pending() == 1
    assert queue.histogram() == {"pending": 1}
    with db.write() as conn:
        conn.execute(
            "insert into crawl_runs (run_id, kind, started_at, status) "
            "values ('run_a', 'detail', '2026-09-07T18:00:00', 'running')"
        )
    with clock.frozen(START):
        first = queue.lease("run_a", 5)
        assert [lease.tmdb_id for lease in first] == [603]
        assert queue.lease("run_a", 5) == []
        assert queue_rows(db)[0]["state"] == "leased"
        assert queue_rows(db)[0]["attempts"] == 1
        assert queue.pending() == 1
    # A killed worker leaves the lease behind and the next run reclaims it.
    with clock.frozen(START + timedelta(seconds=300)):
        reclaimed = queue.lease("run_a", 5)
    assert [lease.tmdb_id for lease in reclaimed] == [603]
    assert queue_rows(db)[0]["attempts"] == 2


def test_a_detail_crawl_fills_films_people_and_credits(db: Database) -> None:
    seed_film(db, 1398, "Stalker")
    assert seed_history(db) == 1
    with clock.frozen(START):
        report = run_crawl(db, FakeTMDB())
    assert (report.n_ok, report.n_err, report.n_dead) == (1, 0, 0)

    row = db.read().execute("select * from films where tmdb_id = 1398").fetchone()
    assert row["title"] == "Stalker"
    assert row["year"] == 1979
    assert row["runtime"] == 162
    assert row["imdb_id"] == "tt0079944"
    assert row["detail_version"] == DETAIL_VERSION
    assert row["etag"] == '"v-1398"'
    assert row["popularity_at_crawl"] == pytest.approx(12.504)

    directors = (
        db.read()
        .execute(
            "select p.name from credits c join people p using (person_id) "
            "where c.tmdb_id = 1398 and c.job = 'Director'"
        )
        .fetchall()
    )
    assert [r["name"] for r in directors] == ["Andrei Tarkovsky"]
    keywords = db.read().execute("select count(*) from film_keywords where tmdb_id = 1398")
    assert keywords.fetchone()[0] == 6
    countries = db.read().execute("select iso_3166_1 from film_countries where tmdb_id = 1398")
    assert [r["iso_3166_1"] for r in countries] == ["SU"]
    assert queue_rows(db)[0]["state"] == "done"


def test_both_directors_survive_normalisation() -> None:
    film = normalize_movie(payload("movie_603_matrix.json"))
    assert len(film.director_ids) == 2
    assert film.primary_region == "US"


def test_a_co_directed_film_keeps_both_directors(db: Database) -> None:
    seed_film(db, 603, "The Matrix")
    seed_history(db)
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    rows = (
        db.read()
        .execute(
            "select p.name from credits c join people p using (person_id) "
            "where c.tmdb_id = 603 and c.job = 'Director' order by p.name"
        )
        .fetchall()
    )
    assert [r["name"] for r in rows] == ["Lana Wachowski", "Lilly Wachowski"]
    film = db.read().execute("select collection_id, collection_name from films where tmdb_id = 603")
    assert dict(film.fetchone()) == {
        "collection_id": 2344,
        "collection_name": "The Matrix Collection",
    }


def test_a_film_with_no_overview_still_lands(db: Database) -> None:
    seed_film(db, 802, "The Bear")
    seed_history(db)
    with clock.frozen(START):
        report = run_crawl(db, FakeTMDB())
    assert report.n_ok == 1
    row = db.read().execute("select overview, detail_version from films where tmdb_id = 802")
    record = row.fetchone()
    assert record["overview"] is None
    assert record["detail_version"] == DETAIL_VERSION


def test_the_crawl_upserts_over_the_stub_rows_ingest_left(db: Database) -> None:
    seed_film(db, 11104, "Chungking Express")
    before = db.read().execute("select detail_version from films where tmdb_id = 11104").fetchone()
    assert before["detail_version"] == 0
    seed_history(db)
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    after = (
        db.read()
        .execute("select detail_version, original_language from films where tmdb_id = 11104")
        .fetchone()
    )
    assert after["detail_version"] == DETAIL_VERSION
    assert after["original_language"] == "cn"
    assert db.read().execute("select count(*) from films").fetchone()[0] == 1


def test_a_second_pass_sends_the_etag_and_takes_the_304(db: Database) -> None:
    seed_film(db, 1398, "Stalker")
    seed_history(db)
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    with db.write() as conn:
        conn.execute("update crawl_queue set state = 'pending', next_attempt_at = 0")

    def not_modified(request: httpx.Request) -> httpx.Response | None:
        if request.headers.get("if-none-match") == '"v-1398"':
            return httpx.Response(304)
        return None

    later = START + timedelta(days=1)
    with clock.frozen(later):
        report = run_crawl(db, FakeTMDB(fail_on=not_modified))
    assert (report.n_ok, report.n_304) == (0, 1)
    row = db.read().execute("select fetched_at, title from films where tmdb_id = 1398").fetchone()
    assert row["fetched_at"].startswith("2026-09-08")
    assert row["title"] == "Stalker"


def test_a_404_is_dead_not_retried(db: Database) -> None:
    with db.write() as conn:
        stub_film(conn, 999999, "Withdrawn Title", 2001, "2026-09-01T00:00:00")
        enqueue_row(conn, "detail", tmdb_id=999999)
    with clock.frozen(START):
        report = run_crawl(db, FakeTMDB())
    assert report.n_dead == 1
    row = queue_rows(db)[0]
    assert row["state"] == "dead"
    assert "404" in row["last_error"]


def test_a_429_is_requeued_without_counting_the_attempt(db: Database) -> None:
    seed_film(db, 603, "The Matrix")
    seed_history(db)

    def throttle(request: httpx.Request) -> httpx.Response | None:
        return httpx.Response(429, headers={"Retry-After": "11"})

    with clock.frozen(START):
        report = run_crawl(db, FakeTMDB(fail_on=throttle), max_items=1)
    assert report.n_err == 1
    row = queue_rows(db)[0]
    assert row["state"] == "pending"
    assert row["attempts"] == 0
    assert row["next_attempt_at"] == pytest.approx(START.timestamp() + 11)


def test_a_server_error_backs_off_then_fails_after_five_attempts(db: Database) -> None:
    seed_film(db, 603, "The Matrix")
    seed_history(db)

    def broken(request: httpx.Request) -> httpx.Response | None:
        return httpx.Response(503, json={"status_code": 1})

    fake = FakeTMDB(fail_on=broken)
    for attempt in range(1, 7):
        with clock.frozen(START + timedelta(hours=attempt)):
            run_crawl(db, fake, max_items=1)
        row = queue_rows(db)[0]
        if attempt < 5:
            assert row["state"] == "pending", attempt
            assert row["attempts"] == attempt
    assert queue_rows(db)[0]["state"] == "failed"
    assert "503" in queue_rows(db)[0]["last_error"]


def test_killing_the_workers_mid_flight_loses_nothing(db: Database) -> None:
    for tmdb_id, name in [(603, "The Matrix"), (1398, "Stalker"), (11104, "Chungking Express")]:
        seed_film(db, tmdb_id, name)
    assert seed_history(db) == 3

    class Boom(Exception):
        pass

    seen: list[int] = []

    def crash_after_one(request: httpx.Request) -> httpx.Response | None:
        seen.append(1)
        if len(seen) > 1:
            raise Boom("the process is gone")
        return None

    with clock.frozen(START), pytest.raises(BaseExceptionGroup) as crash:
        run_crawl(db, FakeTMDB(fail_on=crash_after_one), concurrency=1)
    assert "the process is gone" in str(crash.value.exceptions[0])

    states = {row["state"] for row in queue_rows(db)}
    assert "leased" in states or "pending" in states
    assert (
        db.read().execute("select count(*) from films where detail_version > 0").fetchone()[0] == 1
    )

    # The next run reclaims the expired lease and finishes the set.
    with clock.frozen(START + timedelta(seconds=600)):
        report = run_crawl(db, FakeTMDB())
    assert report.n_ok == 2
    assert [row["state"] for row in queue_rows(db)] == ["done", "done", "done"]
    enriched = db.read().execute("select count(*) from films where detail_version > 0").fetchone()
    assert enriched[0] == 3
    assert db.read().execute("select count(*) from tmdb_raw").fetchone()[0] == 3


def span(window: DiscoverWindow) -> str:
    return f"{window.start} {window.end}"


def probe_from(totals: dict[str, int], *, sparse: int = 120) -> Callable[[DiscoverWindow], int]:
    """A probe answering from a table of spans, with everything else under the cap."""
    return lambda window: totals.get(span(window), sparse)


def test_a_year_under_the_ceiling_stays_one_window() -> None:
    plan = plan_windows(since=1920, until=1922, probe=probe_from({}))
    assert [span(w) for w in plan] == [
        "1920-01-01 1920-12-31",
        "1921-01-01 1921-12-31",
        "1922-01-01 1922-12-31",
    ]


def test_a_year_over_the_ceiling_splits_into_quarters() -> None:
    plan = plan_windows(since=1999, until=1999, probe=probe_from({"1999-01-01 1999-12-31": 24_000}))
    assert [span(w) for w in plan] == [
        "1999-01-01 1999-03-31",
        "1999-04-01 1999-06-30",
        "1999-07-01 1999-09-30",
        "1999-10-01 1999-12-31",
    ]


def test_a_quarter_still_over_the_ceiling_splits_into_months() -> None:
    totals = {"2019-01-01 2019-12-31": 41_000, "2019-07-01 2019-09-30": 12_000}
    plan = plan_windows(since=2019, until=2019, probe=probe_from(totals))
    assert [span(w) for w in plan] == [
        "2019-01-01 2019-03-31",
        "2019-04-01 2019-06-30",
        "2019-07-01 2019-07-31",
        "2019-08-01 2019-08-31",
        "2019-09-01 2019-09-30",
        "2019-10-01 2019-12-31",
    ]


def test_a_month_that_still_overflows_is_reported_not_truncated() -> None:
    with pytest.raises(DiscoverWindowTooLarge, match="2020-01-01 to 2020-01-31") as raised:
        plan_windows(since=2020, until=2020, probe=probe_from({}, sparse=11_000))
    assert raised.value.total == 11_000


def test_the_plan_stops_at_this_year_by_default() -> None:
    with clock.frozen(START):
        plan = plan_windows(since=2024, probe=probe_from({}))
    assert [w.start.year for w in plan] == [2024, 2025, 2026]


def test_a_window_carries_its_dates_into_the_discover_query() -> None:
    window = DiscoverWindow(date(1979, 1, 1), date(1979, 12, 31), vote_count_gte=5)
    query = window.params().as_query()
    assert query["primary_release_date.gte"] == "1979-01-01"
    assert query["primary_release_date.lte"] == "1979-12-31"
    assert query["vote_count.gte"] == "5"
    assert query["sort_by"] == "vote_count.desc"


def test_the_probe_reads_total_results_off_page_one() -> None:
    fake = FakeTMDB()

    async def scenario() -> int:
        async with build_client(transport=fake.transport()) as http:
            window = DiscoverWindow(date(1999, 1, 1), date(1999, 12, 31), 5)
            return await probe_window(TMDBClient(token=TOKEN, client=http), window)

    assert anyio.run(scenario) == 5
    assert len(fake.requests) == 1
    assert fake.requests[0].url.params["primary_release_date.gte"] == "1999-01-01"


def test_the_live_plan_probes_one_request_per_window() -> None:
    fake = FakeTMDB()

    async def scenario() -> list[DiscoverWindow]:
        async with build_client(transport=fake.transport()) as http:
            client = TMDBClient(token=TOKEN, client=http)
            return await probe_plan(client, since=1924, vote_count_gte=1)

    with clock.frozen(START):
        plan = anyio.run(scenario)
    # Every year from 1924 to the year the clock says, and the fixture never overflows.
    assert [w.start.year for w in plan] == list(range(1924, 2027))
    assert len(fake.requests) == len(plan)
    assert fake.requests[0].url.params["vote_count.gte"] == "1"


def test_a_planned_window_queues_details_and_the_next_page(db: Database) -> None:
    plan = plan_windows(since=1999, until=1999, probe=probe_from({}), vote_count_gte=5)
    assert seed_windows(db, plan) == 1
    assert seed_windows(db, plan) == 0
    queued = json.loads(queue_rows(db)[0]["params_json"])["discover"]
    assert queued["primary_release_date_gte"] == "1999-01-01"
    assert queued["vote_count_gte"] == 5
    with clock.frozen(START):
        report = run_crawl(db, FakeTMDB())
    # Two pages of the window, four films that exist, one that TMDB has dropped.
    assert report.n_ok == 2 + 4
    assert report.n_dead == 1
    members = (
        db.read().execute("select tmdb_id, source from corpus_members order by tmdb_id").fetchall()
    )
    assert [(r["tmdb_id"], r["source"]) for r in members] == [
        (603, "discover"),
        (802, "discover"),
        (1398, "discover"),
        (11104, "discover"),
        (999999, "discover"),
    ]
    kinds = {row["kind"] for row in queue_rows(db)}
    assert kinds == {"discover", "detail"}
    assert all(row["state"] in ("done", "dead") for row in queue_rows(db))
    # The window is one row, whatever it takes to page it.
    counted = status(db)
    assert (counted.n_windows, counted.n_windows_done) == (1, 1)


def test_onehop_brings_in_neighbours_of_the_top_rated(db: Database) -> None:
    seed_film(db, 603, "The Matrix", rating=10)
    assert seed_onehop(db, top=1) == 2
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    members = (
        db.read()
        .execute(
            "select tmdb_id, source from corpus_members where source = 'onehop' order by tmdb_id"
        )
        .fetchall()
    )
    assert [r["tmdb_id"] for r in members] == [802, 1398, 11104]
    raw = (
        db.read()
        .execute(
            "select entity, entity_id from tmdb_raw where entity in ('recommendations','similar')"
        )
        .fetchall()
    )
    assert {(r["entity"], r["entity_id"]) for r in raw} == {
        ("recommendations", 603),
        ("similar", 603),
    }


def test_the_id_export_is_a_backstop_enumerator(db: Database) -> None:
    assert seed_export(db, day="2026-09-06") == 1
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    members = (
        db.read()
        .execute("select tmdb_id from corpus_members where source = 'export' order by tmdb_id")
        .fetchall()
    )
    # The adult title in the dump is never queued.
    assert [r["tmdb_id"] for r in members] == [603, 1398]
    receipt = db.read().execute("select payload_z from tmdb_raw where entity = 'export'").fetchone()
    assert orjson.loads(zlib.decompress(receipt["payload_z"]))["n_kept"] == 2


def test_the_export_filename_follows_tmdb_and_waits_for_publication() -> None:
    assert export_url("2026-09-06").endswith("movie_ids_09_06_2026.json.gz")
    assert latest_day(datetime(2026, 9, 7, 9, 0, tzinfo=UTC)) == "2026-09-07"
    assert latest_day(datetime(2026, 9, 7, 3, 0, tzinfo=UTC)) == "2026-09-06"


def test_the_export_reader_reads_gzipped_lines(tmp_path: Path) -> None:
    body = gzip.compress(b'{"id": 5, "original_title": "Five", "popularity": 1.5}\n\n')
    assert [e.tmdb_id for e in iter_entries(body)] == [5]
    dump = tmp_path / "movie_ids_09_06_2026.json.gz"
    dump.write_bytes(body)
    assert [e.original_title for e in read_export(dump)] == ["Five"]


def test_raw_payloads_survive_a_renormalize(db: Database) -> None:
    seed_film(db, 1398, "Stalker")
    seed_history(db)
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    with db.write() as conn:
        conn.execute("update films set detail_version = 0, runtime = null where tmdb_id = 1398")
        conn.execute("delete from film_keywords where tmdb_id = 1398")

    assert renormalize(db) == 1
    row = db.read().execute("select detail_version, runtime from films where tmdb_id = 1398")
    record = row.fetchone()
    assert record["detail_version"] == DETAIL_VERSION
    assert record["runtime"] == 162
    assert (
        db.read().execute("select count(*) from film_keywords where tmdb_id = 1398").fetchone()[0]
        == 6
    )
    stored = load_raw(db.read(), "movie", 1398)
    assert stored is not None
    assert normalize_movie(stored).title == "Stalker"


def test_compact_drops_payloads_outside_the_corpus(db: Database) -> None:
    seed_film(db, 1398, "Stalker")
    seed_history(db)
    with clock.frozen(START):
        run_crawl(db, FakeTMDB())
    with db.write() as conn:
        conn.execute("delete from corpus_members where tmdb_id = 1398")
    assert compact(db) == 1
    assert db.read().execute("select count(*) from tmdb_raw").fetchone()[0] == 0


def test_status_counts_what_the_crawl_is_judged_on(db: Database) -> None:
    seed_film(db, 1398, "Stalker")
    seed_film(db, 603, "The Matrix")
    seed_history(db)
    with clock.frozen(START):
        run_crawl(db, FakeTMDB(), max_items=1)
    report = status(db)
    assert report.n_films == 2
    assert report.n_enriched == 1
    assert report.n_corpus == 2
    assert report.queue["done"] == 1
    assert report.by_kind == {"detail": 1}
    assert report.runs[0][2] == "done"
    assert report.raw_bytes > 0


def test_membership_keeps_the_first_source(db: Database) -> None:
    with db.write() as conn:
        stub_film(conn, 603, "The Matrix", 1999, "2026-09-01T00:00:00")
        record_member(conn, 603, "history", "2026-09-01T00:00:00")
        record_member(conn, 603, "discover", "2026-09-02T00:00:00")
    row = db.read().execute("select source from corpus_members where tmdb_id = 603").fetchone()
    assert row["source"] == "history"
