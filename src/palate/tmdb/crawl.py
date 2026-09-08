"""The crawl queue, its leases, and the disposable workers that drain it."""

from __future__ import annotations

import calendar
import sqlite3
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from typing import Any

import anyio
import orjson

from palate import clock
from palate.clock import now_iso
from palate.db.connect import Database
from palate.errors import DiscoverWindowTooLarge, ProviderTimeout, ProviderUnavailable
from palate.hashing import canonical_json, request_sha, sha256_hex
from palate.ids import new_id
from palate.tmdb.client import TMDBClient, TMDBResponse
from palate.tmdb.exports import fetch_export, iter_entries, latest_day
from palate.tmdb.models import DiscoverParams, MoviePage, MovieSummary
from palate.tmdb.normalize import normalize_movie, write_film

MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 300.0

# /discover/movie stops at 500 pages of 20 and 10000 results, whatever the query
# says it has, so the planner keeps every window under the second number.
DISCOVER_PAGE_CAP = 500
DISCOVER_RESULT_CAP = 10_000

# A year over the ceiling is cut into quarters, and a quarter into months.
_SPLITS = (3, 1)

RAW_LEVEL = 6


@dataclass(frozen=True, slots=True)
class Lease:
    """One queue row, held by one worker until lease_until passes."""

    id: int
    kind: str
    tmdb_id: int | None
    params: dict[str, Any]
    attempts: int


@dataclass(frozen=True, slots=True)
class DiscoverWindow:
    """A release date span narrow enough that the result ceiling never truncates it."""

    start: date
    end: date
    vote_count_gte: int = 0
    sort_by: str = "vote_count.desc"
    region: str | None = None

    def params(self) -> DiscoverParams:
        """The discover query this window stands for."""
        return DiscoverParams(
            sort_by=self.sort_by,
            vote_count_gte=self.vote_count_gte or None,
            primary_release_date_gte=self.start.isoformat(),
            primary_release_date_lte=self.end.isoformat(),
            region=self.region,
        )


@dataclass(frozen=True, slots=True)
class CrawlReport:
    """What one crawl run did, mirroring the crawl_runs counters."""

    run_id: str
    n_ok: int = 0
    n_304: int = 0
    n_err: int = 0
    n_dead: int = 0


@dataclass(frozen=True, slots=True)
class CrawlStatus:
    """What `palate tmdb status` prints."""

    queue: dict[str, int]
    by_kind: dict[str, int]
    n_windows: int
    n_windows_done: int
    n_films: int
    n_enriched: int
    n_corpus: int
    n_raw: int
    raw_bytes: int
    runs: tuple[tuple[str, str, str, int, int, int, int], ...]


def params_text(params: Mapping[str, Any] | None) -> str | None:
    """Canonical text for the queue row, so the unique index actually dedupes."""
    return canonical_json(dict(params)).decode() if params else None


class CrawlQueue:
    """crawl_queue rows are the only crawl state, so a killed worker loses nothing."""

    def __init__(self, db: Database, *, lease_s: int = 120) -> None:
        self.db = db
        self.lease_s = lease_s

    def enqueue(
        self,
        kind: str,
        *,
        tmdb_id: int | None = None,
        params: Mapping[str, Any] | None = None,
        priority: int = 0,
    ) -> int:
        """Add one row, ignoring it if the same work is already queued."""
        with self.db.write() as conn:
            return enqueue_row(conn, kind, tmdb_id=tmdb_id, params=params, priority=priority)

    def lease(self, run_id: str, n: int) -> list[Lease]:
        """Claim up to n rows. Expired leases are reclaimed by the same statement."""
        now = clock.now().timestamp()
        with self.db.write() as conn:
            rows = conn.execute(
                "update crawl_queue set state = 'leased', lease_until = ?, "
                "attempts = attempts + 1, run_id = ? where id in ("
                "select id from crawl_queue "
                "where (state = 'pending' and next_attempt_at <= ?) "
                "or (state = 'leased' and lease_until <= ?) "
                "order by priority desc, id limit ?) "
                "returning id, kind, tmdb_id, params_json, attempts",
                (now + self.lease_s, run_id, now, now, n),
            ).fetchall()
        return [
            Lease(
                id=int(row["id"]),
                kind=str(row["kind"]),
                tmdb_id=None if row["tmdb_id"] is None else int(row["tmdb_id"]),
                params=orjson.loads(row["params_json"]) if row["params_json"] else {},
                attempts=int(row["attempts"]),
            )
            for row in rows
        ]

    def complete(self, lease_id: int) -> None:
        """Mark a row done."""
        self._finish(lease_id, "done", None)

    def fail(self, lease_id: int, error: str) -> None:
        """Give up on a row after too many attempts."""
        self._finish(lease_id, "failed", error)

    def kill(self, lease_id: int, reason: str) -> None:
        """A film TMDB no longer has stays gone tomorrow, so it is never retried."""
        self._finish(lease_id, "dead", reason)

    def requeue(
        self, lease_id: int, *, delay_s: float, error: str | None = None, count_attempt: bool = True
    ) -> None:
        """Put a row back, optionally without holding the attempt against it."""
        now = clock.now().timestamp()
        with self.db.write() as conn:
            conn.execute(
                "update crawl_queue set state = 'pending', next_attempt_at = ?, "
                "lease_until = null, last_error = ?, attempts = case when ? then attempts "
                "else max(attempts - 1, 0) end where id = ?",
                (now + delay_s, error, 1 if count_attempt else 0, lease_id),
            )

    def histogram(self) -> dict[str, int]:
        """Row counts by state."""
        rows = self.db.read().execute("select state, count(*) as n from crawl_queue group by state")
        return {str(r["state"]): int(r["n"]) for r in rows}

    def pending(self) -> int:
        """Rows still to do, leases included."""
        counts = self.histogram()
        return counts.get("pending", 0) + counts.get("leased", 0)

    def _finish(self, lease_id: int, state: str, error: str | None) -> None:
        with self.db.write() as conn:
            conn.execute(
                "update crawl_queue set state = ?, lease_until = null, last_error = ? where id = ?",
                (state, error, lease_id),
            )


def enqueue_row(
    conn: sqlite3.Connection,
    kind: str,
    *,
    tmdb_id: int | None = None,
    params: Mapping[str, Any] | None = None,
    priority: int = 0,
) -> int:
    """Insert one queue row inside an open transaction. Returns rows added."""
    cursor = conn.execute(
        "insert into crawl_queue (kind, tmdb_id, params_json, priority) values (?,?,?,?) "
        "on conflict do nothing",
        (kind, tmdb_id, params_text(params), priority),
    )
    return int(cursor.rowcount or 0)


def store_raw(
    conn: sqlite3.Connection,
    *,
    entity: str,
    payload: Mapping[str, Any],
    entity_id: int | None = None,
    params_sha: str | None = None,
    etag: str | None = None,
    fetched_at: str,
) -> None:
    """Keep the compressed payload so a normalisation change never means re-crawling."""
    body = canonical_json(dict(payload))
    conn.execute(
        "insert into tmdb_raw (entity, entity_id, params_sha, fetched_at, etag, payload_sha, "
        "payload_z) values (?,?,?,?,?,?,?) "
        "on conflict(entity, coalesce(entity_id, -1), coalesce(params_sha, '')) do update set "
        "fetched_at = excluded.fetched_at, etag = excluded.etag, "
        "payload_sha = excluded.payload_sha, payload_z = excluded.payload_z",
        (
            entity,
            entity_id,
            params_sha,
            fetched_at,
            etag,
            sha256_hex(body),
            zlib.compress(body, RAW_LEVEL),
        ),
    )


def load_raw(conn: sqlite3.Connection, entity: str, entity_id: int) -> dict[str, Any] | None:
    """Read one stored payload back."""
    row = conn.execute(
        "select payload_z from tmdb_raw where entity = ? and entity_id = ?", (entity, entity_id)
    ).fetchone()
    if row is None:
        return None
    loaded: dict[str, Any] = orjson.loads(zlib.decompress(row["payload_z"]))
    return loaded


def record_member(conn: sqlite3.Connection, tmdb_id: int, source: str, stamp: str) -> None:
    """Note where a film entered the corpus. The first source wins."""
    conn.execute(
        "insert into corpus_members (tmdb_id, source, added_at) values (?,?,?) "
        "on conflict(tmdb_id) do nothing",
        (tmdb_id, source, stamp),
    )


def stub_film(
    conn: sqlite3.Connection, tmdb_id: int, title: str, year: int | None, stamp: str
) -> None:
    """A placeholder row so membership and history have something to point at."""
    conn.execute(
        "insert into films (tmdb_id, title, year, fetched_at, detail_version) "
        "values (?,?,?,?,0) on conflict(tmdb_id) do nothing",
        (tmdb_id, title or str(tmdb_id), year, stamp),
    )


def register_summary(
    conn: sqlite3.Connection, summary: MovieSummary, source: str, stamp: str, *, priority: int = 0
) -> int:
    """Stub, membership and a detail row for one list result."""
    stub_film(conn, summary.id, summary.title or summary.original_title or "", summary.year, stamp)
    record_member(conn, summary.id, source, stamp)
    return enqueue_row(conn, "detail", tmdb_id=summary.id, priority=priority)


def seed_history(db: Database, *, priority: int = 10) -> int:
    """Queue every film the user has seen, because the eval holdout needs them."""
    rows = db.read().execute("select tmdb_id from user_films order by tmdb_id").fetchall()
    stamp = now_iso()
    added = 0
    with db.write() as conn:
        for row in rows:
            tmdb_id = int(row["tmdb_id"])
            record_member(conn, tmdb_id, "history", stamp)
            added += enqueue_row(conn, "detail", tmdb_id=tmdb_id, priority=priority)
    return added


def seed_onehop(db: Database, *, top: int = 200, priority: int = 5) -> int:
    """Queue recommendations and similar for the user's best rated films."""
    rows = (
        db.read()
        .execute(
            "select tmdb_id from user_films where rating_half is not null "
            "order by rating_half desc, tmdb_id limit ?",
            (top,),
        )
        .fetchall()
    )
    added = 0
    with db.write() as conn:
        for row in rows:
            for endpoint in ("recommendations", "similar"):
                added += enqueue_row(
                    conn,
                    "onehop",
                    tmdb_id=int(row["tmdb_id"]),
                    params={"endpoint": endpoint, "page": 1},
                    priority=priority,
                )
    return added


def plan_windows(
    *,
    probe: Callable[[DiscoverWindow], int],
    since: int = 1920,
    until: int | None = None,
    vote_count_gte: int = 0,
    sort_by: str = "vote_count.desc",
    region: str | None = None,
) -> list[DiscoverWindow]:
    """One window per year, split until every window fits under the result ceiling."""
    plan: list[DiscoverWindow] = []
    for year in range(since, (until or clock.now().year) + 1):
        whole = DiscoverWindow(
            date(year, 1, 1), date(year, 12, 31), vote_count_gte, sort_by, region
        )
        plan.extend(_fit(whole, probe, 0))
    return plan


def _fit(
    window: DiscoverWindow, probe: Callable[[DiscoverWindow], int], depth: int
) -> list[DiscoverWindow]:
    total = probe(window)
    if total <= DISCOVER_RESULT_CAP:
        return [window]
    if depth >= len(_SPLITS):
        raise DiscoverWindowTooLarge(
            window.start.isoformat(), window.end.isoformat(), total, DISCOVER_RESULT_CAP
        )
    fitted: list[DiscoverWindow] = []
    for piece in _divide(window, _SPLITS[depth]):
        fitted.extend(_fit(piece, probe, depth + 1))
    return fitted


def _divide(window: DiscoverWindow, months: int) -> list[DiscoverWindow]:
    pieces: list[DiscoverWindow] = []
    start = window.start
    while start <= window.end:
        last = _add_months(start, months - 1)
        pieces.append(replace(window, start=start, end=min(_month_end(last), window.end)))
        start = _add_months(start, months)
    return pieces


def _add_months(day: date, months: int) -> date:
    index = day.month - 1 + months
    return date(day.year + index // 12, index % 12 + 1, 1)


def _month_end(day: date) -> date:
    return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])


async def probe_window(client: TMDBClient, window: DiscoverWindow) -> int:
    """How many films a window holds. Page one carries it, so a probe is one request."""
    response = await client.discover(window.params(), page=1)
    if not response.ok or response.payload is None:
        raise ProviderUnavailable(
            f"discover probe returned http {response.status}", provider="tmdb", retryable=True
        )
    return MoviePage.model_validate(response.payload).total_results


async def probe_plan(
    client: TMDBClient,
    *,
    since: int = 1920,
    vote_count_gte: int = 0,
) -> list[DiscoverWindow]:
    """The plan for a live crawl. The planner is synchronous, so it runs off the loop."""
    return await anyio.to_thread.run_sync(
        partial(
            plan_windows,
            probe=lambda window: anyio.from_thread.run(probe_window, client, window),
            since=since,
            vote_count_gte=vote_count_gte,
        )
    )


def seed_windows(db: Database, windows: Sequence[DiscoverWindow], *, priority: int = 0) -> int:
    """Queue page one of every planned window. Later pages queue themselves."""
    added = 0
    with db.write() as conn:
        for window in windows:
            added += enqueue_row(
                conn,
                "discover",
                params={"discover": window.params().to_dict(), "page": 1},
                priority=priority,
            )
    return added


def seed_export(db: Database, *, day: str | None = None, priority: int = -5) -> int:
    """Queue the daily id dump as a backstop enumerator."""
    with db.write() as conn:
        return enqueue_row(
            conn, "export", params={"day": day or latest_day(clock.now())}, priority=priority
        )


class Crawler:
    """Leases work, calls TMDB, writes rows. Killing it mid-flight costs one lease."""

    def __init__(
        self,
        db: Database,
        client: TMDBClient,
        *,
        queue: CrawlQueue | None = None,
        concurrency: int = 8,
        lease_s: int = 120,
        max_items: int | None = None,
        page_cap: int = DISCOVER_PAGE_CAP,
        export_ids: int = 5000,
        run_kind: str = "detail",
        default_source: str = "discover",
        poll_s: float = 0.05,
    ) -> None:
        self._db = db
        self._client = client
        self._queue = queue or CrawlQueue(db, lease_s=lease_s)
        self._concurrency = max(1, concurrency)
        self._max_items = max_items
        self._page_cap = page_cap
        self._export_ids = export_ids
        self._run_kind = run_kind
        self._default_source = default_source
        self._poll_s = poll_s
        self._taken = 0
        self._in_flight = 0
        self._ok = 0
        self._not_modified = 0
        self._err = 0
        self._dead = 0

    async def run(self) -> CrawlReport:
        """Drain the queue with the configured number of workers."""
        run_id = new_id("crw_")
        self._open_run(run_id)
        try:
            async with anyio.create_task_group() as group:
                for _ in range(self._concurrency):
                    group.start_soon(self._worker, run_id)
        except BaseException:
            self._close_run(run_id, "failed")
            raise
        self._close_run(run_id, "done")
        return CrawlReport(run_id, self._ok, self._not_modified, self._err, self._dead)

    async def _worker(self, run_id: str) -> None:
        while True:
            if self._max_items is not None and self._taken >= self._max_items:
                return
            leases = self._queue.lease(run_id, 1)
            if not leases:
                # Another worker may still be about to enqueue follow-up work.
                if self._in_flight == 0:
                    return
                await anyio.sleep(self._poll_s)
                continue
            self._taken += 1
            self._in_flight += 1
            try:
                await self._handle(leases[0])
            finally:
                self._in_flight -= 1

    async def _handle(self, lease: Lease) -> None:
        try:
            if lease.kind == "detail":
                await self._detail(lease)
            elif lease.kind == "discover":
                await self._discover(lease)
            elif lease.kind == "onehop":
                await self._onehop(lease)
            elif lease.kind == "export":
                await self._export(lease)
            else:
                self._queue.kill(lease.id, f"unknown kind {lease.kind}")
                self._dead += 1
        except (ProviderTimeout, ProviderUnavailable) as exc:
            self._retry(lease, str(exc))

    async def _detail(self, lease: Lease) -> None:
        if lease.tmdb_id is None:
            self._queue.kill(lease.id, "detail row with no tmdb_id")
            self._dead += 1
            return
        response = await self._client.movie(lease.tmdb_id, etag=self._known_etag(lease.tmdb_id))
        if response.status == 200 and response.payload is not None:
            self._store_movie(lease.tmdb_id, response)
            self._queue.complete(lease.id)
            self._ok += 1
        elif response.status == 304:
            self._touch(lease.tmdb_id)
            self._queue.complete(lease.id)
            self._not_modified += 1
        elif response.status == 429:
            self._rate_limited(lease, response)
        elif response.status == 404:
            self._queue.kill(lease.id, "404 from tmdb")
            self._dead += 1
        else:
            self._retry(lease, f"http {response.status}")

    async def _discover(self, lease: Lease) -> None:
        params = DiscoverParams.from_dict(dict(lease.params.get("discover", {})))
        page = int(lease.params.get("page", 1))
        response = await self._client.discover(params, page=page)
        if not self._list_ok(lease, response):
            return
        payload = response.payload or {}
        listing = MoviePage.model_validate(payload)
        stamp = now_iso()
        with self._db.write() as conn:
            store_raw(
                conn,
                entity="discover",
                payload=payload,
                params_sha=request_sha({"discover": params.to_dict(), "page": page}),
                etag=response.etag,
                fetched_at=stamp,
            )
            for summary in listing.results:
                register_summary(conn, summary, "discover", stamp)
            if page < min(listing.total_pages, self._page_cap):
                enqueue_row(
                    conn,
                    "discover",
                    params={"discover": params.to_dict(), "page": page + 1},
                    priority=1,
                )
        self._queue.complete(lease.id)
        self._ok += 1

    async def _onehop(self, lease: Lease) -> None:
        if lease.tmdb_id is None:
            self._queue.kill(lease.id, "onehop row with no tmdb_id")
            self._dead += 1
            return
        endpoint = str(lease.params.get("endpoint", "recommendations"))
        page = int(lease.params.get("page", 1))
        call = self._client.similar if endpoint == "similar" else self._client.recommendations
        response = await call(lease.tmdb_id, page=page)
        if not self._list_ok(lease, response):
            return
        payload = response.payload or {}
        listing = MoviePage.model_validate(payload)
        stamp = now_iso()
        with self._db.write() as conn:
            store_raw(
                conn,
                entity=endpoint,
                payload=payload,
                entity_id=lease.tmdb_id,
                params_sha=request_sha({"page": page}),
                etag=response.etag,
                fetched_at=stamp,
            )
            for summary in listing.results:
                register_summary(conn, summary, "onehop", stamp)
        self._queue.complete(lease.id)
        self._ok += 1

    async def _export(self, lease: Lease) -> None:
        day = str(lease.params.get("day") or latest_day(clock.now()))
        blob = await fetch_export(self._client.http, day)
        entries = sorted(iter_entries(blob), key=lambda e: e.popularity, reverse=True)
        kept = [e for e in entries if not e.adult and not e.video][: self._export_ids]
        stamp = now_iso()
        with self._db.write() as conn:
            # The dump itself is tens of megabytes of ids, so only the receipt is kept.
            store_raw(
                conn,
                entity="export",
                payload={"day": day, "n_lines": len(entries), "n_kept": len(kept)},
                params_sha=request_sha({"day": day}),
                fetched_at=stamp,
            )
            for entry in kept:
                stub_film(conn, entry.tmdb_id, entry.original_title or "", None, stamp)
                record_member(conn, entry.tmdb_id, "export", stamp)
                enqueue_row(conn, "detail", tmdb_id=entry.tmdb_id, priority=-1)
        self._queue.complete(lease.id)
        self._ok += 1

    def _list_ok(self, lease: Lease, response: TMDBResponse) -> bool:
        if response.status == 200 and response.payload is not None:
            return True
        if response.status == 304:
            self._queue.complete(lease.id)
            self._not_modified += 1
        elif response.status == 429:
            self._rate_limited(lease, response)
        elif response.status == 404:
            self._queue.kill(lease.id, "404 from tmdb")
            self._dead += 1
        else:
            self._retry(lease, f"http {response.status}")
        return False

    def _rate_limited(self, lease: Lease, response: TMDBResponse) -> None:
        delay = response.retry_after if response.retry_after is not None else BACKOFF_BASE_S
        # The server is pacing us, which is not this row's fault, so the attempt is refunded.
        self._queue.requeue(lease.id, delay_s=delay, error="429", count_attempt=False)
        self._err += 1

    def _retry(self, lease: Lease, error: str) -> None:
        self._err += 1
        if lease.attempts >= MAX_ATTEMPTS:
            self._queue.fail(lease.id, error)
            return
        delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (lease.attempts - 1))
        self._queue.requeue(lease.id, delay_s=delay, error=error)

    def _store_movie(self, tmdb_id: int, response: TMDBResponse) -> None:
        payload = response.payload or {}
        film = normalize_movie(payload)
        stamp = now_iso()
        with self._db.write() as conn:
            store_raw(
                conn,
                entity="movie",
                payload=payload,
                entity_id=tmdb_id,
                etag=response.etag,
                fetched_at=stamp,
            )
            write_film(conn, film, fetched_at=stamp, etag=response.etag)
            record_member(conn, tmdb_id, self._default_source, stamp)

    def _known_etag(self, tmdb_id: int) -> str | None:
        row = (
            self._db.read()
            .execute("select etag from films where tmdb_id = ? and detail_version > 0", (tmdb_id,))
            .fetchone()
        )
        return None if row is None or row["etag"] is None else str(row["etag"])

    def _touch(self, tmdb_id: int) -> None:
        with self._db.write() as conn:
            conn.execute("update films set fetched_at = ? where tmdb_id = ?", (now_iso(), tmdb_id))

    def _open_run(self, run_id: str) -> None:
        with self._db.write() as conn:
            conn.execute(
                "insert into crawl_runs (run_id, kind, started_at, status) values (?,?,?,'running')",
                (run_id, self._run_kind, now_iso()),
            )

    def _close_run(self, run_id: str, status: str) -> None:
        with self._db.write() as conn:
            conn.execute(
                "update crawl_runs set finished_at = ?, status = ?, n_ok = ?, n_304 = ?, "
                "n_err = ?, n_dead = ? where run_id = ?",
                (now_iso(), status, self._ok, self._not_modified, self._err, self._dead, run_id),
            )


def renormalize(db: Database, *, limit: int | None = None) -> int:
    """Rebuild every derived row from local payloads, with no API calls."""
    rows = (
        db.read()
        .execute(
            "select entity_id from tmdb_raw where entity = 'movie' and entity_id is not null "
            "order by entity_id" + (" limit ?" if limit is not None else ""),
            (limit,) if limit is not None else (),
        )
        .fetchall()
    )
    done = 0
    for row in rows:
        tmdb_id = int(row["entity_id"])
        stored = (
            db.read()
            .execute(
                "select etag, fetched_at, payload_z from tmdb_raw where entity = 'movie' "
                "and entity_id = ?",
                (tmdb_id,),
            )
            .fetchone()
        )
        if stored is None:
            continue
        payload = orjson.loads(zlib.decompress(stored["payload_z"]))
        film = normalize_movie(payload)
        with db.write() as conn:
            write_film(
                conn,
                film,
                fetched_at=str(stored["fetched_at"]),
                etag=None if stored["etag"] is None else str(stored["etag"]),
            )
        done += 1
    return done


def compact(db: Database) -> int:
    """Drop raw payloads for films outside the corpus, when disk matters more."""
    with db.write() as conn:
        cursor = conn.execute(
            "delete from tmdb_raw where entity = 'movie' and entity_id is not null "
            "and entity_id not in (select tmdb_id from corpus_members)"
        )
        return int(cursor.rowcount or 0)


def status(db: Database, *, runs: int = 5) -> CrawlStatus:
    """The state histogram plus the counts the crawl is judged on."""
    conn = db.read()
    queue = {
        str(r["state"]): int(r["n"])
        for r in conn.execute("select state, count(*) as n from crawl_queue group by state")
    }
    by_kind = {
        str(r["kind"]): int(r["n"])
        for r in conn.execute(
            "select kind, count(*) as n from crawl_queue where state in ('pending','leased') "
            "group by kind"
        )
    }
    # Page one of a window is the window, later pages are the same query continued.
    windows = conn.execute(
        "select count(*) as n_windows, coalesce(sum(state = 'done'), 0) as n_done "
        "from crawl_queue where kind = 'discover' "
        "and json_extract(params_json, '$.page') = 1"
    ).fetchone()
    counts = conn.execute(
        "select (select count(*) from films) as n_films, "
        "(select count(*) from films where detail_version > 0) as n_enriched, "
        "(select count(*) from corpus_members) as n_corpus, "
        "(select count(*) from tmdb_raw) as n_raw, "
        "(select coalesce(sum(length(payload_z)), 0) from tmdb_raw) as raw_bytes"
    ).fetchone()
    recent = conn.execute(
        "select run_id, kind, status, n_ok, n_304, n_err, n_dead from crawl_runs "
        "order by started_at desc limit ?",
        (runs,),
    ).fetchall()
    return CrawlStatus(
        queue=queue,
        by_kind=by_kind,
        n_windows=int(windows["n_windows"]),
        n_windows_done=int(windows["n_done"]),
        n_films=int(counts["n_films"]),
        n_enriched=int(counts["n_enriched"]),
        n_corpus=int(counts["n_corpus"]),
        n_raw=int(counts["n_raw"]),
        raw_bytes=int(counts["raw_bytes"]),
        runs=tuple(
            (
                str(r["run_id"]),
                str(r["kind"]),
                str(r["status"]),
                int(r["n_ok"]),
                int(r["n_304"]),
                int(r["n_err"]),
                int(r["n_dead"]),
            )
            for r in recent
        ),
    )
