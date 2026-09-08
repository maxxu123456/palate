"""`palate tmdb crawl`, `status`, `renormalize` and `compact`."""

from __future__ import annotations

from functools import partial

import anyio
import typer
from rich.console import Console
from rich.table import Table

from palate import paths
from palate.config import Settings, load_settings, require_secret, resolve_secret
from palate.db.connect import Database, open_database
from palate.providers.http import client_session
from palate.tmdb.client import TMDBClient
from palate.tmdb.crawl import (
    Crawler,
    CrawlReport,
    DiscoverWindow,
    compact,
    probe_plan,
    renormalize,
    seed_export,
    seed_history,
    seed_onehop,
    seed_windows,
    status,
)
from palate.tmdb.limiter import AIMDLimiter

console = Console()
tmdb_app = typer.Typer(help="Fill the film corpus from TMDB.", no_args_is_help=True)


def open_db(settings: Settings) -> Database:
    """Open the configured palate.db with migrations applied."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


def build_limiter(settings: Settings) -> AIMDLimiter:
    """One bucket shared by every worker in the process."""
    return AIMDLimiter(rate_per_s=settings.tmdb.rate_per_s, burst=settings.tmdb.burst)


async def plan_discover(settings: Settings) -> list[DiscoverWindow]:
    """Probe one window per release year, splitting whatever overflows the ceiling."""
    token = require_secret(settings.tmdb.token_env)
    async with client_session() as http:
        client = TMDBClient(
            token=token,
            client=http,
            limiter=build_limiter(settings),
            language=settings.tmdb.language,
        )
        return await probe_plan(
            client,
            since=settings.tmdb.discover_since,
            vote_count_gte=settings.tmdb.discover_vote_floor,
        )


async def drive(
    db: Database, settings: Settings, *, concurrency: int, limit: int | None
) -> CrawlReport:
    """Run the workers against a live client until the queue is empty."""
    token = require_secret(settings.tmdb.token_env)
    async with client_session() as http:
        client = TMDBClient(
            token=token,
            client=http,
            limiter=build_limiter(settings),
            language=settings.tmdb.language,
        )
        crawler = Crawler(
            db,
            client,
            concurrency=concurrency,
            lease_s=settings.tmdb.lease_s,
            max_items=limit,
        )
        return await crawler.run()


@tmdb_app.command()
def crawl(
    resume: bool = typer.Option(False, "--resume", help="Work the queue without seeding it."),
    limit: int | None = typer.Option(None, help="Stop after this many queue rows."),
    concurrency: int | None = typer.Option(None, help="Workers, defaults to the config value."),
    history: bool = typer.Option(True, help="Seed every film in your own history."),
    discover: bool = typer.Option(True, help="Plan and seed the discover windows."),
    onehop: bool = typer.Option(
        False, help="Seed recommendations and similar for top rated films."
    ),
    ids: bool = typer.Option(False, help="Seed from the daily id export as a backstop."),
) -> None:
    """Crawl TMDB into the local corpus. Killing it and rerunning finishes the set."""
    settings = load_settings()
    if resolve_secret(settings.tmdb.token_env) is None:
        console.print(f"{settings.tmdb.token_env} is not set, so there is nothing to crawl with")
        raise typer.Exit(code=2)
    db = open_db(settings)
    try:
        if not resume:
            queued = 0
            if history:
                queued += seed_history(db)
            if discover:
                windows = anyio.run(partial(plan_discover, settings))
                console.print(f"planned {len(windows)} discover windows")
                queued += seed_windows(db, windows)
            if onehop:
                queued += seed_onehop(db)
            if ids:
                queued += seed_export(db)
            console.print(f"queued {queued} new rows")
        report = anyio.run(
            partial(
                drive,
                db,
                settings,
                concurrency=concurrency or settings.tmdb.concurrency,
                limit=limit,
            )
        )
        console.print(
            f"{report.n_ok} fetched, {report.n_304} unchanged, "
            f"{report.n_err} retried, {report.n_dead} gone"
        )
        remaining = status(db).queue
        left = remaining.get("pending", 0) + remaining.get("leased", 0)
        if left:
            console.print(f"{left} rows left, run `palate tmdb crawl --resume` to finish")
    finally:
        db.close()


@tmdb_app.command("status")
def show_status() -> None:
    """Print the queue histogram and what the corpus holds."""
    settings = load_settings()
    db = open_db(settings)
    try:
        report = status(db)
    finally:
        db.close()
    queue = Table(box=None, pad_edge=False)
    for column in ("state", "rows"):
        queue.add_column(column)
    for name in ("pending", "leased", "done", "failed", "dead"):
        queue.add_row(name, str(report.queue.get(name, 0)))
    console.print(queue)
    if report.by_kind:
        waiting = ", ".join(f"{k} {n}" for k, n in sorted(report.by_kind.items()))
        console.print(f"waiting: {waiting}")
    if report.n_windows:
        console.print(f"{report.n_windows_done} of {report.n_windows} discover windows complete")
    console.print(
        f"{report.n_films} films, {report.n_enriched} enriched, {report.n_corpus} in the corpus"
    )
    console.print(f"{report.n_raw} raw payloads, {report.raw_bytes / 1e6:.1f} MB compressed")
    for run_id, kind, run_status, n_ok, n_304, n_err, n_dead in report.runs:
        console.print(
            f"{run_id} {kind} {run_status} ok {n_ok} unchanged {n_304} "
            f"retried {n_err} gone {n_dead}"
        )


@tmdb_app.command("renormalize")
def run_renormalize(
    limit: int | None = typer.Option(None, help="Rebuild only this many films."),
) -> None:
    """Rebuild every derived row from the stored payloads, with no API calls."""
    settings = load_settings()
    db = open_db(settings)
    try:
        done = renormalize(db, limit=limit)
    finally:
        db.close()
    console.print(f"rebuilt {done} films from local payloads")


@tmdb_app.command("compact")
def run_compact() -> None:
    """Drop raw payloads for films outside the corpus."""
    settings = load_settings()
    db = open_db(settings)
    try:
        dropped = compact(db)
    finally:
        db.close()
    console.print(f"dropped {dropped} raw payloads")


def register(app: typer.Typer) -> None:
    """Attach the tmdb command group to the CLI."""
    app.add_typer(tmdb_app, name="tmdb")
