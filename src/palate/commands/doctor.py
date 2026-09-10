"""`palate doctor`: one line per check, each with the command that fixes it."""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from functools import partial

import anyio
import typer
from rich.console import Console

from palate import paths
from palate.config import Settings, load_settings, resolve_secret
from palate.db.connect import Database, open_database
from palate.db.sqlvec import probe
from palate.errors import PalateError
from palate.extras import have
from palate.hf.models import load as load_models
from palate.index import verify
from palate.providers.base import EmbeddingProvider
from palate.providers.http import client_session
from palate.providers.registry import build_chat, build_embedder

console = Console()


@dataclass(frozen=True, slots=True)
class Check:
    """One line of the report."""

    name: str
    ok: bool
    detail: str
    fix: str = ""


def open_db(settings: Settings) -> Database:
    """The same database every other command opens."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


def python_check() -> Check:
    """python.org macOS builds ship extension loading disabled, which hides sqlite-vec."""
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return Check(
            "sqlite extensions",
            False,
            "this Python was built without extension loading",
            "uv python install 3.13",
        )
    return Check("sqlite extensions", True, f"python {sys.version.split()[0]}")


def storage_checks(db: Database) -> list[Check]:
    """Linked SQLite, FTS5, sqlite-vec, corpus size and the active index."""
    conn = db.read()
    linked = str(conn.execute("select sqlite_version()").fetchone()[0])
    # The sqlite3 CLI is a different library, so its version proves nothing about this.
    checks = [Check("sqlite linked into python", True, linked)]
    try:
        capability = probe(conn)
    except PalateError as exc:
        checks.append(Check("sqlite-vec", False, str(exc), "uv python install 3.13"))
        return checks
    checks.append(Check("fts5", capability.fts5, "compiled in" if capability.fts5 else "absent"))
    checks.append(
        Check(
            "sqlite-vec",
            True,
            f"{capability.vec_version}, {capability.max_metadata_columns} metadata columns",
        )
    )
    films = int(conn.execute("select count(*) from films").fetchone()[0])
    eligible = int(
        conn.execute("select count(*) from corpus_members where eligible = 1").fetchone()[0]
    )
    docs = int(conn.execute("select count(*) from film_docs").fetchone()[0])
    checks.append(
        Check(
            "corpus",
            films > 0,
            f"{films} films, {eligible} eligible, {docs} documents",
            "palate tmdb crawl",
        )
    )
    checks.append(index_check(db))
    return checks


def index_check(db: Database) -> Check:
    """Whether anything can answer a query at all."""
    index_id = verify.active_id(db)
    if index_id is None:
        return Check("embedding index", False, "none active", "palate index build")
    report = verify.verify(db, index_id)
    return Check(
        "embedding index",
        report.ok,
        f"{index_id} {report.status}, {report.n_present} vectors at dim {report.dim}",
        "palate index verify",
    )


def key_checks(settings: Settings) -> list[Check]:
    """Which secrets are present, by env var name, never by value."""
    names = [settings.tmdb.token_env, settings.chat.api_key_env, settings.embed.api_key_env]
    return [
        Check(
            f"env {name}",
            resolve_secret(name) is not None,
            "set" if resolve_secret(name) else "missing",
        )
        for name in dict.fromkeys(names)
    ]


def model_checks() -> list[Check]:
    """Which Hub aliases are pinned, and whether the local extra is installed."""
    checks = [
        Check(
            f"model {alias}",
            spec.pinned,
            spec.repo_id if spec.pinned else f"{spec.repo_id}, not pinned",
            f"palate models pull {alias}",
        )
        for alias, spec in sorted(load_models().items())
    ]
    torch = have("torch")
    checks.append(
        Check("local extra", torch, "torch present" if torch else "absent", "uv sync --extra local")
    )
    return checks


async def provider_checks(settings: Settings, db: Database) -> list[Check]:
    """Reachability for both providers, plus the fingerprint the index was built with."""
    async with client_session() as http:
        chat = build_chat(settings, client=http)
        embedder = build_embedder(settings, client=http)
        try:
            chat_health = await chat.health()
            embed_health = await embedder.health()
            return [
                Check(f"chat ({chat.name})", chat_health.ok, chat_health.detail),
                Check(f"embeddings ({embedder.provider})", embed_health.ok, embed_health.detail),
                await fingerprint_check(db, embedder),
            ]
        finally:
            await chat.aclose()
            await embedder.aclose()


async def fingerprint_check(db: Database, embedder: EmbeddingProvider) -> Check:
    """The active index against the provider configured right now."""
    index_id = verify.active_id(db)
    record = None if index_id is None else verify.load(db, index_id)
    if record is None:
        return Check("index fingerprint", False, "no active index", "palate index build")
    try:
        moved = await verify.drift(record, embedder)
    except PalateError as exc:
        return Check("index fingerprint", False, f"cannot ask the provider ({exc})")
    if not moved:
        return Check("index fingerprint", True, f"matches {record.index_id}")
    return Check("index fingerprint", False, moved.splitlines()[0], "palate index build")


def run_checks(settings: Settings) -> list[Check]:
    """Every check, in the order a first run needs them."""
    checks = [python_check()]
    db = open_db(settings)
    try:
        checks.extend(storage_checks(db))
        checks.extend(key_checks(settings))
        checks.extend(model_checks())
        checks.extend(anyio.run(partial(provider_checks, settings, db)))
    finally:
        db.close()
    return checks


def register(app: typer.Typer) -> None:
    """Attach `palate doctor` to the CLI."""

    @app.command()
    def doctor() -> None:
        """Check every moving part and print the command that fixes each one."""
        settings = load_settings()
        failures = 0
        for check in run_checks(settings):
            console.print(f"{'ok  ' if check.ok else 'fail'} {check.name}: {check.detail}")
            if not check.ok:
                failures += 1
                if check.fix:
                    console.print(f"     try: {check.fix}")
        if failures:
            raise typer.Exit(code=1)
