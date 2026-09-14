"""`palate index build`, `list`, `activate`, `verify` and `drop`."""

from __future__ import annotations

from functools import partial

import anyio
import typer
from rich.console import Console
from rich.table import Table

from palate import paths
from palate.config import Settings, load_settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.index import fts, verify
from palate.index.build import BuildReport, build
from palate.providers.registry import build_embedder

console = Console()
index_app = typer.Typer(help="Build and manage the embedding index.", no_args_is_help=True)


def open_db(settings: Settings) -> Database:
    """Open palate.db with sqlite-vec loaded, which the index needs."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


async def run_build(
    db: Database, settings: Settings, *, rebuild: bool, only_missing: bool
) -> BuildReport:
    """Build against whichever embedding provider the settings name."""
    embedder = build_embedder(settings)
    try:
        return await build(db, embedder, rebuild=rebuild, only_missing=only_missing)
    finally:
        await embedder.aclose()


async def provider_drift(settings: Settings, record: verify.IndexRecord) -> str:
    """The mismatch report when the configured provider did not build this index."""
    embedder = build_embedder(settings)
    try:
        return await verify.drift(record, embedder)
    except PalateError as exc:
        console.print(f"  could not ask the embedding provider ({exc})")
        return ""
    finally:
        await embedder.aclose()


@index_app.command("build")
def run(
    embed_provider: str | None = typer.Option(
        None, "--embed-provider", help="Override the provider."
    ),
    embed_model: str | None = typer.Option(None, "--embed-model", help="Override the model."),
    no_credits: bool = typer.Option(
        False, "--no-credits", help="Render documents without director and cast names."
    ),
    rebuild: bool = typer.Option(False, help="Re-embed every document, not only the movers."),
) -> None:
    """Render documents, embed them, and activate the resulting index."""
    overrides = {k: v for k, v in (("provider", embed_provider), ("model", embed_model)) if v}
    settings = load_settings()
    if overrides:
        settings = load_settings(embed=settings.embed.model_copy(update=overrides).model_dump())
    db = open_db(settings)
    try:
        docs = fts.rebuild(db, include_credits=not no_credits)
        console.print(f"{docs.n_written} of {docs.n_seen} documents rewritten")
        for kind, n in sorted(docs.by_kind.items()):
            console.print(f"  {kind}: {n}")
        report = anyio.run(
            partial(run_build, db, settings, rebuild=rebuild, only_missing=not rebuild)
        )
    finally:
        db.close()
    console.print(
        f"index {report.index_id} holds {report.n_vectors} vectors at dim {report.dim}, "
        f"{report.n_embedded} embedded and {report.n_cached} from cache"
    )
    if report.activated:
        console.print(f"{report.table_name} is now the active index")


@index_app.command("list")
def show_list() -> None:
    """Every index ever built. The runtime tables are not in the migrations directory."""
    settings = load_settings()
    db = open_db(settings)
    try:
        records = verify.listing(db)
        active = verify.active_id(db)
    finally:
        db.close()
    table = Table(box=None, pad_edge=False)
    for column in ("", "index", "provider", "model", "dim", "vectors", "status", "built"):
        table.add_column(column)
    for record in records:
        table.add_row(
            "*" if record.index_id == active else " ",
            record.index_id,
            record.fingerprint.provider,
            record.fingerprint.model_id,
            str(record.fingerprint.dim),
            str(record.n_vectors),
            record.status,
            record.completed_at or "",
        )
    console.print(table)


@index_app.command("activate")
def run_activate(
    index_id: str = typer.Argument(..., help="Index id from `palate index list`."),
) -> None:
    """Point queries back at an index that was built earlier."""
    settings = load_settings()
    db = open_db(settings)
    try:
        verify.activate(db, index_id)
    finally:
        db.close()
    console.print(f"{index_id} is now active")


@index_app.command("verify")
def run_verify(
    index_id: str | None = typer.Argument(None, help="Defaults to the active index."),
    provider: bool = typer.Option(
        True, "--provider/--no-provider", help="Also ask the configured embedding provider."
    ),
) -> None:
    """Count, dimension, orphans, and whether the current provider built this index."""
    settings = load_settings()
    db = open_db(settings)
    try:
        target = index_id or verify.active_id(db)
        if target is None:
            console.print("no index is active, run: palate index build")
            raise typer.Exit(code=2)
        report = verify.verify(db, target)
        record = verify.load(db, target)
    finally:
        db.close()
    console.print(
        f"{report.index_id} {report.status}: {report.n_present} of {report.n_expected} films "
        f"at dim {report.dim}"
    )
    for problem in report.problems:
        console.print(f"  {problem}")
    moved = (
        anyio.run(partial(provider_drift, settings, record))
        if provider and record is not None
        else ""
    )
    if moved:
        console.print(moved)
    if not report.ok or moved:
        raise typer.Exit(code=1)


@index_app.command("drop")
def run_drop(index_id: str = typer.Argument(..., help="Index id to delete.")) -> None:
    """Delete an index and its vector table. The active one is refused."""
    settings = load_settings()
    db = open_db(settings)
    try:
        dropped = verify.drop(db, index_id)
    finally:
        db.close()
    console.print(f"dropped {index_id}" if dropped else f"no index {index_id}")


def register(app: typer.Typer) -> None:
    """Attach the index command group to the CLI."""
    app.add_typer(index_app, name="index")
