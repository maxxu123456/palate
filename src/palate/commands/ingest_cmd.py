"""`palate ingest <export.zip>` and `palate ingest review`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from palate import paths
from palate.clock import now_iso
from palate.config import load_settings
from palate.db.connect import Database, open_database
from palate.ingest.letterboxd import ImportReport, import_export
from palate.ingest.resolve import record_manual

console = Console()


def open_db() -> Database:
    """Open the configured palate.db with migrations applied."""
    settings = load_settings()
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


def run_import(export: Path) -> ImportReport:
    """Import an export zip and print what landed."""
    if not export.is_file():
        raise typer.BadParameter(f"no such export: {export}")
    db = open_db()
    try:
        report = import_export(db, export)
    finally:
        db.close()
    console.print(
        f"imported {report.n_rows} films from {export.name}, "
        f"{report.n_resolved} resolved ({report.match_rate:.0%}), "
        f"{report.n_needs_review} need review"
    )
    if report.n_unresolved:
        console.print(
            f"{report.n_unresolved} rows kept in unmatched_export_row. "
            "Run `palate ingest review` to see them."
        )
    return report


def run_review(uri: str | None, tmdb_id: int | None, limit: int) -> None:
    """List the titles waiting on a decision, or record one."""
    db = open_db()
    try:
        if uri is not None or tmdb_id is not None:
            if uri is None or tmdb_id is None:
                raise typer.BadParameter("--uri and --tmdb-id go together")
            record_manual(db, uri, tmdb_id, resolved_at=now_iso())
            console.print(f"{uri} is now tmdb {tmdb_id}")
            return
        rows = (
            db.read()
            .execute(
                "select letterboxd_uri, title, year, confidence, method, candidates_json "
                "from title_resolutions where needs_review = 1 order by confidence desc limit ?",
                (limit,),
            )
            .fetchall()
        )
        if not rows:
            console.print("nothing waiting on a decision")
            return
        table = Table(box=None, pad_edge=False)
        for column in ("title", "year", "conf", "method", "candidates"):
            table.add_column(column)
        for row in rows:
            candidates = json.loads(row["candidates_json"] or "[]")
            shown = ", ".join(f"{c[0]} {c[1]} ({c[2]})" for c in candidates[:3]) or "none"
            table.add_row(
                str(row["title"]),
                str(row["year"] or ""),
                f"{float(row['confidence']):.2f}",
                str(row["method"]),
                shown,
            )
        console.print(table)
        console.print("pick one with: palate ingest review --uri <uri> --tmdb-id <id>")
    finally:
        db.close()


def register(app: typer.Typer) -> None:
    """Attach the ingest command to the CLI."""

    @app.command()
    def ingest(
        target: str = typer.Argument(
            ..., metavar="EXPORT|review", help="Path to the export zip, or the word review."
        ),
        uri: str | None = typer.Option(None, help="Letterboxd URI to decide, with --tmdb-id."),
        tmdb_id: int | None = typer.Option(None, help="TMDB id to pin that URI to."),
        limit: int = typer.Option(20, help="How many pending titles to list."),
    ) -> None:
        """Import a Letterboxd export, or review the titles that need a decision."""
        # One argument covers both spellings, because `palate ingest review` reads better
        # than a flag and a zip is never named review.
        if target == "review":
            run_review(uri, tmdb_id, limit)
        else:
            run_import(Path(target).expanduser())
