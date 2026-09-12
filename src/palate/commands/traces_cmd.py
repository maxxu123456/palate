"""`palate traces`, the three surfaces over traces.db."""

from __future__ import annotations

import time
from functools import partial

import anyio
import typer
from rich.console import Console
from rich.table import Table

from palate import paths
from palate.config import Settings, load_settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.obs import report
from palate.obs.replay import ReplayResult, replay
from palate.providers.http import client_session
from palate.providers.registry import build_chat

console = Console()
traces_app = typer.Typer(help="Look at what the agent actually did.", no_args_is_help=True)

POLL_S = 1.0


def register(app: typer.Typer) -> None:
    """Attach the traces command group to the CLI."""
    app.add_typer(traces_app, name="traces")


def open_traces(settings: Settings) -> Database:
    """traces.db, which has its own migrations and never loads sqlite-vec."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(
        resolved.traces_db, migrations=paths.trace_migrations_dir(), load_vec=False
    )


@traces_app.command("list")
def run_list(
    since: str = typer.Option("24h", "--since", help="Window, as 30m, 24h or 7d."),
    kind: str = typer.Option("", "--kind", help="Only runs of this kind, such as chat."),
    limit: int = typer.Option(50, help="How many runs to print."),
) -> None:
    """What ran."""
    db = open_traces(load_settings())
    try:
        rows = report.runs(db, window=since, kind=kind, limit=limit)
    finally:
        db.close()
    if not rows:
        console.print(f"no runs in the last {since}")
        return
    table = Table(box=None, pad_edge=False)
    for column, align in (
        ("run", "left"),
        ("kind", "left"),
        ("started", "left"),
        ("ms", "right"),
        ("turns", "right"),
        ("in", "right"),
        ("out", "right"),
        ("usd", "right"),
        ("status", "left"),
    ):
        table.add_column(column, justify=align)  # type: ignore[arg-type]
    for row in rows:
        table.add_row(
            row.run_id,
            row.kind,
            row.started_at[:19],
            f"{row.latency_ms:.0f}" if row.latency_ms else "",
            str(row.turns),
            str(row.tokens_in),
            str(row.tokens_out),
            f"{row.cost_usd:.4f}" + ("" if row.cost_complete else " ?"),
            row.status,
        )
    console.print(table)


@traces_app.command("show")
def run_show(run_id: str = typer.Argument(..., help="The run to open.")) -> None:
    """The span tree, with a proportional bar per span."""
    db = open_traces(load_settings())
    try:
        spans = report.tree(db, run_id)
    finally:
        db.close()
    if not spans:
        console.print(f"no spans under {run_id}")
        raise typer.Exit(code=2)
    for span, bar in report.bars(spans):
        indent = "  " * span.depth
        timing = f"{span.latency_ms:.0f} ms" if span.latency_ms else ""
        failed = f"  {span.error_type}" if span.status == "error" else ""
        console.print(f"{indent}{span.kind:<9} {span.name:<24} {timing:>9}  {bar}{failed}")


@traces_app.command("costs")
def run_costs(
    since: str = typer.Option("7d", "--since", help="Window, as 30m, 24h or 7d."),
    group_by: str = typer.Option("model", "--group-by", help="model, provider or run."),
) -> None:
    """The rollup, with the unknown-model tokens on their own line."""
    db = open_traces(load_settings())
    try:
        rows = report.costs(db, window=since, group_by=group_by)
    finally:
        db.close()
    if not rows:
        console.print(f"no priced calls in the last {since}")
        return
    table = Table(box=None, pad_edge=False)
    for column in (group_by, "calls", "in", "out", "usd"):
        table.add_column(column, justify="right" if column != group_by else "left")
    for row in rows:
        table.add_row(
            row.bucket, str(row.calls), str(row.tokens_in), str(row.tokens_out), f"{row.usd:.4f}"
        )
    console.print(table)
    unknown = sum(r.unknown_tokens for r in rows)
    if unknown:
        calls = sum(r.unknown_calls for r in rows)
        console.print(f"unpriced: {calls} calls over {unknown} tokens, not in the total above")


@traces_app.command("status")
def run_status() -> None:
    """Queue losses, table sizes and the retention edge."""
    settings = load_settings()
    db = open_traces(settings)
    try:
        found = report.status(db)
    finally:
        db.close()
    console.print(f"{found.runs} runs, {found.spans} spans, {found.llm_calls} model calls")
    console.print(f"{found.db_bytes / 1_048_576:.1f} MB, oldest run {found.oldest or 'none'}")
    console.print(f"{found.dropped} rows dropped, retention {settings.trace.retain_days} days")
    console.print(
        f"payloads {settings.trace.payloads}, tracing {'on' if settings.trace.enabled else 'off'}"
    )


@traces_app.command("tail")
def run_tail(
    kind: str = typer.Option("", "--kind", help="Only runs of this kind."),
) -> None:
    """Follow new runs as they finish. Stop with ctrl-c."""
    db = open_traces(load_settings())
    seen: set[str] = set()
    try:
        while True:
            for row in reversed(report.runs(db, window="5m", kind=kind, limit=20)):
                if row.run_id in seen:
                    continue
                seen.add(row.run_id)
                console.print(
                    f"{row.started_at[:19]}  {row.kind:<6} {row.run_id}  "
                    f"{row.turns} turns  {row.status}"
                )
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        return
    finally:
        db.close()


@traces_app.command("gc")
def run_gc(
    older_than: str = typer.Option("30d", "--older-than", help="Delete runs older than this."),
) -> None:
    """Delete old runs and everything under them. Only ever touches traces.db."""
    db = open_traces(load_settings())
    try:
        removed = report.gc(db, older_than=older_than)
    finally:
        db.close()
    console.print(f"deleted {removed} runs older than {older_than}")


@traces_app.command("replay")
def run_replay(
    span_id: str = typer.Argument(..., help="The llm span to reissue."),
    prompt: str = typer.Option("", "--prompt", help="Replace the system prompt with this text."),
) -> None:
    """Reissue a stored call against the current provider and diff the output."""
    settings = load_settings()
    db = open_traces(settings)
    try:
        result = anyio.run(partial(_replay, settings, db, span_id, prompt or None))
    except PalateError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    finally:
        db.close()
    console.print(f"{result.original_model} -> {result.new_model}")
    console.print(
        f"tokens {result.tokens_delta[0]:+d} in, {result.tokens_delta[1]:+d} out, "
        f"cost {result.cost_delta:+.4f}"
    )
    console.print(result.diff or "identical")


async def _replay(
    settings: Settings, db: Database, span_id: str, prompt: str | None
) -> ReplayResult:
    async with client_session() as http:
        provider = build_chat(settings, client=http)
        try:
            return await replay(db, span_id, provider=provider, prompt_override=prompt)
        finally:
            await provider.aclose()
