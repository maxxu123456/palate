"""`palate recommend`, the unconditioned list and the query conditioned one."""

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
from palate.providers.http import client_session
from palate.providers.registry import build_embedder
from palate.retrieval.candidates import HardFilters
from palate.retrieval.recommend import LocalRecommender, RecommendRequest, RecommendResponse

console = Console()


def open_db(settings: Settings) -> Database:
    """Open palate.db with sqlite-vec loaded, which reading the vectors back needs."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


async def _recommend(db: Database, settings: Settings, req: RecommendRequest) -> RecommendResponse:
    """A query needs an embedder in the index's own space. An empty one needs nothing."""
    async with client_session() as http:
        embedder = build_embedder(settings, client=http) if req.query_text else None
        try:
            local = LocalRecommender(db, embedder=embedder, retrieval=settings.retrieval)
            return await local.recommend(req)
        finally:
            if embedder is not None:
                await embedder.aclose()


def _filters(
    year_min: int | None,
    year_max: int | None,
    runtime_max: int | None,
    language: str,
    not_language: str,
) -> HardFilters:
    return HardFilters(
        year_min=year_min,
        year_max=year_max,
        runtime_max=runtime_max,
        include_languages=_codes(language),
        exclude_languages=_codes(not_language),
    )


def _codes(raw: str) -> frozenset[str]:
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def register(app: typer.Typer) -> None:
    """Attach the recommend command to the CLI."""
    app.command("recommend")(run)


def run(
    query: str = typer.Argument("", help="What you are in the mood for. Empty for no query."),
    n: int = typer.Option(10, "-n", help="How many films to return."),
    offset: int = typer.Option(0, help="Skip this many before listing."),
    year_min: int | None = typer.Option(None, "--year-min", help="Nothing released before this."),
    year_max: int | None = typer.Option(None, "--year-max", help="Nothing released after this."),
    runtime_max: int | None = typer.Option(None, "--runtime-max", help="Minutes, at most."),
    language: str = typer.Option("", "--language", help="Original languages, comma separated."),
    not_language: str = typer.Option(
        "", "--not-language", help="Original languages to drop, comma separated."
    ),
    diversity: bool = typer.Option(True, help="Mode slots, hard caps and the novelty cap."),
    explain: bool = typer.Option(True, help="Print the evidence under each film."),
) -> None:
    """Rank the corpus against your taste, with evidence for every row."""
    settings = load_settings()
    req = RecommendRequest(
        query_text=query or None,
        n=n,
        offset=offset,
        filters=_filters(year_min, year_max, runtime_max, language, not_language),
        diversity="auto" if diversity else "off",
        explain=explain,
    )
    db = open_db(settings)
    try:
        answer = anyio.run(partial(_recommend, db, settings, req))
    except PalateError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    finally:
        db.close()
    _print(answer, explain=explain)


def _print(answer: RecommendResponse, *, explain: bool) -> None:
    if not answer.films:
        hint = answer.diagnostics.most_restrictive_clause
        console.print("nothing came back" + (f", {hint} removed the most" if hint else ""))
        return
    table = Table(box=None, pad_edge=False)
    for column, align in (
        ("", "right"),
        ("title", "left"),
        ("year", "right"),
        ("director", "left"),
        ("min", "right"),
        ("score", "right"),
    ):
        table.add_column(column, justify=align)  # type: ignore[arg-type]
    for rank, film in enumerate(answer.films, start=1 + answer.diagnostics.reranked):
        table.add_row(
            str(rank),
            film.title,
            str(film.year or ""),
            ", ".join(film.directors[:2]),
            str(film.runtime or ""),
            f"{film.score:+.2f}",
        )
    console.print(table)
    if explain:
        for film in answer.films:
            console.print(f"{film.title}")
            for row in film.evidence:
                console.print(f"    {row.text}")
    _summary(answer)


def _summary(answer: RecommendResponse) -> None:
    diagnostics = answer.diagnostics
    console.print(
        f"{diagnostics.after_filter} of {diagnostics.considered} films passed the filters, "
        f"{diagnostics.excluded_watched} already watched, pool {answer.pool_size}"
    )
    console.print(
        f"{answer.condition}, {diagnostics.prefilter_path}, "
        f"{diagnostics.elapsed_ms:.0f} ms"
        + (f", {', '.join(answer.degraded)}" if answer.degraded else "")
    )
