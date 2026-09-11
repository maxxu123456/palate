"""`palate eval`: cut the holdout, run the matrix, print the table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import anyio
import typer
from rich.console import Console

from palate import paths
from palate.config import Settings, load_settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.eval import report as reporting
from palate.eval.harness import (
    EvalCondition,
    EvalContext,
    deltas,
    persist,
    publish_weights,
    run_matrix,
)
from palate.eval.labels import labels_of
from palate.eval.queries import build_review_queries, review_survival
from palate.eval.split import (
    Split,
    SplitSpec,
    build_split,
    detect_catalogue_days,
    freeze_split,
    load_corpus_ids,
    load_regions,
    load_split,
    mark_catalogue_days,
    ratings_fingerprint,
)
from palate.eval.systems import BY_NAME, FULL, SMOKE, SystemConfig, resolve
from palate.providers.http import client_session
from palate.providers.registry import build_embedder
from palate.taste import profile as taste

console = Console()

DEFAULT_SPLIT = "rolling"

app = typer.Typer(name="eval", help="Holdouts, ablations and the table.", no_args_is_help=True)
split_app = typer.Typer(name="split", help="Cut and freeze a holdout.", no_args_is_help=True)
app.add_typer(split_app)


def register(parent: typer.Typer) -> None:
    """Attach the eval command group to the CLI."""
    parent.add_typer(app)


def open_db(settings: Settings) -> Database:
    """Open palate.db with sqlite-vec loaded, which reading the vectors back needs."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


def _spec(settings: Settings, name: str) -> SplitSpec:
    return SplitSpec(
        name=name,
        cuts=settings.eval.cuts,
        catalogue_day_threshold=settings.eval.catalogue_day_threshold,
        min_reliable=settings.eval.min_reliable,
        min_test_per_fold=settings.eval.min_test_per_fold,
        seed=settings.eval.seed,
    )


def _frozen(db: Database, name: str) -> Split:
    rated = taste.load_rated(db.read())
    return load_split(db, name, ratings_sha256=ratings_fingerprint(rated))


@split_app.command("build")
def split_build(
    name: str = typer.Option(DEFAULT_SPLIT, help="Name to freeze the split under."),
    min_reliable: int | None = typer.Option(None, help="Override the temporal protocol floor."),
) -> None:
    """Detect catalogue days, cut the folds, and freeze them against the export's hash."""
    settings = load_settings()
    db = open_db(settings)
    try:
        spec = _spec(settings, name)
        if min_reliable is not None:
            spec = replace(spec, min_reliable=min_reliable)
        rated = taste.load_rated(db.read())
        marked = mark_catalogue_days(
            db, detect_catalogue_days(rated, threshold=spec.catalogue_day_threshold)
        )
        rated = taste.load_rated(db.read())
        split = build_split(
            rated,
            load_corpus_ids(db.read()),
            spec,
            region_of=load_regions(db.read()),
            relevance=labels_of,
        )
        freeze_split(split, db)
    except PalateError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    finally:
        db.close()
    if split.degraded and split.degraded_reason:
        console.print(split.degraded_reason)
    console.print(
        f"{split.name}: {split.strategy}, {len(split.folds)} folds, "
        f"{split.n_reliable} reliably dated, {marked} on a catalogue day"
    )
    console.print(f"test per fold {' / '.join(str(n) for n in split.test_per_fold)}")
    console.print(f"test coverage {split.test_coverage:.2f}")


@app.command("run")
def run(
    suite: str = typer.Option("retrieval", help="retrieval or query."),
    split: str = typer.Option(DEFAULT_SPLIT, "--split", help="Which frozen split to run."),
    systems: str = typer.Option("", help="Comma separated arm names. Empty runs every arm."),
    smoke: bool = typer.Option(False, help="Only the five arms that carry the argument."),
    fit: bool = typer.Option(True, help="Fit fusion weights on the validation slices first."),
    force: bool = typer.Option(False, help="Rerun configurations that are already stored."),
    resamples: int | None = typer.Option(None, help="Bootstrap resamples, overriding config."),
) -> None:
    """Run every arm on every fold and store the rankings, the metrics and the deltas."""
    settings = load_settings()
    db = open_db(settings)
    names = SMOKE if smoke else tuple(n.strip() for n in systems.split(",") if n.strip())
    try:
        frozen = _frozen(db, split)
        chosen = resolve(names or None)
        anyio.run(
            partial(
                _run,
                db,
                settings,
                frozen,
                chosen,
                suite=suite,
                fit=fit,
                force=force,
                resamples=resamples or settings.eval.bootstrap_resamples,
            )
        )
    except PalateError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    finally:
        db.close()


async def _run(
    db: Database,
    settings: Settings,
    split: Split,
    arms: Sequence[SystemConfig],
    *,
    suite: str,
    fit: bool,
    force: bool,
    resamples: int,
) -> None:
    async with client_session() as http:
        embedder = build_embedder(settings, client=http) if suite == "query" else None
        try:
            ctx = EvalContext(
                db=db,
                split=split,
                seed=settings.eval.seed,
                retrieval=settings.retrieval,
                embedder=embedder,
                resamples=resamples,
            )
            if fit and suite == "retrieval":
                published = publish_weights(ctx, FULL)
                console.print(
                    "fusion weights: "
                    + ("not fitted" if published is None else published.fitted_on)
                )
            cases = (
                {f.fold: build_review_queries(db, f) for f in split.folds}
                if suite == "query"
                else None
            )
            conditions: tuple[EvalCondition, ...] = (
                ("query_review",) if suite == "query" else ("unconditioned",)
            )
            results = await run_matrix(arms, ctx, conditions=conditions, cases=cases, force=force)
            persist(ctx, BY_NAME, results)
            if suite == "retrieval":
                for reference in ("full", "director_affinity"):
                    deltas(ctx, results, reference=reference)
            console.print(f"{len(results)} runs stored for split {split.name}")
        finally:
            if embedder is not None:
                await embedder.aclose()


@app.command("report")
def report(
    split: str = typer.Option(DEFAULT_SPLIT, "--split", help="Which frozen split to report."),
    fmt: str = typer.Option("markdown", "--format", help="markdown is the only format."),
    out: str = typer.Option("", help="Write the report here as well as printing it."),
    into: str = typer.Option("", help="Paste the report between the markers in this file."),
) -> None:
    """Render the table from the stored runs. Nothing here is computed, only read."""
    settings = load_settings()
    db = open_db(settings)
    try:
        frozen = _frozen(db, split)
        surviving = _surviving(db, frozen)
        table = reporting.render(db, frozen, surviving=surviving)
    except PalateError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    finally:
        db.close()
    if fmt != "markdown":
        console.print(f"unknown format {fmt!r}, only markdown is rendered")
        raise typer.Exit(code=2)
    if out:
        target = Path(out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(table, encoding="utf-8")
    if into and not reporting.paste_into(Path(into).expanduser(), table):
        console.print(f"{into} has no {reporting.START_MARKER} markers, nothing pasted")
        raise typer.Exit(code=2)
    console.print(table, markup=False, highlight=False)


def _surviving(db: Database, split: Split) -> tuple[int, int]:
    counted = [review_survival(db, fold) for fold in split.folds]
    return sum(c[0] for c in counted), sum(c[1] for c in counted)
