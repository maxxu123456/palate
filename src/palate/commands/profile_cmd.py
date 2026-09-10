"""`palate profile build` and `show`."""

from __future__ import annotations

from datetime import date

import typer
from rich.console import Console
from rich.table import Table

from palate import paths
from palate.config import Settings, load_settings
from palate.db.connect import Database, open_database
from palate.taste import profile as taste
from palate.taste.profile import TasteProfile
from palate.taste.ridge import MIN_RIDGE_N

console = Console()
profile_app = typer.Typer(help="Fit and inspect the taste profile.", no_args_is_help=True)

FACETS = ("director", "keyword", "decade", "language")


def open_db(settings: Settings) -> Database:
    """Open palate.db with sqlite-vec loaded, which reading the vectors back needs."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


@profile_app.command("build")
def run_build(
    alpha: float = typer.Option(0.6, help="Weight on the z score against the residual."),
    seed: int = typer.Option(0, help="Seed for the mode search."),
    cutoff: str | None = typer.Option(
        None, help="Only films watched before this date, as YYYY-MM-DD."
    ),
) -> None:
    """Fit a profile in the active index's space and write it down."""
    settings = load_settings()
    db = open_db(settings)
    try:
        built = taste.build(
            db, alpha=alpha, seed=seed, cutoff=date.fromisoformat(cutoff) if cutoff else None
        )
    finally:
        db.close()
    console.print(
        f"{built.profile_id} is {built.tier} on {built.n_rated} ratings, "
        f"{built.n_reliable_dated} of them reliably dated"
    )
    console.print(
        f"  {len(built.modes)} modes and {len(built.anti_modes)} anti modes, "
        f"generic model explains {built.calibrator['r2']:.2f} of the rating"
    )
    if built.direction is None:
        console.print(f"  no ridge direction, that needs {MIN_RIDGE_N} ratings")
    else:
        console.print(
            f"  ridge over {len(built.direction.feature_names)} features, "
            f"lambda {built.direction.lam:.3g}, leave one out r2 {built.direction.loocv_r2:.3f}"
        )


@profile_app.command("show")
def run_show(
    profile_id: str | None = typer.Argument(None, help="Defaults to the newest fresh profile."),
) -> None:
    """Modes, top entities, and how much the ridge actually learned."""
    settings = load_settings()
    db = open_db(settings)
    try:
        found = taste.load(db, profile_id) if profile_id else taste.latest(db)
    finally:
        db.close()
    if found is None:
        console.print("no profile yet, run: palate profile build")
        raise typer.Exit(code=2)
    _print(found)


def _print(built: TasteProfile) -> None:
    console.print(
        f"{built.profile_id} {built.tier} built {built.built_at[:19]} on {built.index_id}"
    )
    if built.stale:
        console.print(f"  stale: {built.stale_reason}")
    modes = Table(box=None, pad_edge=False)
    for column in ("", "mode", "films", "mass", "coherence", "confidence", "label"):
        modes.add_column(column)
    for mode in (*built.modes, *built.anti_modes):
        modes.add_row(
            "+" if mode.polarity == "like" else "-",
            str(mode.mode_id),
            str(mode.n_members),
            f"{mode.mass:.1f}",
            f"{mode.coherence:.2f}",
            f"{mode.confidence:.2f}",
            mode.label or "",
        )
    console.print(modes)
    for facet in FACETS:
        rows = built.top(facet)
        if not rows:
            continue
        line = ", ".join(f"{row.name} ({row.affinity:+.2f}, n={row.n})" for row in rows)
        console.print(f"{facet}: {line}")


def register(app: typer.Typer) -> None:
    """Attach the profile command group to the CLI."""
    app.add_typer(profile_app, name="profile")
