"""Typer application. Command modules register themselves here."""

import typer

from palate import __version__
from palate.commands import (
    doctor,
    eval_cmd,
    index_cmd,
    ingest_cmd,
    profile_cmd,
    recommend_cmd,
    tmdb_cmd,
)

app = typer.Typer(
    name="palate",
    help="Film recommendations from your own Letterboxd history.",
    no_args_is_help=True,
    add_completion=False,
)


# Without a callback Typer collapses a one-command app into that command.
@app.callback()
def main() -> None:
    """Film recommendations from your own Letterboxd history."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


doctor.register(app)
ingest_cmd.register(app)
tmdb_cmd.register(app)
index_cmd.register(app)
profile_cmd.register(app)
recommend_cmd.register(app)
eval_cmd.register(app)
