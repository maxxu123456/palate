"""Typer application. Command modules register themselves here."""

import typer

from palate import __version__

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
