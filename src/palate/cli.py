"""Typer application. Command modules register themselves here."""

import typer

from palate import __version__
from palate.commands import (
    chat_cmd,
    doctor,
    eval_cmd,
    index_cmd,
    ingest_cmd,
    profile_cmd,
    recommend_cmd,
    tmdb_cmd,
    traces_cmd,
)
from palate.errors import MissingExtra
from palate.extras import require

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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address. Loopback, because there is no auth."),
    port: int = typer.Option(8000, help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Restart on a source change."),
) -> None:
    """Serve the agent over HTTP. Needs the api extra."""
    try:
        # Imported here so a base install still has a working CLI without fastapi.
        uvicorn = require("api", "uvicorn")
    except MissingExtra as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from exc
    uvicorn.run("palate.api.app:create_app", host=host, port=port, reload=reload, factory=True)


doctor.register(app)
ingest_cmd.register(app)
tmdb_cmd.register(app)
index_cmd.register(app)
profile_cmd.register(app)
recommend_cmd.register(app)
chat_cmd.register(app)
eval_cmd.register(app)
traces_cmd.register(app)
