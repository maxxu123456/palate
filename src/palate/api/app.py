"""The FastAPI factory. One local user, no authentication, loopback by default."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response

from palate import __version__
from palate.api import routes_chat, routes_films, routes_prefs, routes_traces
from palate.api.deps import StateBuilder, build_state
from palate.config import Settings, load_settings
from palate.errors import IndexingError, PalateError, ProviderError, ThinHistoryError, ToolFailure

# The Vite dev server is a different origin, so without this the page cannot reach the api.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def status_for(exc: PalateError) -> int:
    """Our own errors carry enough to pick a code without a table of string matches."""
    if isinstance(exc, ToolFailure):
        return 404 if exc.code == "not_found" else 422
    if isinstance(exc, ProviderError):
        return 502
    if isinstance(exc, IndexingError | ThinHistoryError):
        return 503
    return 400


async def as_json(request: Request, exc: Exception) -> Response:
    """Every PalateError answers with its own text, which already names the fix."""
    failure = exc if isinstance(exc, PalateError) else PalateError(str(exc))
    body = {"error": type(failure).__name__, "detail": str(failure)}
    if isinstance(failure, ToolFailure) and failure.hint:
        body["hint"] = failure.hint
    return Response(
        content=orjson.dumps(body),
        status_code=status_for(failure),
        media_type="application/json",
    )


def create_app(settings: Settings | None = None, *, builder: StateBuilder | None = None) -> FastAPI:
    """One app over one data root. Pass a builder to serve a world that is already open."""
    resolved = settings or load_settings()
    build = builder or build_state

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with build(resolved) as state:
            app.state.palate = state
            yield

    app = FastAPI(
        title="palate",
        version=__version__,
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.add_exception_handler(PalateError, as_json)
    for router in (
        routes_chat.router,
        routes_films.router,
        routes_prefs.router,
        routes_traces.router,
    ):
        app.include_router(router)
    return app
