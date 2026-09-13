"""Everything the server holds open, built once at startup and closed once at shutdown."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, cast

import anyio
from fastapi import Depends, Request

from palate import paths
from palate.agent.loop import AgentLoop
from palate.agent.prompts import PromptRegistry
from palate.agent.state import AgentState
from palate.agent.transcript import Transcript
from palate.commands.chat_cmd import build_tracer
from palate.commands.doctor import Check, fingerprint_check, index_check, python_check
from palate.config import Settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.index import verify
from palate.memory.sessions import SessionStore
from palate.obs.cost import seed_rates
from palate.obs.store import TraceStore
from palate.providers.base import ChatProvider, EmbeddingProvider
from palate.providers.http import client_session
from palate.providers.registry import build_chat, build_embedder
from palate.retrieval.recommend import LocalRecommender, Recommender
from palate.retrieval.vocab import Vocabulary
from palate.taste import profile as taste
from palate.taste.memory import PreferenceStore
from palate.taste.profile import TasteProfile
from palate.tools.catalog import build_registry
from palate.tools.context import ToolContext


class RunRegistry:
    """The runs in flight, so a second request can cancel one by id."""

    def __init__(self) -> None:
        self._scopes: dict[str, anyio.CancelScope] = {}

    def add(self, run_id: str, scope: anyio.CancelScope) -> None:
        """Register a scope for the life of one stream."""
        self._scopes[run_id] = scope

    def drop(self, run_id: str) -> None:
        """Forget a finished run."""
        self._scopes.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        """False when nothing by that id is running, which is a 404 and not an error."""
        scope = self._scopes.get(run_id)
        if scope is None:
            return False
        scope.cancel()
        return True


@dataclass(slots=True)
class AppState:
    """One data root, opened once. Routes read it and never open anything themselves."""

    settings: Settings
    db: Database
    traces: Database | None
    chat: ChatProvider
    embedder: EmbeddingProvider
    loop: AgentLoop
    transcript: Transcript
    sessions: SessionStore
    prefs: PreferenceStore
    vocab: Vocabulary
    profile: TasteProfile | None
    runs: RunRegistry

    def recommender(self, session_id: str | None = None) -> Recommender | None:
        """One per session, because a session scoped preference compiles to a filter."""
        if self.profile is None:
            return None
        return LocalRecommender(
            self.db,
            embedder=self.embedder,
            session_id=session_id,
            retrieval=self.settings.retrieval,
            profile=self.profile,
        )

    def context(self, state: AgentState) -> ToolContext:
        """One tool context per run, sharing the server's read connections."""
        return ToolContext(
            session_id=state.session_id,
            run_id=state.run_id,
            user_messages=state.user_messages(),
            settings=self.settings,
            db=self.db,
            recommender=self.recommender(state.session_id),
            profile=self.profile,
            prefs=self.prefs,
            vocab=self.vocab,
        )

    def require_traces(self) -> Database:
        """traces.db, or a refusal naming the setting that turned it off."""
        if self.traces is None:
            raise PalateError("tracing is off, set trace.enabled = true and restart")
        return self.traces


StateBuilder = Callable[[Settings], AbstractAsyncContextManager[AppState]]


def demand(check: Check) -> None:
    """Refuse at startup with the line doctor would print, rather than at the first request."""
    if check.ok:
        return
    fix = f", try: {check.fix}" if check.fix else ""
    raise PalateError(f"{check.name}: {check.detail}{fix}")


def profile_check(profile: TasteProfile | None, index_id: str | None) -> Check:
    """A profile fitted in another vector space loads cleanly and then ranks garbage."""
    if profile is None:
        return Check("taste profile", False, "none fitted", "palate profile build")
    if profile.stale or profile.index_id != index_id:
        detail = f"{profile.profile_id} was fitted for {profile.index_id}"
        return Check("taste profile", False, detail, "palate profile build")
    return Check("taste profile", True, f"{profile.profile_id}, tier {profile.tier}")


@asynccontextmanager
async def build_state(settings: Settings) -> AsyncIterator[AppState]:
    """Open everything a request needs, in the order that makes each failure nameable."""
    demand(python_check())
    resolved = paths.resolve(settings.home, offline=settings.offline)
    db = open_database(resolved.db, migrations=paths.migrations_dir())
    traces: Database | None = None
    store: TraceStore | None = None
    try:
        demand(index_check(db))
        profile = taste.latest(db)
        demand(profile_check(profile, verify.active_id(db)))
        if settings.trace.enabled:
            traces = open_database(
                resolved.traces_db, migrations=paths.trace_migrations_dir(), load_vec=False
            )
            seed_rates(traces, paths.pricing_toml())
        tracer, store = build_tracer(settings, traces)
        async with client_session() as http:
            chat = build_chat(settings, client=http)
            embedder = build_embedder(settings, client=http)
            try:
                demand(await fingerprint_check(db, embedder))
                yield AppState(
                    settings=settings,
                    db=db,
                    traces=traces,
                    chat=chat,
                    embedder=embedder,
                    loop=AgentLoop(
                        chat,
                        build_registry(),
                        PromptRegistry(),
                        settings,
                        transcript=Transcript(db),
                        tracer=tracer,
                    ),
                    transcript=Transcript(db),
                    sessions=SessionStore(db),
                    prefs=PreferenceStore(db),
                    vocab=Vocabulary(db.read()),
                    profile=profile,
                    runs=RunRegistry(),
                )
            finally:
                await chat.aclose()
                await embedder.aclose()
    finally:
        if store is not None:
            store.close()
        if traces is not None:
            traces.close()
        db.close()


def app_state(request: Request) -> AppState:
    """The state the lifespan put on the app, which is the only shared thing routes touch."""
    return cast("AppState", request.app.state.palate)


# A plain alias, not a PEP 695 one, because FastAPI reads the Annotated metadata directly.
State = Annotated[AppState, Depends(app_state)]
