"""`palate chat`, one conversation against the configured provider."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import anyio
import typer
from rich.console import Console

from palate import paths
from palate.agent.budget import Budget
from palate.agent.events import RunFinished, RunStarted, TextDelta, ToolCallFinished
from palate.agent.loop import AgentLoop
from palate.agent.prompts import PromptRegistry
from palate.agent.state import AgentState
from palate.agent.transcript import Transcript
from palate.config import Settings, load_settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.ids import new_run_id
from palate.memory.sessions import SessionStore
from palate.obs.cost import cost_for, seed_rates
from palate.obs.store import TraceStore
from palate.obs.trace import NullTracer, SQLiteTracer, Tracer
from palate.providers.base import ChatProvider, Message
from palate.providers.registry import build_chat, build_embedder
from palate.retrieval.recommend import LocalRecommender
from palate.retrieval.vocab import Vocabulary
from palate.taste import profile as taste
from palate.taste.memory import PreferenceStore
from palate.taste.profile import TasteProfile
from palate.tools.catalog import build_registry
from palate.tools.context import ToolContext

console = Console()

PROMPT = "you"

SPEAKER = "palate"


def register(app: typer.Typer) -> None:
    """Attach the chat command to the CLI."""
    app.command("chat")(run)


@dataclass(slots=True)
class Wiring:
    """Everything one chat session holds open, built once and closed once."""

    db: Database
    settings: Settings
    provider: ChatProvider
    loop: AgentLoop
    transcript: Transcript
    sessions: SessionStore
    session_id: str
    recommender: LocalRecommender | None
    profile: TasteProfile | None
    prefs: PreferenceStore
    vocab: Vocabulary
    traces: Database | None

    def context(self, state: AgentState) -> ToolContext:
        """One tool context per run, sharing this session's read connections."""
        return ToolContext(
            session_id=state.session_id,
            run_id=state.run_id,
            user_messages=state.user_messages(),
            settings=self.settings,
            db=self.db,
            recommender=self.recommender,
            profile=self.profile,
            prefs=self.prefs,
            vocab=self.vocab,
        )


def open_db(settings: Settings) -> Database:
    """Open palate.db with sqlite-vec loaded, which reading the vectors back needs."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    return open_database(resolved.db, migrations=paths.migrations_dir())


def open_traces(settings: Settings) -> Database:
    """traces.db, seeded with the pricing table at startup and never read for prices after."""
    resolved = paths.resolve(settings.home, offline=settings.offline)
    db = open_database(resolved.traces_db, migrations=paths.trace_migrations_dir(), load_vec=False)
    seed_rates(db, paths.pricing_toml())
    return db


def build_tracer(settings: Settings, traces: Database | None) -> tuple[Tracer, TraceStore | None]:
    """A real tracer when tracing is on, and a no-op that costs nothing when it is not."""
    if traces is None:
        return NullTracer(), None
    store = TraceStore(
        traces,
        queue_max=settings.trace.queue_max,
        flush_ms=settings.trace.flush_ms,
        batch=settings.trace.batch,
        payloads=settings.trace.payloads,
    )

    def price(provider: str, model: str, usage: Any, reported: float | None) -> tuple[float, str]:
        found = cost_for(traces, provider, model, usage, reported)
        return found.usd, found.source

    return SQLiteTracer(store, price=price), store


def _recommender(
    db: Database,
    settings: Settings,
    session_id: str,
    profile: TasteProfile | None,
) -> LocalRecommender | None:
    """A recommender only when there is a profile to rank against."""
    if profile is None:
        return None
    return LocalRecommender(
        db,
        embedder=build_embedder(settings),
        session_id=session_id,
        retrieval=settings.retrieval,
        profile=profile,
    )


@asynccontextmanager
async def wire(settings: Settings, db: Database, session_id: str | None) -> AsyncIterator[Wiring]:
    """Build the provider, the recommender and the loop, and close them in order."""
    provider = build_chat(settings)
    sessions = SessionStore(db)
    session = sessions.open(
        provider=settings.chat.provider, model=settings.chat.model, session_id=session_id
    )
    profile = taste.latest(db)
    recommender = _recommender(db, settings, session.session_id, profile)
    traces = open_traces(settings) if settings.trace.enabled else None
    tracer, store = build_tracer(settings, traces)
    try:
        yield Wiring(
            db=db,
            settings=settings,
            provider=provider,
            loop=AgentLoop(
                provider,
                build_registry(),
                PromptRegistry(),
                settings,
                transcript=Transcript(db),
                tracer=tracer,
            ),
            transcript=Transcript(db),
            sessions=sessions,
            session_id=session.session_id,
            recommender=recommender,
            profile=profile,
            prefs=PreferenceStore(db),
            vocab=Vocabulary(db.read()),
            traces=traces,
        )
    finally:
        await provider.aclose()
        if recommender is not None and recommender.embedder is not None:
            await recommender.embedder.aclose()
        if store is not None:
            store.close()
        if traces is not None:
            traces.close()


async def one_turn(wiring: Wiring, text: str, *, quiet: bool) -> str:
    """One user message, its tool trace, and the answer, written as a transcript."""
    state = AgentState(run_id=new_run_id(), session_id=wiring.session_id)
    asked = Message(role="user", content=text)
    await wiring.transcript.append(wiring.session_id, state.run_id, asked)
    wiring.sessions.start_run(state.run_id, wiring.session_id)
    console.print(SPEAKER)
    answer: list[str] = []
    wall = 0
    stream = wiring.loop.run(
        text,
        session_id=wiring.session_id,
        budget=Budget.from_settings(wiring.settings.agent),
        ctx_factory=wiring.context,
        state=state,
    )
    async for event in stream:
        if isinstance(event, ToolCallFinished) and not quiet:
            console.print(f"  > {event.summary}")
        elif isinstance(event, TextDelta) and event.channel == "answer":
            answer.append(event.text)
        elif isinstance(event, RunFinished):
            wall = event.wall_ms
        elif isinstance(event, RunStarted) and not quiet:
            console.print(f"  > {event.model}")
    wiring.sessions.finish_run(state, wall_ms=wall)
    await wiring.transcript.append(
        wiring.session_id, state.run_id, Message(role="assistant", content=state.final_text)
    )
    for line in state.final_text.splitlines() or [""]:
        console.print(f"  {line}")
    console.print()
    return "".join(answer)


def run(
    message: str = typer.Argument("", help="Ask once and exit. Empty to read from stdin."),
    session: str = typer.Option("", "--session", help="Continue an existing session id."),
    quiet: bool = typer.Option(False, "--quiet", help="Hide the tool trace lines."),
) -> None:
    """Talk to your own history. Every film named comes from a tool result."""
    settings = load_settings()
    db = open_db(settings)
    try:
        anyio.run(partial(_chat, settings, db, message, session or None, quiet))
    except PalateError as exc:
        console.print(str(exc))
        raise typer.Exit(code=2) from exc
    finally:
        db.close()


async def _chat(
    settings: Settings, db: Database, message: str, session: str | None, quiet: bool
) -> None:
    async with wire(settings, db, session) as wiring:
        console.print(f"session {wiring.session_id}")
        if message:
            await one_turn(wiring, message, quiet=quiet)
            return
        for line in _lines():
            await one_turn(wiring, line, quiet=quiet)


def _lines() -> Iterator[str]:
    """One message per line until end of input, which is what a pipe gives too."""
    while True:
        console.print(PROMPT)
        raw = sys.stdin.readline()
        if not raw:
            return
        if raw.strip():
            yield raw.strip()
