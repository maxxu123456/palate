"""A tool registry small enough that a loop test needs no database behind it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from palate.agent.budget import Budget
from palate.agent.state import AgentState
from palate.config import Settings
from palate.errors import ToolFailure
from palate.providers.base import ToolCall
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolRegistry, ToolSpec, clamp

BUDGET = Budget(max_turns=4, max_tool_calls=6, max_parallel_calls=4, max_wall_s=30.0)


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=200)
    limit: int = 5

    @field_validator("limit", mode="before")
    @classmethod
    def _limit(cls, value: object, info: ValidationInfo) -> int:
        return clamp(value, info, lo=1, hi=20)


class Film(BaseModel):
    film_id: int
    title: str


class SearchResult(BaseModel):
    films: list[Film]
    meta: dict[str, Any] = Field(default_factory=dict)


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200)


class NoteResult(BaseModel):
    saved: bool
    undo_token: str


class SlowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seconds: float = 0.05


class SlowResult(BaseModel):
    slept: float


class Calls:
    """What the fake tools were asked, and when, so a test can assert on overlap."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.searches: list[str] = []
        self.overlapped = False
        self._running = 0

    def enter(self, name: str) -> None:
        self.order.append(name)
        self._running += 1
        self.overlapped = self.overlapped or self._running > 1

    def leave(self) -> None:
        self._running -= 1


def build_registry(
    seen: Calls,
    *,
    fail: Callable[[SearchArgs], None] | None = None,
) -> ToolRegistry:
    """Two reads and one write, enough to exercise every branch of ACT."""

    async def search(args: SearchArgs, ctx: ToolContext) -> SearchResult:
        seen.enter("search_films")
        seen.searches.append(args.query)
        try:
            await anyio.sleep(0.01)
            if fail is not None:
                fail(args)
            films = [Film(film_id=1000 + i, title=f"film {i}") for i in range(args.limit)]
            return SearchResult(films=films, meta={"count": len(films), "returned": len(films)})
        finally:
            seen.leave()

    async def slow(args: SlowArgs, ctx: ToolContext) -> SlowResult:
        seen.enter("slow_tool")
        try:
            await anyio.sleep(args.seconds)
            return SlowResult(slept=args.seconds)
        finally:
            seen.leave()

    async def note(args: NoteArgs, ctx: ToolContext) -> NoteResult:
        seen.enter("record_note")
        try:
            return NoteResult(saved=True, undo_token="undo_1")
        finally:
            seen.leave()

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_films",
            description="Search films by meaning.",
            args_model=SearchArgs,
            result_model=SearchResult,
            kind=ToolKind.READ,
            handler=search,
            examples=(SearchArgs(query="slow bleak russian cinema", limit=5),),
            max_result_chars=4000,
        )
    )
    registry.register(
        ToolSpec(
            name="slow_tool",
            description="Sleeps, so concurrency is observable.",
            args_model=SlowArgs,
            result_model=SlowResult,
            kind=ToolKind.READ,
            handler=slow,
        )
    )
    registry.register(
        ToolSpec(
            name="record_note",
            description="The only write.",
            args_model=NoteArgs,
            result_model=NoteResult,
            kind=ToolKind.WRITE,
            handler=note,
            invalidates=frozenset({"search_films"}),
        )
    )
    return registry


def refuse(args: SearchArgs) -> None:
    """A handler failure that arrives as a typed envelope rather than an exception."""
    raise ToolFailure("empty_result", "nothing matched", hint="widen the query")


def context_factory(settings: Settings | None = None) -> Callable[[AgentState], ToolContext]:
    """A context with no database behind it, which is all a loop test needs."""
    resolved = settings or Settings()

    def build(state: AgentState) -> ToolContext:
        return ToolContext(
            session_id=state.session_id,
            run_id=state.run_id,
            user_messages=state.user_messages(),
            settings=resolved,
        )

    return build


def call(name: str, arguments: str, *, ident: str = "c1") -> ToolCall:
    """One tool call as a provider would hand it over, arguments parsed if they parse."""
    from palate.providers.streamacc import parse_arguments

    return ToolCall(
        id=ident, name=name, arguments_json=arguments, arguments=parse_arguments(arguments)
    )
