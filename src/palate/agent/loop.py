"""The agent loop. Hand rolled, because the termination guarantees are the design."""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import anyio
import orjson
from pydantic import ValidationError

from palate.agent.answer import (
    StructuredAnswer,
    assemble,
    films_json,
    parse,
    parse_error,
    resolve,
    response_format,
)
from palate.agent.budget import Budget
from palate.agent.events import (
    AgentEvent,
    GroundingChecked,
    PreferenceRecorded,
    Recommendations,
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
    ToolCallFinished,
    ToolCallProposed,
    ToolCallStarted,
    TurnStarted,
)
from palate.agent.prompts import PromptRegistry
from palate.agent.repair import repair
from palate.agent.state import AgentPhase, AgentState, StopReason, next_phase
from palate.agent.transcript import Transcript
from palate.config import Settings
from palate.errors import ProviderContextOverflow, ProviderError
from palate.ground.check import GroundednessChecker, NLIModel
from palate.ids import new_run_id
from palate.obs.trace import LLMRequestRecord, NullTracer, Tracer
from palate.providers.base import (
    ChatCapabilities,
    ChatProvider,
    Completion,
    Message,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolSchema,
    json_int,
)
from palate.providers.contentcalls import extract_tool_calls
from palate.taste.profile import TasteProfile
from palate.tools.context import ToolContext
from palate.tools.envelope import ToolError, ToolErrorCode, ToolResult, failure
from palate.tools.registry import ToolKind, ToolRegistry, ToolSpec, fingerprint
from palate.tools.schema import excerpt, render

APOLOGY = "Something went wrong while I was working on that, so I have no answer for you."

NO_PREFERENCES = "Nothing saved yet. Ask before you assume."

NO_PROFILE = "No taste profile has been fitted, so nothing is known about this user yet."

# Three strikes on the same (tool, error) and the tool stops being offered at all.
WITHDRAW_STRIKE = 3

_CHUNK = re.compile(r"\s*\S+\s*")

# Anything that is not a pure follow up counts as needing retrieval, deliberately generously.
_FOLLOW_UP = re.compile(
    r"^(why|what about|and the|how about|tell me more|more like|the (first|second|third|last))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """One finished run, whole, for tests and the eval harness."""

    run_id: str
    text: str
    state: AgentState
    events: tuple[AgentEvent, ...]
    stop_reason: StopReason
    failed: bool = False

    @property
    def results(self) -> tuple[ToolResult, ...]:
        """Every tool result the run produced, in the order it observed them."""
        return tuple(self.state.results)

    def of_type[E](self, kind: type[E]) -> tuple[E, ...]:
        """Every emitted event of one type, which is how a test reads the stream."""
        return tuple(e for e in self.events if isinstance(e, kind))


def needs_retrieval(text: str) -> bool:
    """A keyword and length heuristic, never a model call, and generous on purpose."""
    stripped = text.strip()
    if not stripped:
        return False
    return not (_FOLLOW_UP.match(stripped) and len(stripped.split()) <= 6)


def profile_digest(profile: TasteProfile | None) -> str:
    """Counts, tier and mode labels. No titles, so it cannot be used as a source of films."""
    if profile is None:
        return NO_PROFILE
    labels = [m.label for m in profile.modes[:3] if m.label]
    line = f"{profile.n_rated} rated films, profile tier {profile.tier}."
    if labels:
        line += " Main taste modes: " + "; ".join(labels) + "."
    return line


def deltas(text: str) -> list[str]:
    """One chunk per word, spacing kept, so the joined deltas are the text again."""
    found = _CHUNK.findall(text)
    return found if found else ([text] if text else [])


class AgentLoop:
    """PLAN, ACT, OBSERVE, with one back edge and seven ways out that all answer."""

    def __init__(
        self,
        provider: ChatProvider,
        registry: ToolRegistry,
        prompts: PromptRegistry,
        settings: Settings,
        *,
        transcript: Transcript | None = None,
        nli: NLIModel | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.prompts = prompts
        self.settings = settings
        self.transcript = transcript
        self.nli = nli
        self.tracer: Tracer = tracer or NullTracer()

    async def run(
        self,
        user_text: str,
        *,
        session_id: str,
        budget: Budget | None = None,
        ctx_factory: Callable[[AgentState], ToolContext],
        state: AgentState | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """The event stream the SSE route iterates.

        Pass a state to own the run id before the first event arrives.
        """
        owned = state or AgentState(run_id=new_run_id(), session_id=session_id)
        async for event in self._execute(owned, user_text, budget, ctx_factory):
            yield event

    async def run_to_completion(
        self,
        user_text: str,
        *,
        session_id: str,
        budget: Budget | None = None,
        ctx_factory: Callable[[AgentState], ToolContext],
    ) -> RunOutcome:
        """The same run, drained. Neither path has logic the other lacks."""
        state = AgentState(run_id=new_run_id(), session_id=session_id)
        events = [e async for e in self._execute(state, user_text, budget, ctx_factory)]
        return RunOutcome(
            run_id=state.run_id,
            text=state.final_text,
            state=state,
            events=tuple(events),
            stop_reason=state.stop_reason or StopReason.ANSWERED,
            failed=state.phase is AgentPhase.FAILED,
        )

    async def _execute(
        self,
        state: AgentState,
        user_text: str,
        budget: Budget | None,
        ctx_factory: Callable[[AgentState], ToolContext],
    ) -> AsyncIterator[AgentEvent]:
        limits = budget or Budget.from_settings(self.settings.agent)
        opened = anyio.current_time()
        state.ledger.start(limits, opened)
        caps = await self.provider.capabilities()
        traced = self.tracer.run("chat", session_id=state.session_id, input_text=user_text)
        run_span = traced.__enter__()
        if self.transcript is not None:
            state.messages.extend(
                self.transcript.load(state.session_id, token_budget=limits.max_prompt_tokens // 2)
            )
        state.messages.append(Message(role="user", content=user_text))
        ctx = ctx_factory(state)
        yield RunStarted(
            run_id=state.run_id,
            session_id=state.session_id,
            model=self.provider.model,
            budget=dict(limits.as_json()),
        )
        ctx.span = run_span
        try:
            while state.phase not in (AgentPhase.DONE, AgentPhase.FAILED):
                ctx.turn = state.turn
                ctx.deadline = state.ledger.deadline
                ctx.user_messages = state.user_messages()
                entered = state.phase
                async for event in self._step(state, ctx, limits, caps):
                    yield event
                if state.phase is entered:
                    state.phase = next_phase(state, limits, anyio.current_time())
        finally:
            self.tracer.finish_run(
                run_span,
                turns=state.turn,
                total_tokens_in=state.ledger.input_tokens,
                total_tokens_out=state.ledger.output_tokens,
                total_cost_usd=state.ledger.cost_usd,
                cost_complete=int(state.ledger.exact),
                stop_reason=str(state.stop_reason or StopReason.ANSWERED),
                output_text=state.final_text,
            )
            traced.__exit__(None, None, None)
        yield RunFinished(
            stop_reason=str(state.stop_reason or StopReason.ANSWERED),
            turns=state.turn,
            tool_calls=state.total_tool_calls,
            input_tokens=state.ledger.input_tokens,
            output_tokens=state.ledger.output_tokens,
            cost_usd=state.ledger.cost_usd,
            wall_ms=int((anyio.current_time() - opened) * 1000.0),
        )

    async def _step(
        self, state: AgentState, ctx: ToolContext, limits: Budget, caps: ChatCapabilities
    ) -> AsyncIterator[AgentEvent]:
        match state.phase:
            case AgentPhase.PLAN:
                async for event in self._plan(state, ctx, limits, caps):
                    yield event
            case AgentPhase.ACT:
                async for event in self._act(state, ctx, limits, caps):
                    yield event
            case AgentPhase.OBSERVE:
                async for event in self._observe(state):
                    yield event
            case AgentPhase.FORCE_ANSWER:
                async for event in self._force_answer(state, ctx):
                    yield event
            case AgentPhase.FINALIZE:
                async for event in self._finalize(state, ctx, caps):
                    yield event
            case AgentPhase.GROUND:
                async for event in self._ground(state):
                    yield event
            case _:
                return

    async def _plan(
        self, state: AgentState, ctx: ToolContext, limits: Budget, caps: ChatCapabilities
    ) -> AsyncIterator[AgentEvent]:
        tools = self._offered(state, caps)
        yield TurnStarted(turn=state.turn, tools_offered=tuple(t.name for t in tools))
        done = await self._plan_call(state, ctx, limits, tools)
        if done is None:
            state.stop_reason = StopReason.PROVIDER_FAILED
            state.phase = AgentPhase.FORCE_ANSWER
            return
        calls, text = self._calls_of(state, done)
        if not calls and self._force_tools(state, tools, caps):
            state.turn += 1
            retried = await self._ask(
                state, ctx, self._request(state, ctx, limits), tools=tools, tool_choice="required"
            )
            calls, text = self._calls_of(state, retried)
        # With a database behind it the answer is assembled in FINALIZE, so this is not it yet.
        talking = not calls and ctx.db is None
        channel: Literal["thinking", "answer"] = "answer" if talking else "thinking"
        for piece in deltas(text):
            yield TextDelta(text=piece, channel=channel)
        state.messages.append(
            Message(role="assistant", content=text, tool_calls=calls, channel=channel)
        )
        state.pending_calls = list(calls)
        if talking:
            state.final_text = text

    async def _plan_call(
        self,
        state: AgentState,
        ctx: ToolContext,
        limits: Budget,
        tools: Sequence[ToolSchema],
    ) -> Completion | None:
        try:
            return await self._ask(state, ctx, self._request(state, ctx, limits), tools=tools)
        except ProviderContextOverflow:
            # Appending "context overflow" to an already overflowing transcript cannot succeed.
            if state.compactions or self.transcript is None:
                return None
            state.messages = self.transcript.compact(state.messages, limits.max_prompt_tokens)
            state.compactions += 1
        except ProviderError:
            return None
        try:
            return await self._ask(state, ctx, self._request(state, ctx, limits), tools=tools)
        except ProviderError:
            return None

    async def _act(
        self, state: AgentState, ctx: ToolContext, limits: Budget, caps: ChatCapabilities
    ) -> AsyncIterator[AgentEvent]:
        for call in state.pending_calls:
            yield ToolCallProposed(
                call_id=call.id,
                name=call.name,
                arguments=dict(call.arguments or {}),
                source=call.source,
            )
        runnable, settled = self._schedule(state, limits, caps)
        for call in runnable:
            yield ToolCallStarted(call_id=call.id, name=call.name)
        produced = await self._dispatch(runnable, ctx, limits)
        for call in runnable:
            result = produced.get(call.id)
            spec = self.registry.get(call.name)
            if result is not None and result.ok and spec is not None and spec.cacheable:
                state.result_cache[fingerprint(call)] = result
        produced.update(settled)
        state.observations = [produced[c.id] for c in state.pending_calls if c.id in produced]

    async def _observe(self, state: AgentState) -> AsyncIterator[AgentEvent]:
        for result in state.observations:
            spec = self.registry.get(result.name)
            shaped = self._escalate(state, result, spec)
            state.messages.append(
                Message(
                    role="tool",
                    content=shaped.render(spec.max_result_chars if spec else 2000),
                    tool_call_id=shaped.call_id,
                    name=shaped.name,
                )
            )
            state.results.append(shaped)
            state.total_tool_calls += 1
            if shaped.ok:
                state.evidence.absorb(shaped.data)
                state.consecutive_tool_errors = 0
                if spec is not None and spec.kind is ToolKind.WRITE:
                    self._invalidate(state, spec.invalidates)
            else:
                state.consecutive_tool_errors += 1
            yield ToolCallFinished(
                call_id=shaped.call_id,
                name=shaped.name,
                ok=shaped.ok,
                summary=shaped.summary(),
                meta=dict(shaped.meta),
                latency_ms=shaped.latency_ms,
                undo_token=shaped.undo_token,
            )
            recorded = _preference_event(shaped)
            if recorded is not None:
                yield recorded
        state.turn += 1
        state.pending_calls = []
        state.observations = []

    async def _force_answer(self, state: AgentState, ctx: ToolContext) -> AsyncIterator[AgentEvent]:
        async for event in self._last_word(state, ctx):
            yield event

    async def _finalize(
        self, state: AgentState, ctx: ToolContext, caps: ChatCapabilities
    ) -> AsyncIterator[AgentEvent]:
        # Prose is assembled from database rows, so with no database the model's own text is it.
        if ctx.db is not None:
            parsed = await self._structured(state, ctx, caps)
            if parsed is not None:
                report = resolve(parsed, ctx.db, state.evidence)
                state.answer = report.answer
                state.resolved = report
                state.constraint_violation = report.watched_ids
                for piece in deltas(report.answer.preamble):
                    yield TextDelta(text=piece, channel="answer")
                return
        if state.final_text.strip():
            return
        # A turn that emitted neither calls nor text still owes the user an answer.
        async for event in self._last_word(state, ctx):
            yield event

    async def _structured(
        self, state: AgentState, ctx: ToolContext, caps: ChatCapabilities
    ) -> StructuredAnswer | None:
        # A model that already answered in the right shape does not owe another call.
        ready = _already_structured(state)
        if ready is not None:
            return ready
        prompt = self.prompts.get("final_answer")
        messages = [*state.messages, Message(role="system", content=prompt.text)]
        wanted = response_format() if caps.json_schema else None
        done = await self._answer_call(state, ctx, messages, wanted)
        if done is None:
            return None
        try:
            return parse(done.content, provider_json=caps.json_schema)
        except ValidationError as exc:
            retry = [*messages, Message(role="system", content=parse_error(exc))]
            second = await self._answer_call(state, ctx, retry, wanted)
            if second is None:
                return None
        try:
            return parse(second.content, provider_json=caps.json_schema)
        except ValidationError:
            return None

    async def _answer_call(
        self,
        state: AgentState,
        ctx: ToolContext,
        messages: Sequence[Message],
        wanted: ResponseFormat | None,
    ) -> Completion | None:
        try:
            return await self._ask(
                state,
                ctx,
                messages,
                tools=(),
                response_format=wanted,
                max_tokens=self.settings.agent.max_completion_tokens,
            )
        except ProviderError:
            return None

    async def _ground(self, state: AgentState) -> AsyncIterator[AgentEvent]:
        if state.answer is None or state.resolved is None:
            return
        checker = GroundednessChecker(state.evidence, self.nli, self.settings.agent.nli_threshold)
        with self.tracer.span("ground", "ground") as span:
            report = await checker.check(state.answer, state.run_id)
            self.tracer.record_grounding(span, report)
        state.grounding = report
        if self.settings.agent.strict_grounding:
            repaired, removed = repair(state.answer, report)
            state.answer = repaired
            state.answer_repaired = bool(removed)
        yield GroundingChecked(
            grounded_ratio=report.grounded_ratio,
            nli_available=report.nli_available,
            unsupported=report.unsupported_sentences,
            dropped_film_ids=report.dropped_film_ids,
        )
        facts = state.resolved.facts
        state.final_text = assemble(state.answer, facts)
        state.messages.append(Message(role="assistant", content=state.final_text))
        yield Recommendations(films=films_json(state.answer, facts), prose=state.final_text)

    async def _last_word(self, state: AgentState, ctx: ToolContext) -> AsyncIterator[AgentEvent]:
        prompt = self.prompts.get("force_answer")
        messages = [*state.messages, Message(role="system", content=prompt.text)]
        try:
            # tools=[] rather than tool_choice="none": several servers ignore the latter.
            done = await self._ask(
                state,
                ctx,
                messages,
                tools=(),
                max_tokens=self.settings.agent.max_completion_tokens,
            )
        except ProviderError as exc:
            state.final_text = APOLOGY
            state.phase = AgentPhase.FAILED
            yield RunFailed(
                error_code=type(exc).__name__,
                message=str(exc),
                stop_reason=str(state.stop_reason or StopReason.PROVIDER_FAILED),
            )
            return
        state.final_text = done.content or APOLOGY
        state.messages.append(Message(role="assistant", content=state.final_text))
        for piece in deltas(state.final_text):
            yield TextDelta(text=piece, channel="answer")

    async def _ask(
        self,
        state: AgentState,
        ctx: ToolContext,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema],
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> Completion:
        state.ledger.observe_prompt(self.provider.count_tokens(messages, tools))
        left = state.ledger.remaining_s(anyio.current_time())
        with self.tracer.span("chat", "llm", parent=ctx.span) as span:
            done = await self.provider.complete(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.settings.chat.temperature,
                max_tokens=max_tokens or self.settings.chat.max_tokens,
                response_format=response_format,
                timeout_s=max(1.0, min(self.settings.chat.timeout_s, left)),
                span=span,
            )
            self.tracer.record_llm(span, self._request_record(messages, tools), done)
        state.ledger.charge(done.usage, done.cost_usd)
        return done

    def _request_record(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> LLMRequestRecord:
        prompt = self.prompts.get("system_agent")
        return LLMRequestRecord(
            provider=self.provider.name,
            model=self.provider.model,
            messages=messages,
            tools=tools,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            prompt_sha=prompt.sha,
            prompt_template=prompt.text,
            temperature=self.settings.chat.temperature,
        )

    def _offered(self, state: AgentState, caps: ChatCapabilities) -> list[ToolSchema]:
        if not caps.tools:
            return []
        return self.registry.schemas(exclude=state.withdrawn_tools, style=caps.schema_style)

    def _request(self, state: AgentState, ctx: ToolContext, limits: Budget) -> list[Message]:
        messages = [self._system(ctx), *state.messages]
        if state.turn == math.ceil(limits.max_turns / 2):
            # Small models measurably stop runaway searching when told what is left.
            left = max(limits.max_tool_calls - state.total_tool_calls, 0)
            messages.append(
                Message(role="system", content=f"you have {left} tool calls left in this run")
            )
        return messages

    def _system(self, ctx: ToolContext) -> Message:
        prompt = self.prompts.get("system_agent")
        block = ctx.prefs.as_prompt_block(ctx.session_id) if ctx.prefs is not None else ""
        return Message(
            role="system",
            content=prompt.render(
                preferences=block or NO_PREFERENCES, profile=profile_digest(ctx.profile)
            ),
        )

    def _calls_of(self, state: AgentState, done: Completion) -> tuple[tuple[ToolCall, ...], str]:
        if done.tool_calls or not self.settings.chat.content_toolcall_parse:
            return done.tool_calls, done.content
        # Small local models routinely emit a tool call as ordinary text.
        return extract_tool_calls(done.content, known=self.registry.names(), turn=state.turn)

    def _force_tools(
        self, state: AgentState, tools: Sequence[ToolSchema], caps: ChatCapabilities
    ) -> bool:
        if state.turn != 0 or not tools or not caps.tools:
            return False
        if not self.settings.chat.force_tools_on_turn0:
            return False
        asked = state.user_messages()
        return bool(asked) and needs_retrieval(asked[-1].content)

    def _schedule(
        self, state: AgentState, limits: Budget, caps: ChatCapabilities
    ) -> tuple[list[ToolCall], dict[str, ToolResult]]:
        runnable: list[ToolCall] = []
        settled: dict[str, ToolResult] = {}
        seen_names: set[str] = set()
        spent = state.ledger.remaining_s(anyio.current_time()) <= 0.0
        for call in state.pending_calls:
            refused = self._refuse(state, call, limits, caps, runnable, seen_names, spent=spent)
            if refused is not None:
                settled[call.id] = refused
                continue
            seen_names.add(call.name)
            runnable.append(call)
        return runnable, settled

    def _refuse(
        self,
        state: AgentState,
        call: ToolCall,
        limits: Budget,
        caps: ChatCapabilities,
        runnable: Sequence[ToolCall],
        seen_names: set[str],
        *,
        spent: bool,
    ) -> ToolResult | None:
        if spent:
            return failure(
                call.id,
                call.name,
                ToolErrorCode.BUDGET_EXHAUSTED,
                "the run is out of time",
                hint="answer with what you already have",
            )
        if len(runnable) >= limits.max_parallel_calls:
            return failure(
                call.id,
                call.name,
                ToolErrorCode.TOO_MANY_CALLS,
                f"{len(state.pending_calls)} calls in one turn",
                hint=f"ask for at most {limits.max_parallel_calls} at a time",
            )
        if not caps.parallel_tool_calls and call.name in seen_names:
            # Results correlate only by tool name here, so two calls to one tool are ambiguous.
            return failure(
                call.id,
                call.name,
                ToolErrorCode.UNSUPPORTED,
                "this model cannot be given two calls to the same tool in one turn",
                hint="this model handles one call per tool per turn",
            )
        key = fingerprint(call)
        seen = state.call_counts.get(key, 0)
        state.call_counts[key] = seen + 1
        if seen >= limits.max_repeat - 1:
            return failure(
                call.id,
                call.name,
                ToolErrorCode.REPEATED_CALL,
                f"{call.name} was already called with these exact arguments {seen} times",
                hint="change the arguments or answer with what you have",
            )
        cached = state.result_cache.get(key)
        if seen and cached is not None:
            return replace(
                cached,
                call_id=call.id,
                cache_hit=True,
                meta={**cached.meta, "repeated_call": True},
            )
        return None

    async def _dispatch(
        self, calls: Sequence[ToolCall], ctx: ToolContext, limits: Budget
    ) -> dict[str, ToolResult]:
        out: dict[str, ToolResult] = {}
        reads = [c for c in calls if self._kind(c) is ToolKind.READ]
        writes = [c for c in calls if self._kind(c) is ToolKind.WRITE]
        limiter = anyio.CapacityLimiter(limits.max_parallel_calls)
        order = {call.id: place for place, call in enumerate(calls)}

        async def one(call: ToolCall) -> None:
            async with limiter:
                out[call.id] = await self._one_tool(call, ctx, order[call.id])

        if reads:
            async with anyio.create_task_group() as group:
                for call in reads:
                    group.start_soon(one, call)
        # Writes run after every read, so a preference cannot race a search that should see it.
        for call in writes:
            out[call.id] = await self._one_tool(call, ctx, order[call.id])
        return out

    async def _one_tool(self, call: ToolCall, ctx: ToolContext, place: int) -> ToolResult:
        with self.tracer.span(call.name, "tool", parent=ctx.span) as span:
            span.set(turn=ctx.turn, seq_in_turn=place)
            result = await self.registry.dispatch(call, ctx)
            if not result.ok and result.error is not None:
                span.fail(str(result.error.code), result.error.message)
            self.tracer.record_tool(span, call, result)
        return result

    def _kind(self, call: ToolCall) -> ToolKind:
        spec = self.registry.get(call.name)
        return ToolKind.READ if spec is None else spec.kind

    def _invalidate(self, state: AgentState, names: Iterable[str]) -> None:
        wanted = set(names)
        for key, result in list(state.result_cache.items()):
            if result.name in wanted:
                del state.result_cache[key]

    def _escalate(
        self, state: AgentState, result: ToolResult, spec: ToolSpec[Any, Any] | None
    ) -> ToolResult:
        if result.ok or result.error is None:
            return result
        key = (result.name, str(result.error.code))
        strike = state.error_strikes.get(key, 0) + 1
        state.error_strikes[key] = strike
        if strike >= WITHDRAW_STRIKE:
            state.withdrawn_tools.add(result.name)
            return result.with_error(
                replace(
                    result.error,
                    hint=f"do not call {result.name} again in this run, answer with what you have",
                )
            )
        if strike == 2 and spec is not None:
            return result.with_error(self._worked_example(result.error, spec))
        return result

    @staticmethod
    def _worked_example(error: ToolError, spec: ToolSpec[Any, Any]) -> ToolError:
        schema = error.schema_excerpt or excerpt(
            render(spec.args_model, style="plain"), _field_of(error.message)
        )
        if not spec.examples:
            return replace(error, schema_excerpt=schema)
        worked = orjson.dumps(spec.examples[0].model_dump(exclude_none=True)).decode()
        joined = f"{error.hint}\na call that works: {worked}" if error.hint else worked
        return replace(error, hint=joined, schema_excerpt=schema)


def _already_structured(state: AgentState) -> StructuredAnswer | None:
    spoken = [m for m in state.messages if m.role == "assistant" and m.content.strip()]
    if not spoken:
        return None
    try:
        return parse(spoken[-1].content, provider_json=False)
    except ValidationError:
        return None


def _field_of(message: str) -> str:
    head, _, _ = message.partition(":")
    return head.strip()


def _preference_event(result: ToolResult) -> PreferenceRecorded | None:
    if not result.ok or result.name != "record_preference" or not isinstance(result.data, dict):
        return None
    data = result.data
    return PreferenceRecorded(
        pref_id=json_int(data.get("pref_id")),
        label=str(data.get("target_label", "")),
        polarity=str(data.get("polarity", "")),
        hardness=str(data.get("hardness", "")),
        affected_films=json_int(data.get("affected_films")),
        undo_token=result.undo_token or "",
    )
