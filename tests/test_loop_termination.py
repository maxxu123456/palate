"""Every way a run can end, as a table over the one function that decides."""

from __future__ import annotations

from collections.abc import Callable
from itertools import count

import pytest
from fixtures.agent import Calls, build_registry, context_factory

from palate.agent.budget import Budget, BudgetLedger
from palate.agent.loop import AgentLoop, RunOutcome
from palate.agent.prompts import PromptRegistry
from palate.agent.state import AgentPhase, AgentState, StopReason, next_phase
from palate.config import Settings
from palate.providers.base import Completion, Message, ToolCall, ToolSchema, Usage
from palate.providers.chat.fake import FakeChatProvider

BUDGET = Budget(max_turns=4, max_tool_calls=6, max_cost_usd=0.10, max_wall_s=30.0)

NOW = 1000.0


def fresh(**fields: object) -> AgentState:
    """A state sitting in OBSERVE with room left on every guard."""
    state = AgentState(run_id="run_1", session_id="ses_1", phase=AgentPhase.OBSERVE)
    state.ledger = BudgetLedger(deadline=NOW + BUDGET.max_wall_s)
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def spend(tokens: int = 0, cost: float = 0.0, prompt: int = 0) -> BudgetLedger:
    ledger = BudgetLedger(deadline=NOW + BUDGET.max_wall_s)
    ledger.observe_prompt(prompt)
    ledger.charge(Usage(output_tokens=tokens), cost)
    return ledger


GUARDS: tuple[tuple[str, Callable[[], AgentState], StopReason], ...] = (
    ("turn cap", lambda: fresh(turn=4), StopReason.MAX_TURNS),
    ("tool call cap", lambda: fresh(total_tool_calls=6), StopReason.MAX_TOOL_CALLS),
    ("prompt tokens", lambda: fresh(ledger=spend(prompt=24_000)), StopReason.TOKEN_BUDGET),
    ("completion tokens", lambda: fresh(ledger=spend(tokens=4_000)), StopReason.TOKEN_BUDGET),
    ("cost", lambda: fresh(ledger=spend(cost=0.11)), StopReason.COST_BUDGET),
    ("deadline", lambda: fresh(ledger=BudgetLedger(deadline=NOW)), StopReason.DEADLINE),
    ("error streak", lambda: fresh(consecutive_tool_errors=3), StopReason.TOOL_ERROR_STREAK),
    ("repeat loop", lambda: fresh(call_counts={"abc": 3}), StopReason.REPEAT_LOOP),
)


@pytest.mark.parametrize(("name", "build", "reason"), GUARDS, ids=[g[0] for g in GUARDS])
def test_every_guard_diverts_to_force_answer(
    name: str, build: Callable[[], AgentState], reason: StopReason
) -> None:
    state = build()
    assert next_phase(state, BUDGET, NOW) is AgentPhase.FORCE_ANSWER
    assert state.stop_reason is reason


def test_an_unexhausted_observe_goes_back_to_plan() -> None:
    state = fresh()
    assert next_phase(state, BUDGET, NOW) is AgentPhase.PLAN
    assert state.stop_reason is None


def test_plan_with_calls_acts_and_plan_without_them_finalizes() -> None:
    call = ToolCall(id="c1", name="search_films", arguments_json="{}", arguments={})
    acting = AgentState(run_id="r", session_id="s", pending_calls=[call])
    assert next_phase(acting, BUDGET, NOW) is AgentPhase.ACT
    talking = AgentState(run_id="r", session_id="s")
    assert next_phase(talking, BUDGET, NOW) is AgentPhase.FINALIZE


def test_the_tail_of_the_graph_always_reaches_done() -> None:
    state = AgentState(run_id="r", session_id="s", phase=AgentPhase.ACT)
    seen = [state.phase]
    for _ in range(8):
        state.phase = next_phase(state, BUDGET, NOW)
        seen.append(state.phase)
        if state.phase is AgentPhase.DONE:
            break
    assert seen[-1] is AgentPhase.DONE
    assert AgentPhase.GROUND in seen


def test_force_answer_is_exempt_from_the_turn_cap() -> None:
    state = fresh(turn=99, phase=AgentPhase.FORCE_ANSWER)
    assert next_phase(state, BUDGET, NOW) is AgentPhase.FINALIZE
    assert state.stop_reason is None


def test_next_phase_touches_nothing_but_the_stop_reason() -> None:
    state = fresh(turn=4, total_tool_calls=2)
    next_phase(state, BUDGET, NOW)
    assert state.turn == 4
    assert state.total_tool_calls == 2
    assert state.phase is AgentPhase.OBSERVE


def test_the_ledger_goes_inexact_the_moment_one_call_is_estimated() -> None:
    ledger = BudgetLedger()
    ledger.charge(Usage(output_tokens=10, exact=True))
    assert ledger.exact
    ledger.charge(Usage(output_tokens=7, exact=False))
    assert not ledger.exact
    assert ledger.output_tokens == 17
    assert ledger.calls == 2


def test_the_deadline_is_pinned_once_and_a_retry_cannot_move_it() -> None:
    ledger = BudgetLedger()
    ledger.start(BUDGET, NOW)
    assert ledger.remaining_s(NOW + 10.0) == pytest.approx(20.0)
    assert ledger.remaining_s(NOW + 999.0) == 0.0


ANSWER = "Here is what I found."


def endless_search() -> Callable[[list[Message], list[ToolSchema]], Completion]:
    """A model that searches forever while it is offered tools, and answers when it is not."""
    turns = count()

    def reply(messages: list[Message], tools: list[ToolSchema]) -> Completion:
        if not tools:
            return _completion(ANSWER, ())
        n = next(turns)
        call = ToolCall(
            id=f"c{n}",
            name="search_films",
            arguments_json=f'{{"query": "query {n}"}}',
            arguments={"query": f"query {n}"},
        )
        return _completion("looking", (call,))

    return reply


def _completion(text: str, calls: tuple[ToolCall, ...]) -> Completion:
    return Completion(
        content=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="fake-model",
        response_model="fake-model",
        cost_usd=0.01,
    )


async def run_until(limits: Budget) -> RunOutcome:
    """One whole run against a model that will not stop, so a guard is seen end to end."""
    seen = Calls()
    provider = FakeChatProvider(endless_search())
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), Settings())
    return await agent.run_to_completion(
        "something slow and cold",
        session_id="ses_guard",
        budget=limits,
        ctx_factory=context_factory(),
    )


async def test_a_runaway_search_loop_stops_at_the_turn_cap_and_still_answers() -> None:
    outcome = await run_until(Budget(max_turns=3, max_tool_calls=20, max_wall_s=30.0))
    assert outcome.stop_reason is StopReason.MAX_TURNS
    assert outcome.state.turn == 3
    assert outcome.text == ANSWER


async def test_fan_out_across_turns_stops_at_the_tool_call_cap() -> None:
    outcome = await run_until(Budget(max_turns=20, max_tool_calls=2, max_wall_s=30.0))
    assert outcome.stop_reason is StopReason.MAX_TOOL_CALLS
    assert outcome.state.total_tool_calls == 2
    assert outcome.text == ANSWER


async def test_a_run_past_its_deadline_answers_from_what_it_has() -> None:
    outcome = await run_until(Budget(max_turns=20, max_tool_calls=20, max_wall_s=0.0))
    assert outcome.stop_reason is StopReason.DEADLINE
    assert outcome.text == ANSWER


async def test_a_paid_endpoint_looping_stops_at_the_cost_cap() -> None:
    outcome = await run_until(
        Budget(max_turns=20, max_tool_calls=20, max_cost_usd=0.015, max_wall_s=30.0)
    )
    assert outcome.stop_reason is StopReason.COST_BUDGET
    assert outcome.text == ANSWER


async def test_a_transcript_that_outgrows_its_token_cap_stops_and_answers() -> None:
    outcome = await run_until(
        Budget(max_turns=20, max_tool_calls=20, max_prompt_tokens=1, max_wall_s=30.0)
    )
    assert outcome.stop_reason is StopReason.TOKEN_BUDGET
    assert outcome.text == ANSWER


async def test_every_guard_leaves_the_run_in_done_with_text() -> None:
    outcome = await run_until(Budget(max_turns=2, max_tool_calls=20, max_wall_s=30.0))
    assert outcome.state.phase is AgentPhase.DONE
    assert outcome.text
