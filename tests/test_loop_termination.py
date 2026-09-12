"""Every way a run can end, as a table over the one function that decides."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from palate.agent.budget import Budget, BudgetLedger
from palate.agent.state import AgentPhase, AgentState, StopReason, next_phase
from palate.providers.base import ToolCall, Usage

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
