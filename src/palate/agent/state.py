"""The agent's phases, the ways a run can end, and the one pure function that decides."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from palate.agent.budget import Budget, BudgetLedger
from palate.providers.base import Message, ToolCall
from palate.tools.envelope import ToolResult


class AgentPhase(StrEnum):
    """One node of the graph. There is exactly one back edge, OBSERVE to PLAN."""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    FORCE_ANSWER = "force_answer"
    FINALIZE = "finalize"
    GROUND = "ground"
    DONE = "done"
    FAILED = "failed"


class StopReason(StrEnum):
    """Why the run ended. Every one of these still produces an answer."""

    ANSWERED = "answered"
    MAX_TURNS = "max_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    TOKEN_BUDGET = "token_budget"
    COST_BUDGET = "cost_budget"
    DEADLINE = "deadline"
    TOOL_ERROR_STREAK = "tool_error_streak"
    REPEAT_LOOP = "repeat_loop"
    CANCELLED = "cancelled"
    PROVIDER_FAILED = "provider_failed"


@dataclass(slots=True)
class AgentState:
    """Everything one run carries. Nothing here is shared between runs."""

    run_id: str
    session_id: str
    phase: AgentPhase = AgentPhase.PLAN
    turn: int = 0
    total_tool_calls: int = 0
    messages: list[Message] = field(default_factory=list)
    pending_calls: list[ToolCall] = field(default_factory=list)
    observations: list[ToolResult] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    result_cache: dict[str, ToolResult] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    error_strikes: dict[tuple[str, str], int] = field(default_factory=dict)
    withdrawn_tools: set[str] = field(default_factory=set)
    consecutive_tool_errors: int = 0
    compactions: int = 0
    ledger: BudgetLedger = field(default_factory=BudgetLedger)
    stop_reason: StopReason | None = None
    final_text: str = ""

    def user_messages(self) -> tuple[Message, ...]:
        """What the user actually typed, which is the only source a quote may come from."""
        return tuple(m for m in self.messages if m.role == "user")


def next_phase(state: AgentState, budget: Budget, now: float) -> AgentPhase:
    """Pure. Every termination decision lives here, which makes it a table test."""
    if state.phase is AgentPhase.PLAN:
        return AgentPhase.ACT if state.pending_calls else AgentPhase.FINALIZE
    if state.phase is AgentPhase.ACT:
        return AgentPhase.OBSERVE
    if state.phase is AgentPhase.OBSERVE:
        stop = _exhausted(state, budget, now)
        if stop is not None:
            state.stop_reason = stop
            return AgentPhase.FORCE_ANSWER
        return AgentPhase.PLAN
    if state.phase is AgentPhase.FORCE_ANSWER:
        return AgentPhase.FINALIZE
    if state.phase is AgentPhase.FINALIZE:
        return AgentPhase.GROUND
    return AgentPhase.DONE


def _exhausted(state: AgentState, budget: Budget, now: float) -> StopReason | None:
    if state.turn >= budget.max_turns:
        return StopReason.MAX_TURNS
    if state.total_tool_calls >= budget.max_tool_calls:
        return StopReason.MAX_TOOL_CALLS
    if state.ledger.tokens_exhausted(budget):
        return StopReason.TOKEN_BUDGET
    if state.ledger.cost_usd >= budget.max_cost_usd:
        return StopReason.COST_BUDGET
    if now >= state.ledger.deadline:
        return StopReason.DEADLINE
    if state.consecutive_tool_errors >= budget.max_consecutive_tool_errors:
        return StopReason.TOOL_ERROR_STREAK
    if any(n >= budget.max_repeat for n in state.call_counts.values()):
        return StopReason.REPEAT_LOOP
    return None
