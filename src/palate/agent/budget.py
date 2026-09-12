"""The limits that end a run, and the ledger that tracks what has been spent against them."""

from __future__ import annotations

import math
from dataclasses import dataclass

from palate.config import AgentSettings
from palate.providers.base import Usage


@dataclass(frozen=True, slots=True)
class Budget:
    """Every bound on one run, frozen so a tool cannot widen its own leash."""

    max_turns: int = 8
    max_tool_calls: int = 16
    max_parallel_calls: int = 4
    max_prompt_tokens: int = 24_000
    max_completion_tokens: int = 4_000
    max_cost_usd: float = 0.10
    max_wall_s: float = 90.0
    max_consecutive_tool_errors: int = 3
    max_repeat: int = 3

    @classmethod
    def from_settings(cls, agent: AgentSettings) -> Budget:
        """The configured limits, which is what the CLI and the API both run with."""
        return cls(
            max_turns=agent.max_turns,
            max_tool_calls=agent.max_tool_calls,
            max_parallel_calls=agent.max_parallel_calls,
            max_prompt_tokens=agent.max_prompt_tokens,
            max_completion_tokens=agent.max_completion_tokens,
            max_cost_usd=agent.max_cost_usd,
            max_wall_s=agent.max_wall_s,
            max_consecutive_tool_errors=agent.max_consecutive_tool_errors,
            max_repeat=agent.max_repeat,
        )

    def as_json(self) -> dict[str, float]:
        """The budget as the run.started event carries it."""
        return {
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_cost_usd": self.max_cost_usd,
            "max_wall_s": self.max_wall_s,
        }


@dataclass(slots=True)
class BudgetLedger:
    """What has been spent so far. Estimates are corrected from real Usage after every call."""

    deadline: float = math.inf
    prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    exact: bool = True

    def start(self, budget: Budget, now: float) -> None:
        """Pin the deadline once, so a retry can never push a run past it."""
        self.deadline = now + budget.max_wall_s

    def observe_prompt(self, tokens: int) -> None:
        """Size of the request about to go out, which is what the context guard reads."""
        self.prompt_tokens = tokens

    def charge(self, usage: Usage, cost_usd: float | None = None) -> None:
        """Fold one finished call in. A dropped usage chunk makes the whole ledger inexact."""
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += cost_usd or 0.0
        self.exact = self.exact and usage.exact

    def tokens_exhausted(self, budget: Budget) -> bool:
        """True when the transcript or the generated text has outgrown its cap."""
        return (
            self.prompt_tokens >= budget.max_prompt_tokens
            or self.output_tokens >= budget.max_completion_tokens
        )

    def remaining_s(self, now: float) -> float:
        """Seconds left before the deadline, never negative."""
        return max(0.0, self.deadline - now)
