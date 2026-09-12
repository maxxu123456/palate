"""The typed events a run emits. The UI never sees raw provider JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from palate.providers.base import JSONObject


@dataclass(frozen=True, slots=True)
class RunStarted:
    type: Literal["run.started"] = "run.started"
    run_id: str = ""
    session_id: str = ""
    model: str = ""
    budget: JSONObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnStarted:
    type: Literal["turn.started"] = "turn.started"
    turn: int = 0
    tools_offered: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TextDelta:
    type: Literal["text.delta"] = "text.delta"
    text: str = ""
    channel: Literal["thinking", "answer"] = "answer"


@dataclass(frozen=True, slots=True)
class ToolCallProposed:
    type: Literal["tool.proposed"] = "tool.proposed"
    call_id: str = ""
    name: str = ""
    arguments: JSONObject = field(default_factory=dict)
    source: Literal["field", "content"] = "field"


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    type: Literal["tool.started"] = "tool.started"
    call_id: str = ""
    name: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    type: Literal["tool.finished"] = "tool.finished"
    call_id: str = ""
    name: str = ""
    ok: bool = True
    # One line for the UI card, never the payload.
    summary: str = ""
    meta: JSONObject = field(default_factory=dict)
    latency_ms: int = 0
    undo_token: str | None = None


@dataclass(frozen=True, slots=True)
class PreferenceRecorded:
    type: Literal["preference.recorded"] = "preference.recorded"
    pref_id: int = 0
    label: str = ""
    polarity: str = ""
    hardness: str = ""
    affected_films: int = 0
    undo_token: str = ""


@dataclass(frozen=True, slots=True)
class GroundingChecked:
    type: Literal["grounding.checked"] = "grounding.checked"
    grounded_ratio: float = 1.0
    nli_available: bool = False
    unsupported: tuple[str, ...] = ()
    dropped_film_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Recommendations:
    type: Literal["recommendations"] = "recommendations"
    films: tuple[JSONObject, ...] = ()
    prose: str = ""


@dataclass(frozen=True, slots=True)
class RunFinished:
    type: Literal["run.finished"] = "run.finished"
    stop_reason: str = ""
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0


@dataclass(frozen=True, slots=True)
class RunFailed:
    type: Literal["run.failed"] = "run.failed"
    error_code: str = ""
    message: str = ""
    stop_reason: str = ""


type AgentEvent = (
    RunStarted
    | TurnStarted
    | TextDelta
    | ToolCallProposed
    | ToolCallStarted
    | ToolCallFinished
    | PreferenceRecorded
    | GroundingChecked
    | Recommendations
    | RunFinished
    | RunFailed
)
