"""list_preferences: so the model never re-asks something the user already told it."""

from __future__ import annotations

from functools import partial
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field

from palate.taste.memory import PreferenceStore
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "Read back saved preferences so you do not re-ask something the user already told you, "
    "and so you can say when a preference was set."
)


class ListPreferencesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["session", "durable", "all"] = "all"


class PreferenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pref_id: int
    polarity: str
    target_kind: str
    target_label: str
    strength: int
    hardness: str
    scope: str
    affected_films: int
    evidence_quote: str
    created_at: str
    compiles_to_filter: bool


class ListPreferencesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferences: list[PreferenceRow]
    meta: dict[str, object] = Field(default_factory=dict)


def read(store: PreferenceStore, session_id: str, scope: str) -> ListPreferencesResult:
    """Live rows only. A superseded row is history, not a preference."""
    rows = [p for p in store.active(session_id) if scope == "all" or p.scope == scope]
    return ListPreferencesResult(
        preferences=[
            PreferenceRow(
                pref_id=p.pref_id,
                polarity=p.polarity,
                target_kind=p.target_kind,
                target_label=p.target_label,
                strength=p.strength,
                hardness=p.hardness,
                scope=p.scope,
                affected_films=p.affected_films,
                evidence_quote=p.evidence_quote,
                created_at=p.created_at,
                compiles_to_filter=p.is_hard,
            )
            for p in rows
        ],
        meta={"count": len(rows), "returned": len(rows), "scope": scope},
    )


async def handler(args: ListPreferencesArgs, ctx: ToolContext) -> ListPreferencesResult:
    """One indexed read off the event loop."""
    ctx.check_deadline()
    store = ctx.require_prefs()
    return await anyio.to_thread.run_sync(partial(read, store, ctx.session_id, args.scope))


SPEC = ToolSpec(
    name="list_preferences",
    description=DESCRIPTION,
    args_model=ListPreferencesArgs,
    result_model=ListPreferencesResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(ListPreferencesArgs(),),
    cost_hint_ms=10,
    max_result_chars=3000,
)
