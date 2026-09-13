"""Durable memory, read back and undone. Nothing here is ever deleted."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from palate.api.deps import State
from palate.tools.catalog import list_preferences

router = APIRouter(tags=["preferences"])


class PreferenceUndone(BaseModel):
    """What a delete actually did, which is supersede rather than remove."""

    pref_id: int
    superseded: bool


@router.get("/preferences")
async def preferences(
    state: State,
    session_id: Annotated[str | None, Query()] = None,
    scope: Annotated[Literal["session", "durable", "all"], Query()] = "all",
) -> list_preferences.ListPreferencesResult:
    """Live rows only. A superseded row is history, not a preference."""
    return list_preferences.read(state.prefs, session_id or "", scope)


@router.delete("/preferences/{pref_id}")
async def undo(pref_id: int, state: State) -> PreferenceUndone:
    """Supersede the row so as_of stays exact. A row already superseded is a 404."""
    if not state.prefs.undo(pref_id):
        raise HTTPException(status_code=404, detail=f"no live preference with id {pref_id}")
    return PreferenceUndone(pref_id=pref_id, superseded=True)
