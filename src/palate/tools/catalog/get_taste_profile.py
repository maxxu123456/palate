"""get_taste_profile: the core quality problem in tool form."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from palate.taste.modes import TasteMode
from palate.taste.profile import TasteProfile
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "The user's taste model: named modes with exemplars and support counts, top and bottom "
    "directors and keywords, rating histogram, and the profile tier. Check this before the "
    "first search on a vague request. Never generalise from a support count of 1."
)

TOP_N = 8


class TasteProfileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode_id: int | None = None
    polarity: Literal["like", "dislike", "both"] = "both"


class ModeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode_id: int
    polarity: str
    label: str | None = None
    n_members: int
    mean_signal: float
    coherence: float
    confidence: float
    exemplar_film_ids: list[int] = Field(default_factory=list)


class EntitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    entity_id: str
    name: str
    n: int
    affinity: float


class TasteProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: str
    n_rated: int
    n_reliable_dated: int
    mean_rating: float
    rating_histogram: dict[str, int] = Field(default_factory=dict)
    modes: list[ModeSummary] = Field(default_factory=list)
    top_directors: list[EntitySummary] = Field(default_factory=list)
    bottom_directors: list[EntitySummary] = Field(default_factory=list)
    top_keywords: list[EntitySummary] = Field(default_factory=list)
    bottom_keywords: list[EntitySummary] = Field(default_factory=list)
    stale: bool = False
    meta: dict[str, object] = Field(default_factory=dict)


def _mode(mode: TasteMode, polarity: str) -> ModeSummary:
    return ModeSummary(
        mode_id=mode.mode_id,
        polarity=polarity,
        label=mode.label,
        n_members=mode.n_members,
        mean_signal=round(float(mode.mean_signal), 4),
        coherence=round(float(mode.coherence), 4),
        confidence=round(float(mode.confidence), 4),
        exemplar_film_ids=list(mode.exemplars),
    )


def _entities(profile: TasteProfile, kind: str, *, best: bool) -> list[EntitySummary]:
    rows = list(profile.affinities.get(kind, ()))
    rows.sort(key=lambda a: a.affinity, reverse=best)
    return [
        EntitySummary(
            kind=row.kind,
            entity_id=row.entity_id,
            name=row.name,
            n=row.n,
            affinity=round(float(row.affinity), 4),
        )
        for row in rows[:TOP_N]
    ]


def summarise(profile: TasteProfile, args: TasteProfileArgs) -> TasteProfileResult:
    """The profile as rows with support counts beside every claim it could support."""
    modes: list[ModeSummary] = []
    if args.polarity in ("like", "both"):
        modes.extend(_mode(m, "like") for m in profile.modes)
    if args.polarity in ("dislike", "both"):
        modes.extend(_mode(m, "dislike") for m in profile.anti_modes)
    if args.mode_id is not None:
        modes = [m for m in modes if m.mode_id == args.mode_id]
    return TasteProfileResult(
        tier=profile.tier,
        n_rated=profile.n_rated,
        n_reliable_dated=profile.n_reliable_dated,
        mean_rating=round(profile.mean_rating, 3),
        rating_histogram=dict(profile.rating_histogram),
        modes=modes,
        top_directors=_entities(profile, "director", best=True),
        bottom_directors=_entities(profile, "director", best=False),
        top_keywords=_entities(profile, "keyword", best=True),
        bottom_keywords=_entities(profile, "keyword", best=False),
        stale=profile.stale,
        meta={"count": len(modes), "returned": len(modes), "profile_id": profile.profile_id},
    )


async def handler(args: TasteProfileArgs, ctx: ToolContext) -> TasteProfileResult:
    """No IO past the already loaded profile, so this is the cheapest tool there is."""
    ctx.check_deadline()
    return summarise(ctx.require_profile(), args)


SPEC = ToolSpec(
    name="get_taste_profile",
    description=DESCRIPTION,
    args_model=TasteProfileArgs,
    result_model=TasteProfileResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(TasteProfileArgs(),),
    cost_hint_ms=5,
    max_result_chars=5000,
)
