"""Repair is deletion. Regenerating costs a call and tends to produce a second bad claim."""

from __future__ import annotations

from palate.agent.answer import Recommendation, StructuredAnswer
from palate.ground.check import GroundingReport
from palate.ground.claims import sentences


def repair(answer: StructuredAnswer, report: GroundingReport) -> tuple[StructuredAnswer, list[str]]:
    """Drop unsupported sentences. Drop a recommendation that loses every supporting one."""
    bad = {_fold(s) for s in report.unsupported_sentences}
    if not bad:
        return answer, []
    kept: list[Recommendation] = []
    removed: list[str] = []
    for recommendation in answer.recommendations:
        surviving = [s for s in sentences(recommendation.why) if _fold(s) not in bad]
        removed.extend(s for s in sentences(recommendation.why) if _fold(s) in bad)
        why = " ".join(surviving).strip()
        if len(why) < _MIN_WHY:
            continue
        kept.append(recommendation.model_copy(update={"why": why}))
    return answer.model_copy(update={"recommendations": kept}), removed


# Recommendation.why has a floor of its own, so a shorter remainder cannot be rebuilt.
_MIN_WHY = 10


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()
