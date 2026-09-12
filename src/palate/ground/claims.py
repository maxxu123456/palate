"""What the answer actually claims, split deterministically rather than by a model."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from palate.agent.answer import StructuredAnswer
from palate.ground.evidence_index import EvidenceIndex, proper_nouns

type ClaimKind = Literal["entity", "number", "user_rating", "plot", "opinion"]

# Reported under the first kind that matches, most checkable first.
PRECEDENCE: tuple[ClaimKind, ...] = ("user_rating", "number", "entity", "plot", "opinion")

# Split on terminal punctuation followed by space, so a 4.5 star rating stays whole.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_OPINION = re.compile(
    r"\b(might|maybe|probably|perhaps|could be|if you|worth|feels?|seems?|i think|"
    r"you may|arguably|likely)\b",
    re.IGNORECASE,
)

_USER_RATING = re.compile(
    r"\b(you (gave|rated|loved|liked|hated|scored)|your \d(\.\d)? ?(star|stars)?)\b",
    re.IGNORECASE,
)

_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")

_RUNTIME = re.compile(r"\b(\d{2,3})\s*(?:minutes|minute|mins|min)\b", re.IGNORECASE)

_STARS = re.compile(r"\b([0-5](?:\.5)?)\s*(?:stars|star)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Claim:
    """One sentence, attached to the recommendation it came from, typed by pattern."""

    sentence: str
    film_id: int | None
    kind: ClaimKind
    spans: tuple[tuple[int, int], ...] = ()


def sentences(text: str) -> list[str]:
    """Plain sentence split. Nothing subtle, and it never silently drops a comparative."""
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def kinds_of(sentence: str, gazetteer: frozenset[str]) -> tuple[ClaimKind, ...]:
    """Every kind this sentence has to pass. Opinion alone means nothing is scored."""
    found: list[ClaimKind] = []
    if _USER_RATING.search(sentence):
        found.append("user_rating")
    if numbers_in(sentence):
        found.append("number")
    if names_in(sentence, gazetteer):
        found.append("entity")
    if found:
        return tuple(found)
    return ("opinion",) if _OPINION.search(sentence) else ("plot",)


def names_in(sentence: str, gazetteer: frozenset[str]) -> list[str]:
    """Capitalised runs the run actually saw. A name nobody retrieved is not checkable."""
    known = {_fold(n): n for n in gazetteer}
    out: list[str] = []
    for candidate in proper_nouns(sentence):
        found = known.get(_fold(candidate))
        if found is not None and found not in out:
            out.append(found)
    return out


def numbers_in(sentence: str) -> list[tuple[str, float]]:
    """Years, runtimes and star ratings, each tagged with what it claims to be."""
    out: list[tuple[str, float]] = []
    out.extend(("runtime", float(m.group(1))) for m in _RUNTIME.finditer(sentence))
    out.extend(("rating", float(m.group(1))) for m in _STARS.finditer(sentence))
    taken = {int(value) for kind, value in out if kind == "runtime"}
    out.extend(
        ("year", float(m.group(1)))
        for m in _YEAR.finditer(sentence)
        if int(m.group(1)) not in taken
    )
    return out


def spans_for(sentence: str, names: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Where each checked name sits, for the detail view."""
    out: list[tuple[int, int]] = []
    for name in names:
        at = sentence.casefold().find(name.casefold())
        if at >= 0:
            out.append((at, at + len(name)))
    return tuple(out)


def extract(answer: StructuredAnswer, evidence: EvidenceIndex) -> list[Claim]:
    """One claim per (sentence, kind). The film attachment cannot fail, it comes from the id."""
    gazetteer = evidence.gazetteer()
    claims: list[Claim] = []
    for recommendation in answer.recommendations:
        for sentence in sentences(recommendation.why):
            names = names_in(sentence, gazetteer)
            for kind in kinds_of(sentence, gazetteer):
                claims.append(
                    Claim(
                        sentence=sentence,
                        film_id=recommendation.film_id,
                        kind=kind,
                        spans=spans_for(sentence, names) if kind == "entity" else (),
                    )
                )
    return claims


def reported(kinds: Sequence[ClaimKind]) -> ClaimKind:
    """The kind a claim is filed under when it matched several."""
    for kind in PRECEDENCE:
        if kind in kinds:
            return kind
    return "plot"


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()
