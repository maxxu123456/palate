"""Five checks, cheapest sufficient method each, and a number that says what it cost."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from palate.agent.answer import StructuredAnswer
from palate.ground.claims import Claim, ClaimKind, extract, names_in, numbers_in
from palate.ground.evidence_index import EvidenceIndex, FilmEvidence


@runtime_checkable
class NLIModel(Protocol):
    """Anything that scores how far a premise entails a hypothesis, in [0, 1]."""

    model_key: str

    async def entails(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...

    async def aclose(self) -> None: ...


type Method = Literal["set", "exact", "span", "nli", "exempt"]

# Tolerance on a runtime claim is zero. A film is 161 minutes or it is not.
RUNTIME_TOLERANCE = 0

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Shared function words never make a plot sentence a substring hit on their own.
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "his",
        "her",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "with",
        "you",
        "your",
    ]
)

SPAN_OVERLAP = 0.6


@dataclass(frozen=True, slots=True)
class CheckedClaim:
    """One claim, the method that settled it, and what settled it."""

    claim: Claim
    supported: bool
    method: Method
    evidence: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """The number, and everything needed to argue with it."""

    grounded_ratio: float
    claims: tuple[CheckedClaim, ...]
    unsupported_sentences: tuple[str, ...]
    dropped_film_ids: tuple[int, ...]
    nli_available: bool
    method_counts: Mapping[str, int]

    @property
    def scored(self) -> int:
        """Claims that counted. An exempt opinion is not one of them."""
        return sum(1 for c in self.claims if c.method != "exempt")


EMPTY = GroundingReport(1.0, (), (), (), False, {})


class GroundednessChecker:
    """Deterministic first, and the model only for a paraphrase the span check missed."""

    def __init__(
        self,
        evidence: EvidenceIndex,
        nli: NLIModel | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.evidence = evidence
        self.nli = nli
        self.threshold = threshold

    async def check(self, answer: StructuredAnswer, run_id: str = "") -> GroundingReport:
        """Every claim under every kind it matched, supported only if all of them pass."""
        claims = extract(answer, self.evidence)
        checked = [self._cheap(claim) for claim in claims]
        checked = await self._paraphrase(checked)
        scored = [c for c in checked if c.method != "exempt"]
        supported = sum(1 for c in scored if c.supported)
        unsupported = tuple(dict.fromkeys(c.claim.sentence for c in checked if not c.supported))
        dropped = tuple(
            sorted({c.claim.film_id for c in checked if not c.supported and c.claim.film_id})
        )
        counts: dict[str, int] = {}
        for check in checked:
            counts[check.method] = counts.get(check.method, 0) + 1
        return GroundingReport(
            grounded_ratio=supported / len(scored) if scored else 1.0,
            claims=tuple(checked),
            unsupported_sentences=unsupported,
            dropped_film_ids=dropped,
            nli_available=self.nli is not None,
            method_counts=counts,
        )

    def _cheap(self, claim: Claim) -> CheckedClaim:
        film = self.evidence.get(claim.film_id) if claim.film_id is not None else None
        if claim.kind == "opinion":
            return CheckedClaim(claim, True, "exempt", "hedged language is not scored")
        if film is None:
            return CheckedClaim(claim, False, "set", "no tool result named this film")
        if claim.kind == "entity":
            return _entities(claim, film, self.evidence.gazetteer())
        if claim.kind in ("number", "user_rating"):
            return _numbers(claim, film)
        return _span(claim, film)

    async def _paraphrase(self, checked: Sequence[CheckedClaim]) -> list[CheckedClaim]:
        pending = [
            (index, check)
            for index, check in enumerate(checked)
            if check.method == "span" and not check.supported
        ]
        out = list(checked)
        if not pending or self.nli is None:
            return out
        pairs = [(self._premise(check.claim), check.claim.sentence) for _, check in pending]
        scores = await self.nli.entails(pairs)
        for (index, check), score in zip(pending, scores, strict=True):
            out[index] = CheckedClaim(
                claim=check.claim,
                supported=score >= self.threshold,
                method="nli",
                evidence=self._premise(check.claim)[:200],
                score=score,
            )
        return out

    def _premise(self, claim: Claim) -> str:
        film = self.evidence.get(claim.film_id) if claim.film_id is not None else None
        return film.premise() if film is not None else ""


def _entities(claim: Claim, film: FilmEvidence, gazetteer: frozenset[str]) -> CheckedClaim:
    """Every name the run saw has to belong to this film, which catches the wrong attachment."""
    names = names_in(claim.sentence, gazetteer)
    wrong = [name for name in names if not film.owns(name)]
    if wrong:
        return CheckedClaim(
            claim, False, "set", f"{', '.join(wrong)} is not credited on {film.title}"
        )
    return CheckedClaim(claim, True, "set", ", ".join(names) or film.title)


def _numbers(claim: Claim, film: FilmEvidence) -> CheckedClaim:
    """Years, runtimes and stars, compared exactly against the row."""
    for kind, value in numbers_in(claim.sentence):
        actual = _actual(kind, film)
        if actual is None:
            return CheckedClaim(claim, False, "exact", f"no {kind} recorded for {film.title}")
        if abs(actual - value) > (RUNTIME_TOLERANCE if kind == "runtime" else 0.0):
            return CheckedClaim(
                claim, False, "exact", f"{film.title} {kind} is {actual:g}, not {value:g}"
            )
    return CheckedClaim(claim, True, "exact", film.title)


def _actual(kind: str, film: FilmEvidence) -> float | None:
    if kind == "year":
        return None if film.year is None else float(film.year)
    if kind == "runtime":
        return None if film.runtime is None else float(film.runtime)
    return film.your_rating


def _span(claim: Claim, film: FilmEvidence) -> CheckedClaim:
    """Substring containment first, at zero cost. A miss is what the NLI arm then sees."""
    premise = film.premise()
    if not premise:
        return CheckedClaim(claim, False, "span", f"no overview retrieved for {film.title}")
    words = [w for w in _words(claim.sentence) if w not in _STOP]
    if not words:
        return CheckedClaim(claim, True, "span", "no content words to check")
    present = sum(1 for w in words if w in _words(premise))
    share = present / len(words)
    return CheckedClaim(
        claim,
        share >= SPAN_OVERLAP,
        "span",
        f"{present} of {len(words)} content words appear in the retrieved text",
        round(share, 3),
    )


def _words(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _WORD.finditer(text)}


def kind_counts(claims: Sequence[CheckedClaim]) -> dict[ClaimKind, tuple[int, int]]:
    """Supported and total per kind, because the entity arm and the plot arm fail differently."""
    out: dict[ClaimKind, tuple[int, int]] = {}
    for check in claims:
        supported, total = out.get(check.claim.kind, (0, 0))
        out[check.claim.kind] = (supported + int(check.supported), total + 1)
    return out
