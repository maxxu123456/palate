"""Five checks over one run's own evidence, and a number that says how it was reached."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from palate.agent.answer import StructuredAnswer
from palate.agent.repair import repair
from palate.ground.check import GroundednessChecker, GroundingReport, kind_counts
from palate.ground.claims import extract, kinds_of, numbers_in, sentences
from palate.ground.evidence_index import EvidenceIndex

STALKER: dict[str, Any] = {
    "film_id": 1398,
    "title": "Stalker",
    "year": 1979,
    "runtime": 161,
    "directors": ["Andrei Tarkovsky"],
    "countries": ["Soviet Union"],
    "keywords": ["zone", "wish"],
    "overview": "Three men journey into the Zone toward a room that grants wishes.",
    "your_rating": 4.5,
}

HORSE: dict[str, Any] = {
    "film_id": 551,
    "title": "The Turin Horse",
    "year": 2011,
    "runtime": 146,
    "directors": ["Bela Tarr"],
    "overview": "A farmer and his daughter live through six days of relentless wind.",
}


class FakeNLI:
    """One fixed entailment score, so the paraphrase arm is testable with no torch."""

    model_key = "fake-nli"

    def __init__(self, score: float) -> None:
        self.score = score
        self.pairs: list[tuple[str, str]] = []

    async def entails(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [self.score] * len(pairs)

    async def aclose(self) -> None:
        return None


def evidence(*records: dict[str, Any]) -> EvidenceIndex:
    index = EvidenceIndex()
    index.absorb({"films": list(records)})
    return index


def answer(film_id: int, why: str) -> StructuredAnswer:
    return StructuredAnswer.model_validate(
        {"preamble": "", "recommendations": [{"film_id": film_id, "why": why}]}
    )


async def check(
    film_id: int, why: str, *, nli: FakeNLI | None = None, records: Sequence[dict[str, Any]] = ()
) -> GroundingReport:
    index = evidence(*(records or (STALKER, HORSE)))
    checker = GroundednessChecker(index, nli)
    return await checker.check(answer(film_id, why))


def test_sentences_never_silently_drop_a_comparative() -> None:
    split = sentences("Colder and even slower than the last one. You rated that 4 stars.")
    assert len(split) == 2
    assert split[0].startswith("Colder and even slower")


def test_a_sentence_is_typed_by_every_pattern_it_matches() -> None:
    gazetteer = frozenset({"Andrei Tarkovsky", "Stalker"})
    kinds = kinds_of("Andrei Tarkovsky made it in 1979.", gazetteer)
    assert set(kinds) == {"entity", "number"}
    assert kinds_of("It might be too slow for you.", gazetteer) == ("opinion",)
    assert kinds_of("Three men cross a wasteland.", gazetteer) == ("plot",)


def test_numbers_are_read_as_what_they_claim_to_be() -> None:
    found = dict(numbers_in("A 161 minute film from 1979 you gave 4.5 stars."))
    assert found["runtime"] == 161.0
    assert found["year"] == 1979.0
    assert found["rating"] == 4.5


async def test_a_name_credited_on_another_film_is_caught() -> None:
    report = await check(551, "Andrei Tarkovsky shoots it with enormous patience.")
    assert report.grounded_ratio == 0.0
    assert report.claims[0].method == "set"
    assert "not credited on The Turin Horse" in report.claims[0].evidence
    assert report.dropped_film_ids == (551,)


async def test_the_same_name_on_its_own_film_passes() -> None:
    report = await check(1398, "Andrei Tarkovsky shoots it with enormous patience.")
    assert report.grounded_ratio == 1.0
    assert report.claims[0].method == "set"


async def test_a_wrong_year_is_an_exact_miss() -> None:
    report = await check(1398, "Released in 1980, it is a very long film indeed.")
    assert report.grounded_ratio == 0.0
    assert report.claims[0].method == "exact"
    assert "year is 1979" in report.claims[0].evidence


async def test_a_wrong_runtime_is_an_exact_miss_with_no_tolerance() -> None:
    report = await check(1398, "It runs 162 minutes, which is a long evening.")
    assert report.grounded_ratio == 0.0
    report = await check(1398, "It runs 161 minutes, which is a long evening.")
    assert report.grounded_ratio == 1.0


async def test_a_claim_about_the_users_own_stars_is_checked_against_the_row() -> None:
    good = await check(1398, "You rated it 4.5 stars, which is near the top for you.")
    assert good.grounded_ratio == 1.0
    bad = await check(1398, "You rated it 2 stars, which is near the bottom for you.")
    assert bad.grounded_ratio == 0.0


async def test_hedged_language_is_exempt_rather_than_scored() -> None:
    report = await check(1398, "It might be too slow for you.")
    assert report.claims[0].method == "exempt"
    assert report.scored == 0
    assert report.grounded_ratio == 1.0


async def test_a_plot_sentence_that_is_in_the_overview_costs_nothing() -> None:
    report = await check(1398, "Three men journey into the Zone toward a room.")
    assert report.claims[0].method == "span"
    assert report.claims[0].supported


async def test_a_paraphrase_misses_the_span_check_and_stops_there_without_nli() -> None:
    report = await check(
        1398, "A trio crosses a post industrial wasteland hunting a wish granting chamber."
    )
    assert report.claims[0].method == "span"
    assert not report.claims[0].supported
    assert report.nli_available is False


async def test_the_same_paraphrase_reaches_nli_when_there_is_one() -> None:
    nli = FakeNLI(0.9)
    report = await check(
        1398,
        "A trio crosses a post industrial wasteland hunting a wish granting chamber.",
        nli=nli,
    )
    assert report.claims[0].method == "nli"
    assert report.claims[0].supported
    assert report.claims[0].score == pytest.approx(0.9)
    assert report.nli_available is True
    assert "Three men journey" in nli.pairs[0][0]


async def test_nli_below_the_threshold_stays_unsupported() -> None:
    report = await check(
        1398,
        "A trio crosses a post industrial wasteland hunting a wish granting chamber.",
        nli=FakeNLI(0.2),
    )
    assert report.claims[0].method == "nli"
    assert not report.claims[0].supported


async def test_a_film_no_tool_ever_returned_cannot_be_supported() -> None:
    report = await check(9999, "Three men journey into the Zone toward a room.")
    assert report.grounded_ratio == 0.0
    assert "no tool result named this film" in report.claims[0].evidence


async def test_the_ratio_counts_only_what_was_scored() -> None:
    why = (
        "Andrei Tarkovsky shoots it with patience. "
        "Released in 1980, which is wrong. "
        "It might be too slow for you."
    )
    report = await check(1398, why)
    assert report.scored == 2
    assert report.grounded_ratio == pytest.approx(0.5)
    assert report.method_counts["exempt"] == 1
    assert len(report.unsupported_sentences) == 1


async def test_the_per_kind_split_is_available_because_the_arms_fail_differently() -> None:
    report = await check(551, "Andrei Tarkovsky made it in 2011.")
    counts = kind_counts(report.claims)
    assert counts["entity"] == (0, 1)
    assert counts["number"] == (1, 1)


async def test_repair_drops_the_bad_sentence_and_keeps_the_good_one() -> None:
    why = "Andrei Tarkovsky shoots it with patience. Released in 1980, which is wrong."
    report = await check(1398, why)
    repaired, removed = repair(answer(1398, why), report)
    assert len(removed) == 1
    assert repaired.recommendations[0].why == "Andrei Tarkovsky shoots it with patience."


async def test_a_recommendation_that_loses_every_sentence_is_dropped_whole() -> None:
    why = "Released in 1980, which is wrong."
    report = await check(1398, why)
    repaired, removed = repair(answer(1398, why), report)
    assert repaired.recommendations == []
    assert removed


async def test_repair_touches_nothing_when_everything_held_up() -> None:
    why = "Three men journey into the Zone toward a room."
    report = await check(1398, why)
    original = answer(1398, why)
    repaired, removed = repair(original, report)
    assert repaired == original
    assert removed == []


def test_the_evidence_index_merges_what_several_tools_said_about_one_film() -> None:
    index = EvidenceIndex()
    index.absorb({"films": [{"film_id": 1398, "title": "Stalker"}]})
    index.absorb({"films": [{"film_id": 1398, "overview": "Three men.", "runtime": 161}]})
    film = index.get(1398)
    assert film is not None
    assert film.title == "Stalker"
    assert film.runtime == 161
    assert "Three men." in film.premise()
    assert len(index) == 1


def test_the_gazetteer_only_holds_names_the_run_actually_saw() -> None:
    index = evidence(STALKER)
    names = index.gazetteer()
    assert "Andrei Tarkovsky" in names
    assert "Bela Tarr" not in names


def test_extraction_attaches_every_sentence_to_the_id_it_came_from() -> None:
    both = StructuredAnswer.model_validate(
        {
            "preamble": "",
            "recommendations": [
                {"film_id": 1398, "why": "Three men journey into the Zone."},
                {"film_id": 551, "why": "Bela Tarr shoots six days of wind."},
            ],
        }
    )
    claims = extract(both, evidence(STALKER, HORSE))
    assert {c.film_id for c in claims} == {1398, 551}
    assert all(c.film_id is not None for c in claims)
