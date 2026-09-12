"""The structural leak stops. An invented title has no path into the output at all."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import orjson
import pytest
from fixtures.agent import BUDGET, call, completion, films_seen, world_context
from fixtures.synth import pipeline

from palate.agent.answer import StructuredAnswer, assemble, facts_for, films_json, resolve
from palate.agent.loop import AgentLoop, RunOutcome, profile_digest
from palate.agent.prompts import PromptRegistry
from palate.config import Settings
from palate.ground.evidence_index import EvidenceIndex
from palate.providers.base import Completion, Message, ToolSchema
from palate.providers.chat.fake import FakeChatProvider
from palate.taste.memory import ensure_session
from palate.tools.catalog import build_registry

SESSION = "ses_leak"

INVENTED = "The Mirror of Nobody, a 1974 masterpiece by a director who never existed."

SEARCH = call("search_films", '{"query": "something slow and cold", "limit": 5}')


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("leak"))
    ensure_session(built.db, SESSION)
    yield built
    built.close()


def scripted(build: Any) -> Any:
    """Search, read, then say whatever the test asked for."""

    def reply(messages: list[Message], tools: list[ToolSchema]) -> Completion:
        records = films_seen(messages)
        if tools and not records:
            return completion("", (SEARCH,))
        if tools and not any("overview" in r for r in records):
            wanted = [r["film_id"] for r in records[:3]]
            return completion(
                "", (call("get_film", orjson.dumps({"film_ids": wanted}).decode(), ident="g1"),)
            )
        return completion(orjson.dumps(build(records)).decode())

    return reply


async def drive(fitted: pipeline.Fitted, build: Any) -> RunOutcome:
    provider = FakeChatProvider(scripted(build))
    agent = AgentLoop(provider, build_registry(), PromptRegistry(), Settings())
    return await agent.run_to_completion(
        "something slow and cold",
        session_id=SESSION,
        budget=BUDGET,
        ctx_factory=world_context(fitted, SESSION),
    )


async def test_a_title_the_model_made_up_never_reaches_the_prose(
    fitted: pipeline.Fitted,
) -> None:
    def invents(records: list[dict[str, Any]]) -> dict[str, Any]:
        read = next(r for r in records if r.get("overview"))
        return {
            "preamble": "",
            "recommendations": [
                {"film_id": read["film_id"], "why": read["overview"]},
                {"film_id": 987_654, "why": INVENTED},
            ],
        }

    outcome = await drive(fitted, invents)
    assert "Mirror of Nobody" not in outcome.text
    assert "987654" not in outcome.text
    assert outcome.state.resolved is not None
    assert outcome.state.resolved.unresolvable_ids == (987_654,)


async def test_every_title_in_the_prose_is_a_row_in_the_database(
    fitted: pipeline.Fitted,
) -> None:
    def plain(records: list[dict[str, Any]]) -> dict[str, Any]:
        read = [r for r in records if r.get("overview")][:2]
        return {
            "preamble": "Two of these.",
            "recommendations": [{"film_id": r["film_id"], "why": r["overview"]} for r in read],
        }

    outcome = await drive(fitted, plain)
    assert outcome.state.answer is not None
    ids = [r.film_id for r in outcome.state.answer.recommendations]
    facts = facts_for(fitted.db.read(), ids)
    for line in outcome.text.splitlines():
        if line and line[0].isdigit() and "." in line[:3]:
            assert any(f.title in line for f in facts.values())


async def test_the_reason_the_model_wrote_is_the_only_free_text_that_survives(
    fitted: pipeline.Fitted,
) -> None:
    def plain(records: list[dict[str, Any]]) -> dict[str, Any]:
        read = next(r for r in records if r.get("overview"))
        return {
            "preamble": "One film.",
            "recommendations": [{"film_id": read["film_id"], "why": read["overview"]}],
            "caveats": ["The corpus here is small."],
            "could_not_check": ["Whether it is streaming anywhere."],
        }

    outcome = await drive(fitted, plain)
    assert "The corpus here is small." in outcome.text
    assert "Whether it is streaming anywhere." in outcome.text
    assert "One film." in outcome.text


def test_assemble_drops_a_recommendation_with_no_row_behind_it() -> None:
    answer = StructuredAnswer.model_validate(
        {
            "preamble": "here you go",
            "recommendations": [{"film_id": 424_242, "why": "a reason long enough to pass"}],
        }
    )
    prose = assemble(answer, {})
    assert prose == "here you go"
    assert films_json(answer, {}) == ()


def test_a_watched_film_is_dropped_before_anything_is_rendered(
    fitted: pipeline.Fitted,
) -> None:
    watched = fitted.world.rated[0].tmdb_id
    answer = StructuredAnswer.model_validate(
        {
            "preamble": "",
            "recommendations": [{"film_id": watched, "why": "a reason long enough to pass"}],
        }
    )
    evidence = EvidenceIndex()
    evidence.absorb({"films": [{"film_id": watched}]})
    report = resolve(answer, fitted.db, evidence)
    assert report.constraint_violation
    assert report.watched_ids == (watched,)
    assert report.answer.recommendations == []
    assert assemble(report.answer, report.facts) == ""


def test_the_profile_digest_carries_counts_and_never_a_title(
    fitted: pipeline.Fitted,
) -> None:
    digest = profile_digest(fitted.profile)
    titles = {f.title for f in fitted.world.films}
    assert not any(title in digest for title in titles)
    assert str(fitted.profile.n_rated) in digest
    assert fitted.profile.tier in digest


async def test_the_system_prompt_names_no_film_at_all(fitted: pipeline.Fitted) -> None:
    sent: list[str] = []

    def watching(messages: list[Message], tools: list[ToolSchema]) -> Completion:
        sent.append(messages[0].content)
        return completion("nothing to add")

    provider = FakeChatProvider(watching)
    agent = AgentLoop(provider, build_registry(), PromptRegistry(), Settings())
    await agent.run_to_completion(
        "why",
        session_id=SESSION,
        budget=BUDGET,
        ctx_factory=world_context(fitted, SESSION),
    )
    titles = {f.title for f in fitted.world.films}
    assert sent
    assert not any(title in sent[0] for title in titles)
