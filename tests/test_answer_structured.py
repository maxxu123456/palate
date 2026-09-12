"""Prose is built from database rows, so an invented title has no path to the user."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import orjson
import pytest
from fixtures.agent import BUDGET, call, completion, films_seen, world_context
from fixtures.synth import pipeline

from palate.agent.answer import StructuredAnswer, assemble, facts_for, parse, resolve
from palate.agent.events import Recommendations, TextDelta
from palate.agent.loop import AgentLoop, RunOutcome
from palate.agent.prompts import PromptRegistry
from palate.config import Settings
from palate.providers.base import ChatCapabilities, Completion, Message, ToolSchema
from palate.providers.chat.fake import FakeChatProvider
from palate.taste.memory import ensure_session
from palate.tools.catalog import build_registry

SESSION = "ses_answer"

SEARCH = call("search_films", '{"query": "something slow and cold", "limit": 5}')

type Builder = Callable[[list[dict[str, Any]]], dict[str, Any]]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("answer"))
    ensure_session(built.db, SESSION)
    yield built
    built.close()


def scripted(build: Builder, *, raw: Sequence[str] = ()) -> Any:
    """Search, read the records, then answer with whatever the test wants."""
    replies = list(raw)

    def reply(messages: list[Message], tools: list[ToolSchema]) -> Completion:
        records = films_seen(messages)
        if tools and not records:
            return completion("", (SEARCH,))
        if tools and not any("overview" in r for r in records):
            wanted = [r["film_id"] for r in records[:3]]
            arguments = orjson.dumps({"film_ids": wanted}).decode()
            return completion("", (call("get_film", arguments, ident="g1"),))
        if replies:
            return completion(replies.pop(0))
        return completion(orjson.dumps(build(records)).decode())

    return reply


async def drive(
    fitted: pipeline.Fitted,
    build: Builder,
    *,
    raw: Sequence[str] = (),
    capabilities: ChatCapabilities | None = None,
) -> RunOutcome:
    """One whole run against the planted world and the real tool registry."""
    provider = FakeChatProvider(scripted(build, raw=raw), capabilities=capabilities)
    agent = AgentLoop(provider, build_registry(), PromptRegistry(), Settings())
    return await agent.run_to_completion(
        "something slow and cold",
        session_id=SESSION,
        budget=BUDGET,
        ctx_factory=world_context(fitted, SESSION),
    )


def one(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One recommendation whose reason is a sentence from the record get_film returned."""
    read = next(r for r in records if r.get("overview"))
    return {
        "preamble": "Three of these sit in the same corner of your history.",
        "recommendations": [{"film_id": read["film_id"], "why": read["overview"]}],
        "caveats": [],
        "could_not_check": [],
    }


async def test_the_title_in_the_prose_comes_from_the_database(
    fitted: pipeline.Fitted,
) -> None:
    outcome = await drive(fitted, one)
    assert outcome.state.answer is not None
    film_id = outcome.state.answer.recommendations[0].film_id
    facts = facts_for(fitted.db.read(), [film_id])
    assert facts[film_id].title in outcome.text
    assert str(facts[film_id].year) in outcome.text


async def test_an_id_the_corpus_does_not_hold_never_reaches_the_prose(
    fitted: pipeline.Fitted,
) -> None:
    def invented(records: list[dict[str, Any]]) -> dict[str, Any]:
        answer = one(records)
        answer["recommendations"].append(
            {"film_id": 999_999, "why": "A masterpiece nobody has ever made, which you would love."}
        )
        return answer

    outcome = await drive(fitted, invented)
    assert outcome.state.resolved is not None
    assert outcome.state.resolved.unresolvable_ids == (999_999,)
    assert "999999" not in outcome.text
    assert outcome.state.resolved.unresolvable_id_rate == pytest.approx(0.5)


async def test_a_film_no_tool_returned_is_dropped_even_though_it_exists(
    fitted: pipeline.Fitted,
) -> None:
    corpus = [f.tmdb_id for f in fitted.world.films]
    hidden: list[int] = []

    def unseen(records: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [r["film_id"] for r in records]
        hidden.append(next(i for i in corpus if i not in ids))
        answer = one(records)
        answer["recommendations"].append(
            {"film_id": hidden[0], "why": "Nothing in this run ever looked this film up at all."}
        )
        return answer

    outcome = await drive(fitted, unseen)
    assert outcome.state.resolved is not None
    dropped = outcome.state.resolved.unseen_ids + outcome.state.resolved.watched_ids
    assert hidden[0] in dropped
    assert outcome.state.answer is not None
    assert hidden[0] not in [r.film_id for r in outcome.state.answer.recommendations]


async def test_recommending_a_watched_film_is_recorded_as_a_violation(
    fitted: pipeline.Fitted,
) -> None:
    watched = fitted.world.rated[0].tmdb_id

    def seen(records: list[dict[str, Any]]) -> dict[str, Any]:
        answer = one(records)
        answer["recommendations"].append(
            {"film_id": watched, "why": "You have already seen this one, which is the bug."}
        )
        return answer

    outcome = await drive(fitted, seen)
    assert outcome.state.constraint_violation == (watched,)
    facts = facts_for(fitted.db.read(), [watched])
    assert facts[watched].title not in outcome.text


async def test_a_broken_json_answer_gets_exactly_one_repair_retry(
    fitted: pipeline.Fitted,
) -> None:
    outcome = await drive(fitted, one, raw=["not json at all"])
    assert outcome.state.answer is not None
    assert outcome.state.answer.recommendations
    assert outcome.text


async def test_two_broken_answers_fall_back_to_the_text_the_model_gave(
    fitted: pipeline.Fitted,
) -> None:
    outcome = await drive(fitted, one, raw=["still not json", "nor is this", "nor this"])
    assert outcome.state.answer is None
    assert outcome.text


async def test_the_preamble_streams_and_the_film_list_arrives_once(
    fitted: pipeline.Fitted,
) -> None:
    outcome = await drive(fitted, one)
    answered = "".join(d.text for d in outcome.of_type(TextDelta) if d.channel == "answer")
    assert answered.strip() == "Three of these sit in the same corner of your history."
    lists = outcome.of_type(Recommendations)
    assert len(lists) == 1
    assert lists[0].films
    assert lists[0].films[0]["title"]
    assert lists[0].prose == outcome.text


async def test_a_provider_with_json_schema_is_given_one(fitted: pipeline.Fitted) -> None:
    caps = ChatCapabilities(json_schema=True, context_window=8192)
    outcome = await drive(fitted, one, capabilities=caps)
    assert outcome.state.answer is not None
    assert outcome.text


def test_parse_finds_the_object_inside_prose() -> None:
    body = '{"preamble": "hi", "recommendations": []}'
    assert parse(f"Here you go:\n```json\n{body}\n```", provider_json=False).preamble == "hi"
    assert parse(f"blah {body} blah", provider_json=False).preamble == "hi"
    assert parse(body, provider_json=True).preamble == "hi"


def test_assemble_never_writes_a_film_the_database_does_not_know(
    fitted: pipeline.Fitted,
) -> None:
    answer = StructuredAnswer.model_validate(
        {"preamble": "", "recommendations": [{"film_id": 42, "why": "a reason long enough"}]}
    )
    assert assemble(answer, {}) == ""


def test_resolve_keeps_the_order_the_model_asked_for(fitted: pipeline.Fitted) -> None:
    ids = [f.tmdb_id for f in fitted.world.films[:3]]
    answer = StructuredAnswer.model_validate(
        {
            "preamble": "",
            "recommendations": [
                {"film_id": film_id, "why": "a reason long enough to pass"} for film_id in ids
            ],
        }
    )
    evidence = _evidence(ids)
    report = resolve(answer, fitted.db, evidence)
    kept = [r.film_id for r in report.answer.recommendations]
    assert kept == [i for i in ids if i in kept]


def _evidence(ids: list[int]) -> Any:
    from palate.ground.evidence_index import EvidenceIndex

    index = EvidenceIndex()
    index.absorb({"films": [{"film_id": film_id} for film_id in ids]})
    return index
