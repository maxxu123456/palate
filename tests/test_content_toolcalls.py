"""Small local models emit tool calls as text, and the default provider is a small local model."""

from __future__ import annotations

import pytest
from fixtures.agent import BUDGET, Calls, build_registry, context_factory

from palate.agent.loop import AgentLoop, RunOutcome, needs_retrieval
from palate.agent.prompts import PromptRegistry
from palate.config import Settings
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn

ANSWER = "Two of those match."

TAGGED = '<tool_call>{"name": "search_films", "arguments": {"query": "slow and cold"}}</tool_call>'

FENCED = '```json\n{"name": "search_films", "arguments": {"query": "long and bleak"}}\n```'

BARE = '{"name": "search_films", "arguments": {"query": "quiet and sad"}}'

NESTED = '{"name": "search_films", "arguments": "{\\"query\\": \\"cold and still\\"}"}'


async def drive(turns: list[ScriptedTurn], seen: Calls, *, asked: str) -> RunOutcome:
    """One run against a provider that answers in prose rather than in tool call fields."""
    provider = FakeChatProvider(turns)
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), Settings())
    return await agent.run_to_completion(
        asked, session_id="ses_content", budget=BUDGET, ctx_factory=context_factory()
    )


@pytest.mark.parametrize(
    ("content", "query"),
    [
        (TAGGED, "slow and cold"),
        (FENCED, "long and bleak"),
        (BARE, "quiet and sad"),
        (NESTED, "cold and still"),
    ],
    ids=["tagged", "fenced", "bare", "nested"],
)
async def test_a_tool_call_written_as_text_is_still_dispatched(content: str, query: str) -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(text=content), ScriptedTurn(text=ANSWER)], seen, asked="something slow"
    )
    assert seen.searches == [query]
    assert outcome.text == ANSWER


async def test_the_lifted_call_says_where_it_came_from() -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(text=TAGGED), ScriptedTurn(text=ANSWER)], seen, asked="something slow"
    )
    proposed = [e for e in outcome.events if type(e).__name__ == "ToolCallProposed"]
    assert proposed[0].source == "content"  # type: ignore[union-attr]


async def test_the_tagged_block_does_not_survive_into_the_transcript() -> None:
    seen = Calls()
    prose = f"Let me look.\n{TAGGED}"
    outcome = await drive(
        [ScriptedTurn(text=prose), ScriptedTurn(text=ANSWER)], seen, asked="something slow"
    )
    assistant = [m for m in outcome.state.messages if m.role == "assistant"]
    assert "<tool_call>" not in assistant[0].content
    assert assistant[0].content == "Let me look."


async def test_content_parsing_can_be_turned_off() -> None:
    seen = Calls()
    settings = Settings()
    settings.chat.content_toolcall_parse = False
    settings.chat.force_tools_on_turn0 = False
    provider = FakeChatProvider([ScriptedTurn(text=TAGGED)])
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), settings)
    outcome = await agent.run_to_completion(
        "something slow",
        session_id="ses_off",
        budget=BUDGET,
        ctx_factory=context_factory(settings),
    )
    assert seen.searches == []
    assert outcome.text == TAGGED


async def test_turn_zero_text_with_no_calls_is_retried_with_tools_required() -> None:
    seen = Calls()
    provider = FakeChatProvider(
        [
            ScriptedTurn(text="I could suggest a few things."),
            ScriptedTurn(text=BARE),
            ScriptedTurn(text=ANSWER),
        ]
    )
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), Settings())
    outcome = await agent.run_to_completion(
        "recommend something slow and cold for tonight",
        session_id="ses_required",
        budget=BUDGET,
        ctx_factory=context_factory(),
    )
    assert seen.searches == ["quiet and sad"]
    assert outcome.text == ANSWER
    assert len(provider.calls) == 3


async def test_a_pure_follow_up_is_not_retried_with_tools_required() -> None:
    seen = Calls()
    provider = FakeChatProvider([ScriptedTurn(text=ANSWER)])
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), Settings())
    outcome = await agent.run_to_completion(
        "why", session_id="ses_follow", budget=BUDGET, ctx_factory=context_factory()
    )
    assert len(provider.calls) == 1
    assert outcome.text == ANSWER


@pytest.mark.parametrize(
    "text",
    ["why", "what about the second one", "tell me more", "the first"],
)
def test_a_follow_up_does_not_need_retrieval(text: str) -> None:
    assert not needs_retrieval(text)


@pytest.mark.parametrize(
    "text",
    [
        "something slow and cold",
        "why do you think I would like anything by Bela Tarr",
        "nothing russian please",
        "a comedy",
    ],
)
def test_anything_else_does(text: str) -> None:
    assert needs_retrieval(text)
