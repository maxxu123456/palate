"""The paths that work: one turn of tools, a fan out, a write, and a cached repeat."""

from __future__ import annotations

import orjson
import pytest
from fixtures.agent import BUDGET, Calls, build_registry, call, context_factory

from palate.agent.events import (
    RunFinished,
    RunStarted,
    TextDelta,
    ToolCallFinished,
    ToolCallProposed,
    ToolCallStarted,
    TurnStarted,
)
from palate.agent.loop import AgentLoop, RunOutcome
from palate.agent.prompts import PromptRegistry
from palate.agent.state import StopReason
from palate.config import Settings
from palate.providers.base import ChatCapabilities, Usage
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn

ANSWER = "Three of these are long and cold, which is what you asked for."

SEARCH = call("search_films", '{"query": "slow and cold", "limit": 3}')


async def drive(
    turns: list[ScriptedTurn],
    seen: Calls,
    *,
    capabilities: ChatCapabilities | None = None,
    loop_last: bool = False,
    asked: str = "something slow and cold",
) -> RunOutcome:
    """One whole run against the scripted provider, with no network anywhere."""
    provider = FakeChatProvider(turns, capabilities=capabilities, loop_last=loop_last)
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), Settings())
    return await agent.run_to_completion(
        asked,
        session_id="ses_happy",
        budget=BUDGET,
        ctx_factory=context_factory(),
    )


async def test_a_tool_turn_then_an_answer_produces_the_answer() -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(text="let me look", tool_calls=(SEARCH,)), ScriptedTurn(text=ANSWER)], seen
    )
    assert outcome.text == ANSWER
    assert outcome.stop_reason is StopReason.ANSWERED
    assert not outcome.failed
    assert seen.searches == ["slow and cold"]


async def test_the_event_stream_is_ordered_and_typed() -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(text="looking", tool_calls=(SEARCH,)), ScriptedTurn(text=ANSWER)], seen
    )
    kinds = [type(event) for event in outcome.events]
    assert kinds[0] is RunStarted
    assert kinds[-1] is RunFinished
    assert kinds.index(TurnStarted) < kinds.index(ToolCallProposed)
    assert kinds.index(ToolCallProposed) < kinds.index(ToolCallStarted)
    assert kinds.index(ToolCallStarted) < kinds.index(ToolCallFinished)
    finished = outcome.of_type(ToolCallFinished)[0]
    assert finished.ok
    assert finished.summary == "search_films returned 3"


async def test_planning_text_is_thinking_and_the_answer_is_not() -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(text="let me look", tool_calls=(SEARCH,)), ScriptedTurn(text=ANSWER)], seen
    )
    channels = {d.channel for d in outcome.of_type(TextDelta)}
    assert channels == {"thinking", "answer"}
    answer = "".join(d.text for d in outcome.of_type(TextDelta) if d.channel == "answer")
    assert answer == ANSWER


async def test_the_tool_result_reaches_the_transcript_as_a_tool_message() -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(text="", tool_calls=(SEARCH,)), ScriptedTurn(text=ANSWER)], seen
    )
    tools = [m for m in outcome.state.messages if m.role == "tool"]
    assert len(tools) == 1
    body = orjson.loads(tools[0].content)
    assert body["ok"] is True
    assert len(body["data"]["films"]) == 3
    assert tools[0].tool_call_id == "c1"


async def test_two_reads_in_one_turn_actually_overlap() -> None:
    seen = Calls()
    pair = (
        call("slow_tool", '{"seconds": 0.05}', ident="a"),
        call("slow_tool", '{"seconds": 0.05}', ident="b"),
    )
    outcome = await drive(
        [ScriptedTurn(tool_calls=pair), ScriptedTurn(text=ANSWER)],
        seen,
        capabilities=ChatCapabilities(context_window=8192),
    )
    assert seen.overlapped
    assert len(outcome.results) == 2


async def test_a_write_runs_after_every_read_in_the_same_turn() -> None:
    seen = Calls()
    mixed = (
        call("record_note", '{"text": "no musicals"}', ident="w"),
        call("search_films", '{"query": "musical free"}', ident="r"),
    )
    await drive([ScriptedTurn(tool_calls=mixed), ScriptedTurn(text=ANSWER)], seen)
    assert seen.order.index("search_films") < seen.order.index("record_note")


async def test_a_write_clears_the_cache_of_what_it_invalidates() -> None:
    seen = Calls()
    query = call("search_films", '{"query": "slow and cold", "limit": 3}', ident="s1")
    again = call("search_films", '{"query": "slow and cold", "limit": 3}', ident="s2")
    write = call("record_note", '{"text": "no musicals"}', ident="w")
    await drive(
        [
            ScriptedTurn(tool_calls=(query,)),
            ScriptedTurn(tool_calls=(write,)),
            ScriptedTurn(tool_calls=(again,)),
            ScriptedTurn(text=ANSWER),
        ],
        seen,
    )
    assert seen.searches == ["slow and cold", "slow and cold"]


async def test_the_second_identical_call_is_served_from_the_cache() -> None:
    seen = Calls()
    again = call("search_films", '{"query": "slow and cold", "limit": 3}', ident="c2")
    outcome = await drive(
        [
            ScriptedTurn(tool_calls=(SEARCH,)),
            ScriptedTurn(tool_calls=(again,)),
            ScriptedTurn(text=ANSWER),
        ],
        seen,
    )
    assert seen.searches == ["slow and cold"]
    assert outcome.results[1].cache_hit
    assert outcome.results[1].meta["repeated_call"] is True


async def test_the_ledger_carries_the_tokens_the_provider_reported() -> None:
    seen = Calls()
    outcome = await drive(
        [
            ScriptedTurn(tool_calls=(SEARCH,), usage=Usage(input_tokens=100, output_tokens=20)),
            ScriptedTurn(text=ANSWER, usage=Usage(input_tokens=300, output_tokens=40)),
        ],
        seen,
    )
    finished = outcome.of_type(RunFinished)[0]
    assert finished.input_tokens == 400
    assert finished.output_tokens == 60
    assert finished.tool_calls == 1
    assert finished.turns == 1


async def test_the_system_prompt_is_sent_once_and_carries_the_preference_block() -> None:
    seen = Calls()
    captured: list[str] = []

    def expect(messages: object, tools: object) -> None:
        assert isinstance(messages, list)
        captured.append(messages[0].content)

    await drive([ScriptedTurn(text=ANSWER, expect=expect)], seen, asked="why")
    assert len(captured) == 1
    assert "Never recommend a film the user has already watched" in captured[0]
    assert "exclude_languages" in captured[0]


async def test_streaming_and_draining_produce_the_same_answer() -> None:
    seen = Calls()
    turns = [ScriptedTurn(text="looking", tool_calls=(SEARCH,)), ScriptedTurn(text=ANSWER)]
    provider = FakeChatProvider(turns)
    agent = AgentLoop(provider, build_registry(seen), PromptRegistry(), Settings())
    streamed = [
        event
        async for event in agent.run(
            "something slow and cold",
            session_id="ses_stream",
            budget=BUDGET,
            ctx_factory=context_factory(),
        )
    ]
    drained = await drive(turns, Calls())
    assert [type(e) for e in streamed] == [type(e) for e in drained.events]


@pytest.mark.parametrize("limit", [0, 99])
async def test_an_out_of_range_limit_is_clamped_rather_than_rejected(limit: int) -> None:
    seen = Calls()
    wide = call("search_films", f'{{"query": "slow", "limit": {limit}}}')
    outcome = await drive([ScriptedTurn(tool_calls=(wide,)), ScriptedTurn(text=ANSWER)], seen)
    assert outcome.results[0].ok
    assert outcome.results[0].meta["clamped"] == ["limit"]
