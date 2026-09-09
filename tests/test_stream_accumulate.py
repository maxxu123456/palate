"""Two streaming dialects, one content side rescue, and the same tool call out of each."""

from __future__ import annotations

from collections.abc import AsyncIterator

from palate.providers.base import ChatChunk, ToolCallDelta
from palate.providers.contentcalls import extract_tool_calls
from palate.providers.streamacc import ToolCallAccumulator, accumulate, parse_arguments

INDEXED = (
    ToolCallDelta(0, id="call_a1", name="search_films"),
    ToolCallDelta(0, arguments_fragment='{"qu'),
    ToolCallDelta(0, arguments_fragment='ery":"slow'),
    ToolCallDelta(0, arguments_fragment=' and cold"}'),
)

UNINDEXED = (
    ToolCallDelta(None, id="call_a1", name="search_films", arguments_fragment='{"qu'),
    ToolCallDelta(None, arguments_fragment='ery":"slow'),
    ToolCallDelta(None, arguments_fragment=' and cold"}'),
)


def fold(deltas: tuple[ToolCallDelta, ...]) -> tuple[tuple[str, str, object], ...]:
    accumulator = ToolCallAccumulator()
    for delta in deltas:
        accumulator.add(delta)
    return tuple((c.id, c.name, c.arguments) for c in accumulator.finish())


async def stream(*chunks: ChatChunk) -> AsyncIterator[ChatChunk]:
    for chunk in chunks:
        yield chunk


def test_both_delta_dialects_fold_to_the_same_call() -> None:
    assert fold(INDEXED) == fold(UNINDEXED)
    assert fold(INDEXED) == (("call_a1", "search_films", {"query": "slow and cold"}),)


def test_two_indexed_calls_do_not_bleed_into_each_other() -> None:
    accumulator = ToolCallAccumulator()
    for delta in (
        ToolCallDelta(0, id="a", name="get_film", arguments_fragment='{"id":'),
        ToolCallDelta(1, id="b", name="check_watched", arguments_fragment='{"ids":'),
        ToolCallDelta(0, arguments_fragment="1398}"),
        ToolCallDelta(1, arguments_fragment="[603]}"),
    ):
        accumulator.add(delta)
    calls = accumulator.finish()
    assert [c.name for c in calls] == ["get_film", "check_watched"]
    assert calls[0].arguments == {"id": 1398}
    assert calls[1].arguments == {"ids": [603]}


def test_an_unindexed_name_starts_a_second_call() -> None:
    accumulator = ToolCallAccumulator()
    for delta in (
        ToolCallDelta(None, name="get_film", arguments_fragment='{"id": 1}'),
        ToolCallDelta(None, name="get_film", arguments_fragment='{"id": 2}'),
    ):
        accumulator.add(delta)
    assert len(accumulator.finish()) == 2


def test_a_missing_id_is_synthesised_from_the_turn() -> None:
    accumulator = ToolCallAccumulator(turn=3)
    accumulator.add(ToolCallDelta(0, name="get_film", arguments_fragment="{}"))
    assert accumulator.finish()[0].id == "call_3_0"


def test_unparseable_arguments_are_kept_verbatim() -> None:
    accumulator = ToolCallAccumulator()
    accumulator.add(ToolCallDelta(0, name="get_film", arguments_fragment='{"id": '))
    call = accumulator.finish()[0]
    assert call.arguments is None
    assert call.arguments_json == '{"id": '


def test_parse_arguments_rejects_a_bare_list() -> None:
    assert parse_arguments("[1, 2]") is None
    assert parse_arguments("") == {}


async def test_a_stream_that_reports_usage_is_exact() -> None:
    from palate.providers.base import Usage

    result = await accumulate(
        stream(
            ChatChunk(delta_text="slow and"),
            ChatChunk(delta_text=" cold"),
            ChatChunk(finish_reason="stop", usage=Usage(input_tokens=12, output_tokens=4)),
        )
    )
    assert result.content == "slow and cold"
    assert result.usage.exact is True
    assert result.usage.output_tokens == 4


async def test_a_stream_that_drops_usage_says_so() -> None:
    result = await accumulate(
        stream(ChatChunk(delta_text="slow and cold"), ChatChunk(finish_reason="stop"))
    )
    assert result.usage.exact is False
    assert result.usage.output_tokens > 0


async def test_a_streamed_tool_call_ends_the_stream_as_tool_calls() -> None:
    result = await accumulate(
        stream(
            *(ChatChunk(tool_call_delta=d) for d in INDEXED),
            ChatChunk(finish_reason="tool_calls"),
        )
    )
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].arguments == {"query": "slow and cold"}


def test_a_tagged_block_in_content_becomes_a_tool_call() -> None:
    calls, left = extract_tool_calls(
        'Let me look.\n<tool_call>\n{"name": "search_films", "arguments": {"query": "x"}}\n'
        "</tool_call>",
        known=["search_films"],
    )
    assert left == "Let me look."
    assert calls[0].source == "content"
    assert calls[0].arguments == {"query": "x"}


def test_a_fenced_json_block_becomes_a_tool_call() -> None:
    calls, _ = extract_tool_calls(
        'Here:\n```json\n{"name": "get_film", "arguments": {"id": 1398}}\n```'
    )
    assert calls[0].name == "get_film"
    assert calls[0].arguments == {"id": 1398}


def test_a_bare_object_becomes_a_tool_call_and_leaves_no_prose() -> None:
    calls, left = extract_tool_calls('{"name": "get_film", "parameters": {"id": 1}}')
    assert (calls[0].name, left) == ("get_film", "")


def test_arguments_that_arrive_as_a_json_string_are_reparsed() -> None:
    calls, _ = extract_tool_calls('{"name": "get_film", "arguments": "{\\"id\\": 7}"}')
    assert calls[0].arguments == {"id": 7}


def test_an_unknown_tool_name_is_not_rescued() -> None:
    calls, left = extract_tool_calls(
        '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', known=["search_films"]
    )
    assert calls == ()
    assert "rm_rf" in left


def test_ordinary_prose_is_left_alone() -> None:
    calls, left = extract_tool_calls("Three films, all long and bleak.")
    assert calls == ()
    assert left == "Three films, all long and bleak."
