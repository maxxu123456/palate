"""The scripted provider the whole offline suite runs against."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from palate.providers.base import ChatProvider, Message, ToolCall
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn


async def drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in stream]


async def test_the_fake_provider_replays_a_script_and_records_the_calls() -> None:
    seen: list[int] = []
    provider = FakeChatProvider(
        [
            ScriptedTurn(tool_calls=(ToolCall("c1", "search_films", '{"query": "x"}'),)),
            ScriptedTurn(text="Two films.", expect=lambda m, t: seen.append(len(m))),
        ]
    )
    first = await provider.complete([Message("user", "hi")])
    second = await provider.complete([Message("user", "hi"), Message("tool", "ok")])
    assert first.finish_reason == "tool_calls"
    assert second.content == "Two films."
    assert seen == [2]
    assert len(provider.calls) == 2


async def test_the_fake_provider_can_repeat_its_last_turn_forever() -> None:
    provider = FakeChatProvider([ScriptedTurn(text="again")], loop_last=True)
    for _ in range(5):
        assert (await provider.complete([Message("user", "hi")])).content == "again"


async def test_the_fake_provider_streams_the_same_answer_it_completes() -> None:
    provider = FakeChatProvider([ScriptedTurn(text="slow and cold")], loop_last=True)
    chunks = await drain(provider.stream([Message("user", "hi")]))
    assert "".join(c.delta_text for c in chunks) == "slow and cold"
    assert chunks[-1].finish_reason == "stop"


async def test_the_fake_provider_satisfies_the_protocol() -> None:
    assert isinstance(FakeChatProvider([]), ChatProvider)
