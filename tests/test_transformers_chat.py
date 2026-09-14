"""The local pipeline, offline. A stub pipeline stands in for the checkpoint, so nothing loads."""

from __future__ import annotations

import json
import queue
from typing import Any

import pytest

from palate.config import Settings
from palate.providers.base import ChatProvider, Message, ToolCall, ToolSchema
from palate.providers.chat.transformers_local import TransformersLocalChat
from palate.providers.registry import build_chat
from palate.providers.streamacc import ToolCallAccumulator
from palate.providers.tokens import count_tokens
from palate.tools.catalog import build_registry

ALIAS = "qwen2.5-3b-instruct"

SEARCH_TOOL = ToolSchema(
    name="search_films",
    description="Hybrid retrieval over the unwatched corpus.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

TOOL_CALL = (
    '<tool_call>\n{"name": "search_films", "arguments": {"query": "slow and cold"}}\n</tool_call>'
)

_STOP = object()


class FakeTokenizer:
    """Records everything apply_chat_template was given, which is what the adapter gets wrong."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> str:
        self.seen.append(
            {
                "conversation": conversation,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
            }
        )
        rendered = json.dumps({"messages": conversation, "tools": tools})
        # A real template names the wrapper it wants a call in, and the adapter sniffs for it.
        return f"{rendered}\ncall inside <tool_call></tool_call>\n" if tools else rendered

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class FakeStreamer:
    """The queue half of TextIteratorStreamer, with no tokenizer and no model behind it."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._queue: queue.Queue[Any] = queue.Queue()

    def put(self, text: str) -> None:
        self._queue.put(text)

    def end(self) -> None:
        self._queue.put(_STOP)

    def __iter__(self) -> FakeStreamer:
        return self

    def __next__(self) -> str:
        item = self._queue.get(timeout=self.timeout)
        if item is _STOP:
            raise StopIteration
        return str(item)


class FakePipeline:
    """Replays one canned reply and records the generate kwargs it was called with."""

    def __init__(self, reply: str, tokenizer: FakeTokenizer) -> None:
        self.reply = reply
        self.tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"prompt": prompt, **kwargs})
        streamer = kwargs.get("streamer")
        if streamer is not None:
            for start in range(0, len(self.reply), 8):
                streamer.put(self.reply[start : start + 8])
            streamer.end()
        return [{"generated_text": self.reply}]


def build(reply: str, **overrides: Any) -> tuple[TransformersLocalChat, FakePipeline]:
    pipe = FakePipeline(reply, FakeTokenizer())
    provider = TransformersLocalChat(
        alias=ALIAS,
        device="cpu",
        pipe=pipe,
        streamer_factory=lambda tokenizer, timeout: FakeStreamer(timeout),
        **overrides,
    )
    return provider, pipe


async def test_the_template_gets_the_messages_and_the_tools_in_its_own_envelope() -> None:
    provider, pipe = build("Two films.")
    await provider.complete(
        [Message("system", "You recommend films."), Message("user", "something slow")],
        tools=[SEARCH_TOOL],
    )
    sent = pipe.tokenizer.seen[-1]
    assert sent["conversation"] == [
        {"role": "system", "content": "You recommend films."},
        {"role": "user", "content": "something slow"},
    ]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_films",
                "description": SEARCH_TOOL.description,
                "parameters": SEARCH_TOOL.parameters,
            },
        }
    ]
    assert sent["add_generation_prompt"] is True
    assert sent["tokenize"] is False


async def test_a_call_and_its_result_go_back_as_template_rows() -> None:
    provider, pipe = build("Two films.")
    await provider.complete(
        [
            Message(
                "assistant",
                "",
                tool_calls=(
                    ToolCall("call_1_0", "search_films", '{"query": "x"}', {"query": "x"}),
                ),
            ),
            Message("tool", "3 films", tool_call_id="call_1_0", name="search_films"),
        ],
        tools=[SEARCH_TOOL],
    )
    rows = pipe.tokenizer.seen[-1]["conversation"]
    assert rows[0]["tool_calls"] == [
        {"type": "function", "function": {"name": "search_films", "arguments": {"query": "x"}}}
    ]
    # No id reaches the model, so the result is correlated by name and by order.
    assert rows[1] == {"role": "tool", "name": "search_films", "content": "3 films"}


async def test_tool_choice_none_offers_no_tools_at_all() -> None:
    provider, pipe = build("Two films.")
    await provider.complete([Message("user", "hi")], tools=[SEARCH_TOOL], tool_choice="none")
    assert pipe.tokenizer.seen[-1]["tools"] is None


async def test_required_seeds_the_call_opener_so_prose_is_not_an_option() -> None:
    # The fake replies with the body only, the way the model does once the opener is in the prompt.
    body = '\n{"name": "search_films", "arguments": {"query": "slow and cold"}}\n</tool_call>'
    provider, pipe = build(body)
    done = await provider.complete(
        [Message("user", "hi")], tools=[SEARCH_TOOL], tool_choice="required"
    )
    assert pipe.calls[-1]["prompt"].endswith("<tool_call>\n")
    assert done.finish_reason == "tool_calls"
    assert done.tool_calls[0].arguments == {"query": "slow and cold"}


async def test_naming_a_tool_seeds_its_name_too() -> None:
    provider, pipe = build(' {"query": "slow and cold"}}\n</tool_call>')
    done = await provider.complete(
        [Message("user", "hi")], tools=[SEARCH_TOOL], tool_choice=("tool", "search_films")
    )
    assert pipe.calls[-1]["prompt"].endswith('<tool_call>\n{"name": "search_films", "arguments": ')
    assert done.tool_calls[0].name == "search_films"


async def test_auto_seeds_nothing_and_a_template_without_the_marker_is_left_alone() -> None:
    provider, pipe = build("Two films.")
    await provider.complete([Message("user", "hi")], tools=[SEARCH_TOOL])
    assert not pipe.calls[-1]["prompt"].endswith("<tool_call>\n")
    # The fake template never writes the marker, so there is nothing to seed and nothing is.
    assert provider._opener("no marker here", [SEARCH_TOOL], "required") == ""


async def test_a_tool_call_is_read_out_of_the_generated_text() -> None:
    provider, _ = build(f"I will look that up.\n{TOOL_CALL}")
    done = await provider.complete([Message("user", "hi")], tools=[SEARCH_TOOL])
    assert done.finish_reason == "tool_calls"
    assert done.tool_calls[0].name == "search_films"
    assert done.tool_calls[0].arguments == {"query": "slow and cold"}
    assert done.tool_calls[0].id == "call_1_0"
    assert done.content == "I will look that up."


async def test_an_answer_object_is_not_mistaken_for_a_call_when_no_tool_is_offered() -> None:
    provider, _ = build('{"preamble": "three films", "recommendations": []}')
    done = await provider.complete([Message("user", "hi")])
    assert done.tool_calls == ()
    assert done.finish_reason == "stop"
    assert done.content.startswith("{")


async def test_streaming_gives_the_text_then_the_call_then_the_counts() -> None:
    provider, _ = build(f"Looking now. {TOOL_CALL}")
    chunks = [c async for c in provider.stream([Message("user", "hi")], tools=[SEARCH_TOOL])]
    streamed = "".join(c.delta_text for c in chunks)
    # The call is text on this transport, so streaming it raw would put json in front of the user.
    assert streamed == "Looking now."
    assert "tool_call" not in streamed
    accumulator = ToolCallAccumulator(turn=1)
    for chunk in chunks:
        if chunk.tool_call_delta is not None:
            accumulator.add(chunk.tool_call_delta)
    calls = accumulator.finish()
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "slow and cold"}
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens > 0


async def test_streaming_with_no_tools_offered_still_arrives_piece_by_piece() -> None:
    provider, _ = build("Two slow films, both wet.")
    chunks = [c async for c in provider.stream([Message("user", "hi")])]
    deltas = [c.delta_text for c in chunks if c.delta_text]
    assert len(deltas) > 1
    assert "".join(deltas) == "Two slow films, both wet."


async def test_the_generation_kwargs_and_the_stop_string_are_applied() -> None:
    provider, pipe = build("Two films. THE END and then some")
    done = await provider.complete(
        [Message("user", "hi")], max_tokens=64, temperature=0.0, stop=["THE END"]
    )
    config = pipe.calls[-1]["generation_config"]
    assert config.max_new_tokens == 64
    assert config.do_sample is False
    assert pipe.calls[-1]["return_full_text"] is False
    # None, so the checkpoint's own sampling settings are still the ones in force.
    assert config.temperature is None
    assert done.content == "Two films. "


async def test_the_registry_schemas_arrive_with_no_ref_left_in_them() -> None:
    registry = build_registry()
    provider, pipe = build("Two films.")
    await provider.complete([Message("user", "hi")], tools=registry.schemas(style="chat_template"))
    sent = pipe.tokenizer.seen[-1]["tools"]
    assert [tool["function"]["name"] for tool in sent] == list(registry.names())
    assert "$ref" not in json.dumps(sent)


async def test_an_injected_pipeline_reaches_for_neither_models_toml_nor_the_hub() -> None:
    provider, _ = build("Two films.")
    done = await provider.complete([Message("user", "hi")])
    # A resolved pin would name the repo and the sha here, so the alias proves nothing loaded.
    assert done.response_model == ALIAS
    assert done.usage.exact is False


def test_token_counting_uses_the_tokenizer_only_once_it_is_loaded() -> None:
    messages = [Message("user", "something slow and cold")]
    loaded, _ = build("Two films.")
    assert loaded.count_tokens(messages) > 0
    cold = TransformersLocalChat(alias=ALIAS, device="cpu")
    assert cold.count_tokens(messages) == count_tokens(messages)


async def test_health_answers_without_loading_anything() -> None:
    provider, pipe = build("Two films.")
    report = await provider.health()
    assert report.ok is True
    assert pipe.calls == []


async def test_health_names_an_alias_models_toml_does_not_have() -> None:
    provider = TransformersLocalChat(alias="not-a-real-alias", device="cpu")
    report = await provider.health()
    assert report.ok is False
    assert "not-a-real-alias" in report.detail


async def test_the_adapter_refuses_to_claim_constrained_decoding() -> None:
    provider, _ = build("Two films.")
    capabilities = await provider.capabilities()
    assert capabilities.json_schema is False
    assert capabilities.parallel_tool_calls is False
    assert capabilities.schema_style == "chat_template"
    assert isinstance(provider, ChatProvider)


async def test_closing_drops_the_pipeline() -> None:
    provider, _ = build("Two films.")
    messages = [Message("user", "hi")]
    assert provider.count_tokens(messages) != count_tokens(messages)
    await provider.aclose()
    assert provider.count_tokens(messages) == count_tokens(messages)


async def test_the_registry_builds_the_local_provider_from_the_default_config() -> None:
    settings = Settings(chat={"device": "cpu"})
    provider = build_chat(settings)
    assert isinstance(provider, TransformersLocalChat)
    assert (provider.name, provider.model) == ("transformers", ALIAS)
    assert provider.max_new_tokens == settings.chat.max_tokens
    assert provider.context_window == settings.chat.num_ctx


@pytest.mark.weights
async def test_the_pinned_checkpoint_answers_on_the_real_device() -> None:
    """Needs several gigabytes of weights in the cache, so it never runs by default."""
    provider = TransformersLocalChat(alias=ALIAS)
    try:
        done = await provider.complete(
            [Message("user", "Name one slow film. Two words.")], max_tokens=32
        )
    finally:
        await provider.aclose()
    assert done.content.strip()
    assert done.usage.input_tokens > 0
    assert done.response_model.startswith("Qwen/Qwen2.5-3B-Instruct@")
