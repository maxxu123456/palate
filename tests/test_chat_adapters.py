"""What we send matters more than what we parse, so the request body is asserted too."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from palate.errors import ModelNotFound, ProviderAuthError, ProviderBadRequest
from palate.providers.base import (
    ChatProvider,
    Message,
    ResponseFormat,
    ToolCall,
    ToolSchema,
)
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.providers.chat.hf_inference import HFInferenceChat, as_payload
from palate.providers.chat.ollama import OllamaChat
from palate.providers.chat.openai_compat import OpenAICompatChat
from palate.providers.chat.openrouter import ATTRIBUTION, OpenRouterChat
from palate.providers.streamacc import ToolCallAccumulator

CASSETTES = Path(__file__).parent / "fixtures" / "http"

SEARCH_TOOL = ToolSchema(
    name="search_films",
    description="Hybrid retrieval over the unwatched corpus.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


def cassette(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((CASSETTES / name).read_text(encoding="utf-8"))
    return loaded


class Recorder:
    """A transport that records the request and replays a cassette response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def body(self) -> dict[str, Any]:
        sent: dict[str, Any] = json.loads(self.requests[-1].content)
        return sent

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = int(self.response.get("status", 200))
        if "json" in self.response:
            return httpx.Response(status, json=self.response["json"])
        if "sse" in self.response:
            text = "".join(f"data: {line}\n\n" for line in self.response["sse"])
            return httpx.Response(status, text=text)
        lines = "".join(json.dumps(item) + "\n" for item in self.response["ndjson"])
        return httpx.Response(status, text=lines)


def client_for(recorder: Recorder) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=recorder.transport())


async def drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in stream]


async def test_openai_compat_sends_the_body_the_cassette_recorded() -> None:
    tape = cassette("openai_compat_complete.json")
    recorder = Recorder(tape["response"])
    async with client_for(recorder) as http:
        provider = OpenAICompatChat(
            base_url="http://127.0.0.1:8000/v1",
            model="qwen3-30b-a3b",
            api_key=SecretStr("scrubbed"),
            client=http,
        )
        done = await provider.complete(
            [
                Message("system", "You recommend films."),
                Message("user", "something slow and cold like Stalker"),
            ],
            tools=[SEARCH_TOOL],
            max_tokens=256,
        )
    assert recorder.body() == tape["request"]["json"]
    assert recorder.requests[-1].headers["authorization"] == "Bearer scrubbed"
    assert done.finish_reason == "tool_calls"
    assert done.tool_calls[0].name == "search_films"
    assert done.tool_calls[0].arguments == {"query": "slow cold contemplative"}
    assert done.usage.input_tokens == 412
    assert done.usage.cached_input_tokens == 128
    # What we asked for and what served it are different fields on purpose.
    assert (done.model, done.response_model) == ("qwen3-30b-a3b", "qwen3-30b-a3b-instruct")


async def test_a_json_schema_response_format_goes_out_whole() -> None:
    recorder = Recorder(cassette("openai_compat_complete.json")["response"])
    schema = {"type": "object", "properties": {"picks": {"type": "array"}}}
    async with client_for(recorder) as http:
        provider = OpenAICompatChat(base_url="http://x/v1", model="m", client=http)
        await provider.complete(
            [Message("user", "hi")],
            response_format=ResponseFormat("json_schema", schema=schema, name="answer"),
        )
    sent = recorder.body()["response_format"]
    assert sent["json_schema"]["name"] == "answer"
    assert sent["json_schema"]["schema"] == schema


async def test_a_tool_result_carries_its_call_id() -> None:
    recorder = Recorder(cassette("openai_compat_complete.json")["response"])
    async with client_for(recorder) as http:
        provider = OpenAICompatChat(base_url="http://x/v1", model="m", client=http)
        await provider.complete(
            [
                Message(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("call_a1", "search_films", '{"query": "x"}'),),
                ),
                Message("tool", "3 films", tool_call_id="call_a1", name="search_films"),
            ]
        )
    messages = recorder.body()["messages"]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'
    assert messages[1] == {"role": "tool", "content": "3 films", "tool_call_id": "call_a1"}


async def test_index_keyed_tool_deltas_fold_into_one_call() -> None:
    recorder = Recorder(cassette("openai_compat_stream_tools.json")["response"])
    accumulator = ToolCallAccumulator()
    async with client_for(recorder) as http:
        provider = OpenAICompatChat(base_url="http://x/v1", model="m", client=http)
        chunks = await drain(provider.stream([Message("user", "hi")], tools=[SEARCH_TOOL]))
    for chunk in chunks:
        if chunk.tool_call_delta is not None:
            accumulator.add(chunk.tool_call_delta)
    calls = accumulator.finish()
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "slow and cold"}
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens == 412
    assert recorder.body()["stream"] is True
    assert recorder.body()["stream_options"] == {"include_usage": True}


async def test_deltas_with_no_index_fold_the_same_way() -> None:
    recorder = Recorder(cassette("openai_compat_stream_noindex.json")["response"])
    accumulator = ToolCallAccumulator()
    async with client_for(recorder) as http:
        provider = OpenAICompatChat(base_url="http://x/v1", model="m", client=http)
        for chunk in await drain(provider.stream([Message("user", "hi")])):
            if chunk.tool_call_delta is not None:
                accumulator.add(chunk.tool_call_delta)
    calls = accumulator.finish()
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "slow and cold"}


async def test_openrouter_asks_for_cost_and_names_itself() -> None:
    tape = cassette("openrouter_complete.json")
    recorder = Recorder(tape["response"])
    async with client_for(recorder) as http:
        provider = OpenRouterChat(
            model="qwen/qwen3-30b-a3b",
            api_key=SecretStr("scrubbed"),
            client=http,
            order=["deepinfra"],
            allow_fallbacks=False,
        )
        done = await provider.complete([Message("user", "name one slow film")])
    assert recorder.body() == tape["request"]["json"]
    for header, value in ATTRIBUTION.items():
        assert recorder.requests[-1].headers[header] == value
    # Provider reported dollars, not a table lookup.
    assert done.cost_usd == pytest.approx(0.0000138)


async def test_ollama_sends_keep_alive_and_reads_the_real_counts() -> None:
    tape = cassette("ollama_stream_tools.json")
    recorder = Recorder({"status": 200, "json": tape["response"]["ndjson"][-1]})
    async with client_for(recorder) as http:
        provider = OllamaChat(model="qwen3:8b", client=http)
        done = await provider.complete(
            [Message("user", "something slow and cold")], tools=[SEARCH_TOOL]
        )
    sent = recorder.body()
    assert sent["keep_alive"] == "10m"
    assert sent["options"]["num_ctx"] == 8192
    assert done.usage.input_tokens == 388
    assert done.usage.output_tokens == 31
    assert done.tool_calls[0].id == "call_1_0"
    assert done.tool_calls[0].arguments == {"query": "slow and cold"}


async def test_ollama_streams_ndjson_with_the_call_on_the_last_line() -> None:
    tape = cassette("ollama_stream_tools.json")
    recorder = Recorder(tape["response"])
    async with client_for(recorder) as http:
        provider = OllamaChat(model="qwen3:8b", client=http)
        chunks = await drain(provider.stream([Message("user", "hi")], tools=[SEARCH_TOOL]))
    assert recorder.body() == tape["request"]["json"] | {
        "messages": [{"role": "user", "content": "hi"}]
    }
    assert "".join(c.delta_text for c in chunks) == "Looking now"
    assert chunks[-1].tool_call_delta is not None
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].usage is not None


async def test_ollama_rescues_a_tool_call_written_into_the_content() -> None:
    recorder = Recorder(cassette("ollama_content_toolcall.json")["response"])
    async with client_for(recorder) as http:
        provider = OllamaChat(model="qwen3:8b", client=http)
        done = await provider.complete([Message("user", "hi")], tools=[SEARCH_TOOL])
    assert done.tool_calls[0].source == "content"
    assert done.tool_calls[0].arguments == {"query": "slow and cold"}
    assert done.content == "I will look that up."


async def test_ollama_has_no_tool_call_ids_on_the_way_out() -> None:
    recorder = Recorder(cassette("ollama_content_toolcall.json")["response"])
    async with client_for(recorder) as http:
        provider = OllamaChat(model="qwen3:8b", client=http)
        await provider.complete(
            [Message("tool", "3 films", tool_call_id="call_a1", name="search_films")]
        )
    assert recorder.body()["messages"] == [
        {"role": "tool", "content": "3 films", "tool_name": "search_films"}
    ]


async def test_ollama_refuses_to_claim_parallel_tool_calls() -> None:
    async with httpx.AsyncClient() as http:
        provider = OllamaChat(model="qwen3:8b", client=http)
        capabilities = await provider.capabilities()
    assert capabilities.parallel_tool_calls is False
    assert capabilities.tool_streaming is False
    assert capabilities.schema_style == "ollama"


async def test_a_missing_ollama_model_names_the_pull_command() -> None:
    recorder = Recorder({"status": 404, "json": {"error": "model not found"}})
    async with client_for(recorder) as http:
        provider = OllamaChat(model="qwen3:8b", client=http)
        with pytest.raises(ModelNotFound) as exc:
            await provider.complete([Message("user", "hi")])
    assert exc.value.pull_hint == "ollama pull qwen3:8b"


async def test_a_bad_key_and_a_bad_request_are_different_errors() -> None:
    async with client_for(Recorder({"status": 401, "json": {"error": "no"}})) as http:
        provider = OpenAICompatChat(base_url="http://x/v1", model="m", client=http)
        with pytest.raises(ProviderAuthError):
            await provider.complete([Message("user", "hi")])
    async with client_for(Recorder({"status": 400, "json": {"error": "bad tools"}})) as http:
        provider = OpenAICompatChat(base_url="http://x/v1", model="m", client=http)
        with pytest.raises(ProviderBadRequest):
            await provider.complete([Message("user", "hi")])


class FakeHubClient:
    """Stands in for AsyncInferenceClient so the router adapter is exercised offline."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.kwargs: dict[str, Any] = {}

    async def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return self.payload


async def test_hf_inference_strips_what_the_hub_client_takes_separately() -> None:
    hub = FakeHubClient(cassette("hf_inference_complete.json")["response"]["json"])
    provider = HFInferenceChat(model="Qwen/Qwen3-30B-A3B-Instruct", client=hub)
    done = await provider.complete([Message("user", "name one bleak film")])
    assert "model" not in hub.kwargs
    assert "stream" not in hub.kwargs
    assert hub.kwargs["messages"] == [{"role": "user", "content": "name one bleak film"}]
    assert done.content == "The Turin Horse."
    assert done.response_model == "Qwen/Qwen3-30B-A3B-Instruct"


def test_as_payload_handles_a_hub_dataclass() -> None:
    from dataclasses import dataclass

    @dataclass
    class Output:
        model: str
        choices: list[dict[str, Any]]

    payload = as_payload(Output("m", [{"index": 0}]))
    assert payload == {"model": "m", "choices": [{"index": 0}]}


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


async def test_every_adapter_satisfies_the_protocol() -> None:
    async with httpx.AsyncClient() as http:
        providers: list[ChatProvider] = [
            FakeChatProvider([]),
            OllamaChat(model="m", client=http),
            OpenAICompatChat(base_url="http://x/v1", model="m", client=http),
            OpenRouterChat(model="m", api_key=SecretStr("k"), client=http),
            HFInferenceChat(model="m", client=object()),
        ]
        assert all(isinstance(p, ChatProvider) for p in providers)
