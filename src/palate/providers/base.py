"""The provider Protocols and every dataclass that crosses them."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from palate.providers.fingerprint import EmbeddingFingerprint

type Role = Literal["system", "user", "assistant", "tool"]
type FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]
type ToolChoice = Literal["auto", "none", "required"] | tuple[Literal["tool"], str]
type SchemaStyle = Literal["openai", "openai_strict", "ollama", "hf"]
type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
type JSONObject = dict[str, JSONValue]
type Vector = tuple[float, ...]


def json_object(value: Any) -> dict[str, Any]:
    """A nested json value as a dict, or an empty dict when it is anything else."""
    return dict(value) if isinstance(value, dict) else {}


def json_int(value: Any, default: int = 0) -> int:
    """A nested json value as an int, tolerating the nulls providers send."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@runtime_checkable
class SpanLike(Protocol):
    """The slice of a trace span a provider is allowed to touch."""

    def event(self, name: str, **fields: Any) -> None: ...

    def set(self, **fields: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call the model asked for, with the raw text kept for the trace."""

    id: str
    name: str
    arguments_json: str
    arguments: JSONObject | None = None
    source: Literal["field", "content"] = "field"


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A streaming fragment. index is None when the server omits it."""

    index: int | None
    id: str | None = None
    name: str | None = None
    arguments_fragment: str = ""


@dataclass(frozen=True, slots=True)
class Message:
    """One transcript entry, in provider neutral form."""

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    channel: Literal["answer", "thinking"] = "answer"


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the model sees it."""

    name: str
    description: str
    parameters: JSONObject
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """What shape the answer has to come back in."""

    kind: Literal["text", "json_object", "json_schema"]
    schema: JSONObject | None = None
    name: str = "response"


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts. exact is False when the stream dropped them and we estimated."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    exact: bool = True


@dataclass(frozen=True, slots=True)
class Completion:
    """One finished model reply."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    model: str
    response_model: str
    cost_usd: float | None = None
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    attempt: int = 1
    cache_hit: bool = False
    raw: JSONObject | None = None


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One streamed fragment. usage rides the terminal chunk."""

    delta_text: str = ""
    tool_call_delta: ToolCallDelta | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class ChatCapabilities:
    """What this endpoint can actually do, which is not what the docs claim."""

    tools: bool = True
    parallel_tool_calls: bool = True
    tool_streaming: bool = True
    json_schema: bool = True
    streaming: bool = True
    context_window: int | None = None
    max_output_tokens: int | None = None
    schema_style: SchemaStyle = "openai"


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """One reachability check, with enough detail to act on."""

    ok: bool
    detail: str
    latency_ms: float | None = None


@runtime_checkable
class ChatProvider(Protocol):
    """Anything that can answer a message list and call tools."""

    name: str
    model: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
        response_format: ResponseFormat | None = None,
        seed: int | None = None,
        timeout_s: float | None = None,
        span: SpanLike | None = None,
    ) -> Completion: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
        response_format: ResponseFormat | None = None,
        seed: int | None = None,
        timeout_s: float | None = None,
        span: SpanLike | None = None,
    ) -> AsyncIterator[ChatChunk]: ...

    async def capabilities(self) -> ChatCapabilities: ...

    async def health(self) -> ProviderHealth: ...

    def count_tokens(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema] = ()
    ) -> int: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """One batch of vectors, always carrying the space they belong to."""

    vectors: tuple[Vector, ...]
    fingerprint: EmbeddingFingerprint
    input_tokens: int | None = None
    latency_ms: float = 0.0
    cache_hits: int = 0


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors, in one declared space."""

    provider: str
    max_batch: int

    @property
    def fingerprint(self) -> EmbeddingFingerprint: ...

    async def ready(self) -> EmbeddingFingerprint: ...

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch: ...

    async def embed_query(self, text: str, *, span: SpanLike | None = None) -> Vector: ...

    async def health(self) -> ProviderHealth: ...

    async def aclose(self) -> None: ...
