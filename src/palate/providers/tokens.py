"""Token estimates. The ledger corrects them from real usage after every call."""

from __future__ import annotations

from collections.abc import Sequence

from palate.hashing import canonical_json
from palate.providers.base import Message, ToolSchema

# Four characters per token is the usual English ratio and is close enough to budget on.
CHARS_PER_TOKEN = 4
# Role, separators and the wrapper the server adds around each message.
PER_MESSAGE_OVERHEAD = 4
# A tool schema costs its json plus the framing the provider wraps it in.
PER_TOOL_OVERHEAD = 12


def estimate_text(text: str) -> int:
    """Rough token count for a string."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def count_tokens(messages: Sequence[Message], tools: Sequence[ToolSchema] = ()) -> int:
    """Estimate the prompt size of a transcript plus its tool schemas."""
    total = 0
    for message in messages:
        total += PER_MESSAGE_OVERHEAD + estimate_text(message.content)
        for call in message.tool_calls:
            total += estimate_text(call.name) + estimate_text(call.arguments_json)
    for tool in tools:
        total += PER_TOOL_OVERHEAD + estimate_text(tool.name) + estimate_text(tool.description)
        total += estimate_text(canonical_json(tool.parameters).decode())
    return total
