"""Reissue a stored call and diff it. This is what makes tracing more than logging."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import orjson

from palate.db.connect import Database
from palate.errors import PalateError
from palate.hashing import sha256_hex
from palate.obs.store import decompress
from palate.providers.base import ChatProvider, Message, Role
from palate.providers.tokens import estimate_text

NEEDS_FULL = 'this call has no stored payload, set trace.payloads = "full" and run it again'

_CALL = (
    "select provider, request_model, prompt_name, prompt_version, prompt_sha, messages_z, "
    "response_z, temperature, seed, tokens_in, tokens_out, cost_usd from llm_calls "
    "where span_id = ?"
)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """The old answer, the new one, and what moved between them."""

    span_id: str
    original_response: str
    new_response: str
    diff: str
    original_model: str
    new_model: str
    original_prompt_sha: str
    new_prompt_sha: str
    tokens_delta: tuple[int, int]
    cost_delta: float


def stored_messages(db: Database, span_id: str) -> list[Message]:
    """The exact messages that produced a stored response, or an error naming the setting."""
    row = db.read().execute(_CALL, (span_id,)).fetchone()
    if row is None:
        raise PalateError(f"no llm call traced under span {span_id}")
    if row["messages_z"] is None:
        raise PalateError(NEEDS_FULL)
    parsed = orjson.loads(decompress(bytes(row["messages_z"])))
    return [
        Message(
            role=_role(item.get("role")),
            content=str(item.get("content", "")),
            tool_call_id=item.get("tool_call_id"),
            name=item.get("name"),
        )
        for item in parsed
    ]


async def replay(
    db: Database,
    span_id: str,
    *,
    provider: ChatProvider,
    prompt_override: str | None = None,
) -> ReplayResult:
    """Reissue a stored llm_call verbatim against the current provider and diff the output."""
    row = db.read().execute(_CALL, (span_id,)).fetchone()
    if row is None:
        raise PalateError(f"no llm call traced under span {span_id}")
    if row["messages_z"] is None or row["response_z"] is None:
        raise PalateError(NEEDS_FULL)
    messages = stored_messages(db, span_id)
    if prompt_override is not None:
        messages = [Message(role="system", content=prompt_override), *messages[1:]]
    before = decompress(bytes(row["response_z"]))
    done = await provider.complete(
        messages,
        temperature=float(row["temperature"] or 0.0),
        seed=row["seed"],
    )
    return ReplayResult(
        span_id=span_id,
        original_response=before,
        new_response=done.content,
        diff="\n".join(
            difflib.unified_diff(
                before.splitlines(), done.content.splitlines(), "stored", "replayed", lineterm=""
            )
        ),
        original_model=str(row["request_model"]),
        new_model=provider.model,
        original_prompt_sha=str(row["prompt_sha"] or ""),
        new_prompt_sha=sha256_hex(prompt_override)
        if prompt_override
        else str(row["prompt_sha"] or ""),
        tokens_delta=(
            done.usage.input_tokens - int(row["tokens_in"]),
            (done.usage.output_tokens or estimate_text(done.content)) - int(row["tokens_out"]),
        ),
        cost_delta=float(done.cost_usd or 0.0) - float(row["cost_usd"] or 0.0),
    )


_ROLES: tuple[Role, ...] = ("system", "user", "assistant", "tool")


def _role(value: object) -> Role:
    return next((r for r in _ROLES if r == value), "user")
