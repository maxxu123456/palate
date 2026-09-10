"""The nine fields that decide whether two vectors are comparable."""

from __future__ import annotations

import array
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from palate.errors import EmbeddingFingerprintMismatch
from palate.hashing import short_hash
from palate.index.documents import DOC_TEMPLATE_VERSION

if TYPE_CHECKING:
    from palate.providers.base import EmbeddingProvider

# One sentence, re-embedded daily, to catch a host swapping weights behind a stable name.
CANARY_TEXT = "a slow, cold film about memory and an empty room"

# Always shown in a mismatch report, because matching dims are the whole trap.
_ALWAYS_SHOWN = ("model_id", "dim")


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    """Everything that changes the vector space, hashed into one index id."""

    provider: str
    model_id: str
    revision: str | None = None
    dim: int = 0
    normalized: bool = True
    query_prompt: str = ""
    document_prompt: str = ""
    pooling: str = "mean"
    doc_template_version: str = DOC_TEMPLATE_VERSION

    @property
    def key(self) -> str:
        """First 16 hex of sha256 over canonical json of every field."""
        return short_hash(self.to_row())

    def diff(self, other: EmbeddingFingerprint) -> list[tuple[str, Any, Any]]:
        """Every field where these two disagree."""
        return [
            (f.name, getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        ]

    def to_row(self) -> dict[str, Any]:
        """The embedding_indexes columns this fingerprint owns."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> EmbeddingFingerprint:
        """Rebuild from a stored row, ignoring columns that are not ours."""
        names = {f.name for f in fields(cls)}
        values = {k: row[k] for k in names if k in row}
        if "normalized" in values:
            values["normalized"] = bool(values["normalized"])
        return cls(**values)


def mismatch_message(index: EmbeddingFingerprint, query: EmbeddingFingerprint) -> str:
    """The report that tells someone which knob moved, dim included even when it matches."""
    changed = {name for name, _, _ in index.diff(query)}
    shown = list(_ALWAYS_SHOWN)
    shown += [f.name for f in fields(index) if f.name in changed and f.name not in _ALWAYS_SHOWN]
    lines = [
        f"index {index.key!r} was built with a different embedding setup "
        "and cannot answer this query"
    ]
    for name in shown:
        mine = _show(getattr(index, name))
        theirs = _show(getattr(query, name))
        lines.append(f"  {name:<18}{mine:<28}  ->  {theirs}")
    lines.append(
        f"run `palate index build --embed-model {query.model_id}` "
        f"or `palate index activate {index.key}`"
    )
    return "\n".join(lines)


def _show(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def require_match(index: EmbeddingFingerprint, query: EmbeddingFingerprint) -> None:
    """Raise unless the two fingerprints are the same space. Never a warning."""
    if index.key != query.key:
        raise EmbeddingFingerprintMismatch(mismatch_message(index, query), index=index, query=query)


def pack_f32(vector: Sequence[float]) -> bytes:
    """Little endian float32, which is the only layout sqlite-vec accepts."""
    return array.array("f", vector).tobytes()


def unpack_f32(blob: bytes) -> tuple[float, ...]:
    """Read a stored vector back."""
    values = array.array("f")
    values.frombytes(blob)
    return tuple(values)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, safe on a zero vector."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


async def canary_check(
    embedder: EmbeddingProvider,
    stored: bytes,
    *,
    tolerance: float = 0.999,
    checked_at: datetime | None = None,
    now: datetime | None = None,
    max_age: timedelta = timedelta(hours=24),
) -> bool:
    """Re-embed a fixed sentence and compare. False means the last check was still fresh."""
    if checked_at is not None and now is not None and now - checked_at < max_age:
        return False
    fresh = await embedder.embed_query(CANARY_TEXT)
    similarity = cosine(unpack_f32(stored), fresh)
    if similarity < tolerance:
        drifted = replace(embedder.fingerprint, revision="canary-drift")
        raise EmbeddingFingerprintMismatch(
            f"the canary moved to {similarity:.5f}, below {tolerance}, so the host "
            f"changed weights behind {embedder.fingerprint.model_id}",
            index=embedder.fingerprint,
            query=drifted,
        )
    return True
