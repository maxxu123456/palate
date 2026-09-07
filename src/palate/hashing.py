"""Canonical JSON and the hashes derived from it."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson


def canonical_json(value: Any) -> bytes:
    """Sorted-key, no-whitespace JSON so equal values always hash equal."""
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)


def sha256_hex(data: bytes | str) -> str:
    """Hex sha256 of bytes or utf-8 text."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, read in chunks so a large export does not land in memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def doc_sha(text: str) -> str:
    """Hash of a rendered film document. A move here means re-embed that film."""
    return sha256_hex(text)


def request_sha(payload: Any) -> str:
    """Cache key for an outbound model request."""
    return sha256_hex(canonical_json(payload))


def short_hash(value: Any, *, length: int = 16) -> str:
    """First `length` hex characters of the canonical hash, used for index ids."""
    return request_sha(value)[:length]
