"""Prompt texts, one markdown file each, loaded from beside this module."""

from __future__ import annotations

from importlib.resources import files

from palate.hashing import sha256_hex


def load(name: str) -> str:
    """One prompt by file stem. A prompt is data, so it lives in a file and not in a string."""
    return (files(__package__) / f"{name}.md").read_text(encoding="utf-8")


def prompt_sha(name: str) -> str:
    """Hash of the prompt text, which belongs in any cache key the prompt shapes."""
    return sha256_hex(load(name))
