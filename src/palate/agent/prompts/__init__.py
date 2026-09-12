"""Prompt texts, one markdown file each, loaded from beside this module."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from palate.hashing import sha256_hex

FENCE = "---"


@dataclass(frozen=True, slots=True)
class Prompt:
    """One template plus the identity a trace row records it under."""

    name: str
    version: str
    text: str
    sha: str

    def render(self, **fields: str) -> str:
        """Fill the template. A missing field is a programming error, not a runtime one."""
        return self.text.format(**fields)


def _read(name: str) -> str:
    return str((files(__package__) / f"{name}.md").read_text(encoding="utf-8"))


def _split(raw: str) -> tuple[dict[str, str], str]:
    """Front matter and body. A file without front matter is all body."""
    if not raw.startswith(FENCE):
        return {}, raw
    end = raw.find(f"\n{FENCE}", len(FENCE))
    if end < 0:
        return {}, raw
    head = raw[len(FENCE) : end]
    meta: dict[str, str] = {}
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, raw[end + len(FENCE) + 1 :].lstrip("\n")


def load(name: str) -> str:
    """One prompt body by file stem. A prompt is data, so it lives in a file and not a string."""
    return _split(_read(name))[1]


def prompt_sha(name: str) -> str:
    """Hash of the prompt text, which belongs in any cache key the prompt shapes."""
    return sha256_hex(load(name))


class PromptRegistry:
    """Prompts by name, cached, each carrying the version and sha a trace row needs."""

    def __init__(self) -> None:
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        """One prompt, read from disk once per process."""
        found = self._cache.get(name)
        if found is None:
            meta, body = _split(_read(name))
            found = Prompt(
                name=meta.get("name", name),
                version=meta.get("version", "1"),
                text=body,
                sha=sha256_hex(body),
            )
            self._cache[name] = found
        return found

    def render(self, name: str, **fields: str) -> str:
        """A filled prompt, for a caller that does not care about the version."""
        return self.get(name).render(**fields)
