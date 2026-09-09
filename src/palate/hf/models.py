"""models.toml: alias to a repo id and a commit sha that cannot be a branch."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from palate.errors import ConfigError
from palate.paths import default_home

type ModelKind = Literal["embedding", "reranker"]

_SHA = re.compile(r"^[0-9a-f]{40}$")

# A moving revision is the one thing a pin exists to prevent.
BRANCHES = frozenset({"main", "master", "HEAD", "latest"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One models.toml entry, whether or not it has been pinned yet."""

    alias: str
    kind: ModelKind
    repo_id: str
    revision: str = ""
    dim: int = 0
    pooling: str = "mean"
    normalized: bool = True
    query_prompt: str = ""
    document_prompt: str = ""

    @property
    def pinned(self) -> bool:
        return bool(_SHA.match(self.revision))


@dataclass(frozen=True, slots=True)
class ModelPin:
    """A resolved model: a repo id plus a real commit sha."""

    alias: str
    repo_id: str
    revision: str
    dim: int = 0
    pooling: str = "mean"
    normalized: bool = True
    query_prompt: str = ""
    document_prompt: str = ""


def search_paths() -> tuple[Path, ...]:
    """Where models.toml is looked for, nearest first."""
    return (
        default_home() / "models.toml",
        Path("models.toml"),
        Path(__file__).resolve().parents[3] / "models.toml",
    )


def default_path() -> Path:
    """The first models.toml that exists, falling back to the checked in one."""
    candidates = search_paths()
    return next((p for p in candidates if p.is_file()), candidates[-1])


def load(path: Path | None = None) -> dict[str, ModelSpec]:
    """Read every alias. An unpinned entry still loads, so `models list` can show it."""
    target = path or default_path()
    try:
        raw: dict[str, Any] = tomllib.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"no models.toml at {target}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"models.toml is not valid toml ({exc})") from exc
    return {alias: _spec(alias, body) for alias, body in raw.items()}


def _spec(alias: str, body: Any) -> ModelSpec:
    if not isinstance(body, dict) or "repo_id" not in body:
        raise ConfigError(f"models.toml entry {alias!r} has no repo_id")
    revision = str(body.get("revision") or "")
    if revision in BRANCHES:
        raise ConfigError(
            f"models.toml pins {alias!r} to {revision!r}, which moves. Pin a commit sha."
        )
    return ModelSpec(
        alias=alias,
        kind=body.get("kind", "embedding"),
        repo_id=str(body["repo_id"]),
        revision=revision,
        dim=int(body.get("dim", 0)),
        pooling=str(body.get("pooling", "mean")),
        normalized=bool(body.get("normalized", True)),
        query_prompt=str(body.get("query_prompt", "")),
        document_prompt=str(body.get("document_prompt", "")),
    )


def pin(alias: str, *, path: Path | None = None) -> ModelPin:
    """Resolve an alias to a pinned model, refusing anything that is not a commit sha."""
    specs = load(path)
    spec = specs.get(alias)
    if spec is None:
        known = ", ".join(sorted(specs)) or "nothing"
        raise ConfigError(f"no model alias {alias!r} in models.toml, which has {known}")
    if not spec.pinned:
        raise ConfigError(
            f"model alias {alias!r} has no commit sha. Run: palate models pull {alias}"
        )
    return ModelPin(
        alias=spec.alias,
        repo_id=spec.repo_id,
        revision=spec.revision,
        dim=spec.dim,
        pooling=spec.pooling,
        normalized=spec.normalized,
        query_prompt=spec.query_prompt,
        document_prompt=spec.document_prompt,
    )
