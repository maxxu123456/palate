"""Where palate keeps its data, and the environment it pins before HF is imported."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_home() -> Path:
    """Data root, PALATE_HOME if set, otherwise ~/.palate."""
    raw = os.environ.get("PALATE_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".palate"


def config_dir() -> Path:
    """Directory holding palate.toml, XDG aware."""
    raw = os.environ.get("XDG_CONFIG_HOME")
    base = Path(raw).expanduser() if raw else Path.home() / ".config"
    return base / "palate"


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved data locations for one run."""

    home: Path

    @property
    def db(self) -> Path:
        return self.home / "palate.db"

    @property
    def traces_db(self) -> Path:
        return self.home / "traces.db"

    @property
    def hf_home(self) -> Path:
        return self.home / "hf"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def exports(self) -> Path:
        return self.home / "exports"

    def ensure(self) -> Paths:
        """Create the data root and its subdirectories."""
        for p in (self.home, self.hf_home, self.cache, self.exports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def resolve(home: Path | None = None, *, offline: bool = False) -> Paths:
    """Resolve paths and pin HF_HOME. Call this before anything imports huggingface_hub."""
    paths = Paths(home.expanduser() if home is not None else default_home()).ensure()
    os.environ.setdefault("HF_HOME", str(paths.hf_home))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return paths


def migrations_dir() -> Path:
    """Directory holding the palate.db migration files."""
    return Path(__file__).parent / "db" / "migrations"
