"""Shared fixtures. Every test runs offline with its own PALATE_HOME."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Names that would otherwise leak a real key or a real endpoint into a test run.
_SCRUBBED = (
    "PALATE_CHAT_API_KEY",
    "PALATE_EMBED_API_KEY",
    "OPENROUTER_API_KEY",
    "HF_TOKEN",
    "TMDB_READ_TOKEN",
)


@pytest.fixture(autouse=True)
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate every test from the developer's own config, keys and data root."""
    home = tmp_path / "palate_home"
    home.mkdir()
    config_home = tmp_path / "config_home"
    (config_home / "palate").mkdir(parents=True)
    monkeypatch.setenv("PALATE_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    for name in _SCRUBBED:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("PALATE_") and name != "PALATE_HOME":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield home


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Path of the user-level palate.toml for this test."""
    return tmp_path / "config_home" / "palate" / "palate.toml"
