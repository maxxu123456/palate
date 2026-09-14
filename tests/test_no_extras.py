"""Someone on OpenRouter must never be made to install torch, and a convention would rot."""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

import palate
from palate.errors import MissingExtra
from palate.extras import have
from palate.hf.models import ModelPin

# Anything here at import time means a base install just paid for a gigabyte it cannot use.
FORBIDDEN = ("torch", "sentence_transformers", "transformers", "scipy")

PIN = ModelPin("embeddinggemma", "google/embeddinggemma-300m", "a" * 40, dim=768)

# A web framework cannot be lazily imported, so these modules exist only with the extra.
NEEDS_EXTRA = {"palate.api": "fastapi"}


def installed(name: str) -> bool:
    return all(not name.startswith(prefix) or have(mod) for prefix, mod in NEEDS_EXTRA.items())


def module_names() -> list[str]:
    found = [m.name for m in pkgutil.walk_packages(palate.__path__, "palate.")]
    return [name for name in found if installed(name)]


def test_there_is_something_to_walk() -> None:
    names = module_names()
    assert "palate.providers.embed.sentence_transformers" in names
    assert ("palate.api.app" in names) == have("fastapi")
    assert len(names) > 30


@pytest.mark.parametrize("name", module_names())
def test_importing_a_module_pulls_in_no_heavy_dependency(name: str) -> None:
    importlib.import_module(name)
    loaded = [heavy for heavy in FORBIDDEN if heavy in sys.modules]
    assert loaded == [], f"{name} imported {loaded} at module scope"


def test_the_local_embedder_names_the_extra_and_the_command() -> None:
    from palate.providers.embed.sentence_transformers import SentenceTransformersEmbedder

    with pytest.raises(MissingExtra) as exc:
        SentenceTransformersEmbedder(pin=PIN)
    assert exc.value.extra == "local"
    assert "uv sync --extra local" in str(exc.value)


def test_the_hub_helpers_name_the_hf_extra() -> None:
    from palate.hf import cache, download

    with pytest.raises(MissingExtra) as exc:
        cache.scan()
    assert "uv sync --extra hf" in str(exc.value)
    with pytest.raises(MissingExtra):
        download.ensure_local(PIN)


def test_device_resolution_says_cpu_without_importing_torch() -> None:
    from palate.hf.device import resolve_device

    assert resolve_device() in ("cpu", "mps", "cuda")
    assert "torch" not in sys.modules or resolve_device("cpu") == "cpu"
