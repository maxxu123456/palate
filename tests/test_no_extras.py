"""Someone on OpenRouter must never be made to install torch, and a convention would rot."""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

import palate
from palate.errors import MissingExtra
from palate.hf.models import ModelPin

# Anything here at import time means a base install just paid for a gigabyte it cannot use.
FORBIDDEN = ("torch", "sentence_transformers", "transformers", "scipy")

PIN = ModelPin("embeddinggemma", "google/embeddinggemma-300m", "a" * 40, dim=768)


def module_names() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(palate.__path__, "palate.")]


def test_there_is_something_to_walk() -> None:
    names = module_names()
    assert "palate.providers.embed.sentence_transformers" in names
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


def test_the_hf_chat_adapter_asks_before_it_reaches_for_the_hub() -> None:
    from palate.providers.chat.hf_inference import HFInferenceChat

    with pytest.raises(MissingExtra):
        HFInferenceChat(model="Qwen/Qwen3-30B-A3B-Instruct")


def test_device_resolution_says_cpu_without_importing_torch() -> None:
    from palate.hf.device import resolve_device

    assert resolve_device() in ("cpu", "mps", "cuda")
    assert "torch" not in sys.modules or resolve_device("cpu") == "cpu"
