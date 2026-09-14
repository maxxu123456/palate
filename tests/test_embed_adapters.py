"""Query and document sides are separate methods because getting them backwards is silent."""

from __future__ import annotations

import math
from typing import Any

import pytest

from palate.config import Settings
from palate.errors import ConfigError
from palate.hf.models import BRANCHES, ModelSpec, load, pin
from palate.providers.base import EmbeddingProvider
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.fingerprint import CANARY_TEXT
from palate.providers.registry import build_embedder


def unit(vector: tuple[float, ...]) -> bool:
    return math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-6)


async def test_the_fake_embedder_is_stable_and_unit_length() -> None:
    embedder = FakeEmbedder(dim=16)
    batch = await embedder.embed_documents(["Stalker", "The Turin Horse"])
    assert len(batch.vectors) == 2
    assert all(len(v) == 16 for v in batch.vectors)
    assert all(unit(v) for v in batch.vectors)
    assert batch.vectors[0] != batch.vectors[1]
    again = await FakeEmbedder(dim=16).embed_documents(["Stalker"])
    assert again.vectors[0] == batch.vectors[0]


async def test_a_different_seed_is_a_different_space() -> None:
    one = await FakeEmbedder(dim=8, seed=1).embed_query("Stalker")
    two = await FakeEmbedder(dim=8, seed=2).embed_query("Stalker")
    assert one != two


def test_models_toml_loads_and_refuses_a_moving_revision(tmp_path: Any) -> None:
    aliases = load()
    assert "embeddinggemma" in aliases
    gemma = aliases["embeddinggemma"]
    assert gemma.repo_id == "google/embeddinggemma-300m"
    assert gemma.query_prompt != gemma.document_prompt
    assert gemma.pinned is False
    moving = tmp_path / "models.toml"
    moving.write_text('[x]\nrepo_id = "a/b"\nrevision = "main"\n')
    with pytest.raises(ConfigError) as exc:
        load(moving)
    assert "moves" in str(exc.value)
    assert "main" in BRANCHES


def test_an_unpinned_alias_names_the_command_that_pins_it(tmp_path: Any) -> None:
    with pytest.raises(ConfigError) as exc:
        pin("embeddinggemma")
    assert "palate models pull embeddinggemma" in str(exc.value)
    pinned = tmp_path / "models.toml"
    pinned.write_text(f'[x]\nrepo_id = "a/b"\nrevision = "{"a" * 40}"\ndim = 768\n')
    assert pin("x", path=pinned).revision == "a" * 40


def test_a_spec_only_counts_as_pinned_with_a_real_sha() -> None:
    assert ModelSpec("x", "embedding", "a/b", revision="abc").pinned is False
    assert ModelSpec("x", "embedding", "a/b", revision="f" * 40).pinned is True


async def test_the_registry_refuses_an_unpinned_default_rather_than_guessing() -> None:
    # The shipped models.toml has no sha for the embedder, so the default cannot be built here.
    with pytest.raises(ConfigError):
        build_embedder(Settings())
    fake = build_embedder(Settings(embed={"provider": "fake"}))
    assert fake.provider == "fake"
    assert isinstance(fake, EmbeddingProvider)
    assert CANARY_TEXT
