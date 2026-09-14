"""Query and document sides are separate methods because getting them backwards is silent."""

from __future__ import annotations

import math
from typing import Any

import pytest

from palate.config import Settings
from palate.errors import ConfigError
from palate.hf.models import BRANCHES, ModelPin, ModelSpec, load, pin
from palate.providers.base import EmbeddingProvider
from palate.providers.embed import sentence_transformers as st_embed
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.fingerprint import CANARY_TEXT
from palate.providers.registry import build_embedder

GEMMA = ModelPin(
    alias="embeddinggemma",
    repo_id="google/embeddinggemma-300m",
    revision="c" * 40,
    dim=4,
    query_prompt="task: search result | query: ",
    document_prompt="title: none | text: ",
)


class StubSentenceTransformer:
    """Records every text it was handed, which is where the prefixes go wrong."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        truncate_dim: int | None = None,
    ) -> None:
        self.repo_id = model_name_or_path
        self.revision = revision
        self.device = device
        self.truncate_dim = truncate_dim
        self.seen: list[list[str]] = []
        self.batches: list[int] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        self.seen.append(list(texts))
        self.batches.append(batch_size)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def stub_st(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st_embed, "SentenceTransformer", StubSentenceTransformer)


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


async def test_the_two_sides_carry_different_prefixes(stub_st: None) -> None:
    embedder = st_embed.SentenceTransformersEmbedder(pin=GEMMA, device="cpu", batch_size=8)
    await embedder.embed_documents(["Stalker"])
    await embedder.embed_query("slow and cold")
    assert embedder.model.seen[0] == ["title: none | text: Stalker"]
    assert embedder.model.seen[1] == ["task: search result | query: slow and cold"]
    assert embedder.model.batches == [8, 8]


async def test_the_fingerprint_carries_the_pinned_commit(stub_st: None) -> None:
    embedder = st_embed.SentenceTransformersEmbedder(pin=GEMMA, device="cpu")
    found = await embedder.ready()
    assert (found.model_id, found.revision) == (GEMMA.repo_id, "c" * 40)
    assert found.dim == 4
    assert found.normalized is True
    assert embedder.model.revision == "c" * 40


async def test_health_encodes_the_canary_and_nothing_else(stub_st: None) -> None:
    embedder = st_embed.SentenceTransformersEmbedder(pin=GEMMA, device="cpu")
    report = await embedder.health()
    assert report.ok is True
    assert embedder.model.seen == [[CANARY_TEXT]]
    await embedder.aclose()
    assert embedder.model is None


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
