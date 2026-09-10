"""Query and document sides are separate methods because getting them backwards is silent."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from palate.config import Settings
from palate.errors import ConfigError
from palate.hf.models import BRANCHES, ModelSpec, load, pin
from palate.providers.base import EmbeddingProvider
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.embed.hf_inference import HFInferenceEmbedder, pool
from palate.providers.embed.ollama import OllamaEmbedder
from palate.providers.embed.openai_compat import OpenAICompatEmbedder, l2_normalize
from palate.providers.fingerprint import CANARY_TEXT
from palate.providers.registry import build_embedder


class FakeHost:
    """Serves embeddings and records what was asked for."""

    def __init__(self, *, dim: int = 4, digest: str = "sha256:abc") -> None:
        self.dim = dim
        self.digest = digest
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"model": "embeddinggemma:300m", "digest": self.digest}]}
            )
        body = json.loads(request.content)
        self.bodies.append(body)
        inputs = body["input"]
        rows = [self._row(i) for i, _ in enumerate(inputs)]
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": rows, "prompt_eval_count": 7})
        return httpx.Response(
            200,
            json={
                "data": [{"index": i, "embedding": row} for i, row in enumerate(rows)],
                "usage": {"prompt_tokens": 7},
            },
        )

    def _row(self, index: int) -> list[float]:
        # Deliberately not unit length, so the normalising adapter has work to do.
        return [float(index + 1) * (j + 1) for j in range(self.dim)]


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


async def test_ollama_reads_its_revision_from_the_tag_digest() -> None:
    host = FakeHost(dim=4)
    async with host.client() as http:
        embedder = OllamaEmbedder(model="embeddinggemma", client=http)
        fingerprint = await embedder.ready()
    assert fingerprint.revision == "sha256:abc"
    assert fingerprint.dim == 4
    assert fingerprint.normalized is True


async def test_ollama_splits_a_batch_at_max_batch() -> None:
    host = FakeHost(dim=4)
    async with host.client() as http:
        embedder = OllamaEmbedder(model="embeddinggemma", client=http, max_batch=2, dim=4)
        batch = await embedder.embed_documents([f"film {i}" for i in range(5)])
    assert len(batch.vectors) == 5
    assert [len(b["input"]) for b in host.bodies if b.get("input")][-3:] == [2, 2, 1]


async def test_the_two_sides_carry_different_prefixes() -> None:
    host = FakeHost(dim=4)
    async with host.client() as http:
        embedder = OllamaEmbedder(
            model="embeddinggemma",
            client=http,
            dim=4,
            revision="r",
            query_prompt="task: search result | query: ",
            document_prompt="title: none | text: ",
        )
        await embedder.embed_documents(["Stalker"])
        await embedder.embed_query("slow and cold")
    assert host.bodies[0]["input"] == ["title: none | text: Stalker"]
    assert host.bodies[1]["input"] == ["task: search result | query: slow and cold"]


async def test_the_openai_route_normalises_whatever_it_is_given() -> None:
    host = FakeHost(dim=4)
    async with host.client() as http:
        embedder = OpenAICompatEmbedder(base_url="http://x/v1", model="m", client=http)
        fingerprint = await embedder.ready()
        batch = await embedder.embed_documents(["a", "b"])
    assert fingerprint.dim == 4
    assert all(unit(v) for v in batch.vectors)
    assert batch.input_tokens == 7


async def test_the_openai_route_keeps_the_server_order() -> None:
    dim = 3

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        embedder = OpenAICompatEmbedder(base_url="http://x/v1", model="m", client=http, dim=dim)
        batch = await embedder.embed_documents(["first", "second"])
    assert batch.vectors[0] == (1.0, 0.0, 0.0)


async def test_a_truncate_dim_is_asked_for_on_the_wire() -> None:
    host = FakeHost(dim=4)
    async with host.client() as http:
        embedder = OpenAICompatEmbedder(
            base_url="http://x/v1", model="m", client=http, dim=4, truncate_dim=2
        )
        await embedder.embed_documents(["a"])
    assert host.bodies[-1]["dimensions"] == 2


def test_l2_normalize_leaves_a_zero_vector_alone() -> None:
    assert l2_normalize([0.0, 0.0]) == (0.0, 0.0)
    assert unit(l2_normalize([3.0, 4.0]))


class FakeFeatureClient:
    """Stands in for AsyncInferenceClient, returning whatever shape the test wants."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.seen: list[Any] = []

    async def feature_extraction(self, *, text: Any) -> Any:
        self.seen.append(text)
        return self.output


def test_pooling_collapses_the_token_axis() -> None:
    assert pool([1.0, 3.0]) == (1.0, 3.0)
    assert pool([[1.0, 3.0], [3.0, 5.0]]) == (2.0, 4.0)


async def test_the_hf_router_output_is_pooled_and_normalised() -> None:
    client = FakeFeatureClient([[[1.0, 0.0], [3.0, 0.0]], [[0.0, 2.0], [0.0, 4.0]]])
    embedder = HFInferenceEmbedder(model="m", client=client)
    batch = await embedder.embed_documents(["a", "b"])
    assert batch.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert embedder.fingerprint.dim == 2


async def test_a_fingerprint_before_the_probe_is_an_error_not_a_guess() -> None:
    async with httpx.AsyncClient() as http:
        embedder = OllamaEmbedder(model="m", client=http)
        with pytest.raises(RuntimeError):
            _ = embedder.fingerprint


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


async def test_the_registry_builds_the_configured_embedder() -> None:
    async with httpx.AsyncClient() as http:
        ollama = build_embedder(Settings(), client=http)
        assert isinstance(ollama, EmbeddingProvider)
        assert ollama.provider == "ollama"
        fake = build_embedder(Settings(embed={"provider": "fake"}), client=http)
        assert fake.provider == "fake"
        with pytest.raises(ConfigError):
            build_embedder(Settings(embed={"provider": "openai_compat"}), client=http)


async def test_every_embedder_satisfies_the_protocol() -> None:
    async with httpx.AsyncClient() as http:
        embedders: list[EmbeddingProvider] = [
            FakeEmbedder(),
            OllamaEmbedder(model="m", client=http),
            OpenAICompatEmbedder(base_url="http://x/v1", model="m", client=http),
            HFInferenceEmbedder(model="m", client=FakeFeatureClient([[0.0]])),
        ]
        assert all(isinstance(e, EmbeddingProvider) for e in embedders)
        assert CANARY_TEXT
