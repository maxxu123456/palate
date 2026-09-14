"""Two models that agree on dimension and disagree on everything else must not mix."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from palate.errors import EmbeddingFingerprintMismatch
from palate.index.documents import DOC_TEMPLATE_VERSION
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.fingerprint import (
    CANARY_TEXT,
    EmbeddingFingerprint,
    canary_check,
    cosine,
    mismatch_message,
    pack_f32,
    require_match,
    unpack_f32,
)

GEMMA = EmbeddingFingerprint(
    provider="sentence_transformers",
    model_id="google/embeddinggemma-300m",
    revision="a" * 40,
    dim=768,
    query_prompt="task: search result | query: ",
    document_prompt="title: none | text: ",
)

# What an index built by an older palate, through a runtime this one no longer has, looks like.
LEGACY = replace(
    GEMMA,
    provider="legacy",
    model_id="nomic-embed-text",
    revision=None,
    document_prompt="search_document: ",
)


def test_the_key_is_sixteen_hex_and_stable() -> None:
    assert len(GEMMA.key) == 16
    assert GEMMA.key == replace(GEMMA).key
    assert int(GEMMA.key, 16) >= 0


def test_the_same_model_through_two_runtimes_is_two_spaces() -> None:
    # Same name, same 768 dims, different templating. This is the whole point.
    same_name = replace(GEMMA, provider="legacy", revision=None)
    assert same_name.dim == GEMMA.dim
    assert same_name.key != GEMMA.key


def test_changing_the_document_template_changes_the_space() -> None:
    assert replace(GEMMA, doc_template_version="v2").key != GEMMA.key


def test_a_round_trip_through_a_row_keeps_the_key() -> None:
    row = GEMMA.to_row()
    row["normalized"] = 1
    assert EmbeddingFingerprint.from_row(row).key == GEMMA.key


def test_a_row_with_extra_columns_still_loads() -> None:
    row = {**GEMMA.to_row(), "table_name": "vec_films_abc", "status": "ready"}
    assert EmbeddingFingerprint.from_row(row) == GEMMA


@given(st.permutations(list(GEMMA.to_row().items())))
def test_field_order_never_changes_the_key(pairs: list[tuple[str, object]]) -> None:
    assert EmbeddingFingerprint(**dict(pairs)).key == GEMMA.key


def test_the_diff_names_every_field_that_moved() -> None:
    moved = {name for name, _, _ in GEMMA.diff(LEGACY)}
    assert moved == {"provider", "model_id", "revision", "document_prompt"}


def test_the_mismatch_report_prints_the_dim_even_when_it_matches() -> None:
    message = mismatch_message(GEMMA, LEGACY)
    lines = message.splitlines()
    assert "cannot answer this query" in lines[0]
    assert any(line.strip().startswith("dim") and "768" in line for line in lines)
    assert any("document_prompt" in line for line in lines)
    assert lines[-1].startswith("run `palate index build")
    assert GEMMA.key in lines[-1]


def test_require_match_raises_rather_than_warning() -> None:
    require_match(GEMMA, replace(GEMMA))
    with pytest.raises(EmbeddingFingerprintMismatch) as exc:
        require_match(GEMMA, LEGACY)
    assert exc.value.index is GEMMA


def test_a_vector_survives_the_round_trip_through_bytes() -> None:
    packed = pack_f32([0.5, -0.25, 0.125])
    assert len(packed) == 12
    assert unpack_f32(packed) == (0.5, -0.25, 0.125)


def test_cosine_is_one_against_itself_and_zero_against_nothing() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0


async def test_the_canary_passes_against_its_own_vector() -> None:
    embedder = FakeEmbedder(dim=32)
    stored = pack_f32(await embedder.embed_query(CANARY_TEXT))
    assert await canary_check(embedder, stored) is True


async def test_a_host_that_swapped_weights_is_caught() -> None:
    embedder = FakeEmbedder(dim=32, seed=1)
    other = FakeEmbedder(dim=32, seed=2)
    stored = pack_f32(await other.embed_query(CANARY_TEXT))
    with pytest.raises(EmbeddingFingerprintMismatch) as exc:
        await canary_check(embedder, stored)
    assert exc.value.query.revision == "canary-drift"


async def test_a_fresh_check_is_skipped_rather_than_repeated() -> None:
    embedder = FakeEmbedder(dim=32)
    stored = pack_f32(await embedder.embed_query(CANARY_TEXT))
    now = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    skipped = await canary_check(embedder, stored, checked_at=now - timedelta(hours=1), now=now)
    assert skipped is False
    ran = await canary_check(embedder, stored, checked_at=now - timedelta(hours=30), now=now)
    assert ran is True


def test_the_template_version_travels_with_the_fingerprint() -> None:
    assert FakeEmbedder().fingerprint.doc_template_version == DOC_TEMPLATE_VERSION
