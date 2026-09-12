"""One authority for pricing, and unknown is not zero."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from palate.clock import now_iso
from palate.db.connect import Database, open_database
from palate.obs.cost import ANY_MODEL, cost_for, seed_rates
from palate.paths import pricing_toml, trace_migrations_dir
from palate.providers.base import Usage

USAGE = Usage(input_tokens=1_000_000, output_tokens=500_000)


@pytest.fixture
def traces(tmp_path: Path) -> Iterator[Database]:
    db = open_database(tmp_path / "traces.db", migrations=trace_migrations_dir(), load_vec=False)
    yield db
    db.close()


def rate(
    db: Database,
    provider: str,
    model: str,
    *,
    inp: float,
    out: float,
    cached: float | None = None,
    stamp: str | None = None,
) -> None:
    with db.write() as conn:
        conn.execute(
            "insert or replace into cost_rates (provider, model, currency, input_per_mtok, "
            "output_per_mtok, cached_input_per_mtok, source, fetched_at) values (?,?,?,?,?,?,?,?)",
            (provider, model, "USD", inp, out, cached, "manual", stamp or now_iso()),
        )


def test_the_providers_own_number_wins_over_any_table(traces: Database) -> None:
    rate(traces, "openrouter", "qwen/qwen3", inp=1.0, out=2.0)
    found = cost_for(traces, "openrouter", "qwen/qwen3", USAGE, 0.0042)
    assert found.usd == pytest.approx(0.0042)
    assert found.source == "provider"


def test_the_table_prices_a_call_the_provider_said_nothing_about(traces: Database) -> None:
    rate(traces, "openai_compat", "local/model", inp=1.0, out=2.0)
    found = cost_for(traces, "openai_compat", "local/model", USAGE, None)
    assert found.usd == pytest.approx(1.0 + 1.0)
    assert found.source == "table"


def test_a_model_nobody_priced_is_unknown_and_never_zero(traces: Database) -> None:
    found = cost_for(traces, "openai_compat", "nobody/knows", USAGE, None)
    assert found.source == "unknown"
    assert not found.known


def test_a_wildcard_row_prices_every_model_that_provider_serves(traces: Database) -> None:
    rate(traces, "ollama", ANY_MODEL, inp=0.0, out=0.0)
    found = cost_for(traces, "ollama", "qwen3:8b", USAGE, None)
    assert found.usd == 0.0
    assert found.source == "table"
    assert found.known


def test_an_exact_model_row_beats_the_wildcard(traces: Database) -> None:
    rate(traces, "openrouter", ANY_MODEL, inp=10.0, out=10.0)
    rate(traces, "openrouter", "cheap/model", inp=1.0, out=1.0)
    found = cost_for(traces, "openrouter", "cheap/model", USAGE, None)
    assert found.usd == pytest.approx(1.0 + 0.5)


def test_cached_input_is_priced_at_its_own_rate_when_there_is_one(traces: Database) -> None:
    rate(traces, "openrouter", "cached/model", inp=10.0, out=0.0, cached=1.0)
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=900_000)
    found = cost_for(traces, "openrouter", "cached/model", usage, None)
    assert found.usd == pytest.approx(0.1 * 10.0 + 0.9 * 1.0)


def test_cached_input_falls_back_to_the_plain_rate_when_there_is_not(traces: Database) -> None:
    rate(traces, "openrouter", "plain/model", inp=10.0, out=0.0)
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=900_000)
    found = cost_for(traces, "openrouter", "plain/model", usage, None)
    assert found.usd == pytest.approx(10.0)


def test_the_newest_row_per_model_is_the_one_that_prices(traces: Database) -> None:
    rate(traces, "openrouter", "moved/model", inp=10.0, out=10.0, stamp="2020-01-01T00:00:00")
    rate(traces, "openrouter", "moved/model", inp=1.0, out=1.0, stamp="2030-01-01T00:00:00")
    found = cost_for(traces, "openrouter", "moved/model", USAGE, None)
    assert found.usd == pytest.approx(1.5)


def test_local_inference_is_zero_from_the_provider_not_from_a_guess(traces: Database) -> None:
    found = cost_for(traces, "ollama", "qwen3:8b", USAGE, 0.0)
    assert found.usd == 0.0
    assert found.source == "provider"


def test_the_shipped_seed_loads_and_prices_the_local_providers(traces: Database) -> None:
    loaded = seed_rates(traces, pricing_toml())
    assert loaded >= 3
    for provider in ("ollama", "sentence_transformers", "fake"):
        found = cost_for(traces, provider, "anything", USAGE, None)
        assert found.source == "table"
        assert found.usd == 0.0


def test_the_seed_ships_no_remote_price_it_would_have_had_to_invent(traces: Database) -> None:
    seed_rates(traces, pricing_toml())
    rows = list(traces.read().execute("select provider from current_rates"))
    assert {str(r["provider"]) for r in rows} == {"ollama", "sentence_transformers", "fake"}
    assert cost_for(traces, "openrouter", "qwen/qwen3", USAGE, None).source == "unknown"


def test_seeding_twice_leaves_one_live_rate_per_model(traces: Database) -> None:
    seed_rates(traces, pricing_toml())
    seed_rates(traces, pricing_toml())
    rows = list(traces.read().execute("select provider, model from current_rates"))
    assert len(rows) == len({(r["provider"], r["model"]) for r in rows})


def test_a_missing_seed_file_is_not_an_error(traces: Database, tmp_path: Path) -> None:
    assert seed_rates(traces, tmp_path / "nothing.toml") == 0
