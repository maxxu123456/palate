"""Traces land in their own file, off the latency path, and say what they dropped."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import pytest
from fixtures.agent import BUDGET, Calls, build_registry, call, context_factory

from palate.agent.loop import AgentLoop
from palate.agent.prompts import PromptRegistry
from palate.clock import frozen
from palate.config import Settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.obs import report
from palate.obs.replay import replay, stored_messages
from palate.obs.store import DROPPED_KEY, TraceRow, TraceStore, decompress
from palate.obs.trace import NullTracer, SQLiteTracer, current_span
from palate.paths import trace_migrations_dir
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn

ANSWER = "Two of those are long and cold."

SEARCH = call("search_films", '{"query": "slow and cold", "limit": 3}')


@pytest.fixture
def traces(tmp_path: Path) -> Iterator[Database]:
    db = open_database(tmp_path / "traces.db", migrations=trace_migrations_dir(), load_vec=False)
    yield db
    db.close()


def store_for(traces: Database, **kwargs: Any) -> TraceStore:
    return TraceStore(traces, flush_ms=20, **kwargs)


async def run_once(
    traces: Database, *, payloads: str = "hashed", turns: list[ScriptedTurn] | None = None
) -> tuple[TraceStore, Calls]:
    """One whole agent run with tracing on, drained and flushed."""
    store = store_for(traces, payloads=payloads)  # type: ignore[arg-type]
    seen = Calls()
    script = turns or [
        ScriptedTurn(text="looking", tool_calls=(SEARCH,)),
        ScriptedTurn(text=ANSWER),
    ]
    agent = AgentLoop(
        FakeChatProvider(script),
        build_registry(seen),
        PromptRegistry(),
        Settings(),
        tracer=SQLiteTracer(store),
    )
    await agent.run_to_completion(
        "something slow and cold",
        session_id="ses_trace",
        budget=BUDGET,
        ctx_factory=context_factory(),
    )
    store.flush()
    return store, seen


def rows(traces: Database, sql: str, *params: Any) -> list[Any]:
    return list(traces.read().execute(sql, params))


async def test_one_run_row_per_turn_with_the_ledger_totals(traces: Database) -> None:
    await run_once(traces)
    found = rows(traces, "select * from runs")
    assert len(found) == 1
    assert found[0]["kind"] == "chat"
    assert found[0]["session_id"] == "ses_trace"
    assert found[0]["status"] == "ok"
    assert found[0]["ended_at"] is not None
    assert found[0]["turns"] == 1


async def test_one_llm_span_per_provider_call_and_one_tool_span_per_call(
    traces: Database,
) -> None:
    await run_once(traces)
    kinds = [r["kind"] for r in rows(traces, "select kind from spans order by seq")]
    assert kinds.count("llm") == 2
    assert kinds.count("tool") == 1
    assert kinds.count("run") == 1
    assert len(rows(traces, "select * from llm_calls")) == 2
    assert len(rows(traces, "select * from tool_calls")) == 1


async def test_the_tool_row_carries_the_fingerprint_and_the_result_size(
    traces: Database,
) -> None:
    await run_once(traces)
    row = rows(traces, "select * from tool_calls")[0]
    assert row["tool_name"] == "search_films"
    assert row["tool_call_id"] == "c1"
    assert len(row["args_fingerprint"]) == 64
    assert row["args_valid"] == 1
    assert row["ok"] == 1
    assert row["result_rows"] == 3


async def test_a_failed_tool_call_carries_the_text_the_model_was_handed(
    traces: Database,
) -> None:
    broken = call("search_films", '{"query": "a"}')
    await run_once(
        traces,
        turns=[ScriptedTurn(tool_calls=(broken,)), ScriptedTurn(text=ANSWER)],
    )
    row = rows(traces, "select * from tool_calls")[0]
    assert row["ok"] == 0
    assert row["error_code"] == "bad_arguments"
    assert "at least 3 characters" in row["validation_error"]
    assert rows(traces, "select status from spans where kind = 'tool'")[0]["status"] == "error"


async def test_hashed_is_the_default_and_keeps_the_proof_not_the_text(
    traces: Database,
) -> None:
    await run_once(traces, payloads="hashed")
    row = rows(traces, "select * from llm_calls")[0]
    assert row["messages_z"] is None
    assert len(row["messages_sha"]) == 64
    assert row["response_z"] is None
    assert len(row["response_sha"]) == 64
    assert "slow and cold" not in orjson.dumps(dict(row), default=str).decode()
    tools = rows(traces, "select args_json, args_fingerprint from tool_calls")[0]
    assert tools["args_json"] is None
    assert tools["args_fingerprint"]


async def test_off_keeps_neither_the_text_nor_a_hash_of_it(traces: Database) -> None:
    await run_once(traces, payloads="off")
    row = rows(traces, "select * from llm_calls")[0]
    assert row["messages_z"] is None
    assert row["messages_sha"] == ""


async def test_full_is_what_makes_replay_possible(traces: Database) -> None:
    await run_once(traces, payloads="full")
    row = rows(traces, "select * from llm_calls order by rowid")[0]
    assert row["messages_z"] is not None
    assert "slow and cold" in decompress(bytes(row["messages_z"]))
    restored = stored_messages(traces, str(row["span_id"]))
    assert restored[0].role == "system"
    assert any(m.content == "something slow and cold" for m in restored)


async def test_replay_reissues_the_stored_call_and_diffs_it(traces: Database) -> None:
    await run_once(traces, payloads="full")
    span_id = str(rows(traces, "select span_id from llm_calls order by rowid")[0]["span_id"])
    provider = FakeChatProvider([ScriptedTurn(text="a different answer")])
    result = await replay(traces, span_id, provider=provider)
    assert result.new_response == "a different answer"
    assert result.original_response != result.new_response
    assert result.diff
    assert result.new_model == "fake-model"


async def test_replay_says_which_setting_is_missing_when_there_is_no_payload(
    traces: Database,
) -> None:
    await run_once(traces, payloads="hashed")
    span_id = str(rows(traces, "select span_id from llm_calls")[0]["span_id"])
    with pytest.raises(PalateError) as exc:
        stored_messages(traces, span_id)
    assert 'trace.payloads = "full"' in str(exc.value)


async def test_the_span_tree_knows_which_span_is_under_which(traces: Database) -> None:
    await run_once(traces)
    run_id = str(rows(traces, "select run_id from runs")[0]["run_id"])
    tree = report.tree(traces, run_id)
    root = [s for s in tree if s.kind == "run"]
    assert len(root) == 1
    assert root[0].depth == 0
    assert all(s.depth == 1 for s in tree if s.kind in ("llm", "tool"))
    assert all(s.parent_id == root[0].span_id for s in tree if s.kind != "run")


def test_a_full_queue_drops_and_counts_rather_than_blocking(traces: Database) -> None:
    store = TraceStore(traces, queue_max=1, flush_ms=10_000, batch=1_000)
    for n in range(200):
        store.submit(TraceRow("trace_stats", {"key": f"k{n}", "value": n}))
    assert store.dropped > 0
    store.close()
    assert report.status(traces).dropped == store.dropped
    assert report.status(traces).counters[DROPPED_KEY] == store.dropped


def test_a_store_that_dropped_nothing_says_so(traces: Database) -> None:
    store = store_for(traces)
    store.close()
    assert report.status(traces).dropped == 0


async def test_the_listing_and_the_rollup_read_the_same_run(traces: Database) -> None:
    await run_once(traces)
    listed = report.runs(traces, window="1d")
    assert len(listed) == 1
    assert listed[0].kind == "chat"
    priced = report.costs(traces, window="1d", group_by="model")
    assert priced
    assert priced[0].bucket == "fake-model"
    assert priced[0].calls == 2


async def test_gc_deletes_old_runs_and_everything_under_them(traces: Database) -> None:
    from datetime import UTC, datetime

    with frozen(datetime(2020, 1, 1, tzinfo=UTC)):
        await run_once(traces)
    assert len(rows(traces, "select * from spans")) > 1
    assert report.gc(traces, older_than="1d") == 1
    assert rows(traces, "select * from runs") == []
    assert rows(traces, "select * from spans") == []
    assert rows(traces, "select * from llm_calls") == []


def test_a_window_that_is_not_a_window_is_refused(traces: Database) -> None:
    with pytest.raises(ValueError, match="number followed by"):
        report.since("last tuesday")


def test_the_null_tracer_is_a_no_op_that_still_hands_back_a_span() -> None:
    tracer = NullTracer()
    with tracer.run("chat") as span:
        assert span.run_id == ""
        with tracer.span("chat", "llm") as child:
            child.set(anything=1)
            child.event("thing", n=1)
    tracer.flush()


async def test_the_current_span_is_reachable_without_passing_it(traces: Database) -> None:
    store = store_for(traces)
    tracer = SQLiteTracer(store)
    with tracer.run("chat") as outer:
        assert current_span() is outer
        with tracer.span("chat", "llm") as inner:
            assert current_span() is inner
        assert current_span() is outer
    assert current_span() is None
    store.close()
