"""The HTTP surface, offline: no network, no keys, and a scripted provider behind the loop."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import orjson
import pytest
from fastapi.testclient import TestClient
from fixtures.agent import call, completion, films_seen
from fixtures.synth import LOVED, pipeline
from fixtures.synth.pipeline import SynthEmbedder

from palate.agent.events import RunStarted
from palate.agent.loop import AgentLoop
from palate.agent.prompts import PromptRegistry
from palate.agent.transcript import Transcript
from palate.api import sse
from palate.api.app import create_app
from palate.api.deps import AppState, RunRegistry, build_state, profile_check
from palate.config import Settings
from palate.db.connect import Database, open_database
from palate.errors import PalateError
from palate.memory.sessions import SessionStore
from palate.obs.cost import seed_rates
from palate.obs.store import TraceStore
from palate.obs.trace import SQLiteTracer
from palate.paths import pricing_toml, trace_migrations_dir
from palate.providers.base import Completion, Message, ToolSchema
from palate.providers.chat.fake import FakeChatProvider
from palate.retrieval.vocab import Vocabulary
from palate.taste.memory import PreferenceStore
from palate.tools.catalog import build_registry

ASKED = "something slow and cold"

SEARCH = call("search_films", '{"query": "something slow and cold", "limit": 5}')

PREAMBLE = "Three of these sit in the same corner of your history."


def scripted(messages: list[Message], tools: list[ToolSchema]) -> Completion:
    """Search, read the records, then answer with ids the run actually retrieved."""
    records = films_seen(messages)
    if tools and not records:
        return completion("", (SEARCH,))
    if tools and not any("overview" in r for r in records):
        wanted = orjson.dumps({"film_ids": [r["film_id"] for r in records[:3]]}).decode()
        return completion("", (call("get_film", wanted, ident="g1"),))
    read = next(r for r in records if r.get("overview"))
    return completion(
        orjson.dumps(
            {
                "preamble": PREAMBLE,
                "recommendations": [{"film_id": read["film_id"], "why": read["overview"]}],
            }
        ).decode()
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("api"))
    yield built
    built.close()


@pytest.fixture(scope="module")
def traces(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Database]:
    db = open_database(
        tmp_path_factory.mktemp("api_traces") / "traces.db",
        migrations=trace_migrations_dir(),
        load_vec=False,
    )
    seed_rates(db, pricing_toml())
    yield db
    db.close()


@pytest.fixture(scope="module")
def store(traces: Database) -> Iterator[TraceStore]:
    opened = TraceStore(traces, flush_ms=20)
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def client(world: pipeline.Fitted, traces: Database, store: TraceStore) -> Iterator[TestClient]:
    provider = FakeChatProvider(scripted)

    @asynccontextmanager
    async def builder(settings: Settings) -> AsyncIterator[AppState]:
        yield AppState(
            settings=settings,
            db=world.db,
            traces=traces,
            chat=provider,
            embedder=SynthEmbedder(world.world, cluster=LOVED[0]),
            loop=AgentLoop(
                provider,
                build_registry(),
                PromptRegistry(),
                settings,
                transcript=Transcript(world.db),
                tracer=SQLiteTracer(store),
            ),
            transcript=Transcript(world.db),
            sessions=SessionStore(world.db),
            prefs=PreferenceStore(world.db),
            vocab=Vocabulary(world.db.read()),
            profile=world.profile,
            runs=RunRegistry(),
        )

    with TestClient(create_app(Settings(), builder=builder)) as opened:
        yield opened


def events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """The stream parsed back into (tag, payload) pairs, comments dropped."""
    out: list[tuple[str, dict[str, Any]]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        tag = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                tag = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        if tag and data:
            out.append((tag, orjson.loads(data)))
    return out


def test_health_reports_the_index_and_the_corpus(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["corpus"]["films"] > 0
    assert body["index"]["dim"] > 0
    assert body["profile"]["tier"] in ("cold", "thin", "full")


def test_a_film_record_carries_the_overview_offset(
    client: TestClient, world: pipeline.Fitted
) -> None:
    film_id = world.world.films[0].tmdb_id
    body = client.get(f"/films/{film_id}").json()
    assert body["film_id"] == film_id
    assert body["title"]
    assert body["overview_length"] == len(body["overview"])


def test_an_unknown_film_id_is_a_404(client: TestClient) -> None:
    assert client.get("/films/999999").status_code == 404


def test_title_search_finds_the_film_by_its_own_name(
    client: TestClient, world: pipeline.Fitted
) -> None:
    title = world.world.films[0].title
    found = client.get("/films/search", params={"q": title, "limit": 5}).json()
    assert [row["title"] for row in found] == [title]


def test_a_percent_sign_in_the_query_matches_no_wildcard(client: TestClient) -> None:
    assert client.get("/films/search", params={"q": "%"}).json() == []


def test_compare_returns_set_algebra_and_not_prose(
    client: TestClient, world: pipeline.Fitted
) -> None:
    a, b = (f.tmdb_id for f in world.world.films[:2])
    body = client.get(f"/films/{a}/compare/{b}").json()
    assert body["a"]["film_id"] == a
    assert body["b"]["film_id"] == b
    assert isinstance(body["shared_keywords"], list)


def test_recommend_ranks_without_the_agent(client: TestClient, world: pipeline.Fitted) -> None:
    body = client.post("/recommend", json={"search": {"query": ASKED, "limit": 5}}).json()
    watched = {f.tmdb_id for f in world.world.rated}
    assert body["films"]
    assert not {f["film_id"] for f in body["films"]} & watched
    assert body["diagnostics"]["returned"] == len(body["films"])
    assert all(f["title"] for f in body["films"])
    assert all("poster_path" in f for f in body["films"])


def test_recommend_rejects_a_genre_the_corpus_does_not_have(client: TestClient) -> None:
    answer = client.post(
        "/recommend", json={"search": {"query": ASKED, "include_genres": ["klezmer"]}}
    )
    assert answer.status_code == 422
    assert "klezmer" in answer.json()["detail"]


def test_the_profile_and_one_of_its_modes_read_back(client: TestClient) -> None:
    profile = client.get("/profile").json()
    assert profile["n_rated"] > 0
    assert profile["modes"]
    mode_id = profile["modes"][0]["mode_id"]
    detail = client.get(f"/profile/modes/{mode_id}").json()
    assert detail["mode_id"] == mode_id
    assert detail["members"]
    assert client.get("/profile/modes/9999").status_code == 404


def test_vocabulary_resolution_carries_affected_counts(client: TestClient) -> None:
    body = client.get("/vocabulary/resolve", params={"text": "musicals"}).json()
    assert body["meta"]["asked"] == "musicals"
    assert all(m["affected_films"] >= 0 for m in body["matches"])


def test_the_chat_stream_is_framed_and_ordered(client: TestClient) -> None:
    answer = client.post("/chat", json={"message": ASKED})
    assert answer.headers["content-type"].startswith("text/event-stream")
    framed = events(answer.text)
    tags = [tag for tag, _ in framed]
    assert tags[0] == "run.started"
    assert tags[-1] == "run.finished"
    assert "recommendations" in tags
    assert all(tag == payload["type"] for tag, payload in framed)


def test_the_stream_names_only_films_the_run_retrieved(
    client: TestClient, world: pipeline.Fitted
) -> None:
    framed = events(client.post("/chat", json={"message": ASKED}).text)
    films = next(payload for tag, payload in framed if tag == "recommendations")
    known = {f.tmdb_id: f.title for f in world.world.films}
    assert films["films"]
    for film in films["films"]:
        assert known[film["film_id"]] == film["title"]


def test_a_run_writes_its_session_and_replays_its_messages(client: TestClient) -> None:
    framed = events(client.post("/chat", json={"message": ASKED}).text)
    started = next(payload for tag, payload in framed if tag == "run.started")
    session_id = started["session_id"]
    listed = client.get("/sessions").json()
    assert session_id in [row["session_id"] for row in listed]
    messages = client.get(f"/sessions/{session_id}/messages").json()
    assert [m["role"] for m in messages][:1] == ["user"]
    assert messages[0]["content"] == ASKED
    assert any(m["role"] == "assistant" and PREAMBLE in m["content"] for m in messages)


def test_a_second_turn_continues_the_same_session(client: TestClient) -> None:
    first = events(client.post("/chat", json={"message": ASKED}).text)
    session_id = next(p for t, p in first if t == "run.started")["session_id"]
    second = client.post("/chat", json={"message": ASKED, "session_id": session_id})
    again = next(p for t, p in events(second.text) if t == "run.started")
    assert again["session_id"] == session_id
    assert again["run_id"] != next(p for t, p in first if t == "run.started")["run_id"]


def test_cancelling_a_run_that_is_not_running_is_a_404(client: TestClient) -> None:
    assert client.post("/chat/run_nothing/cancel").status_code == 404


def test_a_budget_override_narrows_and_never_widens(client: TestClient) -> None:
    answer = client.post("/chat", json={"message": ASKED, "budget": {"max_turns": 32}})
    started = next(p for t, p in events(answer.text) if t == "run.started")
    assert started["budget"]["max_turns"] == Settings().agent.max_turns


def test_preferences_start_empty_and_an_unknown_undo_is_a_404(client: TestClient) -> None:
    body = client.get("/preferences").json()
    assert body["preferences"] == []
    assert client.delete("/preferences/424242").status_code == 404


def test_traces_list_the_run_and_open_its_span_tree(client: TestClient, store: TraceStore) -> None:
    framed = events(client.post("/chat", json={"message": ASKED}).text)
    session_id = next(p for t, p in framed if t == "run.started")["session_id"]
    store.flush()
    listed = client.get("/traces", params={"since": "1h"}).json()
    # The tracer mints its own run id, so a session is what joins a chat turn to its trace.
    traced = [row for row in listed if row["session_id"] == session_id]
    assert traced and traced[0]["kind"] == "chat"
    detail = client.get(f"/traces/{traced[0]['run_id']}").json()
    assert detail["spans"]
    assert detail["llm_calls"]
    assert [c["tool_name"] for c in detail["tool_calls"]] == ["search_films", "get_film"]
    assert client.get("/traces/run_nothing").status_code == 404


def test_the_cost_rollup_answers_even_when_nothing_was_billed(client: TestClient) -> None:
    rows = client.get("/traces/costs", params={"since": "1h"}).json()
    assert all(row["usd"] >= 0.0 for row in rows)


def test_startup_refuses_with_the_line_doctor_would_print(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "empty")
    with pytest.raises(PalateError) as exc, TestClient(create_app(settings)):
        pass
    assert "palate index build" in str(exc.value)


def test_a_profile_fitted_for_another_index_is_refused(world: pipeline.Fitted) -> None:
    check = profile_check(world.profile, "some_other_index")
    assert not check.ok
    assert check.fix == "palate profile build"
    assert profile_check(None, None).detail == "none fitted"


def test_build_state_is_the_default_builder() -> None:
    assert create_app(Settings()) is not None
    assert build_state is not None


async def test_the_registry_cancels_the_scope_it_was_given() -> None:
    registry = RunRegistry()
    scope = anyio.CancelScope()
    registry.add("run_1", scope)
    assert registry.cancel("run_1")
    assert scope.cancel_called
    registry.drop("run_1")
    assert not registry.cancel("run_1")


def test_an_event_is_framed_under_its_own_type_tag() -> None:
    framed = sse.frame(RunStarted(run_id="run_1", model="qwen3:8b")).encode().decode()
    assert framed.startswith("event: run.started")
    assert '"run_id":"run_1"' in framed
    assert sse.heartbeat().encode().decode().startswith(": heartbeat")
