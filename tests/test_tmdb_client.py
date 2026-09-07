"""The TMDB client sends a bearer token, one request per film, and reuses ETags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from palate.errors import ProviderTimeout, ProviderUnavailable, TMDBError
from palate.providers.http import build_client
from palate.tmdb.client import TMDBClient, TMDBTitleSearch, to_candidates
from palate.tmdb.models import DiscoverParams, MovieDetail, MoviePage

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"
API = "https://api.themoviedb.org/3"
TOKEN = SecretStr("read-token-value")


def payload(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


class RecordingLimiter:
    """Counts what the client told the limiter, in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.retry_after: float | None = None

    async def acquire(self) -> None:
        self.events.append("acquire")

    def penalise(self, *, retry_after: float | None) -> None:
        self.events.append("penalise")
        self.retry_after = retry_after

    def succeed(self) -> None:
        self.events.append("succeed")


def make_client(http: httpx.AsyncClient, limiter: RecordingLimiter | None = None) -> TMDBClient:
    return TMDBClient(token=TOKEN, client=http, limiter=limiter)


@respx.mock
async def test_movie_sends_a_bearer_token_and_one_appended_request() -> None:
    route = respx.get(f"{API}/movie/1398").mock(
        return_value=httpx.Response(200, json=payload("movie_1398_stalker.json"))
    )
    async with build_client() as http:
        response = await make_client(http).movie(1398)
    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {TOKEN.get_secret_value()}"
    assert "api_key" not in request.url.query.decode()
    assert request.url.params["append_to_response"] == "credits,keywords,release_dates,external_ids"
    assert response.ok
    assert response.path == "/movie/1398"
    assert MovieDetail.model_validate(response.payload).title == "Stalker"


@respx.mock
async def test_an_etag_comes_back_and_a_second_call_sends_it() -> None:
    respx.get(f"{API}/movie/1398").mock(
        return_value=httpx.Response(
            200, json=payload("movie_1398_stalker.json"), headers={"ETag": '"v1"'}
        )
    )
    async with build_client() as http:
        first = await make_client(http).movie(1398)
    assert first.etag == '"v1"'

    route = respx.get(f"{API}/movie/1398").mock(return_value=httpx.Response(304))
    async with build_client() as http:
        second = await make_client(http).movie(1398, etag=first.etag)
    assert route.calls.last.request.headers["if-none-match"] == '"v1"'
    assert second.status == 304
    assert second.from_cache
    assert second.payload is None
    assert second.etag == '"v1"'


@respx.mock
async def test_a_429_penalises_the_limiter_and_carries_retry_after() -> None:
    respx.get(f"{API}/movie/603").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "7"})
    )
    limiter = RecordingLimiter()
    async with build_client() as http:
        response = await make_client(http, limiter).movie(603)
    assert response.status == 429
    assert response.retry_after == 7.0
    assert limiter.events == ["acquire", "penalise"]
    assert limiter.retry_after == 7.0


@respx.mock
async def test_a_200_tells_the_limiter_it_may_climb() -> None:
    respx.get(f"{API}/movie/603").mock(
        return_value=httpx.Response(200, json=payload("movie_603_matrix.json"))
    )
    limiter = RecordingLimiter()
    async with build_client() as http:
        await make_client(http, limiter).movie(603)
    assert limiter.events == ["acquire", "succeed"]


@respx.mock
async def test_a_404_is_returned_not_raised() -> None:
    respx.get(f"{API}/movie/999999").mock(
        return_value=httpx.Response(404, json={"status_code": 34})
    )
    async with build_client() as http:
        response = await make_client(http).movie(999999)
    assert response.status == 404
    assert response.payload is None


@respx.mock
async def test_a_rejected_token_stops_the_crawl() -> None:
    respx.get(f"{API}/movie/603").mock(return_value=httpx.Response(401, json={"status_code": 7}))
    async with build_client() as http:
        with pytest.raises(TMDBError, match="read token"):
            await make_client(http).movie(603)


@respx.mock
async def test_a_body_that_is_not_json_is_an_error() -> None:
    respx.get(f"{API}/movie/603").mock(return_value=httpx.Response(200, text="<html>nope</html>"))
    async with build_client() as http:
        with pytest.raises(TMDBError, match="not json"):
            await make_client(http).movie(603)


@respx.mock
async def test_transport_failures_map_onto_provider_errors() -> None:
    respx.get(f"{API}/movie/1").mock(side_effect=httpx.ReadTimeout("slow"))
    respx.get(f"{API}/movie/2").mock(side_effect=httpx.ConnectError("refused"))
    async with build_client() as http:
        client = make_client(http)
        with pytest.raises(ProviderTimeout):
            await client.movie(1)
        with pytest.raises(ProviderUnavailable):
            await client.movie(2)


@respx.mock
async def test_discover_spells_its_parameters_the_way_tmdb_does() -> None:
    route = respx.get(f"{API}/discover/movie").mock(
        return_value=httpx.Response(200, json=payload("discover_page1.json"))
    )
    params = DiscoverParams(
        vote_count_gte=25,
        primary_release_date_gte="1979-01-01",
        primary_release_date_lte="1979-12-31",
        sort_by="vote_count.desc",
    )
    async with build_client() as http:
        response = await make_client(http).discover(params, page=2)
    query = route.calls.last.request.url.params
    assert query["vote_count.gte"] == "25"
    assert query["primary_release_date.gte"] == "1979-01-01"
    assert query["sort_by"] == "vote_count.desc"
    assert query["page"] == "2"
    assert MoviePage.model_validate(response.payload).total_results == 5


@respx.mock
async def test_search_pins_the_year_and_yields_candidates() -> None:
    route = respx.get(f"{API}/search/movie").mock(
        return_value=httpx.Response(200, json=payload("search_stalker.json"))
    )
    async with build_client() as http:
        response = await make_client(http).search_movie("Stalker", year=1979)
    assert route.calls.last.request.url.params["primary_release_year"] == "1979"
    candidates = to_candidates(response.payload or {})
    assert [c.tmdb_id for c in candidates] == [1398, 47116]
    assert candidates[0].year == 1979
    assert candidates[0].vote_count == 1547


@respx.mock
async def test_recommendations_and_similar_are_separate_pages() -> None:
    respx.get(f"{API}/movie/603/recommendations").mock(
        return_value=httpx.Response(200, json=payload("recommendations_603.json"))
    )
    respx.get(f"{API}/movie/603/similar").mock(
        return_value=httpx.Response(200, json=payload("similar_603.json"))
    )
    async with build_client() as http:
        client = make_client(http)
        recommended = await client.recommendations(603)
        similar = await client.similar(603)
    assert [r["id"] for r in (recommended.payload or {})["results"]] == [1398, 11104]
    assert [r["id"] for r in (similar.payload or {})["results"]] == [802]


def test_title_search_resolves_a_row_without_an_event_loop_of_its_own() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=payload("search_stalker.json"))

    with TMDBTitleSearch(TOKEN, transport=httpx.MockTransport(handler)) as search:
        first = search.search("Stalker", 1979)
        second = search.search("Stalker", None)
    assert [c.tmdb_id for c in first] == [1398, 47116]
    assert len(second) == 2
    # One portal, two calls, so the pool is shared rather than rebuilt per title.
    assert len(seen) == 2
    assert "primary_release_year" not in seen[1].params


def test_title_search_outside_its_context_is_a_programming_error() -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        TMDBTitleSearch(TOKEN).search("Stalker", 1979)
