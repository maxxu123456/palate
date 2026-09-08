"""TMDB read client: bearer auth, one request per film, ETag reuse."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import httpx
from anyio.from_thread import BlockingPortal, start_blocking_portal
from pydantic import SecretStr

from palate.errors import ProviderTimeout, ProviderUnavailable, TMDBError
from palate.ingest.resolve import Candidate
from palate.providers.http import build_client
from palate.tmdb.models import DiscoverParams, MovieDetail, MoviePage

BASE_URL = "https://api.themoviedb.org/3"

# Four appends is one request and one rate limit token instead of five.
DEFAULT_APPEND = ("credits", "keywords", "release_dates", "external_ids")

# How many search hits are worth a release window lookup when the year disagrees.
WINDOW_LOOKUPS = 3
WINDOW_APPEND = ("release_dates",)

JSONObject = dict[str, Any]


@runtime_checkable
class RateLimiter(Protocol):
    """Whatever paces outbound requests and reacts to a 429."""

    async def acquire(self) -> None: ...

    def penalise(self, *, retry_after: float | None) -> None: ...

    def succeed(self) -> None: ...


class NullLimiter:
    """No pacing at all, for tests and one-off lookups."""

    async def acquire(self) -> None:
        return None

    def penalise(self, *, retry_after: float | None) -> None:
        return None

    def succeed(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class TMDBResponse:
    """One TMDB reply, including the ones that carry no body."""

    status: int
    payload: JSONObject | None
    etag: str | None
    from_cache: bool
    latency_ms: float
    retry_after: float | None = None
    path: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.payload is not None


def parse_retry_after(value: str | None) -> float | None:
    """TMDB sends whole seconds. Anything else is treated as absent."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class TMDBClient:
    """Reads TMDB with the v4 token in a header, never in a query string."""

    def __init__(
        self,
        *,
        token: SecretStr,
        client: httpx.AsyncClient,
        limiter: RateLimiter | None = None,
        language: str = "en-US",
        base_url: str = BASE_URL,
    ) -> None:
        self._token = token
        self._http = client
        self._limiter: RateLimiter = limiter or NullLimiter()
        self._language = language
        self._base = base_url.rstrip("/")

    @property
    def http(self) -> httpx.AsyncClient:
        """The underlying client, for the id export which lives on another host."""
        return self._http

    async def movie(
        self,
        tmdb_id: int,
        *,
        append: Sequence[str] = DEFAULT_APPEND,
        etag: str | None = None,
    ) -> TMDBResponse:
        """Full detail for one film, with credits, keywords and ids in the same call."""
        params = {"language": self._language}
        if append:
            params["append_to_response"] = ",".join(append)
        return await self._get(f"/movie/{tmdb_id}", params, etag=etag)

    async def discover(self, params: DiscoverParams, *, page: int = 1) -> TMDBResponse:
        """One page of /discover/movie."""
        query = {"language": self._language, "page": str(page), **params.as_query()}
        return await self._get("/discover/movie", query)

    async def search_movie(self, query: str, *, year: int | None = None) -> TMDBResponse:
        """Title search. `year` matches any release, `primary_release_year` only the first."""
        params = {"language": self._language, "query": query, "include_adult": "false"}
        if year is not None:
            params["year"] = str(year)
        return await self._get("/search/movie", params)

    async def recommendations(self, tmdb_id: int, *, page: int = 1) -> TMDBResponse:
        """TMDB's own recommendations, the niche neighbour channel."""
        return await self._get(
            f"/movie/{tmdb_id}/recommendations", {"language": self._language, "page": str(page)}
        )

    async def similar(self, tmdb_id: int, *, page: int = 1) -> TMDBResponse:
        """Keyword and genre neighbours, which overlap recommendations only partly."""
        return await self._get(
            f"/movie/{tmdb_id}/similar", {"language": self._language, "page": str(page)}
        )

    async def _get(
        self, path: str, params: dict[str, str], *, etag: str | None = None
    ) -> TMDBResponse:
        await self._limiter.acquire()
        headers = {"Authorization": f"Bearer {self._token.get_secret_value()}"}
        if etag:
            headers["If-None-Match"] = etag
        started = time.perf_counter()
        try:
            response = await self._http.get(self._base + path, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"tmdb {path} timed out", provider="tmdb", retryable=True
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(
                f"tmdb {path} unreachable ({exc})", provider="tmdb", retryable=True
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._interpret(path, response, latency_ms, etag)

    def _interpret(
        self, path: str, response: httpx.Response, latency_ms: float, sent_etag: str | None
    ) -> TMDBResponse:
        status = response.status_code
        etag = response.headers.get("etag") or sent_etag
        if status == 304:
            self._limiter.succeed()
            return TMDBResponse(304, None, etag, True, latency_ms, path=path)
        if status == 200:
            self._limiter.succeed()
            return TMDBResponse(200, _json(response, path), etag, False, latency_ms, path=path)
        if status == 429:
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            self._limiter.penalise(retry_after=retry_after)
            return TMDBResponse(429, None, etag, False, latency_ms, retry_after, path)
        if status in (401, 403):
            # A bad token fails every request, so stopping beats 40k identical rejections.
            raise TMDBError(f"tmdb rejected the read token on {path} ({status})")
        return TMDBResponse(status, None, etag, False, latency_ms, path=path)


def _json(response: httpx.Response, path: str) -> JSONObject:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TMDBError(f"tmdb {path} returned a body that is not json") from exc
    if not isinstance(payload, dict):
        raise TMDBError(f"tmdb {path} returned {type(payload).__name__}, expected an object")
    return payload


class TMDBTitleSearch:
    """Synchronous title search for the import path. Use it as a context manager."""

    def __init__(
        self,
        token: SecretStr,
        *,
        language: str = "en-US",
        limit: int = 10,
        limiter: RateLimiter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token
        self._language = language
        self._limit = limit
        self._limiter = limiter
        self._transport = transport
        self._stack = ExitStack()
        self._portal: BlockingPortal | None = None
        self._client: TMDBClient | None = None

    def __enter__(self) -> TMDBTitleSearch:
        # One portal for the whole import, so the pool is not rebuilt per title.
        portal = self._stack.enter_context(start_blocking_portal())
        http = build_client(transport=self._transport)
        self._stack.callback(partial(portal.call, http.aclose))
        self._portal = portal
        self._client = TMDBClient(
            token=self._token, client=http, limiter=self._limiter, language=self._language
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._portal = None
        self._client = None
        self._stack.close()

    def search(self, title: str, year: int | None) -> Sequence[Candidate]:
        """Candidates for one export row, ordered as TMDB ordered them."""
        portal, client = self._portal, self._client
        if portal is None or client is None:
            raise RuntimeError("use TMDBTitleSearch as a context manager")
        response = portal.call(partial(client.search_movie, title, year=year))
        if not response.ok or response.payload is None:
            return ()
        found = list(to_candidates(response.payload, limit=self._limit))
        if year is None or any(year in c.years for c in found):
            return tuple(found)
        # A restoration dates the primary release to the reissue, so read the windows.
        for index, candidate in enumerate(found[:WINDOW_LOOKUPS]):
            detail = portal.call(partial(client.movie, candidate.tmdb_id, append=WINDOW_APPEND))
            if not detail.ok or detail.payload is None:
                continue
            found[index] = to_candidate(detail.payload)
            if year in found[index].years:
                break
        return tuple(found)


def to_candidate(payload: JSONObject) -> Candidate:
    """One /movie/{id} payload as a candidate, release windows included."""
    detail = MovieDetail.model_validate(payload)
    return Candidate(
        tmdb_id=detail.id,
        title=detail.title,
        original_title=detail.original_title,
        year=detail.year,
        vote_count=detail.vote_count,
        release_years=detail.release_years,
    )


def to_candidates(payload: JSONObject, *, limit: int = 10) -> tuple[Candidate, ...]:
    """Search results as resolver candidates."""
    page = MoviePage.model_validate(payload)
    return tuple(
        Candidate(
            tmdb_id=result.id,
            title=result.title,
            original_title=result.original_title,
            year=result.year,
            vote_count=result.vote_count,
        )
        for result in page.results[:limit]
    )
