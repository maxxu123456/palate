"""One httpx client factory so every outbound call shares timeouts and a user agent."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from palate import __version__

USER_AGENT = f"palate/{__version__}"

# Long reads, short connects: a slow TMDB page is normal, an unreachable host is not.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
DEFAULT_LIMITS = httpx.Limits(max_connections=16, max_keepalive_connections=16)


def build_client(
    *,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
    limits: httpx.Limits = DEFAULT_LIMITS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """An AsyncClient carrying the shared timeout, pool limits and user agent."""
    merged = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    merged.update(headers or {})
    return httpx.AsyncClient(
        base_url=base_url,
        headers=merged,
        timeout=timeout,
        limits=limits,
        transport=transport,
        follow_redirects=True,
    )


@asynccontextmanager
async def client_session(**kwargs: Any) -> AsyncIterator[httpx.AsyncClient]:
    """Open a client for the duration of a block and close it on the way out."""
    client = build_client(**kwargs)
    try:
        yield client
    finally:
        await client.aclose()
