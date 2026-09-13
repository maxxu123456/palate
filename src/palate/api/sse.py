"""Agent events as text/event-stream. The tag is the event's own type string."""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from palate.agent.events import AgentEvent

# A comment this often, so a proxy does not close an idle stream during a slow generation.
HEARTBEAT_S = 15


def frame(event: AgentEvent) -> ServerSentEvent:
    """One event, tagged with its type and carrying the dataclass as its data line."""
    return ServerSentEvent(data=orjson.dumps(event).decode(), event=event.type)


def heartbeat() -> ServerSentEvent:
    """The keepalive, which carries no data and so is ignored by every client."""
    return ServerSentEvent(comment="heartbeat")


async def _frames(events: AsyncIterator[AgentEvent]) -> AsyncIterator[ServerSentEvent]:
    async for event in events:
        yield frame(event)


def stream(events: AsyncIterator[AgentEvent]) -> EventSourceResponse:
    """The response. A client disconnect cancels the generator, which cancels the run."""
    return EventSourceResponse(_frames(events), ping=HEARTBEAT_S, ping_message_factory=heartbeat)
