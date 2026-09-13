"""The client mirrors the server's event union by hand, so something has to check it."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_args

from palate.agent import events
from palate.agent.answer import FilmFacts, StructuredAnswer, films_json

EVENTS_TS = Path(__file__).resolve().parents[1] / "web" / "src" / "api" / "events.ts"

_INTERFACE = re.compile(r"export interface (\w+) \{(.*?)\n\}", re.DOTALL)

_UNION = re.compile(r"export type AgentEvent =([^\n]*(?:\n\s*\|[^\n]*)*)")

_FIELD = re.compile(r"^\s{2}(\w+)[?]?:", re.MULTILINE)

_TAG = re.compile(r'^\s{2}type: "([\w.]+)"', re.MULTILINE)

_LISTED_TAG = re.compile(r'^  "([\w.]+)",$', re.MULTILINE)


def source() -> str:
    """The mirror, read as text because this suite has no node behind it."""
    return EVENTS_TS.read_text(encoding="utf-8")


def server_members() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Class name to (event tag, field names) for every member of the python union."""
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for member in get_args(events.AgentEvent.__value__):
        assert is_dataclass(member)
        out[member.__name__] = (member().type, tuple(f.name for f in fields(member)))
    return out


def client_members() -> dict[str, tuple[str, tuple[str, ...]]]:
    """The same, parsed out of events.ts."""
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for name, body in _INTERFACE.findall(source()):
        tag = _TAG.search(body)
        if tag is not None:
            out[name] = (tag.group(1), tuple(_FIELD.findall(body)))
    return out


def union_names(text: str) -> set[str]:
    """The interface names listed in the exported union."""
    found = _UNION.search(text)
    assert found is not None, "events.ts has no exported AgentEvent union"
    return {part.strip() for part in found.group(1).split("|") if part.strip()}


def body_of(name: str) -> str:
    return source().split(f"export interface {name} {{")[1].split("\n}")[0]


def test_the_two_unions_have_the_same_member_names() -> None:
    assert union_names(source()) == set(server_members())


def test_every_member_carries_the_same_tag_and_the_same_fields() -> None:
    server = server_members()
    client = client_members()
    assert set(client) == set(server)
    for name, expected in server.items():
        assert client[name] == expected, name


def test_the_tags_the_client_accepts_at_runtime_are_the_server_tags() -> None:
    assert set(_LISTED_TAG.findall(source())) == {tag for tag, _ in server_members().values()}


def test_dropping_one_member_from_the_mirror_fails_the_comparison() -> None:
    assert union_names(source().replace("  | RunFailed\n", "")) != set(server_members())


def test_the_answered_film_shape_is_what_the_recommendations_event_carries() -> None:
    facts = {7: FilmFacts(7, "Stalker", 1979, 161, ("Andrei Tarkovsky",), watched=False)}
    answer = StructuredAnswer.model_validate(
        {"recommendations": [{"film_id": 7, "why": "a reason long enough to validate"}]}
    )
    sent = films_json(answer, facts)[0]
    assert set(_FIELD.findall(body_of("AnsweredFilm"))) == set(sent)
