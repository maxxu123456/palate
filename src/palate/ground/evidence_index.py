"""Every film record any tool returned this run, which is what grounding checks against."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

# Fields that carry names a sentence could attach to the wrong film.
NAME_FIELDS = ("directors", "writers", "cast", "countries", "genres", "companies")

TEXT_FIELDS = ("overview", "tagline")

_PROPER = re.compile(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*")


@dataclass(slots=True)
class FilmEvidence:
    """One film as the run actually saw it, merged across every tool that returned it."""

    film_id: int
    title: str = ""
    year: int | None = None
    runtime: int | None = None
    original_language: str | None = None
    names: set[str] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)
    overview: str = ""
    tagline: str = ""
    your_rating: float | None = None
    watched: bool = False

    def premise(self) -> str:
        """Everything a plot claim is allowed to be checked against."""
        parts = [self.overview, self.tagline, ", ".join(sorted(self.keywords))]
        return " ".join(p for p in parts if p)

    def owns(self, name: str) -> bool:
        """Whether this film actually carries the name, case and spacing folded."""
        wanted = _fold(name)
        return wanted in {_fold(n) for n in self.names} or wanted == _fold(self.title)


@dataclass(slots=True)
class EvidenceIndex:
    """Grows as tool results arrive. Nothing the model said ever gets in here."""

    films: dict[int, FilmEvidence] = field(default_factory=dict)

    def absorb(self, data: Any) -> None:
        """Walk a tool result for film records and merge each one in."""
        for record in _records(data):
            self._merge(record)

    def get(self, film_id: int) -> FilmEvidence | None:
        """One film, or None when no tool ever returned it."""
        return self.films.get(film_id)

    def seen(self, film_id: int) -> bool:
        """Whether any tool this run actually returned this film."""
        return film_id in self.films

    def gazetteer(self) -> frozenset[str]:
        """Every proper noun the run saw, which is the only vocabulary entity checks trust."""
        found: set[str] = set()
        for film in self.films.values():
            found.update(film.names)
            if film.title:
                found.add(film.title)
        return frozenset(found)

    def ratings(self) -> Mapping[int, float]:
        """The user's own stars, per film, as some tool reported them."""
        return {i: f.your_rating for i, f in self.films.items() if f.your_rating is not None}

    def __len__(self) -> int:
        return len(self.films)

    def _merge(self, record: dict[str, Any]) -> None:
        film_id = int(record["film_id"])
        film = self.films.get(film_id)
        if film is None:
            film = FilmEvidence(film_id=film_id)
            self.films[film_id] = film
        film.title = _text(record.get("title")) or film.title
        film.year = _int(record.get("year")) or film.year
        film.runtime = _int(record.get("runtime")) or film.runtime
        film.original_language = _text(record.get("original_language")) or film.original_language
        for key in NAME_FIELDS:
            film.names.update(_strings(record.get(key)))
        film.keywords.update(_strings(record.get("keywords")))
        for key in TEXT_FIELDS:
            setattr(film, key, _text(record.get(key)) or getattr(film, key))
        rating = record.get("your_rating")
        if isinstance(rating, int | float):
            film.your_rating = float(rating)
        film.watched = bool(record.get("watched")) or film.watched or film.your_rating is not None


def proper_nouns(sentence: str) -> list[str]:
    """Capitalised runs. A sentence initial word is kept, the gazetteer filters it out."""
    return [m.group(0).strip() for m in _PROPER.finditer(sentence) if m.group(0).strip()]


def _records(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if isinstance(node.get("film_id"), int):
            yield node
        for value in node.values():
            yield from _records(value)
    elif isinstance(node, list):
        for item in node:
            yield from _records(item)


def _strings(value: Any) -> set[str]:
    if isinstance(value, list):
        return {v.strip() for v in value if isinstance(v, str) and v.strip()}
    return set()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()
