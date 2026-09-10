"""Free text to corpus ids, with the two translations a naive lookup gets wrong."""

from __future__ import annotations

import difflib
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import orjson

type Kind = Literal["genre", "keyword", "language", "country", "person", "collection"]

# Language before country, because a nationality word is about the language far more often.
RESOLUTION_ORDER: tuple[Kind, ...] = (
    "language",
    "country",
    "genre",
    "keyword",
    "person",
    "collection",
)

EXACT = 1.0
DEMONYM = 0.9
FAMILY = 0.85
PARTIAL = 0.55

_NEGATIONS = frozenset({"not", "no", "non", "anti", "without"})
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Family:
    """A word whose honest resolution is a union rather than one id."""

    label: str
    keywords: tuple[str, ...]
    genres: tuple[str, ...] = ()


# TMDB has no Musical genre for films, and genre Music alone deletes Amadeus and Whiplash.
FAMILIES: Mapping[str, Family] = {
    "musical": Family(
        "musicals",
        ("musical", "stage musical", "song and dance", "broadway", "musical number"),
        ("music",),
    ),
    "anime": Family("anime", ("anime", "japanese animation"), ()),
}

# Country adjectives, which no TMDB table carries. Successor states sit beside the current one.
DEMONYMS: Mapping[str, tuple[str, ...]] = {
    "russian": ("RU", "SU"),
    "soviet": ("SU", "RU"),
    "american": ("US",),
    "british": ("GB",),
    "english": ("GB",),
    "french": ("FR",),
    "italian": ("IT",),
    "german": ("DE", "DD"),
    "japanese": ("JP",),
    "korean": ("KR", "KP"),
    "chinese": ("CN", "HK", "TW"),
    "taiwanese": ("TW",),
    "indian": ("IN",),
    "iranian": ("IR",),
    "swedish": ("SE",),
    "danish": ("DK",),
    "polish": ("PL",),
    "hungarian": ("HU",),
    "spanish": ("ES",),
    "mexican": ("MX",),
    "brazilian": ("BR",),
    "argentine": ("AR",),
}


@dataclass(frozen=True, slots=True)
class VocabMatch:
    """One resolved corpus set, with what it would actually touch."""

    kind: Kind
    ids: tuple[str, ...]
    label: str
    affected_films: int
    confidence: float
    alternatives: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _Entry:
    """One vocabulary row as the corpus knows it."""

    entity_id: str
    label: str
    films: int


_COUNTS: Mapping[str, str] = {
    "genre": (
        "select cast(fg.genre_id as text) as id, g.name as label, "
        "count(distinct fg.tmdb_id) as films from film_genres fg "
        "join genres g on g.genre_id = fg.genre_id "
        "join corpus_members m on m.tmdb_id = fg.tmdb_id and m.eligible = 1 group by 1, 2"
    ),
    "keyword": (
        "select cast(fk.keyword_id as text) as id, k.name as label, "
        "count(distinct fk.tmdb_id) as films from film_keywords fk "
        "join keywords k on k.keyword_id = fk.keyword_id "
        "join corpus_members m on m.tmdb_id = fk.tmdb_id and m.eligible = 1 group by 1, 2"
    ),
    "language": (
        "select f.original_language as id, coalesce(l.name, f.original_language) as label, "
        "count(*) as films from films f "
        "join corpus_members m on m.tmdb_id = f.tmdb_id and m.eligible = 1 "
        "left join languages l on l.iso_639_1 = f.original_language "
        "where f.original_language is not null and f.original_language <> '' group by 1, 2"
    ),
    "country": (
        "select fc.iso_3166_1 as id, co.name as label, count(distinct fc.tmdb_id) as films "
        "from film_countries fc join countries co on co.iso_3166_1 = fc.iso_3166_1 "
        "join corpus_members m on m.tmdb_id = fc.tmdb_id and m.eligible = 1 group by 1, 2"
    ),
    "person": (
        "select cast(c.person_id as text) as id, p.name as label, "
        "count(distinct c.tmdb_id) as films from credits c "
        "join people p on p.person_id = c.person_id "
        "join corpus_members m on m.tmdb_id = c.tmdb_id and m.eligible = 1 "
        "where c.job = 'Director' or (c.credit_kind = 'cast' and c.ord < 10) group by 1, 2"
    ),
    "collection": (
        "select cast(f.collection_id as text) as id, f.collection_name as label, "
        "count(*) as films from films f "
        "join corpus_members m on m.tmdb_id = f.tmdb_id and m.eligible = 1 "
        "where f.collection_id is not null group by 1, 2"
    ),
}

_AFFECTED: Mapping[str, str] = {
    "genre": (
        "select count(distinct fg.tmdb_id) from film_genres fg "
        "join corpus_members m on m.tmdb_id = fg.tmdb_id and m.eligible = 1 "
        "where fg.genre_id in (select value from json_each(?))"
    ),
    "keyword": (
        "select count(distinct fk.tmdb_id) from film_keywords fk "
        "join corpus_members m on m.tmdb_id = fk.tmdb_id and m.eligible = 1 "
        "where fk.keyword_id in (select value from json_each(?))"
    ),
    "language": (
        "select count(*) from films f "
        "join corpus_members m on m.tmdb_id = f.tmdb_id and m.eligible = 1 "
        "where f.original_language in (select value from json_each(?))"
    ),
    "country": (
        "select count(distinct fc.tmdb_id) from film_countries fc "
        "join corpus_members m on m.tmdb_id = fc.tmdb_id and m.eligible = 1 "
        "where fc.iso_3166_1 in (select value from json_each(?))"
    ),
    "person": (
        "select count(distinct c.tmdb_id) from credits c "
        "join corpus_members m on m.tmdb_id = c.tmdb_id and m.eligible = 1 "
        "where c.person_id in (select value from json_each(?))"
    ),
    "collection": (
        "select count(*) from films f "
        "join corpus_members m on m.tmdb_id = f.tmdb_id and m.eligible = 1 "
        "where f.collection_id in (select value from json_each(?))"
    ),
}

_NUMERIC = frozenset({"genre", "keyword", "person", "collection"})


def normalise(text: str) -> str:
    """Lowercase, punctuation free, and stripped of a leading negation."""
    words = _PUNCTUATION.sub(" ", text).casefold().split()
    while words and words[0] in _NEGATIONS:
        words = words[1:]
    return " ".join(words)


class Vocabulary:
    """The corpus's own names, and what each of them would touch."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._entries: dict[str, tuple[_Entry, ...]] = {}

    def entries(self, kind: str) -> tuple[_Entry, ...]:
        """Every id of one kind that at least one eligible corpus film carries."""
        cached = self._entries.get(kind)
        if cached is None:
            rows = self.conn.execute(_COUNTS[kind])
            cached = tuple(
                _Entry(str(r["id"]), str(r["label"] or r["id"]), int(r["films"])) for r in rows
            )
            self._entries[kind] = cached
        return cached

    def counts(self, kind: str) -> Mapping[str, int]:
        """Corpus film count per id."""
        return {e.entity_id: e.films for e in self.entries(kind)}

    def affected(self, kind: str, ids: Sequence[str]) -> int:
        """Distinct corpus films a resolved set touches, which is not the sum of its parts."""
        if not ids:
            return 0
        values = [int(i) for i in ids] if kind in _NUMERIC else list(ids)
        row = self.conn.execute(_AFFECTED[kind], (orjson.dumps(values).decode(),)).fetchone()
        return int(row[0])

    def suggest(self, kind: str, raw: str, n: int = 3) -> list[tuple[str, str, int]]:
        """Closest names of one kind, for a did you mean line."""
        needle = normalise(raw)
        by_label = {e.label.casefold(): e for e in self.entries(kind)}
        close = difflib.get_close_matches(needle, list(by_label), n=n, cutoff=0.6)
        found = [by_label[label] for label in close]
        return [(e.entity_id, e.label, e.films) for e in found]

    def resolve(self, text: str, *, prefer: Sequence[str] = ()) -> list[VocabMatch]:
        """Every corpus set this phrase could mean, best first."""
        needle = normalise(text)
        if not needle:
            return []
        family = FAMILIES.get(needle) or FAMILIES.get(_singular(needle))
        if family is not None:
            return self._family(family)
        matches = [m for kind in RESOLUTION_ORDER for m in self._of_kind(kind, needle)]
        rank = _ranker(prefer)
        matches.sort(key=lambda m: (rank(m.kind), -m.confidence, -m.affected_films))
        return matches

    def _family(self, family: Family) -> list[VocabMatch]:
        out: list[VocabMatch] = []
        pairs: tuple[tuple[Kind, tuple[str, ...]], ...] = (
            ("keyword", family.keywords),
            ("genre", family.genres),
        )
        for kind, names in pairs:
            wanted = {n.casefold() for n in names}
            ids = tuple(
                sorted(e.entity_id for e in self.entries(kind) if e.label.casefold() in wanted)
            )
            if ids:
                out.append(VocabMatch(kind, ids, family.label, self.affected(kind, ids), FAMILY))
        return out

    def _of_kind(self, kind: Kind, needle: str) -> list[VocabMatch]:
        entries = self.entries(kind)
        exact = [e for e in entries if e.label.casefold() == needle]
        if kind == "country" and not exact:
            exact = self._demonym(needle, entries)
            if exact:
                return [self._match(kind, exact, needle, DEMONYM, entries)]
        if exact:
            return [self._match(kind, exact, needle, EXACT, entries)]
        partial = [e for e in entries if needle in e.label.casefold()]
        if not partial:
            return []
        partial.sort(key=lambda e: (-e.films, e.entity_id))
        return [self._match(kind, partial[:1], needle, PARTIAL, entries)]

    @staticmethod
    def _demonym(needle: str, entries: Sequence[_Entry]) -> list[_Entry]:
        codes = DEMONYMS.get(needle) or DEMONYMS.get(_singular(needle))
        if not codes:
            return []
        known = {e.entity_id: e for e in entries}
        return [known[code] for code in codes if code in known]

    def _match(
        self,
        kind: Kind,
        found: Sequence[_Entry],
        needle: str,
        confidence: float,
        entries: Sequence[_Entry],
    ) -> VocabMatch:
        ids = tuple(e.entity_id for e in found)
        label = found[0].label if len(found) == 1 else needle
        return VocabMatch(
            kind=kind,
            ids=ids,
            label=label,
            affected_films=self.affected(kind, ids),
            confidence=confidence,
            alternatives=_alternatives(needle, entries, ids),
        )


def _alternatives(
    needle: str, entries: Sequence[_Entry], taken: Iterable[str]
) -> tuple[tuple[str, str, int], ...]:
    skip = set(taken)
    near = [e for e in entries if e.entity_id not in skip and needle in e.label.casefold()]
    near.sort(key=lambda e: (-e.films, e.entity_id))
    return tuple((e.entity_id, e.label, e.films) for e in near[:3])


def _ranker(prefer: Sequence[str]) -> Callable[[str], int]:
    order = [*prefer, *(k for k in RESOLUTION_ORDER if k not in prefer)]

    def rank(kind: str) -> int:
        return order.index(kind)

    return rank


def _singular(needle: str) -> str:
    return needle[:-1] if needle.endswith("s") else needle
