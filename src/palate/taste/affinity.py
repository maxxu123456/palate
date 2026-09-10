"""Shrunk empirical Bayes per entity, because one film by a director is not a pattern."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import orjson

from palate.taste.signals import PreferenceSignal

type Kind = Literal[
    "director",
    "writer",
    "actor",
    "keyword",
    "genre",
    "language",
    "country",
    "decade",
    "runtime_bucket",
    "collection",
]

KAPPA: dict[str, float] = {
    "director": 2.0,
    "writer": 3.0,
    "actor": 4.0,
    "keyword": 6.0,
    "genre": 8.0,
    "language": 4.0,
    "country": 5.0,
    "decade": 6.0,
    "runtime_bucket": 6.0,
    "collection": 2.0,
}

# A single keyword or bit-part credit is noise, and those two kinds are most of the rows.
MIN_N: dict[str, int] = {"keyword": 2, "actor": 2}

# Only the top billing is a reason anyone chose a film.
BILLED = 10

MAX_SUPPORT = 12


@dataclass(frozen=True, slots=True)
class EntityAffinity:
    """One entity, how often it was watched and how it was rated, both shrunk."""

    kind: str
    entity_id: str
    name: str
    n: int
    raw_sum: float
    affinity: float
    exposure_logodds: float
    support_films: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Facet:
    """One way of slicing a film into entities, as SQL fragments."""

    kind: str
    join: str
    entity: str
    name: str
    where: str = "1 = 1"


FACETS: tuple[_Facet, ...] = (
    # Every job = 'Director' credit. Co-direction is common and a first-credited id drops one.
    _Facet(
        "director",
        "join credits c on c.tmdb_id = f.tmdb_id and c.job = 'Director' "
        "join people p on p.person_id = c.person_id",
        "cast(c.person_id as text)",
        "p.name",
    ),
    _Facet(
        "writer",
        "join credits c on c.tmdb_id = f.tmdb_id and c.department = 'Writing' "
        "join people p on p.person_id = c.person_id",
        "cast(c.person_id as text)",
        "p.name",
    ),
    _Facet(
        "actor",
        f"join credits c on c.tmdb_id = f.tmdb_id and c.credit_kind = 'cast' and c.ord < {BILLED} "
        "join people p on p.person_id = c.person_id",
        "cast(c.person_id as text)",
        "p.name",
    ),
    _Facet(
        "keyword",
        "join film_keywords fk on fk.tmdb_id = f.tmdb_id "
        "join keywords k on k.keyword_id = fk.keyword_id",
        "cast(fk.keyword_id as text)",
        "k.name",
    ),
    _Facet(
        "genre",
        "join film_genres fg on fg.tmdb_id = f.tmdb_id join genres g on g.genre_id = fg.genre_id",
        "cast(fg.genre_id as text)",
        "g.name",
    ),
    _Facet(
        "language",
        "left join languages l on l.iso_639_1 = f.original_language",
        "f.original_language",
        "coalesce(l.name, f.original_language)",
        "f.original_language is not null and f.original_language <> ''",
    ),
    _Facet(
        "country",
        "join film_countries fc on fc.tmdb_id = f.tmdb_id "
        "join countries co on co.iso_3166_1 = fc.iso_3166_1",
        "fc.iso_3166_1",
        "co.name",
    ),
    _Facet("decade", "", "cast(f.decade as text)", "f.decade || 's'", "f.year is not null"),
    _Facet(
        "runtime_bucket",
        "",
        "cast(f.runtime_bucket as text)",
        "case f.runtime_bucket when 0 then 'under 85 min' when 1 then '85 to 105 min' "
        "when 2 then '105 to 130 min' when 3 then '130 to 160 min' else 'over 160 min' end",
        "f.runtime is not null",
    ),
    _Facet(
        "collection",
        "",
        "cast(f.collection_id as text)",
        "f.collection_name",
        "f.collection_id is not null",
    ),
)


def build_affinities(
    conn: sqlite3.Connection,
    signals: Sequence[PreferenceSignal],
    *,
    kinds: Sequence[str] | None = None,
) -> dict[str, tuple[EntityAffinity, ...]]:
    """Rating affinity and watch exposure per entity, both shrunk toward the corpus."""
    wanted = set(kinds) if kinds is not None else {f.kind for f in FACETS}
    by_film = {s.tmdb_id: s for s in signals}
    ids = orjson.dumps(list(by_film)).decode()
    corpus_total = _corpus_total(conn)
    out: dict[str, tuple[EntityAffinity, ...]] = {}
    for facet in FACETS:
        if facet.kind not in wanted:
            continue
        base = _corpus_counts(conn, facet)
        out[facet.kind] = _facet_affinities(conn, facet, ids, by_film, base, corpus_total)
    return out


def _facet_affinities(
    conn: sqlite3.Connection,
    facet: _Facet,
    ids: str,
    by_film: Mapping[int, PreferenceSignal],
    base: Mapping[str, int],
    corpus_total: int,
) -> tuple[EntityAffinity, ...]:
    grouped: dict[str, tuple[str, list[int]]] = {}
    for row in conn.execute(_rated_sql(facet), (ids,)):
        entity = str(row["entity_id"])
        name, films = grouped.setdefault(entity, (str(row["name"] or entity), []))
        films.append(int(row["tmdb_id"]))
    kappa = KAPPA[facet.kind]
    floor = MIN_N.get(facet.kind, 1)
    watched = len(by_film)
    out: list[EntityAffinity] = []
    for entity, (name, films) in grouped.items():
        if len(films) < floor:
            continue
        raw_sum = sum(by_film[f].z for f in films)
        ordered = sorted(films, key=lambda f: abs(by_film[f].z), reverse=True)
        out.append(
            EntityAffinity(
                kind=facet.kind,
                entity_id=entity,
                name=name,
                n=len(films),
                raw_sum=raw_sum,
                affinity=raw_sum / (len(films) + kappa),
                exposure_logodds=_exposure(
                    len(films), watched, base.get(entity, 0), corpus_total, kappa
                ),
                support_films=tuple(ordered[:MAX_SUPPORT]),
            )
        )
    out.sort(key=lambda a: a.affinity, reverse=True)
    return tuple(out)


def _exposure(n: int, watched: int, corpus_n: int, corpus_total: int, kappa: float) -> float:
    """Log odds of watching this entity against the corpus base rate, shrunk toward it."""
    if watched == 0 or corpus_total == 0:
        return 0.0
    # Half a film rather than zero, so an entity the corpus never lists is not a log of nothing.
    base = min(max(corpus_n if corpus_n else 0.5, 0.5) / corpus_total, 1.0 - 1e-6)
    mine = min(max((n + kappa * base) / (watched + kappa), 1e-6), 1.0 - 1e-6)
    return math.log(mine / (1.0 - mine)) - math.log(base / (1.0 - base))


def _rated_sql(facet: _Facet) -> str:
    return (
        f"select f.tmdb_id, {facet.entity} as entity_id, {facet.name} as name "
        f"from films f {facet.join} "
        f"where {facet.where} and f.tmdb_id in (select value from json_each(?))"
    )


def _corpus_sql(facet: _Facet) -> str:
    return (
        f"select {facet.entity} as entity_id, count(distinct f.tmdb_id) as n "
        "from films f join corpus_members m on m.tmdb_id = f.tmdb_id and m.eligible = 1 "
        f"{facet.join} where {facet.where} group by 1"
    )


def _corpus_counts(conn: sqlite3.Connection, facet: _Facet) -> dict[str, int]:
    return {str(r["entity_id"]): int(r["n"]) for r in conn.execute(_corpus_sql(facet))}


def _corpus_total(conn: sqlite3.Connection) -> int:
    row = conn.execute("select count(*) from corpus_members where eligible = 1").fetchone()
    return int(row[0])
