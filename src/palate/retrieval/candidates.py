"""Hard filters, the six candidate channels, and the pool their ranks agree on."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import orjson

from palate.index import fts
from palate.index.vecstore import MetadataFilter, PrefilterPlan, VecStore, choose_prefilter
from palate.taste.memory import PreferenceFilter
from palate.taste.profile import TasteProfile

# Recall only, so one weight per credit kind rather than the scoring model's betas.
PEOPLE_KIND_WEIGHT = {"director": 1.0, "writer": 0.6, "actor": 0.5}

MAX_PEOPLE = 80
MAX_KEYWORDS = 120

_CREDIT_PREDICATE = {
    "director": "job = 'Director'",
    "writer": "department = 'Writing'",
    "actor": "credit_kind = 'cast' and ord < 10",
}

# A film counts as watched once it carries a rating or a date. Watchlist rows do not.
_WATCHED = (
    "select tmdb_id from user_films where rating_half is not null "
    "or watched_date is not null or logged_date is not null"
)


class Channel(StrEnum):
    """The six ways a film can reach the pool."""

    DENSE_MODE = "dense_mode"
    DENSE_QUERY = "dense_query"
    BM25 = "bm25"
    PEOPLE = "people"
    KEYWORD = "keyword"
    POPULAR = "popular"


@dataclass(frozen=True, slots=True)
class ChannelHit:
    """One film as one channel found it."""

    tmdb_id: int
    channel: Channel
    rank: int
    raw_score: float
    source_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelBudget:
    """How deep each channel is allowed to go before the pool is truncated."""

    dense_per_mode: int = 200
    dense_query: int = 400
    bm25: int = 300
    people: int = 300
    keyword: int = 300
    popular: int = 200
    pool_max: int = 1200


@dataclass(frozen=True, slots=True)
class HardFilters:
    """Everything that removes a film outright, before any score exists."""

    year_min: int | None = None
    year_max: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None
    include_languages: frozenset[str] = frozenset()
    exclude_languages: frozenset[str] = frozenset()
    include_genres: frozenset[int] = frozenset()
    exclude_genres: frozenset[int] = frozenset()
    include_keywords: frozenset[int] = frozenset()
    exclude_keywords: frozenset[int] = frozenset()
    include_countries: frozenset[str] = frozenset()
    exclude_countries: frozenset[str] = frozenset()
    include_people: frozenset[int] = frozenset()
    exclude_people: frozenset[int] = frozenset()
    include_decades: frozenset[int] = frozenset()
    exclude_decades: frozenset[int] = frozenset()
    min_vote_count: int = 0
    exclude_ids: frozenset[int] = frozenset()
    exclude_watched: bool = True
    unsatisfiable: bool = False

    def merge(self, other: HardFilters) -> HardFilters:
        """Union the excludes, intersect the includes, tighten the ranges. Never loosens."""
        if self.unsatisfiable or other.unsatisfiable:
            return IMPOSSIBLE
        changes: dict[str, Any] = {}
        for name in _INCLUDE_FIELDS:
            kept = _intersect(getattr(self, name), getattr(other, name))
            if kept is None:
                return IMPOSSIBLE
            changes[name] = kept
        for name in _EXCLUDE_FIELDS:
            changes[name] = getattr(self, name) | getattr(other, name)
        changes["year_min"] = _tighter(self.year_min, other.year_min, high=True)
        changes["runtime_min"] = _tighter(self.runtime_min, other.runtime_min, high=True)
        changes["year_max"] = _tighter(self.year_max, other.year_max, high=False)
        changes["runtime_max"] = _tighter(self.runtime_max, other.runtime_max, high=False)
        changes["min_vote_count"] = max(self.min_vote_count, other.min_vote_count)
        changes["exclude_watched"] = self.exclude_watched or other.exclude_watched
        return replace(self, **changes)

    def most_restrictive(self, removed: Mapping[str, int]) -> str | None:
        """The clause that removed the most candidates, for the empty result hint."""
        live = {name: n for name, n in removed.items() if n > 0}
        if not live:
            return None
        return min(live, key=lambda name: (-live[name], name))


IMPOSSIBLE = HardFilters(unsatisfiable=True)
NO_FILTERS = HardFilters()
DEFAULT_BUDGET = ChannelBudget()

_INCLUDE_FIELDS = (
    "include_languages",
    "include_genres",
    "include_keywords",
    "include_countries",
    "include_people",
    "include_decades",
)
_EXCLUDE_FIELDS = (
    "exclude_languages",
    "exclude_genres",
    "exclude_keywords",
    "exclude_countries",
    "exclude_people",
    "exclude_decades",
    "exclude_ids",
)


def _intersect(a: frozenset[Any], b: frozenset[Any]) -> frozenset[Any] | None:
    """An empty set means unconstrained, so two disjoint includes are impossible, not free."""
    if not a:
        return b
    if not b:
        return a
    both = a & b
    return both or None


def _tighter(a: int | None, b: int | None, *, high: bool) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b) if high else min(a, b)


def from_preferences(conn: sqlite3.Connection, stated: PreferenceFilter) -> HardFilters:
    """Compile the hard half of stated memory into clauses the SQL can run."""
    changes: dict[str, Any] = {}
    if stated.max_runtime is not None:
        changes["runtime_max"] = stated.max_runtime
    for side, table in (("exclude", stated.exclude), ("require", stated.require)):
        for kind, targets in table.items():
            field, values = _preference_clause(conn, side, kind, targets)
            if field is None:
                continue
            changes[field] = frozenset(values) | changes.get(field, frozenset())
    return HardFilters(**changes)


# Every preference kind that compiles to a clause, and which side of it can carry one.
_PREFERENCE_FIELD = {
    ("exclude", "genre"): "exclude_genres",
    ("exclude", "keyword"): "exclude_keywords",
    ("exclude", "language"): "exclude_languages",
    ("exclude", "country"): "exclude_countries",
    ("exclude", "decade"): "exclude_decades",
    ("exclude", "film"): "exclude_ids",
    ("exclude", "director"): "exclude_people",
    ("exclude", "actor"): "exclude_people",
    ("exclude", "writer"): "exclude_people",
    ("exclude", "collection"): "exclude_ids",
    ("require", "genre"): "include_genres",
    ("require", "keyword"): "include_keywords",
    ("require", "language"): "include_languages",
    ("require", "country"): "include_countries",
    ("require", "decade"): "include_decades",
    ("require", "director"): "include_people",
    ("require", "actor"): "include_people",
    ("require", "writer"): "include_people",
}

_TEXT_FIELDS = frozenset(
    {"exclude_languages", "include_languages", "exclude_countries", "include_countries"}
)


def _preference_clause(
    conn: sqlite3.Connection, side: str, kind: str, targets: frozenset[str]
) -> tuple[str | None, Sequence[Any]]:
    field = _PREFERENCE_FIELD.get((side, kind))
    if field is None:
        return None, ()
    if kind == "collection":
        return field, _collection_films(conn, targets)
    return field, list(targets) if field in _TEXT_FIELDS else [int(t) for t in targets]


def _collection_films(conn: sqlite3.Connection, targets: frozenset[str]) -> list[int]:
    rows = conn.execute(
        "select tmdb_id from films where collection_id in (select value from json_each(?))",
        (orjson.dumps([int(t) for t in targets]).decode(),),
    )
    return [int(r["tmdb_id"]) for r in rows]


@dataclass(frozen=True, slots=True)
class AllowSet:
    """Which films survived the hard filters, and what each clause cost."""

    ids: frozenset[int]
    n_corpus: int
    removed_by_clause: Mapping[str, int]
    excluded_watched: int

    @property
    def considered(self) -> int:
        return self.n_corpus


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """The films worth scoring, and the record of how they got here."""

    ids: tuple[int, ...]
    hits: Mapping[int, tuple[ChannelHit, ...]]
    rrf: Mapping[int, float]
    per_channel_counts: Mapping[Channel, int]
    per_channel_unique: Mapping[Channel, int]
    prefilter_path: str
    overfetch_factor: float
    truncated: bool
    elapsed_ms: float
    allow: AllowSet


class CandidateStore:
    """The corpus, the documents and the vectors, as the channels read them."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        vecs: VecStore,
        id_cap: int = 5000,
        overfetch_floor: float = 2.0,
        overfetch_cap: float = 6.0,
    ) -> None:
        self.conn = conn
        self.vecs = vecs
        self.id_cap = id_cap
        self.overfetch_floor = overfetch_floor
        self.overfetch_cap = overfetch_cap

    def eligible(self) -> frozenset[int]:
        """Every film the recommender is allowed to name."""
        rows = self.conn.execute("select tmdb_id from corpus_members where eligible = 1")
        return frozenset(int(r["tmdb_id"]) for r in rows)

    def watched(self) -> frozenset[int]:
        """Films the history says were seen, which never reach a result."""
        return frozenset(int(r["tmdb_id"]) for r in self.conn.execute(_WATCHED))

    def allow(self, filters: HardFilters) -> AllowSet:
        """Apply every hard clause, keeping what each one removed on its own."""
        base = self.eligible()
        if filters.unsatisfiable:
            return AllowSet(frozenset(), len(base), {"unsatisfiable": len(base)}, 0)
        kept = base
        removed: dict[str, int] = {}
        watched = self.watched() if filters.exclude_watched else frozenset()
        if filters.exclude_watched:
            removed["watched"] = len(base & watched)
            kept -= watched
        for name, passing in self._clause_sets(filters):
            removed[name] = len(base - passing)
            kept &= passing
        return AllowSet(kept, len(base), removed, len(base & watched))

    def plan(self, n_allowed: int) -> PrefilterPlan:
        """Which pre-filter mechanism this allow set earns."""
        return choose_prefilter(
            n_allowed,
            self.vecs.count(),
            id_cap=self.id_cap,
            overfetch_floor=self.overfetch_floor,
            overfetch_cap=self.overfetch_cap,
        )

    def _clause_sets(self, filters: HardFilters) -> Iterator[tuple[str, frozenset[int]]]:
        for name, sql, params in _clauses(filters):
            rows = self.conn.execute(sql, params)
            yield name, frozenset(int(r[0]) for r in rows)


def _json(values: Iterable[Any]) -> str:
    return orjson.dumps(sorted(values)).decode()


def _clauses(filters: HardFilters) -> list[tuple[str, str, tuple[Any, ...]]]:
    """Each active clause as the set of films that pass it, on its own."""
    out: list[tuple[str, str, tuple[Any, ...]]] = []
    scalars = (
        ("year_min", "year >= ?", filters.year_min),
        ("year_max", "year <= ?", filters.year_max),
        ("runtime_min", "runtime >= ?", filters.runtime_min),
        ("runtime_max", "runtime <= ?", filters.runtime_max),
        ("min_vote_count", "vote_count >= ?", filters.min_vote_count or None),
    )
    for name, predicate, value in scalars:
        if value is not None:
            out.append((name, f"select tmdb_id from films where {predicate}", (value,)))
    if filters.exclude_ids:
        out.append(
            (
                "exclude_ids",
                "select tmdb_id from films where tmdb_id not in (select value from json_each(?))",
                (_json(filters.exclude_ids),),
            )
        )
    for name, column, values in (
        ("include_languages", "original_language", filters.include_languages),
        ("include_decades", "decade", filters.include_decades),
    ):
        if values:
            out.append(
                (
                    name,
                    f"select tmdb_id from films where {column} in (select value from json_each(?))",
                    (_json(values),),
                )
            )
    for name, column, values in (
        ("exclude_languages", "coalesce(original_language, '')", filters.exclude_languages),
        ("exclude_decades", "coalesce(decade, -1)", filters.exclude_decades),
    ):
        if values:
            out.append(
                (
                    name,
                    f"select tmdb_id from films "
                    f"where {column} not in (select value from json_each(?))",
                    (_json(values),),
                )
            )
    for name, source, column, values in (
        ("genres", "film_genres", "genre_id", filters.include_genres),
        ("keywords", "film_keywords", "keyword_id", filters.include_keywords),
        ("countries", "film_countries", "iso_3166_1", filters.include_countries),
        ("people", "credits", "person_id", filters.include_people),
    ):
        if values:
            out.append(
                (
                    f"include_{name}",
                    f"select tmdb_id from {source} "
                    f"where {column} in (select value from json_each(?))",
                    (_json(values),),
                )
            )
    for name, source, column, values in (
        ("genres", "film_genres", "genre_id", filters.exclude_genres),
        ("keywords", "film_keywords", "keyword_id", filters.exclude_keywords),
        ("countries", "film_countries", "iso_3166_1", filters.exclude_countries),
        ("people", "credits", "person_id", filters.exclude_people),
    ):
        if values:
            out.append(
                (
                    f"exclude_{name}",
                    f"select tmdb_id from films where tmdb_id not in (select tmdb_id from {source} "
                    f"where {column} in (select value from json_each(?)))",
                    (_json(values),),
                )
            )
    return out


@dataclass(frozen=True, slots=True)
class _Search:
    """The pre-filter mechanism, resolved once and reused by every dense call."""

    store: CandidateStore
    allowed: frozenset[int]
    plan: PrefilterPlan
    metadata: MetadataFilter
    allow_list: tuple[int, ...] | None
    exclude_list: tuple[int, ...] | None

    def knn(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        """Nearest allowed neighbours, over-fetching only when the id lists do not fit."""
        wanted = max(1, int(k * self.plan.overfetch))
        hits = self.store.vecs.knn(
            vector,
            k=wanted,
            allow=self.allow_list,
            exclude=self.exclude_list,
            where=self.metadata,
        )
        kept = [(h.film_id, h.similarity) for h in hits if h.film_id in self.allowed]
        return kept[:k]


def _metadata_filter(filters: HardFilters) -> MetadataFilter:
    """Push the single valued predicates into vec0, where they are a real pre-filter."""
    clauses: list[tuple[str, str, Any]] = [("in_corpus", "eq", 1)]
    if filters.exclude_watched:
        clauses.append(("is_watched", "eq", 0))
    for column, operator, value in (
        ("year", "gte", filters.year_min),
        ("year", "lte", filters.year_max),
        ("runtime", "gte", filters.runtime_min),
        ("runtime", "lte", filters.runtime_max),
        ("vote_count", "gte", filters.min_vote_count or None),
    ):
        if value is not None:
            clauses.append((column, operator, value))
    if len(filters.include_languages) == 1:
        clauses.append(("original_language", "eq", next(iter(filters.include_languages))))
    return MetadataFilter(tuple(clauses))


def _plan_search(store: CandidateStore, allowed: frozenset[int], filters: HardFilters) -> _Search:
    plan = store.plan(len(allowed))
    allow_list: tuple[int, ...] | None = None
    exclude_list: tuple[int, ...] | None = None
    if plan.path == "allow_json":
        allow_list = tuple(sorted(allowed))
    elif plan.path == "exclude_json":
        exclude_list = tuple(sorted(store.vecs.ids() - allowed))
    return _Search(store, allowed, plan, _metadata_filter(filters), allow_list, exclude_list)


def _ranked(
    scored: Mapping[int, tuple[float, str | None]], channel: Channel, limit: int
) -> list[ChannelHit]:
    """One channel's hits, best first, ties on tmdb_id so the pool never drifts."""
    order = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))
    return [
        ChannelHit(tmdb_id, channel, rank, score, source)
        for rank, (tmdb_id, (score, source)) in enumerate(order[:limit], start=1)
    ]


def _dense_mode_hits(
    search: _Search, profile: TasteProfile, budget: ChannelBudget
) -> list[ChannelHit]:
    hits: list[ChannelHit] = []
    for mode in profile.modes:
        found = search.knn(mode.centroid.tolist(), budget.dense_per_mode)
        scored = {tmdb_id: (similarity, f"mode:{mode.mode_id}") for tmdb_id, similarity in found}
        hits.extend(_ranked(scored, Channel.DENSE_MODE, budget.dense_per_mode))
    return hits


def _dense_query_hits(
    search: _Search, query_embedding: Sequence[float], budget: ChannelBudget
) -> list[ChannelHit]:
    found = search.knn(query_embedding, budget.dense_query)
    return _ranked({i: (s, None) for i, s in found}, Channel.DENSE_QUERY, budget.dense_query)


def _bm25_hits(
    store: CandidateStore, text: str, allowed: frozenset[int], budget: ChannelBudget
) -> list[ChannelHit]:
    found = fts.search(store.conn, text, limit=budget.bm25 * 3)
    scored = {h.tmdb_id: (h.score, None) for h in found if h.tmdb_id in allowed}
    return _ranked(scored, Channel.BM25, budget.bm25)


def _entity_hits(
    store: CandidateStore,
    wanted: Mapping[str, Mapping[int, float]],
    allowed: frozenset[int],
    channel: Channel,
    limit: int,
) -> list[ChannelHit]:
    best: dict[int, tuple[float, str | None]] = {}
    for kind, weights in wanted.items():
        if not weights:
            continue
        for tmdb_id, entity in _entity_films(store.conn, kind, weights):
            if tmdb_id not in allowed:
                continue
            score = weights[entity]
            if score > best.get(tmdb_id, (float("-inf"), None))[0]:
                best[tmdb_id] = (score, f"{kind}:{entity}")
    return _ranked(best, channel, limit)


def _entity_films(
    conn: sqlite3.Connection, kind: str, weights: Mapping[int, float]
) -> Iterator[tuple[int, int]]:
    ids = _json(list(weights))
    if kind == "keyword":
        sql = (
            "select tmdb_id, keyword_id as entity from film_keywords "
            "where keyword_id in (select value from json_each(?))"
        )
    else:
        sql = (
            "select tmdb_id, person_id as entity from credits "
            f"where {_CREDIT_PREDICATE[kind]} and person_id in (select value from json_each(?))"
        )
    for row in conn.execute(sql, (ids,)):
        yield int(row["tmdb_id"]), int(row["entity"])


def _liked_entities(profile: TasteProfile, kind: str, cap: int, weight: float) -> dict[int, float]:
    rows = [a for a in profile.affinities.get(kind, ()) if a.affinity > 0.0]
    return {int(a.entity_id): weight * a.affinity for a in rows[:cap]}


_POPULAR = (
    "select f.tmdb_id, coalesce(f.popularity_at_crawl, f.popularity, 0.0) as pop "
    "from films f join corpus_members m on m.tmdb_id = f.tmdb_id and m.eligible = 1 "
    "order by pop desc, f.tmdb_id"
)


def _popular_hits(
    store: CandidateStore, allowed: frozenset[int], budget: ChannelBudget
) -> list[ChannelHit]:
    out: list[ChannelHit] = []
    for row in store.conn.execute(_POPULAR):
        tmdb_id = int(row["tmdb_id"])
        if tmdb_id not in allowed:
            continue
        out.append(ChannelHit(tmdb_id, Channel.POPULAR, len(out) + 1, float(row["pop"]), None))
        if len(out) >= budget.popular:
            break
    return out


def reciprocal_rank_fusion(
    hits: Sequence[ChannelHit], *, k: int = 60, weights: Mapping[Channel, float] | None = None
) -> dict[int, float]:
    """sum over channels of w_ch / (k + rank)."""
    out: dict[int, float] = {}
    for hit in hits:
        weight = 1.0 if weights is None else weights.get(hit.channel, 1.0)
        out[hit.tmdb_id] = out.get(hit.tmdb_id, 0.0) + weight / (k + hit.rank)
    return out


def generate_candidates(
    store: CandidateStore,
    profile: TasteProfile,
    *,
    query_embedding: Sequence[float] | None = None,
    query_text: str | None = None,
    filters: HardFilters = NO_FILTERS,
    budget: ChannelBudget = DEFAULT_BUDGET,
    channels: Sequence[Channel] | None = None,
    weights: Mapping[Channel, float] | None = None,
    rrf_k: int = 60,
) -> CandidatePool:
    """Run every enabled channel over the allowed set and keep what their ranks agree on."""
    started = time.perf_counter()
    wanted = frozenset(channels) if channels is not None else frozenset(Channel)
    allow = store.allow(filters)
    search = _plan_search(store, allow.ids, filters)
    hits: list[ChannelHit] = []
    if Channel.DENSE_MODE in wanted and profile.modes:
        hits.extend(_dense_mode_hits(search, profile, budget))
    if Channel.DENSE_QUERY in wanted and query_embedding is not None:
        hits.extend(_dense_query_hits(search, query_embedding, budget))
    if Channel.BM25 in wanted and query_text:
        hits.extend(_bm25_hits(store, query_text, allow.ids, budget))
    if Channel.PEOPLE in wanted:
        people = {
            kind: _liked_entities(profile, kind, MAX_PEOPLE, weight)
            for kind, weight in PEOPLE_KIND_WEIGHT.items()
        }
        hits.extend(_entity_hits(store, people, allow.ids, Channel.PEOPLE, budget.people))
    if Channel.KEYWORD in wanted:
        keywords = {"keyword": _liked_entities(profile, "keyword", MAX_KEYWORDS, 1.0)}
        hits.extend(_entity_hits(store, keywords, allow.ids, Channel.KEYWORD, budget.keyword))
    if Channel.POPULAR in wanted:
        hits.extend(_popular_hits(store, allow.ids, budget))
    return _assemble(hits, allow, search, budget, rrf_k, weights, started)


def _assemble(
    hits: Sequence[ChannelHit],
    allow: AllowSet,
    search: _Search,
    budget: ChannelBudget,
    rrf_k: int,
    weights: Mapping[Channel, float] | None,
    started: float,
) -> CandidatePool:
    by_film: dict[int, list[ChannelHit]] = {}
    counts: dict[Channel, int] = {}
    for hit in hits:
        by_film.setdefault(hit.tmdb_id, []).append(hit)
        counts[hit.channel] = counts.get(hit.channel, 0) + 1
    rrf = reciprocal_rank_fusion(hits, k=rrf_k, weights=weights)
    order = sorted(rrf, key=lambda tmdb_id: (-rrf[tmdb_id], tmdb_id))
    kept = tuple(order[: budget.pool_max])
    unique: dict[Channel, int] = {}
    for tmdb_id in kept:
        found = {h.channel for h in by_film[tmdb_id]}
        if len(found) == 1:
            only = next(iter(found))
            unique[only] = unique.get(only, 0) + 1
    return CandidatePool(
        ids=kept,
        hits={i: tuple(by_film[i]) for i in kept},
        rrf={i: rrf[i] for i in kept},
        per_channel_counts=counts,
        per_channel_unique=unique,
        prefilter_path=search.plan.path,
        overfetch_factor=search.plan.overfetch,
        truncated=len(order) > len(kept),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        allow=allow,
    )
