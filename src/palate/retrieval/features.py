"""The pool feature matrix, scaled over the rows a channel actually reached."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import orjson

from palate.retrieval.candidates import CandidatePool, Channel
from palate.taste.modes import mode_affinity, mode_margin
from palate.taste.profile import TasteProfile, metadata_features
from palate.taste.ridge import PreferenceDirection, predict_with_leverage

FEATURES: tuple[str, ...] = (
    "mode_affinity",
    "mode_margin",
    "anti_affinity",
    "ridge_pref",
    "ridge_leverage",
    "director_aff",
    "writer_aff",
    "actor_aff",
    "keyword_aff",
    "decade_aff",
    "decade_exposure",
    "runtime_aff",
    "lang_aff",
    "country_aff",
    "country_penalty",
    "soft_pref_penalty",
    "popularity",
    "vote_quality",
    "bm25",
    "query_sim",
    "ce_score",
)

PENALTY_FEATURES: frozenset[str] = frozenset(
    {"anti_affinity", "country_penalty", "soft_pref_penalty", "ridge_leverage"}
)

MIN_SUPPORT = 8

# Shrinks the ridge term where the candidate sits outside every direction the user sampled.
TAU = 1.0

# Votes needed before the crowd's average is worth as much as the corpus prior.
VOTE_PRIOR = 200.0

_PEOPLE_KIND = {"director": "directors", "writer": "writers", "actor": "actors"}

_PENALTY_HANDLE = {
    "genre": "genres",
    "keyword": "keywords",
    "director": "directors",
    "actor": "actors",
    "writer": "writers",
    "country": "countries",
}


@dataclass(frozen=True, slots=True)
class FilmFacets:
    """Everything about one candidate that a feature reads."""

    tmdb_id: int
    decade: int
    runtime_bucket: int
    language: str
    popularity: float | None
    vote_average: float
    vote_count: int
    collection_id: int | None
    directors: tuple[int, ...] = ()
    writers: tuple[int, ...] = ()
    actors: tuple[int, ...] = ()
    keywords: tuple[int, ...] = ()
    genres: tuple[int, ...] = ()
    countries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    """What the channels and the reranker produced, as the matrix builder needs it."""

    vectors: Mapping[int, Sequence[float]] = field(default_factory=dict)
    bm25: Mapping[int, float] = field(default_factory=dict)
    query_embedding: Sequence[float] | None = None
    ce_scores: Mapping[int, float] = field(default_factory=dict)
    soft_countries: frozenset[str] = frozenset()
    penalties: Mapping[str, float] = field(default_factory=dict)
    shrink_ridge: bool = True


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """One pool, scaled, with the mask that says which values were ever produced."""

    X: np.ndarray
    raw: np.ndarray
    ids: tuple[int, ...]
    names: tuple[str, ...]
    support: np.ndarray
    scaled_by: tuple[str, ...]

    def column(self, name: str) -> np.ndarray:
        """One scaled column by name."""
        return np.asarray(self.X[:, self.names.index(name)])

    def raw_column(self, name: str) -> np.ndarray:
        """One unscaled column by name, which is what evidence quotes."""
        return np.asarray(self.raw[:, self.names.index(name)])

    def supported(self, name: str) -> np.ndarray:
        """The mask for one column."""
        return np.asarray(self.support[:, self.names.index(name)])

    @property
    def active(self) -> tuple[str, ...]:
        """Features that survived scaling, which is what the weight fitter may touch."""
        return tuple(
            n for n, how in zip(self.names, self.scaled_by, strict=True) if how != "zeroed"
        )


def _empty() -> FeatureMatrix:
    """A pool the filters emptied still has to answer every question about its columns."""
    values = np.zeros((0, len(FEATURES)))
    mask = np.zeros((0, len(FEATURES)), dtype=bool)
    return FeatureMatrix(values, values, (), FEATURES, mask, ("zeroed",) * len(FEATURES))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so a column of one repeated value stays constant."""
    order = np.argsort(values, kind="stable")
    positions = np.empty(values.size, dtype=np.float64)
    positions[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    totals = np.zeros(unique.size)
    counts = np.zeros(unique.size)
    np.add.at(totals, inverse, positions)
    np.add.at(counts, inverse, 1.0)
    return np.asarray((totals / counts)[inverse])


def scale_feature(
    v: np.ndarray, support: np.ndarray, *, clip: float = 3.0
) -> tuple[np.ndarray, str]:
    """Robust-scale over the SUPPORT SET only, with a rank fallback."""
    out = np.zeros(v.shape[0], dtype=np.float64)
    inside = np.asarray(v, dtype=np.float64)[support]
    if inside.size < MIN_SUPPORT:
        return out, "zeroed"
    median = float(np.median(inside))
    spread = 1.4826 * float(np.median(np.abs(inside - median)))
    if spread > 1e-9:
        out[support] = np.clip((inside - median) / spread, -clip, clip)
        return out, "mad"
    ranked = 2.0 * (_ranks(inside) / (inside.size - 1)) - 1.0
    if float(np.ptp(ranked)) < 1e-12:
        return np.zeros(v.shape[0]), "zeroed"
    out[support] = np.clip(ranked, -clip, clip)
    return out, "rank"


_FACTS = (
    "select f.tmdb_id, coalesce(f.decade, -1) as decade, f.runtime_bucket, "
    "coalesce(f.original_language, '') as language, "
    "coalesce(f.popularity_at_crawl, f.popularity) as popularity, "
    "coalesce(f.vote_average, 0.0) as vote_average, f.vote_count, f.collection_id "
    "from films f where f.tmdb_id in (select value from json_each(?))"
)

_FACET_SQL = {
    "directors": (
        "select tmdb_id, person_id as entity from credits "
        "where job = 'Director' and tmdb_id in (select value from json_each(?))"
    ),
    "writers": (
        "select tmdb_id, person_id as entity from credits "
        "where department = 'Writing' and tmdb_id in (select value from json_each(?))"
    ),
    "actors": (
        "select tmdb_id, person_id as entity from credits "
        "where credit_kind = 'cast' and ord < 10 and tmdb_id in (select value from json_each(?))"
    ),
    "keywords": (
        "select tmdb_id, keyword_id as entity from film_keywords "
        "where tmdb_id in (select value from json_each(?))"
    ),
    "genres": (
        "select tmdb_id, genre_id as entity from film_genres "
        "where tmdb_id in (select value from json_each(?))"
    ),
    "countries": (
        "select tmdb_id, iso_3166_1 as entity from film_countries "
        "where tmdb_id in (select value from json_each(?))"
    ),
}


def load_facets(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, FilmFacets]:
    """Every column and credit the feature matrix reads, in one pass per facet."""
    payload = orjson.dumps(list(ids)).decode()
    grouped: dict[str, dict[int, list[Any]]] = {}
    for facet, sql in _FACET_SQL.items():
        found: dict[int, list[Any]] = {}
        for row in conn.execute(sql, (payload,)):
            found.setdefault(int(row["tmdb_id"]), []).append(row["entity"])
        grouped[facet] = found
    out: dict[int, FilmFacets] = {}
    for row in conn.execute(_FACTS, (payload,)):
        tmdb_id = int(row["tmdb_id"])
        out[tmdb_id] = FilmFacets(
            tmdb_id=tmdb_id,
            decade=int(row["decade"]),
            runtime_bucket=int(row["runtime_bucket"]),
            language=str(row["language"]),
            popularity=None if row["popularity"] is None else float(row["popularity"]),
            vote_average=float(row["vote_average"]),
            vote_count=int(row["vote_count"]),
            collection_id=None if row["collection_id"] is None else int(row["collection_id"]),
            directors=tuple(int(e) for e in grouped["directors"].get(tmdb_id, ())),
            writers=tuple(int(e) for e in grouped["writers"].get(tmdb_id, ())),
            actors=tuple(int(e) for e in grouped["actors"].get(tmdb_id, ())),
            keywords=tuple(int(e) for e in grouped["keywords"].get(tmdb_id, ())),
            genres=tuple(int(e) for e in grouped["genres"].get(tmdb_id, ())),
            countries=tuple(str(e) for e in grouped["countries"].get(tmdb_id, ())),
        )
    return out


def bm25_scores(pool: CandidatePool) -> dict[int, float]:
    """What the lexical channel scored each candidate, keyed by film."""
    return {
        tmdb_id: hit.raw_score
        for tmdb_id, hits in pool.hits.items()
        for hit in hits
        if hit.channel is Channel.BM25
    }


def _mode_dim(profile: TasteProfile) -> int:
    return len(profile.modes[0].centroid) if profile.modes else 0


def _table(profile: TasteProfile, kind: str, *, exposure: bool = False) -> dict[str, float]:
    rows = profile.affinities.get(kind, ())
    return {r.entity_id: (r.exposure_logodds if exposure else r.affinity) for r in rows}


def _best_of(entities: Sequence[Any], table: Mapping[str, float]) -> tuple[float, bool]:
    found = [table[str(e)] for e in entities if str(e) in table]
    return (max(found), True) if found else (0.0, False)


def _matrix(vectors: Mapping[int, Sequence[float]], ids: Sequence[int], dim: int) -> np.ndarray:
    out = np.zeros((len(ids), dim))
    for i, tmdb_id in enumerate(ids):
        vector = vectors.get(tmdb_id)
        if vector is not None:
            out[i] = vector
    return out


def _fill_modes(
    raw: np.ndarray,
    support: np.ndarray,
    at: Mapping[str, int],
    profile: TasteProfile,
    rows: np.ndarray,
    seen: np.ndarray,
) -> None:
    if rows.size == 0:
        return
    if profile.modes:
        raw[:, at["mode_affinity"]] = mode_affinity(rows, profile.modes)
        support[:, at["mode_affinity"]] = seen
    if profile.modes and profile.anti_modes:
        raw[:, at["mode_margin"]] = mode_margin(rows, profile.modes, profile.anti_modes)
        support[:, at["mode_margin"]] = seen
    if profile.anti_modes:
        raw[:, at["anti_affinity"]] = mode_affinity(rows, profile.anti_modes)
        support[:, at["anti_affinity"]] = seen


def _ridge_rows(
    conn: sqlite3.Connection,
    direction: PreferenceDirection,
    ids: Sequence[int],
    vectors: Mapping[int, Sequence[float]],
) -> np.ndarray:
    """A candidate row in the direction's own feature order, zero where a name is unknown."""
    names = direction.feature_names
    embedded = sum(1 for n in names if n.startswith("emb:"))
    out = np.zeros((len(ids), len(names)))
    out[:, :embedded] = _matrix(vectors, ids, embedded)
    meta, meta_names = metadata_features(conn, ids)
    columns = {name: i for i, name in enumerate(meta_names)}
    for j, name in enumerate(names[embedded:], start=embedded):
        source = columns.get(name)
        if source is not None:
            out[:, j] = meta[:, source]
    return out


def _fill_ridge(
    raw: np.ndarray,
    support: np.ndarray,
    at: Mapping[str, int],
    conn: sqlite3.Connection,
    profile: TasteProfile,
    ids: Sequence[int],
    inputs: FeatureInputs,
    seen: np.ndarray,
) -> None:
    direction = profile.direction
    if direction is None or not direction.feature_names:
        return
    rows = _ridge_rows(conn, direction, ids, inputs.vectors)
    prediction, leverage = predict_with_leverage(direction, rows)
    if inputs.shrink_ridge:
        prediction = prediction / (1.0 + TAU * np.clip(leverage, 0.0, None))
    raw[:, at["ridge_pref"]] = prediction
    raw[:, at["ridge_leverage"]] = leverage
    support[:, at["ridge_pref"]] = seen
    support[:, at["ridge_leverage"]] = seen


def _fill_affinities(
    raw: np.ndarray,
    support: np.ndarray,
    at: Mapping[str, int],
    profile: TasteProfile,
    facets: Sequence[FilmFacets],
) -> None:
    people = {kind: _table(profile, kind) for kind in _PEOPLE_KIND}
    keywords = _table(profile, "keyword")
    countries = _table(profile, "country")
    decades = _table(profile, "decade")
    exposure = _table(profile, "decade", exposure=True)
    runtimes = _table(profile, "runtime_bucket")
    languages = _table(profile, "language")
    for i, film in enumerate(facets):
        for kind, attribute in _PEOPLE_KIND.items():
            value, found = _best_of(getattr(film, attribute), people[kind])
            raw[i, at[f"{kind}_aff"]] = value
            support[i, at[f"{kind}_aff"]] = found
        for name, entities, table in (
            ("keyword_aff", film.keywords, keywords),
            ("country_aff", film.countries, countries),
            ("decade_aff", (film.decade,), decades),
            ("decade_exposure", (film.decade,), exposure),
            ("runtime_aff", (film.runtime_bucket,), runtimes),
            ("lang_aff", (film.language,), languages),
        ):
            value, found = _best_of(entities, table)
            raw[i, at[name]] = value
            support[i, at[name]] = found


def _fill_penalties(
    raw: np.ndarray,
    support: np.ndarray,
    at: Mapping[str, int],
    facets: Sequence[FilmFacets],
    inputs: FeatureInputs,
) -> None:
    for i, film in enumerate(facets):
        if inputs.soft_countries:
            raw[i, at["country_penalty"]] = float(bool(set(film.countries) & inputs.soft_countries))
            support[i, at["country_penalty"]] = True
        total = 0.0
        matched = False
        for key, weight in inputs.penalties.items():
            kind, _, target = key.partition(":")
            if target in _handles(film, kind):
                # as_penalties signs a dislike negative, and this column is a penalty.
                total -= weight
                matched = True
        raw[i, at["soft_pref_penalty"]] = total
        support[i, at["soft_pref_penalty"]] = matched


def _handles(film: FilmFacets, kind: str) -> set[str]:
    attribute = _PENALTY_HANDLE.get(kind)
    if attribute is not None:
        return {str(e) for e in getattr(film, attribute)}
    if kind == "language":
        return {film.language}
    if kind == "decade":
        return {str(film.decade)}
    if kind == "film":
        return {str(film.tmdb_id)}
    if kind == "collection":
        return set() if film.collection_id is None else {str(film.collection_id)}
    return set()


def _fill_crowd(
    raw: np.ndarray, support: np.ndarray, at: Mapping[str, int], facets: Sequence[FilmFacets]
) -> None:
    voted = [f.vote_average for f in facets if f.vote_count > 0]
    prior = float(np.mean(voted)) if voted else 0.0
    for i, film in enumerate(facets):
        if film.popularity is not None:
            raw[i, at["popularity"]] = float(np.log1p(max(film.popularity, 0.0)))
            support[i, at["popularity"]] = True
        if film.vote_count > 0:
            weight = film.vote_count / (film.vote_count + VOTE_PRIOR)
            raw[i, at["vote_quality"]] = weight * film.vote_average + (1.0 - weight) * prior
            support[i, at["vote_quality"]] = True


def _fill_channels(
    raw: np.ndarray,
    support: np.ndarray,
    at: Mapping[str, int],
    ids: Sequence[int],
    inputs: FeatureInputs,
    rows: np.ndarray,
    seen: np.ndarray,
) -> None:
    for name, values in (("bm25", inputs.bm25), ("ce_score", inputs.ce_scores)):
        for i, tmdb_id in enumerate(ids):
            if tmdb_id in values:
                raw[i, at[name]] = values[tmdb_id]
                support[i, at[name]] = True
    query = np.asarray(inputs.query_embedding or (), dtype=np.float64)
    if query.size == 0 or rows.shape[1] != query.size:
        return
    scale = float(np.linalg.norm(query)) or 1.0
    norms = np.linalg.norm(rows, axis=1)
    raw[:, at["query_sim"]] = (rows @ query) / (scale * np.where(norms > 0.0, norms, 1.0))
    support[:, at["query_sim"]] = seen


def build_matrix(
    conn: sqlite3.Connection,
    profile: TasteProfile,
    ids: Sequence[int],
    inputs: FeatureInputs,
    *,
    clip: float = 3.0,
) -> FeatureMatrix:
    """Every feature for every candidate, scaled inside this pool and nowhere else."""
    known = load_facets(conn, ids)
    kept = tuple(i for i in ids if i in known)
    if not kept:
        return _empty()
    facets = [known[i] for i in kept]
    at = {name: i for i, name in enumerate(FEATURES)}
    raw = np.zeros((len(kept), len(FEATURES)))
    support = np.zeros((len(kept), len(FEATURES)), dtype=bool)
    dim = len(next(iter(inputs.vectors.values()), ())) or _mode_dim(profile)
    rows = _matrix(inputs.vectors, kept, dim)
    seen = np.array([i in inputs.vectors for i in kept], dtype=bool)
    _fill_modes(raw, support, at, profile, rows, seen)
    _fill_ridge(raw, support, at, conn, profile, kept, inputs, seen)
    _fill_affinities(raw, support, at, profile, facets)
    _fill_penalties(raw, support, at, facets, inputs)
    _fill_crowd(raw, support, at, facets)
    _fill_channels(raw, support, at, kept, inputs, rows, seen)
    scaled = np.zeros_like(raw)
    how: list[str] = []
    for j in range(len(FEATURES)):
        column, method = scale_feature(raw[:, j], support[:, j], clip=clip)
        scaled[:, j] = column
        how.append(method)
    return FeatureMatrix(scaled, raw, kept, FEATURES, support, tuple(how))
