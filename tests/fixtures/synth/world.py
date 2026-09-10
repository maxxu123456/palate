"""Builds the world from a seed rather than checking in blobs, so it stays readable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from palate.taste.signals import FilmFacts, RatedFilm

DIM = 48
SEED = 20260909

# Far enough off the centre to be a spread, close enough that coherence clears the floor.
CLUSTER_NOISE = 0.12

FIRST_WATCH = date(2018, 1, 1)
GENRE_POOL = (12, 14, 16, 27, 35, 36, 53, 99, 878, 10402, 10749, 10752)


@dataclass(frozen=True, slots=True)
class Cluster:
    """One planted corner of the sphere and the taste the user has for it."""

    name: str
    effect: float
    decade: int
    language: str
    country: str
    genre: int


CLUSTERS = (
    Cluster("slow soviet", 1.3, 1970, "ru", "SU", 18),
    Cluster("hong kong crime", 1.1, 1990, "cn", "HK", 80),
    Cluster("loud blockbuster", -1.5, 2010, "en", "US", 28),
    Cluster("french comedy", 0.0, 1960, "fr", "FR", 35),
    Cluster("nordic drama", 0.0, 2000, "sv", "SE", 10749),
)
LOVED = (0, 1)
HATED = (2,)

BACKGROUND = -1


@dataclass(frozen=True, slots=True)
class SynthFilm:
    """One film, with every column the taste model reads."""

    tmdb_id: int
    title: str
    cluster: int
    year: int
    runtime: int
    language: str
    country: str
    genres: tuple[int, ...]
    keywords: tuple[int, ...]
    director_id: int
    director_name: str
    vote_average: float
    vote_count: int
    popularity: float
    overview: str

    @property
    def decade(self) -> int:
        return (self.year // 10) * 10


class SynthStore:
    """FilmStore over the synthetic films."""

    def __init__(self, films: Sequence[SynthFilm]) -> None:
        self._facts = {
            f.tmdb_id: FilmFacts(f.tmdb_id, f.vote_average, f.vote_count, f.decade, f.genres)
            for f in films
        }

    def facts(self, tmdb_ids: Sequence[int]) -> Mapping[int, FilmFacts]:
        """Only the ids this world knows about."""
        return {i: self._facts[i] for i in tmdb_ids if i in self._facts}


@dataclass(frozen=True, slots=True)
class SynthWorld:
    """Films, unit embeddings, the ratings that follow from the planted taste."""

    films: tuple[SynthFilm, ...]
    vectors: dict[int, np.ndarray]
    centres: np.ndarray
    rated: tuple[RatedFilm, ...]
    true_stars: dict[int, float]

    @property
    def by_id(self) -> dict[int, SynthFilm]:
        return {f.tmdb_id: f for f in self.films}

    def store(self) -> SynthStore:
        """The FilmStore the calibrator fits against."""
        return SynthStore(self.films)

    def matrix(self, tmdb_ids: Sequence[int]) -> np.ndarray:
        """Embeddings for these ids, in order, as one (n, dim) array."""
        return np.array([self.vectors[i] for i in tmdb_ids], dtype=np.float64)

    def members(self, cluster: int) -> tuple[int, ...]:
        """Every film planted in one cluster."""
        return tuple(f.tmdb_id for f in self.films if f.cluster == cluster)

    def cluster_of(self, tmdb_id: int) -> int:
        return self.by_id[tmdb_id].cluster


def build_world(
    *,
    seed: int = SEED,
    dim: int = DIM,
    per_cluster: int = 240,
    background: int = 700,
    rated_per_cluster: int = 100,
    rated_background: int = 100,
) -> SynthWorld:
    """A corpus, its embeddings and a history whose clusters the mode fitter should find."""
    rng = np.random.default_rng(seed)
    centres = _centres(rng, len(CLUSTERS), dim)
    films: list[SynthFilm] = []
    vectors: dict[int, np.ndarray] = {}
    next_id = 100_000
    for index, cluster in enumerate(CLUSTERS):
        for _ in range(per_cluster):
            films.append(_film(rng, next_id, index, cluster))
            vectors[next_id] = _near(rng, centres[index], dim)
            next_id += 1
    for _ in range(background):
        films.append(_film(rng, next_id, BACKGROUND, None))
        vectors[next_id] = _unit(rng, dim)
        next_id += 1

    by_cluster: dict[int, list[int]] = {}
    for film in films:
        by_cluster.setdefault(film.cluster, []).append(film.tmdb_id)
    chosen: list[int] = []
    for index in range(len(CLUSTERS)):
        pool = by_cluster[index]
        chosen.extend(rng.choice(pool, size=rated_per_cluster, replace=False).tolist())
    chosen.extend(rng.choice(by_cluster[BACKGROUND], size=rated_background, replace=False).tolist())
    rng.shuffle(chosen)

    lookup = {f.tmdb_id: f for f in films}
    rated: list[RatedFilm] = []
    true_stars: dict[int, float] = {}
    for offset, tmdb_id in enumerate(chosen):
        film = lookup[tmdb_id]
        effect = 0.0 if film.cluster == BACKGROUND else CLUSTERS[film.cluster].effect
        stars = 1.40 + 0.32 * film.vote_average + effect + float(rng.normal(0.0, 0.45))
        stars = min(5.0, max(0.5, stars))
        true_stars[tmdb_id] = stars
        rated.append(
            RatedFilm(
                tmdb_id=tmdb_id,
                rating_half=int(min(10, max(1, round(stars * 2)))),
                watched_at=FIRST_WATCH + timedelta(days=offset * 3),
                date_source="diary",
                date_reliable=True,
                is_rewatch=offset % 17 == 0,
            )
        )
    return SynthWorld(tuple(films), vectors, centres, tuple(rated), true_stars)


def _centres(rng: np.random.Generator, k: int, dim: int) -> np.ndarray:
    """Orthonormal centres, so two planted clusters never share a direction."""
    basis, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return np.asarray(basis.T[:k], dtype=np.float64)


def _unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.standard_normal(dim)
    return raw / float(np.linalg.norm(raw))


def _near(rng: np.random.Generator, centre: np.ndarray, dim: int) -> np.ndarray:
    raw = centre + CLUSTER_NOISE * rng.standard_normal(dim)
    return raw / float(np.linalg.norm(raw))


def _film(rng: np.random.Generator, tmdb_id: int, cluster: int, spec: Cluster | None) -> SynthFilm:
    # Cluster identity only tilts the metadata, so the generic model cannot absorb the taste.
    if spec is not None and rng.random() < 0.7:
        year = spec.decade + int(rng.integers(0, 10))
    else:
        year = int(rng.integers(1930, 2025))
    genres = {int(rng.choice(GENRE_POOL)) for _ in range(int(rng.integers(1, 3)))}
    if spec is not None and rng.random() < 0.65:
        genres.add(spec.genre)
    if spec is not None and rng.random() < 0.85:
        director_id = 500 + cluster * 6 + int(rng.integers(0, 6))
    else:
        director_id = 900 + int(rng.integers(0, 20))
    keywords = set()
    for _ in range(3):
        if spec is not None and rng.random() < 0.7:
            keywords.add(2000 + cluster * 8 + int(rng.integers(0, 8)))
        else:
            keywords.add(3000 + int(rng.integers(0, 60)))
    language = spec.language if spec is not None and rng.random() < 0.7 else "en"
    country = spec.country if spec is not None and rng.random() < 0.7 else "US"
    return SynthFilm(
        tmdb_id=tmdb_id,
        title=f"Film {tmdb_id}",
        cluster=cluster,
        year=year,
        runtime=int(rng.integers(72, 205)),
        language=language,
        country=country,
        genres=tuple(sorted(genres)),
        keywords=tuple(sorted(keywords)),
        director_id=director_id,
        director_name=f"Director {director_id}",
        vote_average=float(rng.uniform(5.0, 8.6)),
        vote_count=int(rng.integers(30, 40_000)),
        popularity=float(rng.uniform(0.5, 90.0)),
        overview=f"A film about {tmdb_id} and what happens in it.",
    )
