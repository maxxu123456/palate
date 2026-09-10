"""A star rating is not a preference weight, so it is standardised and de-generified first."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

import numpy as np
import orjson

# Stars. A user who rates everything 3.5 to 4.0 would otherwise get z scores of plus or minus 8.
SIGMA_FLOOR = 0.35

MASS_CAP = 3.0

# Below this the generic model fits the user rather than the crowd, so it is switched off.
MIN_CALIBRATION_N = 20


@dataclass(frozen=True, slots=True)
class RatedFilm:
    """One rated film out of the history."""

    tmdb_id: int
    rating_half: int
    watched_at: date | None = None
    date_source: Literal["diary", "ratings", "none"] = "none"
    date_reliable: bool = False
    is_rewatch: bool = False

    @property
    def stars(self) -> float:
        """Letterboxd keeps half stars as 1 to 10."""
        return self.rating_half / 2.0


@dataclass(frozen=True, slots=True)
class FilmFacts:
    """The public facts a generic viewer's rating can be predicted from."""

    tmdb_id: int
    vote_average: float
    vote_count: int
    decade: int
    genres: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PreferenceSignal:
    """One rating turned into a signed weight, in units of the user's own spread."""

    tmdb_id: int
    z: float
    residual: float
    s: float
    mass: float


class FilmStore(Protocol):
    """Where the calibrator looks up the crowd's opinion of a film."""

    def facts(self, tmdb_ids: Sequence[int]) -> Mapping[int, FilmFacts]:
        """Facts for these films. Ids the corpus has never seen are simply absent."""
        ...


_FACTS = (
    "select f.tmdb_id, coalesce(f.vote_average, 0.0) as vote_average, f.vote_count, "
    "coalesce(f.decade, -1) as decade, "
    "coalesce(group_concat(g.genre_id), '') as genre_ids "
    "from films f left join film_genres g on g.tmdb_id = f.tmdb_id "
    "where f.tmdb_id in (select value from json_each(?)) "
    "group by f.tmdb_id"
)


class SqliteFilmStore:
    """FilmStore over the films table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def facts(self, tmdb_ids: Sequence[int]) -> Mapping[int, FilmFacts]:
        """One grouped query, so a 900 film history is one round trip."""
        ids = orjson.dumps(list(tmdb_ids)).decode()
        out: dict[int, FilmFacts] = {}
        for row in self.conn.execute(_FACTS, (ids,)):
            raw = str(row["genre_ids"])
            out[int(row["tmdb_id"])] = FilmFacts(
                tmdb_id=int(row["tmdb_id"]),
                vote_average=float(row["vote_average"]),
                vote_count=int(row["vote_count"]),
                decade=int(row["decade"]),
                genres=tuple(sorted(int(g) for g in raw.split(",") if g)),
            )
        return out


class ResidualCalibrator:
    """Rating minus what a generic viewer would give, in the user's own scale."""

    def __init__(self, alpha: float = 0.6, ridge: float = 1.0) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha
        self.ridge = ridge
        self._store: FilmStore | None = None
        self._decades: tuple[int, ...] = ()
        self._genres: tuple[int, ...] = ()
        self._coef = np.zeros(0)
        self._center = np.zeros(0)
        self._scale = np.ones(0)
        self._mu = 0.0
        self._sigma = SIGMA_FLOOR
        self._r2 = 0.0

    @property
    def mu(self) -> float:
        """Mean rating in stars."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Spread in stars, floored."""
        return self._sigma

    @property
    def r2(self) -> float:
        """How much of the rating is generic taste rather than this person's."""
        return self._r2

    def fit(self, rated: Sequence[RatedFilm], store: FilmStore) -> ResidualCalibrator:
        """Ridge on [vote_average, log1p(vote_count), decade one-hot, genre multi-hot]."""
        if not rated:
            raise ValueError("cannot calibrate on an empty history")
        self._store = store
        stars = np.array([r.stars for r in rated], dtype=np.float64)
        self._mu = float(stars.mean())
        self._sigma = max(float(stars.std()), SIGMA_FLOOR)
        facts = store.facts([r.tmdb_id for r in rated])
        known = [r for r in rated if r.tmdb_id in facts]
        self._decades = tuple(sorted({facts[r.tmdb_id].decade for r in known}))
        self._genres = tuple(sorted({g for r in known for g in facts[r.tmdb_id].genres}))
        raw = self._raw(known, facts)
        self._center = raw.mean(axis=0) if raw.size else np.zeros(raw.shape[1])
        spread = raw.std(axis=0) if raw.size else np.ones(raw.shape[1])
        self._scale = np.where(spread > 1e-9, spread, 1.0)
        if len(known) < MIN_CALIBRATION_N or raw.shape[1] == 0:
            self._coef = np.zeros(raw.shape[1])
            self._r2 = 0.0
            return self
        design = (raw - self._center) / self._scale
        centered = np.array([r.stars for r in known]) - self._mu
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self._coef = np.linalg.solve(gram, design.T @ centered)
        left = centered - design @ self._coef
        total = float(centered @ centered)
        self._r2 = float(1.0 - float(left @ left) / total) if total > 0.0 else 0.0
        return self

    def transform(self, rated: Sequence[RatedFilm]) -> list[PreferenceSignal]:
        """Signed weights for these films. A film the corpus does not know predicts mu."""
        if self._store is None:
            raise ValueError("fit the calibrator before transforming")
        if not rated:
            return []
        facts = self._store.facts([r.tmdb_id for r in rated])
        stars = np.array([r.stars for r in rated], dtype=np.float64)
        predicted = np.full(len(rated), self._mu)
        rows = [i for i, r in enumerate(rated) if r.tmdb_id in facts]
        if rows and self._coef.size:
            design = (self._raw([rated[i] for i in rows], facts) - self._center) / self._scale
            predicted[rows] = self._mu + design @ self._coef
        z = (stars - self._mu) / self._sigma
        residual = (stars - predicted) / self._sigma
        s = self.alpha * z + (1.0 - self.alpha) * residual
        mass = np.minimum(np.abs(s), MASS_CAP)
        return [
            PreferenceSignal(
                tmdb_id=film.tmdb_id,
                z=float(z[i]),
                residual=float(residual[i]),
                s=float(s[i]),
                mass=float(mass[i]),
            )
            for i, film in enumerate(rated)
        ]

    def to_row(self) -> dict[str, Any]:
        """Everything taste_profiles.calibrator_json has to hold."""
        return {
            "alpha": self.alpha,
            "ridge": self.ridge,
            "mu": self._mu,
            "sigma": self._sigma,
            "r2": self._r2,
            "decades": list(self._decades),
            "genres": list(self._genres),
            "coef": [float(v) for v in self._coef],
            "center": [float(v) for v in self._center],
            "scale": [float(v) for v in self._scale],
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any], store: FilmStore) -> ResidualCalibrator:
        """Rebuild a stored calibrator against a store, without refitting."""
        self = cls(alpha=float(row["alpha"]), ridge=float(row["ridge"]))
        self._store = store
        self._mu = float(row["mu"])
        self._sigma = float(row["sigma"])
        self._r2 = float(row["r2"])
        self._decades = tuple(int(d) for d in row["decades"])
        self._genres = tuple(int(g) for g in row["genres"])
        self._coef = np.array(row["coef"], dtype=np.float64)
        self._center = np.array(row["center"], dtype=np.float64)
        self._scale = np.array(row["scale"], dtype=np.float64)
        return self

    def _raw(self, rated: Sequence[RatedFilm], facts: Mapping[int, FilmFacts]) -> np.ndarray:
        width = 2 + len(self._decades) + len(self._genres)
        out = np.zeros((len(rated), width))
        decade_at = {d: i for i, d in enumerate(self._decades)}
        genre_at = {g: i for i, g in enumerate(self._genres)}
        for i, film in enumerate(rated):
            fact = facts[film.tmdb_id]
            out[i, 0] = fact.vote_average
            out[i, 1] = math.log1p(fact.vote_count)
            column = decade_at.get(fact.decade)
            if column is not None:
                out[i, 2 + column] = 1.0
            for genre in fact.genres:
                slot = genre_at.get(genre)
                if slot is not None:
                    out[i, 2 + len(self._decades) + slot] = 1.0
        return out
