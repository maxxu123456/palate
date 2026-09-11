"""The arms the work is judged against. The director baseline is the one that has to fall."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import orjson

from palate.eval.labels import graded_relevance
from palate.eval.split import Fold
from palate.taste.affinity import BILLED, EntityAffinity
from palate.taste.profile import TasteProfile

# Director loyalty is the strongest of the three, and a bit part is not a reason to watch.
PEOPLE_WEIGHT: Mapping[str, float] = {"director": 1.0, "writer": 0.6, "actor": 0.5}

MIN_DIRECTOR_FILMS = 2

DIRECTOR_KAPPA = 2.0

# Four stars and up. The centroid is the naive build, so it sees likes and nothing else.
LIKED_REL = 2

type Baseline = Callable[["FoldInputs", Fold], list[int]]


@dataclass(frozen=True, slots=True)
class FoldInputs:
    """One fold's whole view of the world. Nothing here was fitted after the cut."""

    conn: sqlite3.Connection
    profile: TasteProfile
    candidates: tuple[int, ...]
    train: tuple[int, ...] = ()
    vectors: Mapping[int, Sequence[float]] = field(default_factory=dict)


def _json(values: Sequence[int]) -> str:
    return orjson.dumps(list(values)).decode()


def _popularity(ctx: FoldInputs) -> dict[int, int]:
    """Vote count per candidate, which is the only hype signal the corpus carries."""
    rows = ctx.conn.execute(
        "select tmdb_id, vote_count from films where tmdb_id in (select value from json_each(?))",
        (_json(ctx.candidates),),
    )
    return {int(r["tmdb_id"]): int(r["vote_count"]) for r in rows}


def _decades(ctx: FoldInputs) -> dict[int, int]:
    rows = ctx.conn.execute(
        "select tmdb_id, coalesce(decade, -1) as decade from films "
        "where tmdb_id in (select value from json_each(?))",
        (_json(ctx.candidates),),
    )
    return {int(r["tmdb_id"]): int(r["decade"]) for r in rows}


_CREDITS = (
    "select c.tmdb_id as tmdb_id, c.person_id as person_id from credits c "
    "where c.tmdb_id in (select value from json_each(?)) and {predicate}"
)

_PREDICATE = {
    "director": "c.job = 'Director'",
    "writer": "c.department = 'Writing'",
    "actor": f"c.credit_kind = 'cast' and c.ord < {BILLED}",
}


def credits_of(ctx: FoldInputs, kind: str) -> dict[int, list[int]]:
    """Every credit of one kind per candidate, never a denormalized first-credited id."""
    out: dict[int, list[int]] = {}
    sql = _CREDITS.format(predicate=_PREDICATE[kind])
    for row in ctx.conn.execute(sql, (_json(ctx.candidates),)):
        out.setdefault(int(row["tmdb_id"]), []).append(int(row["person_id"]))
    return out


def shrunk(entities: Sequence[EntityAffinity], *, min_films: int, kappa: float) -> dict[int, float]:
    """Entity affinities re-shrunk at this baseline's own kappa, thin entities dropped."""
    return {
        int(e.entity_id): e.raw_sum / (e.n + kappa)
        for e in entities
        if e.n >= min_films and e.entity_id.lstrip("-").isdigit()
    }


def _ordered(scored: Mapping[int, float], popularity: Mapping[int, int]) -> list[int]:
    """Score first, hype as the tiebreak, tmdb_id so two runs agree exactly."""
    return sorted(scored, key=lambda i: (-scored[i], -popularity.get(i, 0), i))


def baseline_random(ctx: FoldInputs, fold: Fold, *, seed: int = 0) -> list[int]:
    """The floor that proves the metric is wired up at all."""
    rng = np.random.default_rng(seed + fold.fold)
    order = rng.permutation(len(ctx.candidates))
    return [ctx.candidates[int(i)] for i in order]


def baseline_popularity(ctx: FoldInputs, fold: Fold) -> list[int]:
    """TMDB vote count descending, which is hype and nothing else."""
    popularity = _popularity(ctx)
    return _ordered({i: float(popularity.get(i, 0)) for i in ctx.candidates}, popularity)


def baseline_popularity_era(ctx: FoldInputs, fold: Fold) -> list[int]:
    """Hype inside the decades the user over-indexes on, which is the fair version."""
    popularity = _popularity(ctx)
    decade_of = _decades(ctx)
    liked = preferred_decades(ctx.profile)
    return sorted(
        ctx.candidates,
        key=lambda i: (0 if decade_of.get(i, -1) in liked else 1, -popularity.get(i, 0), i),
    )


def preferred_decades(profile: TasteProfile) -> frozenset[int]:
    """Decades the user watches more of than the corpus holds, or the best rated ones."""
    entries = profile.affinities.get("decade", ())
    over = {int(e.entity_id) for e in entries if e.exposure_logodds > 0.0}
    if over:
        return frozenset(over)
    return frozenset(int(e.entity_id) for e in entries[:3])


def baseline_director_affinity(
    ctx: FoldInputs,
    fold: Fold,
    *,
    min_films: int = MIN_DIRECTOR_FILMS,
    kappa: float = DIRECTOR_KAPPA,
) -> list[int]:
    """Unwatched films by directors the user has rated twice or more. The bar to beat."""
    affinity = shrunk(ctx.profile.affinities.get("director", ()), min_films=min_films, kappa=kappa)
    directors = credits_of(ctx, "director")
    popularity = _popularity(ctx)
    scored = {
        i: max((affinity[p] for p in directors.get(i, ()) if p in affinity), default=0.0)
        for i in ctx.candidates
    }
    return _ordered(scored, popularity)


def baseline_people_affinity(ctx: FoldInputs, fold: Fold) -> list[int]:
    """Directors, writers and top billing together, which is stronger and more awkward."""
    popularity = _popularity(ctx)
    tables = {
        kind: shrunk(
            ctx.profile.affinities.get(kind, ()),
            min_films=1 if kind != "director" else MIN_DIRECTOR_FILMS,
            kappa=DIRECTOR_KAPPA,
        )
        for kind in PEOPLE_WEIGHT
    }
    credits = {kind: credits_of(ctx, kind) for kind in PEOPLE_WEIGHT}
    scored: dict[int, float] = {}
    for tmdb_id in ctx.candidates:
        total = 0.0
        for kind, weight in PEOPLE_WEIGHT.items():
            table = tables[kind]
            best = [table[p] for p in credits[kind].get(tmdb_id, ()) if p in table]
            total += weight * max(best, default=0.0)
        scored[tmdb_id] = total
    return _ordered(scored, popularity)


def baseline_single_centroid(ctx: FoldInputs, fold: Fold) -> list[int]:
    """Mean of the liked embeddings, cosine nearest neighbours, no negatives and no modes."""
    centre = liked_centroid(ctx, fold)
    popularity = _popularity(ctx)
    if centre is None:
        return baseline_popularity(ctx, fold)
    scored = {
        i: float(np.asarray(ctx.vectors[i], dtype=np.float64) @ centre)
        if i in ctx.vectors
        else -1.0
        for i in ctx.candidates
    }
    return _ordered(scored, popularity)


def liked_centroid(ctx: FoldInputs, fold: Fold) -> np.ndarray | None:
    """The naive taste vector: one mean over everything the user rated four stars or more."""
    ratings = _train_ratings(ctx, fold)
    rows = [
        np.asarray(ctx.vectors[i], dtype=np.float64)
        for i, half in ratings.items()
        if graded_relevance(half) >= LIKED_REL and i in ctx.vectors
    ]
    if not rows:
        return None
    centre = np.vstack(rows).mean(axis=0)
    norm = float(np.linalg.norm(centre))
    return centre / norm if norm > 0.0 else None


def _train_ratings(ctx: FoldInputs, fold: Fold) -> dict[int, int]:
    ids = ctx.train or fold.train
    rows = ctx.conn.execute(
        "select tmdb_id, rating_half from user_films where rating_half is not null "
        "and tmdb_id in (select value from json_each(?))",
        (_json(ids),),
    )
    return {int(r["tmdb_id"]): int(r["rating_half"]) for r in rows}


BASELINES: Mapping[str, Baseline] = {
    "random": baseline_random,
    "popularity": baseline_popularity,
    "popularity_era": baseline_popularity_era,
    "director_affinity": baseline_director_affinity,
    "people_affinity": baseline_people_affinity,
    "single_centroid_dense": baseline_single_centroid,
}
