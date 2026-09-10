"""Assembling a profile, writing it down, and refusing one fitted in a space that has moved."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import orjson

from palate.clock import now_iso
from palate.db.connect import Database
from palate.errors import StaleArtifact, ThinHistoryError
from palate.hashing import short_hash
from palate.ids import new_id
from palate.index import verify
from palate.index.vecstore import VecStore
from palate.providers.fingerprint import pack_f32, unpack_f32
from palate.taste.affinity import KAPPA, EntityAffinity, build_affinities
from palate.taste.modes import ModeMember, Polarity, TasteMode, build_modes
from palate.taste.ridge import MIN_RIDGE_N, PreferenceDirection, fit_preference_direction
from palate.taste.signals import (
    PreferenceSignal,
    RatedFilm,
    ResidualCalibrator,
    SqliteFilmStore,
)

type Tier = Literal["cold", "thin", "full"]
type DateSource = Literal["diary", "ratings", "none"]

THIN_FLOOR = 30
FULL_FLOOR = 150


@dataclass(frozen=True, slots=True)
class TasteProfile:
    """One fitted profile, always tied to the index it was fitted in."""

    profile_id: str
    tier: Tier
    cutoff_date: date | None
    index_id: str
    doc_template_version: str
    n_rated: int
    n_reliable_dated: int
    rating_histogram: dict[str, int]
    modes: tuple[TasteMode, ...]
    anti_modes: tuple[TasteMode, ...]
    direction: PreferenceDirection | None
    affinities: Mapping[str, tuple[EntityAffinity, ...]]
    calibrator: dict[str, Any]
    alpha: float
    params_sha: str
    built_at: str
    stale: bool = False
    stale_reason: str | None = None
    split_name: str | None = None
    fold: int | None = None

    @property
    def mean_rating(self) -> float:
        """Mean rating in stars, which is the calibrator's own centre."""
        return float(self.calibrator["mu"])

    def top(self, kind: str, *, limit: int = 5) -> tuple[EntityAffinity, ...]:
        """Highest affinity entities of one kind."""
        return tuple(self.affinities.get(kind, ())[:limit])


def tier_for(n_rated: int) -> Tier:
    """Cold refuses an unconditioned recommendation, thin drops the ridge, full does everything."""
    if n_rated < THIN_FLOOR:
        return "cold"
    return "thin" if n_rated < FULL_FLOOR else "full"


_RATED = (
    "select tmdb_id, rating_half, coalesce(watched_date, logged_date) as watched_at, "
    "date_source, date_reliable, is_rewatch from user_films "
    "where rating_half is not null "
    "and (? is null or (coalesce(watched_date, logged_date) is not null "
    "and coalesce(watched_date, logged_date) < ?)) "
    "order by tmdb_id"
)

_META = (
    "select f.tmdb_id, coalesce(f.decade, -1) as decade, coalesce(f.runtime, 0) as runtime, "
    "coalesce(f.original_language, '') as original_language, "
    "coalesce(f.popularity_at_crawl, f.popularity, 0.0) as popularity, f.vote_count "
    "from films f where f.tmdb_id in (select value from json_each(?))"
)


def load_rated(conn: sqlite3.Connection, *, cutoff: date | None = None) -> list[RatedFilm]:
    """Every rated film, optionally only those watched before a cutoff."""
    stamp = cutoff.isoformat() if cutoff else None
    out: list[RatedFilm] = []
    for row in conn.execute(_RATED, (stamp, stamp)):
        raw = row["watched_at"]
        out.append(
            RatedFilm(
                tmdb_id=int(row["tmdb_id"]),
                rating_half=int(row["rating_half"]),
                watched_at=date.fromisoformat(str(raw)[:10]) if raw else None,
                date_source=cast("DateSource", row["date_source"]),
                date_reliable=bool(row["date_reliable"]),
                is_rewatch=bool(row["is_rewatch"]),
            )
        )
    return out


def metadata_features(
    conn: sqlite3.Connection, tmdb_ids: Sequence[int]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Era, length, language and popularity, explicit so the ridge cannot steal them."""
    rows = {
        int(r["tmdb_id"]): r for r in conn.execute(_META, (orjson.dumps(list(tmdb_ids)).decode(),))
    }
    decades = sorted({int(r["decade"]) for r in rows.values()})
    languages = sorted({str(r["original_language"]) for r in rows.values()})
    names = (
        *(f"meta:decade:{d}" for d in decades),
        "meta:log_runtime",
        *(f"meta:lang:{code or 'unknown'}" for code in languages),
        "meta:log_popularity",
        "meta:log_vote_count",
    )
    out = np.zeros((len(tmdb_ids), len(names)))
    decade_at = {d: i for i, d in enumerate(decades)}
    language_at = {c: len(decades) + 1 + i for i, c in enumerate(languages)}
    for i, tmdb_id in enumerate(tmdb_ids):
        row = rows.get(tmdb_id)
        if row is None:
            continue
        out[i, decade_at[int(row["decade"])]] = 1.0
        out[i, len(decades)] = np.log1p(float(row["runtime"]))
        out[i, language_at[str(row["original_language"])]] = 1.0
        out[i, -2] = np.log1p(float(row["popularity"]))
        out[i, -1] = np.log1p(float(row["vote_count"]))
    return out, names


def build(
    db: Database,
    *,
    cutoff: date | None = None,
    alpha: float = 0.6,
    seed: int = 0,
    split_name: str | None = None,
    fold: int | None = None,
) -> TasteProfile:
    """Fit a profile in the active index's space, then write it down."""
    record = verify.active(db)
    conn = db.read()
    rated = load_rated(conn, cutoff=cutoff)
    if not rated:
        raise ThinHistoryError("no rated films, run: palate ingest <letterboxd export>")
    calibrator = ResidualCalibrator(alpha=alpha).fit(rated, SqliteFilmStore(conn))
    signals = calibrator.transform(rated)
    tier = tier_for(len(rated))
    store = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
    vectors = store.vectors([r.tmdb_id for r in rated]) if tier != "cold" else {}
    modes = build_modes(signals, vectors, polarity="like", seed=seed) if vectors else []
    anti = build_modes(signals, vectors, polarity="dislike", seed=seed) if vectors else []
    direction = (
        _direction(conn, signals, vectors, record.fingerprint.dim) if tier == "full" else None
    )
    ratings = [r.rating_half for r in rated]
    profile = TasteProfile(
        profile_id=new_id("prof_"),
        tier=tier,
        cutoff_date=cutoff,
        index_id=record.index_id,
        doc_template_version=record.fingerprint.doc_template_version,
        n_rated=len(rated),
        n_reliable_dated=sum(1 for r in rated if r.date_reliable and r.watched_at),
        rating_histogram={str(half): ratings.count(half) for half in sorted(set(ratings))},
        modes=tuple(modes),
        anti_modes=tuple(anti),
        direction=direction,
        affinities=build_affinities(conn, signals),
        calibrator=calibrator.to_row(),
        alpha=alpha,
        params_sha=_params_sha(record.index_id, alpha, seed, cutoff),
        built_at=now_iso(),
        split_name=split_name,
        fold=fold,
    )
    _persist(db, profile)
    return profile


def assert_fresh(db: Database, profile: TasteProfile) -> None:
    """A direction fitted under another model loads cleanly and ranks garbage, so it is refused."""
    active = verify.active_id(db)
    if profile.stale or active != profile.index_id:
        raise StaleArtifact("profile", profile.profile_id, profile.index_id, active or "none")


def latest(db: Database, *, cutoff: date | None = None, stale: bool = False) -> TasteProfile | None:
    """Newest profile for this cutoff, fresh ones only unless asked otherwise."""
    row = (
        db.read()
        .execute(
            "select profile_id from taste_profiles "
            "where cutoff_date is ? and (? or stale = 0) order by built_at desc limit 1",
            (cutoff.isoformat() if cutoff else None, int(stale)),
        )
        .fetchone()
    )
    return None if row is None else load(db, str(row["profile_id"]))


def load(db: Database, profile_id: str) -> TasteProfile | None:
    """Read one profile back, with the ridge factors gone and only the diagonal left."""
    conn = db.read()
    row = conn.execute(
        "select * from taste_profiles where profile_id = ?", (profile_id,)
    ).fetchone()
    if row is None:
        return None
    modes = _load_modes(conn, profile_id)
    return TasteProfile(
        profile_id=profile_id,
        tier=cast("Tier", row["tier"]),
        cutoff_date=date.fromisoformat(str(row["cutoff_date"])) if row["cutoff_date"] else None,
        index_id=str(row["index_id"]),
        doc_template_version=str(row["doc_template_version"]),
        n_rated=int(row["n_rated"]),
        n_reliable_dated=int(row["n_reliable_dated"]),
        rating_histogram=orjson.loads(str(row["histogram_json"])),
        modes=modes["like"],
        anti_modes=modes["dislike"],
        direction=_load_direction(row),
        affinities=_load_affinities(conn, profile_id),
        calibrator=orjson.loads(str(row["calibrator_json"])),
        alpha=float(row["alpha"]),
        params_sha=str(row["params_sha"]),
        built_at=str(row["built_at"]),
        stale=bool(row["stale"]),
        stale_reason=None if row["stale_reason"] is None else str(row["stale_reason"]),
        split_name=None if row["split_name"] is None else str(row["split_name"]),
        fold=None if row["fold"] is None else int(row["fold"]),
    )


def _direction(
    conn: sqlite3.Connection,
    signals: Sequence[PreferenceSignal],
    vectors: Mapping[int, Sequence[float]],
    dim: int,
) -> PreferenceDirection | None:
    rows = [s for s in signals if s.tmdb_id in vectors]
    if len(rows) < MIN_RIDGE_N:
        return None
    meta, meta_names = metadata_features(conn, [s.tmdb_id for s in rows])
    design = np.hstack([np.array([vectors[s.tmdb_id] for s in rows], dtype=np.float64), meta])
    names = (*(f"emb:{i}" for i in range(dim)), *meta_names)
    return fit_preference_direction(design, np.array([s.s for s in rows]), feature_names=names)


def _params_sha(index_id: str, alpha: float, seed: int, cutoff: date | None) -> str:
    return short_hash(
        {
            "index_id": index_id,
            "alpha": alpha,
            "seed": seed,
            "cutoff": cutoff.isoformat() if cutoff else None,
            "kappa": KAPPA,
            "min_ridge_n": MIN_RIDGE_N,
            "tiers": [THIN_FLOOR, FULL_FLOOR],
        }
    )


def _persist(db: Database, profile: TasteProfile) -> None:
    direction = profile.direction
    with db.write() as conn:
        conn.execute(
            "insert into taste_profiles (profile_id, built_at, cutoff_date, index_id, "
            "doc_template_version, split_name, fold, tier, n_rated, n_reliable_dated, alpha, "
            "calibrator_json, histogram_json, direction_blob, direction_meta_json, "
            "leverage_diag_blob, params_sha) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                profile.profile_id,
                profile.built_at,
                profile.cutoff_date.isoformat() if profile.cutoff_date else None,
                profile.index_id,
                profile.doc_template_version,
                profile.split_name,
                profile.fold,
                profile.tier,
                profile.n_rated,
                profile.n_reliable_dated,
                profile.alpha,
                orjson.dumps(profile.calibrator).decode(),
                orjson.dumps(profile.rating_histogram).decode(),
                None if direction is None else pack_f32(direction.w.tolist()),
                None if direction is None else orjson.dumps(_direction_meta(direction)).decode(),
                None if direction is None else pack_f32(direction.leverage_diag.tolist()),
                profile.params_sha,
            ),
        )
        conn.executemany(
            "insert into taste_modes (profile_id, polarity, mode_id, centroid, mass, n_members, "
            "mean_signal, coherence, confidence, exemplars_json) values (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    profile.profile_id,
                    mode.polarity,
                    mode.mode_id,
                    pack_f32(mode.centroid.tolist()),
                    mode.mass,
                    mode.n_members,
                    mode.mean_signal,
                    mode.coherence,
                    mode.confidence,
                    orjson.dumps(list(mode.exemplars)).decode(),
                )
                for mode in (*profile.modes, *profile.anti_modes)
            ],
        )
        conn.executemany(
            "insert into taste_mode_members (profile_id, polarity, mode_id, tmdb_id, cosine, "
            "signal) values (?,?,?,?,?,?)",
            [
                (profile.profile_id, mode.polarity, mode.mode_id, m.tmdb_id, m.cosine, m.signal)
                for mode in (*profile.modes, *profile.anti_modes)
                for m in mode.members
            ],
        )
        conn.executemany(
            "insert into taste_affinities (profile_id, kind, entity_id, name, n, raw_sum, "
            "affinity, exposure_logodds, support_json) values (?,?,?,?,?,?,?,?,?)",
            [
                (
                    profile.profile_id,
                    row.kind,
                    row.entity_id,
                    row.name,
                    row.n,
                    row.raw_sum,
                    row.affinity,
                    row.exposure_logodds,
                    orjson.dumps(list(row.support_films)).decode(),
                )
                for rows in profile.affinities.values()
                for row in rows
            ],
        )


def _direction_meta(direction: PreferenceDirection) -> dict[str, Any]:
    return {
        "b": direction.b,
        "lam": direction.lam,
        "sigma2": direction.sigma2,
        "n_fit": direction.n_fit,
        "loocv_r2": direction.loocv_r2,
        "feature_names": list(direction.feature_names),
        "x_mean": [float(v) for v in direction.x_mean],
    }


def _load_direction(row: sqlite3.Row) -> PreferenceDirection | None:
    if row["direction_blob"] is None or row["direction_meta_json"] is None:
        return None
    meta = orjson.loads(str(row["direction_meta_json"]))
    return PreferenceDirection(
        w=np.array(unpack_f32(bytes(row["direction_blob"])), dtype=np.float64),
        b=float(meta["b"]),
        lam=float(meta["lam"]),
        leverage_diag=np.array(unpack_f32(bytes(row["leverage_diag_blob"])), dtype=np.float64),
        x_mean=np.array(meta["x_mean"], dtype=np.float64),
        sigma2=float(meta["sigma2"]),
        n_fit=int(meta["n_fit"]),
        loocv_r2=float(meta["loocv_r2"]),
        feature_names=tuple(meta["feature_names"]),
    )


def _load_modes(conn: sqlite3.Connection, profile_id: str) -> dict[str, tuple[TasteMode, ...]]:
    members: dict[tuple[str, int], list[ModeMember]] = {}
    for row in conn.execute(
        "select polarity, mode_id, tmdb_id, cosine, signal from taste_mode_members "
        "where profile_id = ?",
        (profile_id,),
    ):
        key = (str(row["polarity"]), int(row["mode_id"]))
        members.setdefault(key, []).append(
            ModeMember(int(row["tmdb_id"]), float(row["cosine"]), float(row["signal"]))
        )
    out: dict[str, list[TasteMode]] = {"like": [], "dislike": []}
    for row in conn.execute(
        "select * from taste_modes where profile_id = ? order by polarity, mode_id",
        (profile_id,),
    ):
        polarity: Polarity = "like" if row["polarity"] == "like" else "dislike"
        mode_id = int(row["mode_id"])
        out[polarity].append(
            TasteMode(
                mode_id=mode_id,
                polarity=polarity,
                centroid=np.array(unpack_f32(bytes(row["centroid"])), dtype=np.float32),
                mass=float(row["mass"]),
                n_members=int(row["n_members"]),
                mean_signal=float(row["mean_signal"]),
                coherence=float(row["coherence"]),
                confidence=float(row["confidence"]),
                exemplars=tuple(orjson.loads(str(row["exemplars_json"]))),
                members=tuple(members.get((polarity, mode_id), ())),
                label=None if row["label"] is None else str(row["label"]),
            )
        )
    return {key: tuple(value) for key, value in out.items()}


def _load_affinities(
    conn: sqlite3.Connection, profile_id: str
) -> dict[str, tuple[EntityAffinity, ...]]:
    out: dict[str, list[EntityAffinity]] = {}
    for row in conn.execute(
        "select * from taste_affinities where profile_id = ? order by kind, affinity desc",
        (profile_id,),
    ):
        out.setdefault(str(row["kind"]), []).append(
            EntityAffinity(
                kind=str(row["kind"]),
                entity_id=str(row["entity_id"]),
                name=str(row["name"]),
                n=int(row["n"]),
                raw_sum=float(row["raw_sum"]),
                affinity=float(row["affinity"]),
                exposure_logodds=float(row["exposure_logodds"]),
                support_films=tuple(orjson.loads(str(row["support_json"]))),
            )
        )
    return {key: tuple(value) for key, value in out.items()}
