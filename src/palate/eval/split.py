"""Rolling origin holdouts over watch time, and the catalogue days that make them honest."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal, cast

import orjson

from palate.clock import now_iso
from palate.db.connect import Database
from palate.errors import EvalError, StaleSplitError, TemporalProtocolUnavailable
from palate.eval.metrics import ideal_dcg
from palate.hashing import canonical_json, sha256_hex
from palate.taste.signals import RatedFilm

type Strategy = Literal["rolling_origin", "single_temporal", "leave_last_k"]
type Bucket = Literal["inner", "val", "test", "excluded"]

# A rated film to its (graded, watch) labels. Owned by the labels module, not by the geometry.
type Relevance = Callable[[RatedFilm], tuple[int, int]]

# Below this a fold's NDCG is one lucky film away from anything, so it is reported not headlined.
UNDERPOWERED_POSITIVES = 20

POSITIVE_REL = 2

SINGLE_CUT = 0.8

NDCG_K = 10


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Every knob of the holdout protocol, frozen with the split it produced."""

    name: str
    strategy: Strategy = "rolling_origin"
    cuts: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
    inner_frac: float = 0.8
    catalogue_day_threshold: float = 0.02
    min_reliable: int = 300
    min_test_per_fold: int = 25
    leave_last_k: int = 100
    seed: int = 0

    def to_row(self) -> dict[str, Any]:
        """The json shape stored beside the split."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "cuts": list(self.cuts),
            "inner_frac": self.inner_frac,
            "catalogue_day_threshold": self.catalogue_day_threshold,
            "min_reliable": self.min_reliable,
            "min_test_per_fold": self.min_test_per_fold,
            "leave_last_k": self.leave_last_k,
            "seed": self.seed,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SplitSpec:
        """Rebuild from the stored json."""
        return cls(
            name=str(row["name"]),
            strategy=cast("Strategy", row["strategy"]),
            cuts=tuple(float(c) for c in row["cuts"]),
            inner_frac=float(row["inner_frac"]),
            catalogue_day_threshold=float(row["catalogue_day_threshold"]),
            min_reliable=int(row["min_reliable"]),
            min_test_per_fold=int(row["min_test_per_fold"]),
            leave_last_k=int(row["leave_last_k"]),
            seed=int(row["seed"]),
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    """Where one rated film landed in one fold, and why it landed there."""

    tmdb_id: int
    bucket: Bucket
    rel: int = 0
    rel_watch: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Fold:
    """One origin: everything at or before t_split trains, (t_split, t_end] is judged."""

    fold: int
    t_start: date
    t_split: date
    t_end: date
    assignments: tuple[Assignment, ...]
    min_test: int = 25

    def bucket(self, name: Bucket) -> tuple[int, ...]:
        """Ids in one bucket, in assignment order."""
        return tuple(a.tmdb_id for a in self.assignments if a.bucket == name)

    @property
    def inner(self) -> tuple[int, ...]:
        return self.bucket("inner")

    @property
    def val(self) -> tuple[int, ...]:
        return self.bucket("val")

    @property
    def test(self) -> tuple[int, ...]:
        return self.bucket("test")

    @property
    def train(self) -> tuple[int, ...]:
        """Inner plus validation, which is what the final refit is allowed to see."""
        return tuple(a.tmdb_id for a in self.assignments if a.bucket in ("inner", "val"))

    @property
    def rel(self) -> dict[int, int]:
        """Graded relevance over the test bucket, hard negatives included at zero."""
        return {a.tmdb_id: a.rel for a in self.assignments if a.bucket == "test"}

    @property
    def rel_watch(self) -> dict[int, int]:
        """Binary watch labels over the same bucket."""
        return {a.tmdb_id: a.rel_watch for a in self.assignments if a.bucket == "test"}

    @property
    def n_test_pos(self) -> int:
        return sum(1 for a in self.assignments if a.bucket == "test" and a.rel >= POSITIVE_REL)

    @property
    def n_test_neg(self) -> int:
        return sum(1 for a in self.assignments if a.bucket == "test" and a.rel == 0)

    @property
    def idcg10(self) -> float:
        """The best DCG this fold's labels allow, so every arm divides by the same number."""
        return ideal_dcg(self.rel.values(), NDCG_K)

    @property
    def underpowered(self) -> bool:
        """Too few positives or too few items for the headline to mean anything."""
        return self.n_test_pos < UNDERPOWERED_POSITIVES or len(self.test) < self.min_test


@dataclass(frozen=True, slots=True)
class Split:
    """A frozen protocol: the folds, the counts behind them and every caveat they carry."""

    name: str
    spec: SplitSpec
    strategy: Strategy
    folds: tuple[Fold, ...]
    ratings_sha256: str
    corpus_size: int
    n_reliable: int
    catalogue_days: Mapping[str, int] = field(default_factory=dict)
    n_dropped_unreliable: int = 0
    n_dropped_rewatch: int = 0
    n_dropped_not_in_corpus: int = 0
    test_coverage: float = 1.0
    coverage_by_region: Mapping[str, float] = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str | None = None
    created_at: str = ""

    @property
    def test_per_fold(self) -> tuple[int, ...]:
        return tuple(len(f.test) for f in self.folds)


def detect_catalogue_days(rated: Sequence[RatedFilm], *, threshold: float = 0.02) -> set[date]:
    """Calendar days holding more than threshold of the whole history, which are bulk imports."""
    if not rated:
        return set()
    counts = Counter(r.watched_at for r in rated if r.watched_at is not None)
    limit = threshold * len(rated)
    return {day for day, n in counts.items() if n > limit}


def ratings_fingerprint(rated: Sequence[RatedFilm]) -> str:
    """Hash of the rating rows a split was cut from, so a new export invalidates it."""
    rows = sorted(
        (
            r.tmdb_id,
            r.rating_half,
            r.watched_at.isoformat() if r.watched_at else "",
            r.date_source,
            int(r.is_rewatch),
        )
        for r in rated
    )
    return sha256_hex(canonical_json(rows))


def _reliable(rated: Sequence[RatedFilm], days: set[date]) -> list[RatedFilm]:
    """Rows whose ordering is real: a genuine date, off every catalogue day."""
    kept = [
        r
        for r in rated
        if r.watched_at is not None and r.date_reliable and r.watched_at not in days
    ]
    kept.sort(key=lambda r: (r.watched_at or date.min, r.tmdb_id))
    return kept


def _marks(times: Sequence[date], cuts: Sequence[float]) -> list[date]:
    last = len(times) - 1
    return [times[min(last, max(0, round(c * last)))] for c in cuts]


def _windows(reliable: Sequence[RatedFilm], spec: SplitSpec) -> list[tuple[date, date]]:
    """The (t_split, t_end] pairs each fold is judged on."""
    times = [r.watched_at for r in reliable if r.watched_at is not None]
    end = times[-1]
    if spec.strategy == "leave_last_k":
        k = min(spec.leave_last_k, len(times) - 1)
        return [(times[max(0, len(times) - k - 1)], end)]
    cuts = (SINGLE_CUT,) if spec.strategy == "single_temporal" else spec.cuts
    marks = _marks(times, cuts)
    bounds = [*marks[1:], end]
    return list(zip(marks, bounds, strict=True))


def _labels(film: RatedFilm, relevance: Relevance | None) -> tuple[int, int]:
    return relevance(film) if relevance is not None else (0, 1)


def _assign(
    rated: Sequence[RatedFilm],
    reliable_ids: frozenset[int],
    corpus_ids: frozenset[int],
    window: tuple[date, date],
    spec: SplitSpec,
    relevance: Relevance | None,
) -> list[Assignment]:
    t_split, t_end = window
    train: list[RatedFilm] = []
    out: list[Assignment] = []
    for film in rated:
        when = film.watched_at
        if when is None or film.tmdb_id not in reliable_ids:
            why = "no_date" if when is None else "catalogue_day"
            out.append(Assignment(film.tmdb_id, "inner", reason=why))
            continue
        if when <= t_split:
            train.append(film)
        elif when > t_end:
            out.append(Assignment(film.tmdb_id, "excluded", reason="after_window"))
        elif film.is_rewatch:
            out.append(Assignment(film.tmdb_id, "excluded", reason="rewatch_dup"))
        elif film.tmdb_id not in corpus_ids:
            out.append(Assignment(film.tmdb_id, "excluded", reason="not_in_corpus"))
        else:
            rel, rel_watch = _labels(film, relevance)
            out.append(Assignment(film.tmdb_id, "test", rel=rel, rel_watch=rel_watch))
    cut = max(1, round(spec.inner_frac * len(train))) if train else 0
    for i, film in enumerate(train):
        out.append(Assignment(film.tmdb_id, "inner" if i < cut else "val"))
    return out


def _no_overlap(assignments: Sequence[Assignment], fold: int) -> None:
    train = {a.tmdb_id for a in assignments if a.bucket in ("inner", "val")}
    test = {a.tmdb_id for a in assignments if a.bucket == "test"}
    if train & test:
        raise EvalError(f"fold {fold} puts {len(train & test)} films in both train and test")


def _coverage(
    rated: Sequence[RatedFilm],
    reliable_ids: frozenset[int],
    corpus_ids: frozenset[int],
    region_of: Mapping[int, str] | None,
) -> tuple[float, dict[str, float], int]:
    """How many judgeable films the corpus can actually reach, overall and per region."""
    seen = [r for r in rated if r.tmdb_id in reliable_ids and not r.is_rewatch]
    if not seen:
        return 1.0, {}, 0
    missing = sum(1 for r in seen if r.tmdb_id not in corpus_ids)
    by_region: dict[str, list[int]] = {}
    for film in seen:
        region = (region_of or {}).get(film.tmdb_id, "??")
        by_region.setdefault(region, []).append(int(film.tmdb_id in corpus_ids))
    return (
        (len(seen) - missing) / len(seen),
        {r: sum(v) / len(v) for r, v in sorted(by_region.items())},
        missing,
    )


def _refusal(n_reliable: int, n_rated: int, days: Mapping[str, int], k: int) -> str:
    held = sum(days.values())
    return (
        f"TEMPORAL PROTOCOL REFUSED. Only {n_reliable} of {n_rated} rated films have a usable "
        f"date after catalogue-day removal ({len(days)} catalogue days held {held} films). "
        f"Falling back to leave_last_k over the reliable subset, k = {k}. These numbers are "
        "NOT a temporal holdout and the ordering they imply is weaker than it looks."
    )


def build_split(
    rated: Sequence[RatedFilm],
    corpus_ids: frozenset[int],
    spec: SplitSpec,
    *,
    region_of: Mapping[int, str] | None = None,
    relevance: Relevance | None = None,
) -> Split:
    """Cut the folds, or refuse the temporal protocol and say so in the split itself."""
    days = detect_catalogue_days(rated, threshold=spec.catalogue_day_threshold)
    counts = Counter(r.watched_at for r in rated if r.watched_at in days)
    day_counts = {day.isoformat(): n for day, n in sorted(counts.items()) if day is not None}
    reliable = _reliable(rated, days)
    if not reliable:
        raise TemporalProtocolUnavailable(0, spec.min_reliable)
    degraded = len(reliable) < spec.min_reliable
    strategy: Strategy = "leave_last_k" if degraded else spec.strategy
    working = replace(spec, strategy="leave_last_k") if degraded else spec
    reliable_ids = frozenset(r.tmdb_id for r in reliable)
    coverage, by_region, missing = _coverage(rated, reliable_ids, corpus_ids, region_of)
    folds: list[Fold] = []
    for number, window in enumerate(_windows(reliable, working), start=1):
        assignments = _assign(rated, reliable_ids, corpus_ids, window, working, relevance)
        _no_overlap(assignments, number)
        if not any(a.bucket == "test" for a in assignments):
            continue
        folds.append(
            Fold(
                fold=number,
                t_start=reliable[0].watched_at or window[0],
                t_split=window[0],
                t_end=window[1],
                assignments=tuple(assignments),
                min_test=working.min_test_per_fold,
            )
        )
    if not folds:
        raise TemporalProtocolUnavailable(len(reliable), spec.min_reliable)
    return Split(
        name=spec.name,
        spec=spec,
        strategy=strategy,
        folds=tuple(folds),
        ratings_sha256=ratings_fingerprint(rated),
        corpus_size=len(corpus_ids),
        n_reliable=len(reliable),
        catalogue_days=day_counts,
        n_dropped_unreliable=len(rated) - len(reliable),
        n_dropped_rewatch=sum(1 for r in rated if r.is_rewatch and r.tmdb_id in reliable_ids),
        n_dropped_not_in_corpus=missing,
        test_coverage=coverage,
        coverage_by_region=by_region,
        degraded=degraded,
        degraded_reason=(
            _refusal(len(reliable), len(rated), day_counts, working.leave_last_k)
            if degraded
            else None
        ),
        created_at=now_iso(),
    )


def mark_catalogue_days(db: Database, days: Iterable[date]) -> int:
    """Stamp date_reliable = 0 on the bulk-imported rows, so the profile counts honestly."""
    stamps = [(d.isoformat(),) for d in sorted(days)]
    with db.write() as conn:
        conn.execute("update user_films set date_reliable = 1")
        conn.executemany(
            "update user_films set date_reliable = 0 "
            "where coalesce(watched_date, logged_date) like ? || '%'",
            stamps,
        )
        row = conn.execute("select count(*) from user_films where date_reliable = 0").fetchone()
    return int(row[0])


def load_corpus_ids(conn: sqlite3.Connection) -> frozenset[int]:
    """Every film the recommender could name, which bounds what a test film can reach."""
    rows = conn.execute("select tmdb_id from corpus_members where eligible = 1")
    return frozenset(int(r["tmdb_id"]) for r in rows)


_REGIONS = (
    "select f.tmdb_id as tmdb_id, coalesce(s.primary_region, ("
    "  select c.iso_3166_1 from film_countries c where c.tmdb_id = f.tmdb_id"
    "  order by c.iso_3166_1 limit 1)) as region "
    "from films f left join film_stats s on s.tmdb_id = f.tmdb_id"
)


def load_regions(conn: sqlite3.Connection) -> dict[int, str]:
    """Production region per film, which is where a thin crawl shows up in the report."""
    return {int(r["tmdb_id"]): str(r["region"] or "??") for r in conn.execute(_REGIONS)}


def freeze_split(split: Split, db: Database) -> None:
    """Write the split down. Every arm then scores against byte identical labels."""
    with db.write() as conn:
        conn.execute("delete from eval_split where split_name = ?", (split.name,))
        conn.execute(
            "delete from eval_assignment where split_name = ?",
            (split.name,),
        )
        conn.execute(
            "insert into eval_split (split_name, strategy, spec_json, ratings_sha256, "
            "corpus_size, n_reliable, n_catalogue_days, catalogue_days_json, "
            "n_dropped_unreliable, n_dropped_rewatch, n_dropped_not_in_corpus, test_coverage, "
            "coverage_by_region_json, degraded, degraded_reason, created_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                split.name,
                split.strategy,
                orjson.dumps(split.spec.to_row()).decode(),
                split.ratings_sha256,
                split.corpus_size,
                split.n_reliable,
                len(split.catalogue_days),
                orjson.dumps(dict(split.catalogue_days)).decode(),
                split.n_dropped_unreliable,
                split.n_dropped_rewatch,
                split.n_dropped_not_in_corpus,
                split.test_coverage,
                orjson.dumps(dict(split.coverage_by_region)).decode(),
                int(split.degraded),
                split.degraded_reason,
                split.created_at or now_iso(),
            ),
        )
        conn.executemany(
            "insert into eval_fold (split_name, fold, t_start, t_split, t_end, n_train, "
            "n_inner, n_val, n_test, n_test_pos, n_test_neg, idcg10, underpowered) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    split.name,
                    fold.fold,
                    fold.t_start.isoformat(),
                    fold.t_split.isoformat(),
                    fold.t_end.isoformat(),
                    len(fold.train),
                    len(fold.inner),
                    len(fold.val),
                    len(fold.test),
                    fold.n_test_pos,
                    fold.n_test_neg,
                    fold.idcg10,
                    int(fold.underpowered),
                )
                for fold in split.folds
            ],
        )
        conn.executemany(
            "insert into eval_assignment (split_name, fold, tmdb_id, bucket, rel, rel_watch, "
            "reason) values (?,?,?,?,?,?,?)",
            [
                (split.name, fold.fold, a.tmdb_id, a.bucket, a.rel, a.rel_watch, a.reason)
                for fold in split.folds
                for a in fold.assignments
            ],
        )


def load_split(db: Database, name: str, *, ratings_sha256: str) -> Split:
    """Read a frozen split back, refusing one cut from a history that has since moved."""
    conn = db.read()
    row = conn.execute("select * from eval_split where split_name = ?", (name,)).fetchone()
    if row is None:
        raise StaleSplitError(f"no frozen split named {name!r}, run: palate eval split build")
    if str(row["ratings_sha256"]) != ratings_sha256:
        raise StaleSplitError(
            f"split {name!r} was cut from a different ratings export "
            f"({str(row['ratings_sha256'])[:12]} != {ratings_sha256[:12]})"
        )
    spec = SplitSpec.from_row(orjson.loads(str(row["spec_json"])))
    by_fold: dict[int, list[Assignment]] = {}
    for line in conn.execute(
        "select fold, tmdb_id, bucket, rel, rel_watch, reason from eval_assignment "
        "where split_name = ? order by fold, tmdb_id",
        (name,),
    ):
        by_fold.setdefault(int(line["fold"]), []).append(
            Assignment(
                tmdb_id=int(line["tmdb_id"]),
                bucket=cast("Bucket", line["bucket"]),
                rel=int(line["rel"]),
                rel_watch=int(line["rel_watch"]),
                reason=None if line["reason"] is None else str(line["reason"]),
            )
        )
    folds = tuple(
        Fold(
            fold=int(f["fold"]),
            t_start=date.fromisoformat(str(f["t_start"])),
            t_split=date.fromisoformat(str(f["t_split"])),
            t_end=date.fromisoformat(str(f["t_end"])),
            assignments=tuple(by_fold.get(int(f["fold"]), ())),
            min_test=spec.min_test_per_fold,
        )
        for f in conn.execute("select * from eval_fold where split_name = ? order by fold", (name,))
    )
    return Split(
        name=name,
        spec=spec,
        strategy=cast("Strategy", row["strategy"]),
        folds=folds,
        ratings_sha256=str(row["ratings_sha256"]),
        corpus_size=int(row["corpus_size"]),
        n_reliable=int(row["n_reliable"]),
        catalogue_days=orjson.loads(str(row["catalogue_days_json"])),
        n_dropped_unreliable=int(row["n_dropped_unreliable"]),
        n_dropped_rewatch=int(row["n_dropped_rewatch"]),
        n_dropped_not_in_corpus=int(row["n_dropped_not_in_corpus"]),
        test_coverage=float(row["test_coverage"]),
        coverage_by_region=orjson.loads(str(row["coverage_by_region_json"])),
        degraded=bool(row["degraded"]),
        degraded_reason=None if row["degraded_reason"] is None else str(row["degraded_reason"]),
        created_at=str(row["created_at"]),
    )
