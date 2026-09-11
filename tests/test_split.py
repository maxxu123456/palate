"""Catalogue days, fold geometry, the no-overlap invariant and the refusal below the floor."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from palate.db.connect import Database, open_database
from palate.errors import StaleSplitError, TemporalProtocolUnavailable
from palate.eval.split import (
    SplitSpec,
    build_split,
    detect_catalogue_days,
    freeze_split,
    load_corpus_ids,
    load_regions,
    load_split,
    mark_catalogue_days,
    ratings_fingerprint,
)
from palate.paths import migrations_dir
from palate.taste.signals import RatedFilm

START = date(2020, 1, 1)


def spread(n: int, *, first_id: int = 1, start: date = START) -> list[RatedFilm]:
    """n rated films, one every three days, every rating half star present."""
    return [
        RatedFilm(
            tmdb_id=first_id + i,
            rating_half=1 + (i % 10),
            watched_at=start + timedelta(days=3 * i),
            date_source="diary",
            date_reliable=True,
        )
        for i in range(n)
    ]


def bulk(n: int, *, first_id: int, day: date) -> list[RatedFilm]:
    """n films all stamped with the same day, which is what an import looks like."""
    return [
        RatedFilm(
            tmdb_id=first_id + i,
            rating_half=8,
            watched_at=day,
            date_source="ratings",
            date_reliable=True,
        )
        for i in range(n)
    ]


def graded(film: RatedFilm) -> tuple[int, int]:
    """Enough of a label to exercise the plumbing, the real cutpoints live elsewhere."""
    return (max(0, film.rating_half - 7), 1)


def make_db(tmp_path: Path) -> Database:
    return open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)


def test_a_catalogue_day_is_detected_and_never_reaches_test() -> None:
    rated = spread(400) + bulk(300, first_id=10_000, day=date(2021, 6, 5))
    days = detect_catalogue_days(rated)
    assert days == {date(2021, 6, 5)}
    ids = frozenset(r.tmdb_id for r in rated)
    split = build_split(rated, ids, SplitSpec(name="s"), relevance=graded)
    assert split.catalogue_days == {"2021-06-05": 300}
    assert split.n_dropped_unreliable == 300
    for fold in split.folds:
        assert not set(fold.test) & {r.tmdb_id for r in rated if r.tmdb_id >= 10_000}
        assert {r.tmdb_id for r in rated if r.tmdb_id >= 10_000} <= set(fold.train)


def test_rolling_origin_cuts_five_folds_that_march_forward() -> None:
    rated = spread(500)
    ids = frozenset(r.tmdb_id for r in rated)
    split = build_split(rated, ids, SplitSpec(name="s"), relevance=graded)
    assert split.strategy == "rolling_origin"
    assert len(split.folds) == 5
    assert [f.fold for f in split.folds] == [1, 2, 3, 4, 5]
    splits = [f.t_split for f in split.folds]
    assert splits == sorted(splits)
    # Fold 4 is the conventional 80/20 cut, which is the number readers expect to see.
    assert len(split.folds[3].train) == pytest.approx(0.8 * 500, abs=5)
    for fold in split.folds:
        assert all(fold.t_split < r for r in _dates(rated, fold.test))
        assert all(r <= fold.t_end for r in _dates(rated, fold.test))


def _dates(rated: list[RatedFilm], ids: tuple[int, ...]) -> list[date]:
    lookup = {r.tmdb_id: r.watched_at for r in rated}
    return [d for i in ids if (d := lookup[i]) is not None]


def test_train_and_test_never_share_a_film() -> None:
    rated = spread(450)
    split = build_split(
        rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"), relevance=graded
    )
    for fold in split.folds:
        assert not set(fold.train) & set(fold.test)
        assert set(fold.inner) & set(fold.val) == set()
        assert len(fold.train) == len(fold.inner) + len(fold.val)


def test_inner_is_the_earlier_part_of_train() -> None:
    rated = spread(400)
    split = build_split(
        rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"), relevance=graded
    )
    fold = split.folds[2]
    assert max(_dates(rated, fold.inner)) <= min(_dates(rated, fold.val))
    assert len(fold.inner) == pytest.approx(0.8 * len(fold.train), abs=2)


def test_a_rewatch_is_train_only() -> None:
    rated = spread(400)
    rated[380] = replace(rated[380], is_rewatch=True)
    marked = rated[380].tmdb_id
    split = build_split(
        rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"), relevance=graded
    )
    assert split.n_dropped_rewatch == 1
    for fold in split.folds:
        assert marked not in fold.test
    dropped = [
        a
        for fold in split.folds
        for a in fold.assignments
        if a.tmdb_id == marked and a.reason == "rewatch_dup"
    ]
    assert dropped


def test_a_film_outside_the_corpus_is_dropped_and_counted() -> None:
    rated = spread(400)
    missing = {r.tmdb_id for r in rated[300:340]}
    corpus = frozenset(r.tmdb_id for r in rated) - missing
    regions = {r.tmdb_id: "HK" if r.tmdb_id in missing else "US" for r in rated}
    split = build_split(rated, corpus, SplitSpec(name="s"), region_of=regions, relevance=graded)
    for fold in split.folds:
        assert not set(fold.test) & missing
    assert split.n_dropped_not_in_corpus == 40
    assert split.test_coverage == pytest.approx(360 / 400)
    assert split.coverage_by_region["HK"] == 0.0
    assert split.coverage_by_region["US"] == 1.0


def test_labels_and_idcg_come_from_the_frozen_assignment() -> None:
    rated = spread(400)
    split = build_split(
        rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"), relevance=graded
    )
    fold = split.folds[0]
    assert set(fold.rel.values()) <= {0, 1, 2, 3}
    assert fold.n_test_pos + fold.n_test_neg <= len(fold.test)
    assert fold.idcg10 > 0.0
    assert all(v == 1 for v in fold.rel_watch.values())


def test_too_few_reliable_dates_refuses_the_temporal_protocol() -> None:
    rated = spread(80) + bulk(400, first_id=50_000, day=date(2022, 2, 2))
    split = build_split(
        rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"), relevance=graded
    )
    assert split.degraded is True
    assert split.strategy == "leave_last_k"
    assert len(split.folds) == 1
    assert split.degraded_reason is not None
    assert split.degraded_reason.startswith("TEMPORAL PROTOCOL REFUSED.")
    assert "80 of 480" in split.degraded_reason
    assert split.spec.strategy == "rolling_origin"


def test_no_reliable_date_at_all_is_an_error() -> None:
    rated = bulk(50, first_id=1, day=date(2022, 2, 2))
    with pytest.raises(TemporalProtocolUnavailable):
        build_split(rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"))


def test_underpowered_folds_are_flagged_not_hidden() -> None:
    rated = spread(320)
    split = build_split(
        rated, frozenset(r.tmdb_id for r in rated), SplitSpec(name="s"), relevance=graded
    )
    fold = split.folds[0]
    assert fold.underpowered == (fold.n_test_pos < 20 or len(fold.test) < 25)


def test_freeze_then_load_reproduces_the_split(tmp_path: Path) -> None:
    rated = spread(400)
    ids = frozenset(r.tmdb_id for r in rated)
    split = build_split(rated, ids, SplitSpec(name="rolling"), relevance=graded)
    db = make_db(tmp_path)
    freeze_split(split, db)
    back = load_split(db, "rolling", ratings_sha256=ratings_fingerprint(rated))
    assert back.strategy == split.strategy
    assert back.spec == split.spec
    assert back.test_per_fold == split.test_per_fold
    assert [f.rel for f in back.folds] == [f.rel for f in split.folds]
    assert [f.train for f in back.folds] == [f.train for f in split.folds]
    assert back.catalogue_days == split.catalogue_days
    assert back.test_coverage == pytest.approx(split.test_coverage)
    db.close()


def test_freezing_twice_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    rated = spread(400)
    ids = frozenset(r.tmdb_id for r in rated)
    split = build_split(rated, ids, SplitSpec(name="rolling"), relevance=graded)
    db = make_db(tmp_path)
    freeze_split(split, db)
    freeze_split(split, db)
    n = db.read().execute("select count(*) from eval_fold").fetchone()[0]
    assert n == len(split.folds)
    db.close()


def test_a_new_export_invalidates_the_frozen_split(tmp_path: Path) -> None:
    rated = spread(400)
    ids = frozenset(r.tmdb_id for r in rated)
    db = make_db(tmp_path)
    freeze_split(build_split(rated, ids, SplitSpec(name="rolling"), relevance=graded), db)
    grown = [*rated, *spread(5, first_id=9000, start=date(2024, 1, 1))]
    with pytest.raises(StaleSplitError):
        load_split(db, "rolling", ratings_sha256=ratings_fingerprint(grown))
    db.close()


def test_an_unknown_split_name_is_refused(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    with pytest.raises(StaleSplitError):
        load_split(db, "nothing", ratings_sha256="0" * 64)
    db.close()


def test_marking_catalogue_days_moves_date_reliable(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    with db.write() as conn:
        for tmdb_id, day in ((1, "2021-06-05"), (2, "2021-06-05"), (3, "2021-07-01")):
            conn.execute(
                "insert into films (tmdb_id, title, year, fetched_at) values (?,?,?,?)",
                (tmdb_id, f"Film {tmdb_id}", 2000, "2026-09-11T00:00:00"),
            )
            conn.execute(
                "insert into user_films (tmdb_id, rating_half, watched_date, date_source) "
                "values (?,?,?,'diary')",
                (tmdb_id, 8, day),
            )
    assert mark_catalogue_days(db, [date(2021, 6, 5)]) == 2
    rows = db.read().execute("select tmdb_id, date_reliable from user_films order by tmdb_id")
    assert [(int(r["tmdb_id"]), int(r["date_reliable"])) for r in rows] == [(1, 0), (2, 0), (3, 1)]
    assert mark_catalogue_days(db, []) == 0
    db.close()


def test_corpus_and_region_loaders_read_the_real_tables(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    with db.write() as conn:
        conn.execute("insert into countries (iso_3166_1, name) values ('HK','Hong Kong')")
        for tmdb_id in (1, 2):
            conn.execute(
                "insert into films (tmdb_id, title, fetched_at) values (?,?,?)",
                (tmdb_id, f"Film {tmdb_id}", "2026-09-11T00:00:00"),
            )
            conn.execute(
                "insert into corpus_members (tmdb_id, source, eligible, added_at) "
                "values (?,'discover',?,?)",
                (tmdb_id, int(tmdb_id == 1), "2026-09-11T00:00:00"),
            )
        conn.execute("insert into film_countries (tmdb_id, iso_3166_1) values (1,'HK')")
    assert load_corpus_ids(db.read()) == frozenset({1})
    regions = load_regions(db.read())
    assert regions[1] == "HK"
    assert regions[2] == "??"
    db.close()
