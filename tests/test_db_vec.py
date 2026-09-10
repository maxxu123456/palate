"""sqlite-vec only checks dimension, so everything else is checked here."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from palate.db.connect import Database, open_database
from palate.db.sqlvec import probe, serialize_f32
from palate.index.vecstore import (
    METADATA_COLUMNS,
    MetadataFilter,
    VecRow,
    VecStore,
    choose_prefilter,
)
from palate.paths import migrations_dir


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=True)
    yield database
    database.close()


def store(db: Database, *, dim: int = 4) -> VecStore:
    with db.write() as conn:
        vec = VecStore(conn, table="vec_films_test", dim=dim)
        vec.create()
    return VecStore(db.read(), table="vec_films_test", dim=dim)


def row(film_id: int, vector: tuple[float, ...], **kwargs: object) -> VecRow:
    return VecRow(film_id=film_id, embedding=vector, **kwargs)  # type: ignore[arg-type]


def test_the_capability_probe_finds_the_real_metadata_ceiling(db: Database) -> None:
    capability = probe(db.read())
    assert capability.vec_version.startswith("v")
    assert capability.fts5 is True
    # The seventeenth column is where vec0 stops, and we use ten plus a partition key.
    assert capability.max_metadata_columns == 16
    assert len(METADATA_COLUMNS) + 1 <= capability.max_metadata_columns


def test_a_probe_without_the_extension_says_what_to_install(tmp_path: Path) -> None:
    plain = open_database(tmp_path / "plain.db", migrations=migrations_dir(), load_vec=False)
    try:
        with pytest.raises(Exception) as exc:
            probe(plain.read())
        assert "uv python install" in str(exc.value)
    finally:
        plain.close()


def test_serialize_is_four_bytes_a_float() -> None:
    assert len(serialize_f32([0.0, 1.0, 2.0])) == 12


def test_nearest_neighbours_come_back_closest_first(db: Database) -> None:
    vec = store(db)
    with db.write() as conn:
        VecStore(conn, table="vec_films_test", dim=4).upsert(
            [
                row(1, (1.0, 0.0, 0.0, 0.0)),
                row(2, (0.9, 0.1, 0.0, 0.0)),
                row(3, (0.0, 1.0, 0.0, 0.0)),
            ]
        )
    hits = vec.knn((1.0, 0.0, 0.0, 0.0), k=3)
    assert [h.film_id for h in hits] == [1, 2, 3]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_a_vector_of_the_wrong_width_is_refused(db: Database) -> None:
    store(db)
    with pytest.raises(sqlite3.Error), db.write() as conn:
        VecStore(conn, table="vec_films_test", dim=4).upsert([row(1, (1.0, 0.0))])


def test_an_allow_list_is_a_real_prefilter(db: Database) -> None:
    vec = store(db)
    with db.write() as conn:
        VecStore(conn, table="vec_films_test", dim=4).upsert(
            [row(i, (1.0, 0.0, 0.0, 0.0)) for i in (1, 2, 3)]
        )
    assert {h.film_id for h in vec.knn((1.0, 0.0, 0.0, 0.0), k=3, allow=[2, 3])} == {2, 3}
    assert {h.film_id for h in vec.knn((1.0, 0.0, 0.0, 0.0), k=3, exclude=[1])} == {2, 3}


def test_the_two_universal_filters_are_metadata_columns(db: Database) -> None:
    vec = store(db)
    with db.write() as conn:
        VecStore(conn, table="vec_films_test", dim=4).upsert(
            [
                row(1, (1.0, 0.0, 0.0, 0.0), is_watched=1, in_corpus=1),
                row(2, (1.0, 0.0, 0.0, 0.0), is_watched=0, in_corpus=1),
                row(3, (1.0, 0.0, 0.0, 0.0), is_watched=0, in_corpus=0),
            ]
        )
    unwatched = vec.knn(
        (1.0, 0.0, 0.0, 0.0), k=5, where=MetadataFilter.of(is_watched=0, in_corpus=1)
    )
    assert [h.film_id for h in unwatched] == [2]


def test_a_range_predicate_is_pushed_into_the_knn(db: Database) -> None:
    vec = store(db)
    with db.write() as conn:
        VecStore(conn, table="vec_films_test", dim=4).upsert(
            [
                row(1, (1.0, 0.0, 0.0, 0.0), year=1979, decade=1970),
                row(2, (1.0, 0.0, 0.0, 0.0), year=2011, decade=2010),
            ]
        )
    older = vec.knn((1.0, 0.0, 0.0, 0.0), k=5, where=MetadataFilter((("year", "lt", 2000),)))
    assert [h.film_id for h in older] == [1]


def test_a_predicate_on_a_column_vec0_does_not_have_is_refused(db: Database) -> None:
    vec = store(db)
    with pytest.raises(ValueError, match="genre"):
        vec.knn((1.0, 0.0, 0.0, 0.0), k=1, where=MetadataFilter((("genre", "eq", 18),)))


def test_an_upsert_replaces_rather_than_duplicates(db: Database) -> None:
    vec = store(db)
    for value in (1.0, -1.0):
        with db.write() as conn:
            VecStore(conn, table="vec_films_test", dim=4).upsert([row(1, (value, 0.0, 0.0, 0.0))])
    assert vec.count() == 1
    assert vec.knn((-1.0, 0.0, 0.0, 0.0), k=1)[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_a_knn_with_no_k_never_leaves_this_module(db: Database) -> None:
    vec = store(db)
    assert vec.knn((1.0, 0.0, 0.0, 0.0), k=0) == ()


def test_a_bad_table_name_is_refused_before_it_reaches_sql(db: Database) -> None:
    with pytest.raises(ValueError, match="bad vec table name"):
        VecStore(db.read(), table="vec; drop table films", dim=4)


def test_the_prefilter_path_follows_the_set_size() -> None:
    # A small allow list is exact and cheap.
    assert choose_prefilter(400, 40_000).path == "allow_json"
    # Almost everything allowed, so the excluded side is the small one.
    assert choose_prefilter(39_000, 40_000).path == "exclude_json"
    # The common case is neither, which is why the naive rule almost never fires.
    plan = choose_prefilter(20_000, 40_000)
    assert plan.path == "metadata_overfetch"
    assert plan.overfetch == pytest.approx(2.0)


def test_the_overfetch_factor_is_clamped() -> None:
    assert choose_prefilter(6_000, 100_000).overfetch == pytest.approx(6.0)
    assert choose_prefilter(90_000, 100_000, id_cap=10).overfetch == pytest.approx(2.0)
