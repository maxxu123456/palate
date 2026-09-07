"""Migrations apply once, are idempotent, and refuse to run over an edited file."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from palate.db.connect import Database, open_database
from palate.db.migrate import applied, current_version, discover, migrate
from palate.errors import MigrationChecksumMismatch, StorageError
from palate.paths import migrations_dir

MIGRATIONS = migrations_dir()


def make_db(tmp_path: Path, name: str = "palate.db", migrations: Path = MIGRATIONS) -> Database:
    return Database(tmp_path / name, migrations=migrations, load_vec=False)


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("select name from sqlite_master where type = 'table'")
    return {str(r["name"]) for r in rows}


def schema_text(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        "select type, name, coalesce(sql, '') as sql from sqlite_master "
        "where name not like 'sqlite_%' order by type, name"
    )
    return [(str(r["type"]), str(r["name"]), str(r["sql"])) for r in rows]


def test_discover_orders_and_hashes() -> None:
    found = discover(MIGRATIONS)
    assert [m.version for m in found] == [1, 2, 3]
    assert [m.name for m in found] == ["raw", "core", "user"]
    assert len({m.checksum for m in found}) == 3
    assert all(len(m.checksum) == 64 for m in found)


def test_discover_rejects_a_gap(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("select 1;")
    (tmp_path / "0003_c.sql").write_text("select 1;")
    with pytest.raises(StorageError):
        discover(tmp_path)


def test_discover_rejects_a_bad_filename(tmp_path: Path) -> None:
    (tmp_path / "initial.sql").write_text("select 1;")
    with pytest.raises(StorageError):
        discover(tmp_path)


def test_migrate_creates_every_table(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    report = migrate(db, MIGRATIONS)
    assert report.applied == (1, 2, 3)
    assert report.version == 3
    names = table_names(db.read())
    assert {"films", "credits", "user_films", "title_resolutions", "tmdb_raw"} <= names
    db.close()


def test_migrate_twice_is_a_no_op(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    migrate(db, MIGRATIONS)
    first = schema_text(db.read())
    report = migrate(db, MIGRATIONS)
    assert report.applied == ()
    assert report.already == (1, 2, 3)
    assert schema_text(db.read()) == first
    db.close()


def test_applying_a_prefix_then_the_rest_matches_applying_all(tmp_path: Path) -> None:
    stepped = make_db(tmp_path, "stepped.db")
    migrate(stepped, MIGRATIONS, target=1)
    assert current_version(stepped.read()) == 1
    migrate(stepped, MIGRATIONS, target=2)
    migrate(stepped, MIGRATIONS)
    whole = make_db(tmp_path, "whole.db")
    migrate(whole, MIGRATIONS)
    assert schema_text(stepped.read()) == schema_text(whole.read())
    stepped.close()
    whole.close()


def test_edited_migration_is_refused(tmp_path: Path) -> None:
    work = tmp_path / "migrations"
    work.mkdir()
    for path in sorted(MIGRATIONS.glob("*.sql")):
        (work / path.name).write_text(path.read_text())
    db = make_db(tmp_path, migrations=work)
    migrate(db, work)
    target = work / "0002_core.sql"
    target.write_text(target.read_text() + "\ncreate table sneaky(x integer) strict;\n")
    with pytest.raises(MigrationChecksumMismatch) as exc:
        migrate(db, work)
    assert "0002_core.sql" in str(exc.value)
    assert "sneaky" not in table_names(db.read())
    db.close()


def test_a_failing_migration_leaves_nothing_behind(tmp_path: Path) -> None:
    work = tmp_path / "migrations"
    work.mkdir()
    (work / "0001_a.sql").write_text("create table good(x integer) strict;")
    (work / "0002_b.sql").write_text("create table halfway(x integer) strict;\nthis is not sql;\n")
    db = make_db(tmp_path, migrations=work)
    with pytest.raises(sqlite3.Error):
        migrate(db, work)
    names = table_names(db.read())
    assert "good" in names
    assert "halfway" not in names
    assert current_version(db.read()) == 1
    db.close()


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    db = open_database(tmp_path / "palate.db", migrations=MIGRATIONS, load_vec=False)
    with pytest.raises(sqlite3.IntegrityError), db.write() as conn:
        conn.execute(
            "insert into user_films (tmdb_id, date_source) values (?, ?)", (999_999, "none")
        )
    db.close()


def test_rating_half_must_be_an_integer_in_range(tmp_path: Path) -> None:
    db = open_database(tmp_path / "palate.db", migrations=MIGRATIONS, load_vec=False)
    with db.write() as conn:
        conn.execute(
            "insert into films (tmdb_id, title, fetched_at) values (?, ?, ?)",
            (1, "Stalker", "2026-09-07T00:00:00"),
        )
    with pytest.raises(sqlite3.IntegrityError), db.write() as conn:
        conn.execute(
            "insert into user_films (tmdb_id, rating_half, date_source) values (?, ?, ?)",
            (1, 11, "none"),
        )
    with pytest.raises(sqlite3.IntegrityError), db.write() as conn:
        conn.execute(
            "insert into user_films (tmdb_id, rating_half, date_source) values (?, ?, ?)",
            (1, 4.5, "none"),
        )
    db.close()


def test_generated_columns_bucket_a_film(tmp_path: Path) -> None:
    db = open_database(tmp_path / "palate.db", migrations=MIGRATIONS, load_vec=False)
    with db.write() as conn:
        conn.execute(
            "insert into films (tmdb_id, title, year, runtime, fetched_at) values (?,?,?,?,?)",
            (1, "Stalker", 1979, 161, "2026-09-07T00:00:00"),
        )
    row = db.read().execute("select decade, runtime_bucket from films where tmdb_id = 1").fetchone()
    assert row["decade"] == 1970
    assert row["runtime_bucket"] == 4
    db.close()


def test_write_rolls_back_on_failure(tmp_path: Path) -> None:
    db = open_database(tmp_path / "palate.db", migrations=MIGRATIONS, load_vec=False)
    with pytest.raises(RuntimeError), db.write() as conn:
        conn.execute("insert into app_state (key, value, updated_at) values ('k','v','t')")
        raise RuntimeError("boom")
    assert db.read().execute("select count(*) from app_state").fetchone()[0] == 0
    db.close()


def test_ensure_migrated_runs_once(tmp_path: Path) -> None:
    db = Database(tmp_path / "palate.db", migrations=MIGRATIONS, load_vec=False)
    db.ensure_migrated()
    db.ensure_migrated()
    rows = applied(db.read())
    assert sorted(rows) == [1, 2, 3]
    db.close()


def test_sqlite_vec_loads(tmp_path: Path) -> None:
    db = open_database(tmp_path / "palate.db", migrations=MIGRATIONS, load_vec=True)
    version = db.read().execute("select vec_version()").fetchone()[0]
    assert isinstance(version, str)
    db.close()
