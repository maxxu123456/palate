"""Forward-only numbered migrations, each checksummed and applied in its own transaction."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from palate.clock import now_iso
from palate.db.connect import Database
from palate.errors import MigrationChecksumMismatch, StorageError
from palate.hashing import sha256_hex

_FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

_BOOTSTRAP = """
create table if not exists schema_migrations (
  version    integer primary key,
  name       text not null,
  checksum   text not null,
  applied_at text not null
) strict
"""


@dataclass(frozen=True, slots=True)
class Migration:
    """One numbered SQL file and the hash of its text."""

    version: int
    name: str
    sql: str
    checksum: str


@dataclass(frozen=True, slots=True)
class MigrateReport:
    """What one migrate() call did."""

    applied: tuple[int, ...]
    already: tuple[int, ...]
    version: int


def discover(directory: Path) -> list[Migration]:
    """Read every NNNN_name.sql in the directory, ordered by version."""
    found: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if match is None:
            raise StorageError(f"migration file {path.name} is not NNNN_name.sql")
        version = int(match.group(1))
        text = path.read_text(encoding="utf-8")
        found[version] = Migration(version, match.group(2), text, sha256_hex(text))
    ordered = [found[v] for v in sorted(found)]
    for expected, migration in enumerate(ordered, start=1):
        if migration.version != expected:
            raise StorageError(f"migration versions jump from {expected} to {migration.version}")
    return ordered


def applied(conn: sqlite3.Connection) -> dict[int, str]:
    """Version to checksum for everything already on this database."""
    conn.execute(_BOOTSTRAP)
    rows = conn.execute("select version, checksum from schema_migrations order by version")
    return {int(row["version"]): str(row["checksum"]) for row in rows}


def current_version(conn: sqlite3.Connection) -> int:
    """Highest applied migration version, 0 on a fresh database."""
    return max(applied(conn), default=0)


def migrate(db: Database, directory: Path, *, target: int | None = None) -> MigrateReport:
    """Apply pending migrations in order under the write lock.

    Raises MigrationChecksumMismatch if an already-applied file changed on disk.
    """
    migrations = discover(directory)
    with db.raw_writer() as conn:
        done = _read_applied(conn)
        _verify_checksums(migrations, done)
        pending = [
            m
            for m in migrations
            if m.version not in done and (target is None or m.version <= target)
        ]
        for migration in pending:
            _apply(conn, migration)
    return MigrateReport(
        applied=tuple(m.version for m in pending),
        already=tuple(sorted(done)),
        version=max([*done, *(m.version for m in pending)], default=0),
    )


def _read_applied(conn: sqlite3.Connection) -> dict[int, str]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        done = applied(conn)
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return done


def _verify_checksums(migrations: list[Migration], done: dict[int, str]) -> None:
    for migration in migrations:
        recorded = done.get(migration.version)
        if recorded is not None and recorded != migration.checksum:
            raise MigrationChecksumMismatch(
                f"{migration.version:04d}_{migration.name}.sql changed after it was applied "
                f"({recorded[:12]} -> {migration.checksum[:12]}). "
                "Migrations are forward only, add a new file instead."
            )


def _apply(conn: sqlite3.Connection, migration: Migration) -> None:
    # The BEGIN lives in the script because executescript commits any transaction it finds open.
    try:
        conn.executescript("BEGIN IMMEDIATE;\n" + migration.sql)
        conn.execute(
            "insert into schema_migrations (version, name, checksum, applied_at) values (?,?,?,?)",
            (migration.version, migration.name, migration.checksum, now_iso()),
        )
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
