"""SQLite access: per-thread readers, one process-wide writer."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from palate.errors import SqliteExtensionUnavailable

PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 268435456",
    "PRAGMA cache_size = -65536",
)

_EXTENSION_FIX = "fix it with: uv python install 3.13"


def load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into an open connection."""
    if not hasattr(conn, "enable_load_extension"):
        raise SqliteExtensionUnavailable(
            f"this Python cannot load SQLite extensions, {_EXTENSION_FIX}"
        )
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, ImportError, sqlite3.Error) as exc:
        raise SqliteExtensionUnavailable(
            f"could not load sqlite-vec ({exc}), {_EXTENSION_FIX}"
        ) from exc
    finally:
        conn.enable_load_extension(False)


class Database:
    """One SQLite file, opened once per thread for reads and behind a lock for writes."""

    def __init__(
        self,
        path: Path,
        *,
        migrations: Path,
        load_vec: bool = True,
        read_only: bool = False,
    ) -> None:
        self.path = path
        self.migrations = migrations
        self.load_vec = load_vec
        self.read_only = read_only
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._write_depth = 0
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._migrated = False
        self._attached: dict[str, Path] = {}
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self) -> sqlite3.Connection:
        # isolation_level=None turns off the implicit BEGIN entirely, so write() can
        # take the lock up front with BEGIN IMMEDIATE instead of guessing.
        if self.read_only:
            conn = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, isolation_level=None, check_same_thread=False
            )
        else:
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        for pragma in PRAGMAS:
            if self.read_only and "journal_mode" in pragma:
                continue
            conn.execute(pragma)
        if self.load_vec:
            load_vec_extension(conn)
        for alias, target in self._attached.items():
            self._attach(conn, alias, target)
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    @staticmethod
    def _attach(conn: sqlite3.Connection, alias: str, target: Path) -> None:
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (f"file:{target}?mode=ro",))

    def read(self) -> sqlite3.Connection:
        """Per-thread connection. Concurrent readers are fine under WAL."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
            self._local.conn = conn
        return conn

    @contextmanager
    def write(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Serialised through a process-wide lock. WAL allows exactly one writer."""
        if self.read_only:
            raise sqlite3.OperationalError("database opened read only")
        with self._write_lock:
            conn = self.read()
            nested = self._write_depth > 0
            if not nested:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            self._write_depth += 1
            try:
                yield conn
            except BaseException:
                self._write_depth -= 1
                if not nested:
                    conn.rollback()
                raise
            else:
                self._write_depth -= 1
                if not nested:
                    conn.commit()

    @contextmanager
    def raw_writer(self) -> Iterator[sqlite3.Connection]:
        """Hold the writer lock without opening a transaction, for scripts that BEGIN themselves."""
        if self.read_only:
            raise sqlite3.OperationalError("database opened read only")
        with self._write_lock:
            yield self.read()

    def attach_read_only(self, alias: str, path: Path) -> None:
        """Attach another palate database for lookups, read only."""
        if not alias.isidentifier():
            raise ValueError(f"bad attach alias {alias!r}")
        self._attached[alias] = path
        with self._conns_lock:
            conns = list(self._conns)
        for conn in conns:
            self._attach(conn, alias, path)

    def ensure_migrated(self) -> None:
        """Apply pending migrations once per open, behind the write lock."""
        with self._write_lock:
            if self._migrated:
                return
            from palate.db.migrate import migrate

            migrate(self, self.migrations)
            self._migrated = True

    def close(self) -> None:
        """Close every connection this Database opened."""
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            conn.close()
        self._local = threading.local()


def open_database(
    path: Path,
    *,
    migrations: Path,
    load_vec: bool = True,
    migrate_now: bool = True,
) -> Database:
    """Open a database and bring it up to the latest migration."""
    db = Database(path, migrations=migrations, load_vec=load_vec)
    if migrate_now:
        db.ensure_migrated()
    return db
