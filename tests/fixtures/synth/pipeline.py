"""One migrated database with the planted world installed, indexed and fitted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fixtures.synth import sqlite as synth_sqlite
from fixtures.synth.world import SynthWorld, build_world
from palate.db.connect import Database, open_database
from palate.index import fts, verify
from palate.index.vecstore import VecStore
from palate.paths import migrations_dir
from palate.retrieval.candidates import CandidateStore
from palate.taste import profile as taste
from palate.taste.profile import TasteProfile

SMALL = {"per_cluster": 60, "background": 200, "rated_per_cluster": 40, "rated_background": 40}


def small_world(**overrides: Any) -> SynthWorld:
    """The planted world shrunk to what a retrieval test needs."""
    return build_world(**{**SMALL, **overrides})


@dataclass(frozen=True, slots=True)
class Fitted:
    """A database, the world inside it, and the profile fitted against that index."""

    db: Database
    world: SynthWorld
    profile: TasteProfile

    def store(self, **overrides: Any) -> CandidateStore:
        """A candidate store over the active index."""
        record = verify.active(self.db)
        conn = self.db.read()
        vecs = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
        return CandidateStore(conn, vecs=vecs, **overrides)

    def close(self) -> None:
        self.db.close()


def fit(tmp_path: Path, world: SynthWorld | None = None, *, docs: bool = True) -> Fitted:
    """Install the world, render its documents, and fit a profile in that index."""
    planted = world if world is not None else small_world()
    db = open_database(tmp_path / "palate.db", migrations=migrations_dir())
    synth_sqlite.install(db, planted)
    if docs:
        fts.rebuild(db)
    return Fitted(db, planted, taste.build(db))
