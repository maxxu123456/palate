"""One migrated database with the planted world installed, indexed and fitted."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fixtures.synth import sqlite as synth_sqlite
from fixtures.synth.world import SynthWorld, build_world
from palate.db.connect import Database, open_database
from palate.index import fts, verify
from palate.index.vecstore import VecStore
from palate.paths import migrations_dir
from palate.providers.base import EmbeddingBatch, ProviderHealth, SpanLike, Vector
from palate.providers.fingerprint import EmbeddingFingerprint
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


class SynthEmbedder:
    """Queries land on a planted cluster centre, in the same space as the installed index."""

    provider = "synth"

    def __init__(self, world: SynthWorld, *, cluster: int = 0, max_batch: int = 64) -> None:
        self.world = world
        self.cluster = cluster
        self.max_batch = max_batch

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        """The fingerprint install_index wrote, so the active index accepts these queries."""
        return synth_sqlite.fingerprint(dim=self.world.centres.shape[1])

    async def ready(self) -> EmbeddingFingerprint:
        return self.fingerprint

    async def embed_documents(
        self, texts: Sequence[str], *, span: SpanLike | None = None
    ) -> EmbeddingBatch:
        vectors = tuple(tuple(float(x) for x in self.world.centres[self.cluster]) for _ in texts)
        return EmbeddingBatch(vectors=vectors, fingerprint=self.fingerprint)

    async def embed_query(self, text: str, *, span: SpanLike | None = None) -> Vector:
        return tuple(float(x) for x in self.world.centres[self.cluster])

    async def health(self) -> ProviderHealth:
        return ProviderHealth(True, "synthetic", 0.0)

    async def aclose(self) -> None:
        return None
