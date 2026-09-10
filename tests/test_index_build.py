"""The whole path, on the real payloads, with the fake embedder: docs, vectors, activation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from palate.db.connect import Database, open_database
from palate.errors import EmbeddingFingerprintMismatch, IndexIncomplete, NoActiveIndex
from palate.index import fts, verify
from palate.index.build import build
from palate.index.vecstore import MetadataFilter, VecStore
from palate.ingest.corpus import apply_eligibility
from palate.paths import migrations_dir
from palate.providers.embed.fake import FakeEmbedder
from palate.providers.fingerprint import CANARY_TEXT, unpack_f32
from palate.tmdb.crawl import record_member
from palate.tmdb.normalize import normalize_movie, write_film

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"
STAMP = "2026-09-09T09:00:00.000000+00:00"

SAMPLE = (
    "movie_1398_stalker.json",
    "movie_603_matrix.json",
    "movie_11104_chungking_express.json",
    "movie_802_no_overview.json",
)


def payload(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=True)
    for name in SAMPLE:
        film = normalize_movie(payload(name))
        with database.write() as conn:
            write_film(conn, film, fetched_at=STAMP)
            record_member(conn, film.tmdb_id, "discover", STAMP)
    apply_eligibility(database)
    fts.rebuild(database)
    yield database
    database.close()


def run_build(db: Database, embedder: FakeEmbedder, **kwargs: Any) -> Any:
    async def scenario() -> Any:
        return await build(db, embedder, **kwargs)

    return anyio.run(scenario)


def test_a_build_fills_the_table_and_flips_the_pointer(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    assert report.n_targets == 4
    assert report.n_embedded == 4
    assert report.n_vectors == 4
    assert report.dim == 32
    assert report.activated is True
    assert verify.active_id(db) == report.index_id
    assert report.table_name == f"vec_films_{report.index_id}"


def test_the_vector_table_is_named_after_the_fingerprint(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    names = {
        str(r["name"])
        for r in db.read().execute("select name from sqlite_master where type = 'table'")
    }
    assert report.table_name in names
    # vec0 fixes the dimension at CREATE time, so no migration can create this table.
    sql = "\n".join(p.read_text() for p in sorted(migrations_dir().glob("*.sql")))
    statements = "\n".join(line.split("--")[0] for line in sql.splitlines())
    assert "vec_films" not in statements


def test_a_second_build_embeds_nothing_new(db: Database) -> None:
    first = run_build(db, FakeEmbedder(dim=32))
    second = run_build(db, FakeEmbedder(dim=32))
    assert second.index_id == first.index_id
    assert second.n_embedded == 0
    assert second.n_vectors == 4


def test_only_the_film_whose_document_moved_is_re_embedded(db: Database) -> None:
    run_build(db, FakeEmbedder(dim=32))
    with db.write() as conn:
        conn.execute("update films set overview = 'a quiet room' where tmdb_id = 1398")
    fts.rebuild(db)
    report = run_build(db, FakeEmbedder(dim=32))
    assert report.n_embedded == 1


def test_the_cache_serves_a_document_that_came_back(db: Database) -> None:
    run_build(db, FakeEmbedder(dim=32))
    original = db.read().execute("select overview from films where tmdb_id = 1398").fetchone()[0]
    with db.write() as conn:
        conn.execute("update films set overview = 'a quiet room' where tmdb_id = 1398")
    fts.rebuild(db)
    run_build(db, FakeEmbedder(dim=32))
    with db.write() as conn:
        conn.execute("update films set overview = ? where tmdb_id = 1398", (original,))
    fts.rebuild(db)
    report = run_build(db, FakeEmbedder(dim=32))
    assert (report.n_embedded, report.n_cached) == (0, 1)


def test_a_different_embedder_builds_a_second_index_beside_the_first(db: Database) -> None:
    first = run_build(db, FakeEmbedder(dim=32))
    second = run_build(db, FakeEmbedder(dim=64))
    assert second.index_id != first.index_id
    assert {r.index_id for r in verify.listing(db)} == {first.index_id, second.index_id}
    # The old table is not dropped, which is what makes a rebuild reversible.
    assert verify.verify(db, first.index_id).n_present == 4
    verify.activate(db, first.index_id)
    assert verify.active_id(db) == first.index_id


def test_an_index_built_by_another_embedder_is_refused(db: Database) -> None:
    run_build(db, FakeEmbedder(dim=32))
    other = FakeEmbedder(dim=32, seed=9)
    anyio.run(other.ready)
    with pytest.raises(EmbeddingFingerprintMismatch) as exc:
        verify.require_active_match(db, other)
    assert "cannot answer this query" in str(exc.value)
    assert "dim" in str(exc.value)


def test_the_drift_report_names_the_field_that_moved(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    record = verify.load(db, report.index_id)
    assert record is not None
    assert anyio.run(lambda: verify.drift(record, FakeEmbedder(dim=32))) == ""
    moved = anyio.run(lambda: verify.drift(record, FakeEmbedder(dim=64)))
    assert "dim" in moved
    assert "palate index activate" in moved


def test_the_canary_vector_is_stored_at_the_right_width(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    record = verify.load(db, report.index_id)
    assert record is not None
    assert record.canary_text == CANARY_TEXT
    assert len(unpack_f32(record.canary_vec)) == 32
    assert record.canary_checked_at is not None


def test_verify_notices_a_film_with_no_vector(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    record = verify.load(db, report.index_id)
    assert record is not None
    with db.write() as conn:
        conn.execute(f"delete from {record.table_name} where film_id = 1398")
    checked = verify.verify(db, report.index_id)
    assert checked.ok is False
    assert checked.missing == 1
    assert any("no vector" in p for p in checked.problems)


def test_an_incomplete_build_is_marked_failed_and_never_activated(db: Database) -> None:
    with db.write() as conn:
        conn.execute("delete from film_docs where tmdb_id = 802")
    with pytest.raises(IndexIncomplete):
        run_build(db, FakeEmbedder(dim=32))
    assert verify.active_id(db) is None
    statuses = {str(r["status"]) for r in db.read().execute("select status from embedding_indexes")}
    assert statuses == {"failed"}


def test_the_active_index_cannot_be_dropped(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    with pytest.raises(NoActiveIndex, match="is active"):
        verify.drop(db, report.index_id)
    second = run_build(db, FakeEmbedder(dim=64))
    assert verify.drop(db, report.index_id) is True
    assert {r.index_id for r in verify.listing(db)} == {second.index_id}


def test_a_building_index_is_never_activated(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    with db.write() as conn:
        conn.execute(
            "update embedding_indexes set status = 'building' where index_id = ?",
            (report.index_id,),
        )
    with pytest.raises(NoActiveIndex, match="not ready"):
        verify.activate(db, report.index_id)


def test_the_watched_flag_reaches_the_vector_table(db: Database) -> None:
    with db.write() as conn:
        conn.execute(
            "insert into user_films (tmdb_id, rating_half, date_source) values (1398, 9, 'ratings')"
        )
    fts.rebuild(db)
    report = run_build(db, FakeEmbedder(dim=32))
    store = VecStore(db.read(), table=report.table_name, dim=32)
    unwatched = store.knn(
        [1.0] + [0.0] * 31, k=10, where=MetadataFilter.of(is_watched=0, in_corpus=1)
    )
    assert 1398 not in {h.film_id for h in unwatched}
    assert len(unwatched) == 3


def test_an_ineligible_film_keeps_its_vector_but_leaves_the_corpus(db: Database) -> None:
    with db.write() as conn:
        conn.execute("update films set vote_count = 0 where tmdb_id = 802")
    apply_eligibility(db)
    fts.prune(db)
    report = run_build(db, FakeEmbedder(dim=32))
    assert report.n_vectors == 3
    assert verify.verify(db, report.index_id).ok is True


def test_a_query_vector_lands_in_the_same_space(db: Database) -> None:
    report = run_build(db, FakeEmbedder(dim=32))
    embedder = FakeEmbedder(dim=32)
    vector = anyio.run(lambda: embedder.embed_query("slow and cold"))
    store = VecStore(db.read(), table=report.table_name, dim=32)
    hits = store.knn(vector, k=4)
    assert len(hits) == 4
    assert all(-1.0 <= h.similarity <= 1.0001 for h in hits)
