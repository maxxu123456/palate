"""The document renderer is pinned to a golden file, because a silent change costs an index."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from palate.db.connect import Database, open_database
from palate.index.documents import (
    DOC_TEMPLATE_VERSION,
    MAX_CAST,
    Credits,
    RenderedDoc,
    load_inputs,
    render,
    render_document,
)
from palate.index.fts import match_expression, prune, rebuild, search
from palate.ingest.corpus import FilmRow
from palate.paths import migrations_dir
from palate.tmdb.crawl import record_member
from palate.tmdb.normalize import normalize_movie, write_film

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"
GOLDEN = Path(__file__).parent / "fixtures" / "golden" / f"{DOC_TEMPLATE_VERSION}.txt"
STAMP = "2026-09-08T12:00:00.000000+00:00"

SAMPLE = (
    "movie_1398_stalker.json",
    "movie_603_matrix.json",
    "movie_11104_chungking_express.json",
    "movie_802_no_overview.json",
)

STALE_GOLDEN = (
    "the document renderer changed but DOC_TEMPLATE_VERSION is still "
    f"{DOC_TEMPLATE_VERSION}, which would silently invalidate every built index"
)


def payload(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)
    for name in SAMPLE:
        film = normalize_movie(payload(name))
        with database.write() as conn:
            write_film(conn, film, fetched_at=STAMP)
            record_member(conn, film.tmdb_id, "discover", STAMP)
    yield database
    database.close()


def block(tmdb_id: int, doc: RenderedDoc) -> str:
    """One golden entry: the identity line, then the text the embedder sees."""
    return f"# {tmdb_id} {doc.doc_kind} {doc.doc_sha[:16]}\n{doc.full_text}"


def render_sample(db: Database) -> str:
    ids = [normalize_movie(payload(name)).tmdb_id for name in SAMPLE]
    items = load_inputs(db.read(), ids)
    return "\n\n".join(block(i.film.tmdb_id, render(i)) for i in items) + "\n"


def test_rendered_documents_match_the_golden_file(db: Database) -> None:
    assert GOLDEN.read_text(encoding="utf-8") == render_sample(db), STALE_GOLDEN


def test_the_full_template_puts_the_overview_last(db: Database) -> None:
    doc = render(load_inputs(db.read(), [1398])[0])
    assert doc.doc_kind == "full"
    lines = doc.full_text.splitlines()
    assert lines[0] == "Stalker (1979). Directed by Andrei Tarkovsky."
    assert "Tagline: none." in doc.full_text
    assert lines[-1].startswith("A guide leads two men")
    # Discriminative fields come before the overview because cross encoders truncate.
    assert doc.full_text.index("Keywords:") < doc.overview_offset


def test_overview_offset_slices_back_to_the_overview(db: Database) -> None:
    doc = render(load_inputs(db.read(), [603])[0])
    assert doc.full_text[doc.overview_offset :] == doc.overview_text


def test_a_film_with_nothing_to_say_gets_the_minimal_template(db: Database) -> None:
    doc = render(load_inputs(db.read(), [802])[0])
    assert doc.doc_kind == "minimal"
    assert doc.overview_offset == -1
    assert "Tagline" not in doc.full_text
    assert doc.full_text.splitlines()[0] == "The Bear (1970)."


def test_keywords_without_an_overview_still_render() -> None:
    film = FilmRow(1, title="Nameless", year=1968, runtime=90, status="Released")
    doc = render_document(film, Credits(), ["giallo"], ["Horror"], ["Italy"])
    assert doc.doc_kind == "no_overview"
    assert doc.overview_text == ""
    assert "Keywords: giallo." in doc.full_text


def test_dropping_credits_drops_every_name(db: Database) -> None:
    item = load_inputs(db.read(), [1398])[0]
    with_names = render(item)
    without = render(item, include_credits=False)
    assert "Tarkovsky" in with_names.full_text
    assert "Tarkovsky" not in without.full_text
    assert without.people_text == ""
    # Same doc_kind either way, so the ablation compares like for like.
    assert without.doc_kind == with_names.doc_kind
    assert without.doc_sha != with_names.doc_sha


def test_the_keyword_list_is_capped() -> None:
    film = FilmRow(1, title="Many", year=2001, runtime=90, overview="A film.", status="Released")
    keywords = [f"k{i}" for i in range(40)]
    doc = render_document(film, Credits(), keywords, [], [], max_keywords=3)
    assert "Keywords: k0, k1, k2." in doc.full_text
    assert "k3" not in doc.full_text


def test_a_long_overview_is_cut_and_the_offset_still_holds() -> None:
    film = FilmRow(
        1,
        title="Long",
        year=2001,
        runtime=90,
        overview=" ".join(["word"] * 400),
        status="Released",
    )
    doc = render_document(film, Credits(), [], ["Drama"], [], max_chars=200)
    assert len(doc.full_text) <= 200
    assert doc.full_text[doc.overview_offset :] == doc.overview_text
    assert doc.overview_text.endswith("word")


def test_a_cast_of_forty_is_trimmed() -> None:
    film = FilmRow(1, title="Crowd", year=2001, runtime=90, overview="A film.", status="Released")
    cast = tuple(f"Actor {i}" for i in range(40))
    doc = render_document(film, Credits(("Dir",), cast), [], [], [])
    cast_line = next(line for line in doc.full_text.splitlines() if line.startswith("Cast: "))
    assert cast_line.count(",") == MAX_CAST - 1
    assert "Actor 8" not in doc.full_text


def test_rebuild_writes_a_document_per_eligible_film(db: Database) -> None:
    report = rebuild(db, corpus_only=False)
    assert report.n_seen == 4
    assert report.n_written == 4
    assert report.by_kind == {"full": 3, "minimal": 1}
    again = rebuild(db, corpus_only=False)
    assert again.n_written == 0
    assert again.n_unchanged == 4


def test_search_ranks_a_title_match_over_a_plot_match(db: Database) -> None:
    rebuild(db, corpus_only=False)
    hits = search(db.read(), "stalker")
    assert hits[0].tmdb_id == 1398
    assert hits[0].score > 0


def test_search_finds_a_film_by_its_director(db: Database) -> None:
    rebuild(db, corpus_only=False)
    assert [h.tmdb_id for h in search(db.read(), "wong kar wai")] == [11104]


def test_negation_removes_a_film_from_the_results(db: Database) -> None:
    rebuild(db, corpus_only=False)
    assert 1398 in {h.tmdb_id for h in search(db.read(), "dystopia")}
    assert 1398 not in {h.tmdb_id for h in search(db.read(), "dystopia", exclude=["russian"])}


def test_match_expression_parenthesises_both_sides() -> None:
    assert match_expression("slow cold") == '"slow" OR "cold"'
    assert match_expression("slow -russian") == '("slow") NOT ("russian")'
    assert match_expression("  ") == ""
    # A quote in the query is a token boundary, never a way into the syntax.
    assert match_expression('slow" OR "x') == '"slow" OR "OR" OR "x"'


def test_an_updated_document_stops_matching_its_old_text(db: Database) -> None:
    rebuild(db, corpus_only=False)
    with db.write() as conn:
        conn.execute("update films set overview = 'a quiet room' where tmdb_id = 1398")
    rebuild(db, corpus_only=False)
    # External content plus delete triggers is the only arrangement where this holds.
    assert search(db.read(), "wishes") == ()
    assert {h.tmdb_id for h in search(db.read(), "quiet")} == {1398}


def test_pruning_a_document_clears_its_tokens(db: Database) -> None:
    rebuild(db, corpus_only=False)
    with db.write() as conn:
        conn.execute("update corpus_members set eligible = 0 where tmdb_id = 1398")
    assert prune(db) == 1
    assert search(db.read(), "stalker") == ()
