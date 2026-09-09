"""The text a film turns into before it is embedded or indexed."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import orjson

from palate.hashing import doc_sha
from palate.ingest.corpus import FilmRow

# Inside the fingerprint, because changing this text changes the vector space.
DOC_TEMPLATE_VERSION = "v1"

MAX_KEYWORDS = 20
MAX_CHARS = 1200
# A blockbuster credits forty people and the tail says nothing about the film.
MAX_CAST = 8
# Under this many credits, with no keywords and no overview, there is nothing to render.
MIN_CREDITS = 3

type DocKind = Literal["full", "no_overview", "minimal"]


@dataclass(frozen=True, slots=True)
class Credits:
    """The names a document is allowed to mention."""

    directors: tuple[str, ...] = ()
    cast: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.directors) + len(self.cast)


@dataclass(frozen=True, slots=True)
class RenderedDoc:
    """One film as text, plus the offsets and hash the rest of the system needs."""

    doc_kind: DocKind
    title_text: str
    people_text: str
    keyword_text: str
    overview_text: str
    full_text: str
    overview_offset: int
    doc_sha: str


@dataclass(frozen=True, slots=True)
class DocInputs:
    """Everything the renderer reads for one film, gathered from the joins."""

    film: FilmRow
    credits: Credits = Credits()
    keywords: tuple[str, ...] = ()
    genres: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()


def classify(film: FilmRow, credits: Credits, keywords: Sequence[str]) -> DocKind:
    """Which template this film has enough material for."""
    if film.overview:
        return "full"
    if not keywords and credits.size < MIN_CREDITS:
        return "minimal"
    return "no_overview"


def render_document(
    film: FilmRow,
    credits: Credits,
    keywords: Sequence[str],
    genres: Sequence[str],
    countries: Sequence[str],
    *,
    languages: Sequence[str] = (),
    include_credits: bool = True,
    max_keywords: int = MAX_KEYWORDS,
    max_chars: int = MAX_CHARS,
) -> RenderedDoc:
    """Render one film into the text that gets embedded and indexed."""
    # Classified on the underlying data, so the no-credits ablation compares like for like.
    kind = classify(film, credits, keywords)
    people = Credits(credits.directors, credits.cast[:MAX_CAST]) if include_credits else Credits()
    kept = tuple(keywords[:max_keywords]) if kind != "minimal" else ()
    facts = _facts_line(film, genres, countries, languages)

    lines = [_opening(film, people)]
    if kind != "minimal" and people.cast:
        lines.append(f"Cast: {', '.join(people.cast)}.")
    if facts:
        lines.append(facts)
    if kept:
        lines.append(f"Keywords: {', '.join(kept)}.")
    if kind != "minimal":
        lines.append(f"Tagline: {_tagline(film)}.")

    head = "\n".join(lines)
    overview = ""
    if kind == "full" and film.overview:
        overview = _fit(film.overview.strip(), max_chars - len(head) - 1)
    full = f"{head}\n{overview}" if overview else head
    people_text = ", ".join((*people.directors, *people.cast)) if kind != "minimal" else ""
    return RenderedDoc(
        doc_kind=kind,
        title_text=film.title,
        people_text=people_text,
        keyword_text=", ".join((*genres, *countries, *languages, *kept)),
        overview_text=overview,
        full_text=full,
        overview_offset=len(full) - len(overview) if overview else -1,
        doc_sha=doc_sha(full),
    )


def _opening(film: FilmRow, credits: Credits) -> str:
    opening = f"{film.title} ({film.year})." if film.year else f"{film.title}."
    if credits.directors:
        opening += f" Directed by {', '.join(credits.directors)}."
    return opening


def _tagline(film: FilmRow) -> str:
    # Most taglines already end in a stop, and two of them reads like a typo.
    return (film.tagline or "").strip().rstrip(".") or "none"


def _facts_line(
    film: FilmRow, genres: Sequence[str], countries: Sequence[str], languages: Sequence[str]
) -> str:
    parts = []
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")
    if countries:
        parts.append(f"Country: {', '.join(countries)}.")
    if languages:
        parts.append(f"Language: {', '.join(languages)}.")
    if film.runtime:
        parts.append(f"{film.runtime} minutes.")
    return " ".join(parts)


def _fit(text: str, budget: int) -> str:
    """Cut an overview to the character budget on a word boundary."""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    cut = text[:budget]
    space = cut.rfind(" ")
    return cut[:space] if space > 0 else cut


_FILMS = (
    "select tmdb_id, title, year, runtime, vote_count, adult, status, overview, tagline, "
    "original_language from films where tmdb_id in (select value from json_each(?))"
)
_CREDITS = (
    "select c.tmdb_id, c.credit_kind, c.job, p.name from credits c "
    "join people p on p.person_id = c.person_id "
    "where c.tmdb_id in (select value from json_each(?)) "
    "and (c.credit_kind = 'cast' or c.job = 'Director') order by c.tmdb_id, c.ord, p.name"
)


def _names(conn: sqlite3.Connection, sql: str, ids: Sequence[int]) -> dict[int, tuple[str, ...]]:
    grouped: dict[int, list[str]] = {}
    for row in conn.execute(sql, (orjson.dumps(list(ids)).decode(),)):
        grouped.setdefault(int(row["tmdb_id"]), []).append(str(row["name"]))
    return {k: tuple(v) for k, v in grouped.items()}


_KEYWORDS = (
    "select fk.tmdb_id, k.name from film_keywords fk join keywords k "
    "on k.keyword_id = fk.keyword_id "
    "where fk.tmdb_id in (select value from json_each(?)) order by fk.tmdb_id, k.name"
)
_GENRES = (
    "select fg.tmdb_id, g.name from film_genres fg join genres g on g.genre_id = fg.genre_id "
    "where fg.tmdb_id in (select value from json_each(?)) order by fg.tmdb_id, g.name"
)
_COUNTRIES = (
    "select fc.tmdb_id, c.name from film_countries fc join countries c "
    "on c.iso_3166_1 = fc.iso_3166_1 "
    "where fc.tmdb_id in (select value from json_each(?)) order by fc.tmdb_id, c.name"
)
_LANGUAGES = (
    "select fl.tmdb_id, l.name from film_languages fl join languages l "
    "on l.iso_639_1 = fl.iso_639_1 "
    "where fl.tmdb_id in (select value from json_each(?)) order by fl.tmdb_id, l.name"
)


def load_inputs(conn: sqlite3.Connection, ids: Sequence[int]) -> list[DocInputs]:
    """Gather the joins for a batch of films, in the order the ids were given."""
    if not ids:
        return []
    payload = orjson.dumps(list(ids)).decode()
    films = {
        int(r["tmdb_id"]): FilmRow(
            tmdb_id=int(r["tmdb_id"]),
            title=str(r["title"]),
            year=None if r["year"] is None else int(r["year"]),
            runtime=None if r["runtime"] is None else int(r["runtime"]),
            vote_count=int(r["vote_count"]),
            adult=bool(r["adult"]),
            status=None if r["status"] is None else str(r["status"]),
            overview=None if r["overview"] is None else str(r["overview"]),
            tagline=None if r["tagline"] is None else str(r["tagline"]),
            original_language=None
            if r["original_language"] is None
            else str(r["original_language"]),
        )
        for r in conn.execute(_FILMS, (payload,))
    }
    directors: dict[int, list[str]] = {}
    cast: dict[int, list[str]] = {}
    for row in conn.execute(_CREDITS, (payload,)):
        bucket = directors if row["job"] == "Director" else cast
        bucket.setdefault(int(row["tmdb_id"]), []).append(str(row["name"]))
    keywords = _names(conn, _KEYWORDS, ids)
    genres = _names(conn, _GENRES, ids)
    countries = _names(conn, _COUNTRIES, ids)
    languages = _names(conn, _LANGUAGES, ids)
    return [
        DocInputs(
            film=films[tmdb_id],
            credits=Credits(
                tuple(dict.fromkeys(directors.get(tmdb_id, ()))),
                tuple(dict.fromkeys(cast.get(tmdb_id, ()))),
            ),
            keywords=keywords.get(tmdb_id, ()),
            genres=genres.get(tmdb_id, ()),
            countries=countries.get(tmdb_id, ()),
            languages=languages.get(tmdb_id, ()),
        )
        for tmdb_id in ids
        if tmdb_id in films
    ]


def render(inputs: DocInputs, *, include_credits: bool = True) -> RenderedDoc:
    """Render a gathered film without restating every join at the call site."""
    return render_document(
        inputs.film,
        inputs.credits,
        inputs.keywords,
        inputs.genres,
        inputs.countries,
        languages=inputs.languages,
        include_credits=include_credits,
    )
