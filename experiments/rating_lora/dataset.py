"""Turns the rated history into jsonl for the rating head. Stdlib and numpy, nothing else."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DB = Path.home() / ".palate" / "palate.db"
OVERVIEW_CHARS = 400
MAX_KEYWORDS = 12

ROWS = (
    "select u.tmdb_id, u.rating_half, f.title, f.year, f.runtime, f.original_language, "
    "coalesce(f.overview, '') as overview, "
    "(select group_concat(p.name, ', ') from credits c join people p "
    "  on p.person_id = c.person_id where c.tmdb_id = f.tmdb_id and c.job = 'Director') "
    "  as directors, "
    "(select group_concat(g.name, ', ') from film_genres fg join genres g "
    "  on g.genre_id = fg.genre_id where fg.tmdb_id = f.tmdb_id) as genres, "
    "(select group_concat(k.name, ', ') from film_keywords fk join keywords k "
    "  on k.keyword_id = fk.keyword_id where fk.tmdb_id = f.tmdb_id) as keywords "
    "from user_films u join films f on f.tmdb_id = u.tmdb_id "
    "where u.rating_half is not null order by u.tmdb_id"
)


@dataclass(frozen=True, slots=True)
class Example:
    """One film as the model sees it, with the star rating as the target."""

    tmdb_id: int
    text: str
    rating: float
    split: str


@dataclass(frozen=True, slots=True)
class Summary:
    """What the build wrote, and the number a finetune has to beat."""

    n_train: int
    n_val: int
    mean_rating: float
    baseline_mae: float


def clip(text: str, limit: int) -> str:
    """Cut on a word boundary so the tail is not half a word."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def render(row: sqlite3.Row) -> str:
    """The metadata block. No rating and no review text, those are the label and a leak."""
    year = row["year"]
    lines = [row["title"] if year is None else f"{row['title']} ({year})"]
    if row["directors"]:
        lines.append(f"Directed by {row['directors']}")
    if row["genres"]:
        lines.append(f"Genres: {row['genres']}")
    facts = []
    if row["runtime"]:
        facts.append(f"{row['runtime']} minutes")
    if row["original_language"]:
        facts.append(f"language {row['original_language']}")
    if facts:
        lines.append(", ".join(facts))
    keywords = [k for k in (row["keywords"] or "").split(", ") if k][:MAX_KEYWORDS]
    if keywords:
        lines.append("Keywords: " + ", ".join(keywords))
    overview = " ".join(row["overview"].split())
    if overview:
        lines.append(clip(overview, OVERVIEW_CHARS))
    return "\n".join(lines)


# TODO: a random holdout leaks. The eval harness folds on reliable dates and drops
# catalogue days, and this should split the same way before the MAE means anything.
def held_out(tmdb_ids: list[int], *, seed: int, fraction: float) -> set[int]:
    """The validation ids, drawn once from the seed."""
    rng = np.random.default_rng(seed)
    n = round(len(tmdb_ids) * fraction)
    return {int(i) for i in rng.permutation(np.array(tmdb_ids, dtype=np.int64))[:n]}


def examples(conn: sqlite3.Connection, *, seed: int, fraction: float) -> list[Example]:
    """Every rated film, rendered and assigned a split."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(ROWS).fetchall()
    held = held_out([int(r["tmdb_id"]) for r in rows], seed=seed, fraction=fraction)
    return [
        Example(
            tmdb_id=int(row["tmdb_id"]),
            text=render(row),
            rating=int(row["rating_half"]) / 2.0,
            split="val" if int(row["tmdb_id"]) in held else "train",
        )
        for row in rows
    ]


def summarise(rows: list[Example]) -> Summary:
    """The train mean predicted everywhere is the baseline the head is measured against."""
    train = np.array([r.rating for r in rows if r.split == "train"])
    val = np.array([r.rating for r in rows if r.split == "val"])
    mean = float(train.mean()) if train.size else 0.0
    mae = float(np.abs(val - mean).mean()) if val.size else 0.0
    return Summary(int(train.size), int(val.size), mean, mae)


def build(db: Path, out: Path, *, seed: int = 0, fraction: float = 0.2) -> Summary:
    """Write the jsonl and report what is in it."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = examples(conn, seed=seed, fraction=fraction)
    finally:
        conn.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "tmdb_id": row.tmdb_id,
                        "text": row.text,
                        "rating": row.rating,
                        "split": row.split,
                    }
                )
                + "\n"
            )
    return summarise(rows)


def main() -> None:
    """Build the dataset from a palate database."""
    parser = argparse.ArgumentParser(description="build the rating prediction dataset")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=Path("data/ratings.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    summary = build(args.db, args.out, seed=args.seed, fraction=args.val_fraction)
    print(f"{summary.n_train} train, {summary.n_val} val, mean {summary.mean_rating:.2f}")
    print(f"baseline mae {summary.baseline_mae:.3f}")


if __name__ == "__main__":
    main()
