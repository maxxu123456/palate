"""The rating dataset builder, over the planted world. The package never imports it."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fixtures.synth.pipeline import small_world
from fixtures.synth.sqlite import install
from fixtures.synth.world import SynthWorld

from palate.db.connect import open_database
from palate.paths import migrations_dir

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "rating_lora" / "dataset.py"


def load() -> ModuleType:
    """experiments/ is outside the package and off sys.path, so load the file itself."""
    spec = importlib.util.spec_from_file_location("rating_dataset", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A slots dataclass looks its own module up by name while it is being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dataset = load()


@pytest.fixture
def planted(tmp_path: Path) -> tuple[SynthWorld, Path]:
    """A migrated database holding the synthetic history."""
    world = small_world()
    path = tmp_path / "palate.db"
    db = open_database(path, migrations=migrations_dir())
    install(db, world)
    db.close()
    return world, path


def written(out: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_one_line_per_rated_film(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    world, path = planted
    summary = dataset.build(path, tmp_path / "ratings.jsonl")
    rows = written(tmp_path / "ratings.jsonl")
    assert {row["tmdb_id"] for row in rows} == {film.tmdb_id for film in world.rated}
    assert summary.n_train + summary.n_val == len(world.rated)
    assert {row["rating"] for row in rows} <= {n / 2.0 for n in range(1, 11)}


def test_an_unrated_row_is_not_an_example(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    world, path = planted
    spare = next(
        f.tmdb_id for f in world.films if f.tmdb_id not in {r.tmdb_id for r in world.rated}
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            "insert into user_films (tmdb_id, date_source, in_watchlist) values (?,'none',1)",
            (spare,),
        )
    dataset.build(path, tmp_path / "ratings.jsonl")
    assert spare not in {row["tmdb_id"] for row in written(tmp_path / "ratings.jsonl")}


def test_the_text_carries_the_metadata(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    world, path = planted
    dataset.build(path, tmp_path / "ratings.jsonl")
    rows = {int(row["tmdb_id"]): str(row["text"]) for row in written(tmp_path / "ratings.jsonl")}
    film = world.by_id[world.rated[0].tmdb_id]
    text = rows[film.tmdb_id]
    assert film.title in text
    assert str(film.year) in text
    assert film.director_name in text
    assert f"{film.runtime} minutes" in text
    assert film.overview.split(" ")[0] in text


def test_a_long_overview_is_clipped(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    world, path = planted
    wordy = world.rated[0].tmdb_id
    with sqlite3.connect(path) as conn:
        conn.execute("update films set overview = ? where tmdb_id = ?", ("plot " * 400, wordy))
    dataset.build(path, tmp_path / "ratings.jsonl")
    rows = {int(row["tmdb_id"]): str(row["text"]) for row in written(tmp_path / "ratings.jsonl")}
    assert rows[wordy].endswith("...")
    assert len(rows[wordy].splitlines()[-1]) <= dataset.OVERVIEW_CHARS + 3


def test_the_split_follows_the_seed(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    _, path = planted

    def val_ids(seed: int, name: str) -> set[int]:
        dataset.build(path, tmp_path / name, seed=seed)
        return {int(r["tmdb_id"]) for r in written(tmp_path / name) if r["split"] == "val"}

    again = val_ids(7, "a.jsonl")
    assert val_ids(7, "b.jsonl") == again
    assert val_ids(8, "c.jsonl") != again


def test_the_fraction_is_held_out(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    world, path = planted
    summary = dataset.build(path, tmp_path / "ratings.jsonl", fraction=0.25)
    assert summary.n_val == round(len(world.rated) * 0.25)


def test_the_baseline_is_the_train_mean(planted: tuple[SynthWorld, Path], tmp_path: Path) -> None:
    _, path = planted
    summary = dataset.build(path, tmp_path / "ratings.jsonl")
    rows = written(tmp_path / "ratings.jsonl")
    train = np.array([float(r["rating"]) for r in rows if r["split"] == "train"])
    val = np.array([float(r["rating"]) for r in rows if r["split"] == "val"])
    assert summary.mean_rating == pytest.approx(float(train.mean()))
    assert summary.baseline_mae == pytest.approx(float(np.abs(val - train.mean()).mean()))
