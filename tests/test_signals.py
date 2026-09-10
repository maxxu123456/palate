"""A four star for a film everyone gives four stars says nothing, which is what this checks."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.synth import HATED, LOVED, build_world

from palate.db.connect import Database, open_database
from palate.paths import migrations_dir
from palate.taste.signals import (
    SIGMA_FLOOR,
    FilmFacts,
    RatedFilm,
    ResidualCalibrator,
    SqliteFilmStore,
)
from palate.tmdb.normalize import normalize_movie, write_film

FIXTURES = Path(__file__).parent / "fixtures" / "tmdb"
STAMP = "2026-09-09T09:00:00.000000+00:00"


@pytest.fixture(scope="module")
def world() -> Any:
    return build_world()


@pytest.fixture
def fitted(world: Any) -> ResidualCalibrator:
    return ResidualCalibrator().fit(world.rated, world.store())


def test_the_signal_is_centred_on_the_users_own_mean(
    world: Any, fitted: ResidualCalibrator
) -> None:
    z = np.array([s.z for s in fitted.transform(world.rated)])
    assert abs(float(z.mean())) < 1e-9
    assert 0.9 < float(z.std()) < 1.1


def test_a_narrow_rater_never_reaches_eight_sigma() -> None:
    rated = [RatedFilm(i, 7 if i % 2 else 8) for i in range(40)]
    store = _Store({r.tmdb_id: FilmFacts(r.tmdb_id, 7.0, 500, 2010, (18,)) for r in rated})
    calibrator = ResidualCalibrator().fit(rated, store)
    assert calibrator.sigma == SIGMA_FLOOR
    assert max(abs(s.z) for s in calibrator.transform(rated)) < 2.0


def test_the_generic_model_explains_part_of_the_rating_but_not_all(
    fitted: ResidualCalibrator,
) -> None:
    assert 0.3 < fitted.r2 < 0.95


def test_what_the_crowd_explains_is_taken_out_of_the_signal() -> None:
    rng = np.random.default_rng(7)
    facts: dict[int, FilmFacts] = {}
    rated: list[RatedFilm] = []
    for i in range(200):
        votes = float(rng.uniform(4.0, 9.0))
        facts[i] = FilmFacts(i, votes, 1000, 2010, (18,))
        stars = min(5.0, max(0.5, 0.35 * votes + 1.0 + float(rng.normal(0.0, 0.12))))
        rated.append(RatedFilm(i, round(stars * 2)))
    calibrator = ResidualCalibrator().fit(rated, _Store(facts))
    signals = calibrator.transform(rated)
    z = np.array([s.z for s in signals])
    residual = np.array([s.residual for s in signals])
    assert calibrator.r2 > 0.7
    assert float(residual.std()) < 0.5 * float(z.std())


def test_the_signal_separates_the_planted_taste(world: Any, fitted: ResidualCalibrator) -> None:
    films = world.by_id
    signals = fitted.transform(world.rated)
    grouped: dict[int, list[float]] = {}
    for signal in signals:
        grouped.setdefault(films[signal.tmdb_id].cluster, []).append(signal.s)
    for cluster in LOVED:
        assert float(np.mean(grouped[cluster])) > 0.5
    for cluster in HATED:
        assert float(np.mean(grouped[cluster])) < -0.5
    assert abs(float(np.mean(grouped[3]))) < 0.3


def test_alpha_one_is_the_pure_z_score(world: Any) -> None:
    pure = ResidualCalibrator(alpha=1.0).fit(world.rated, world.store())
    assert all(s.s == pytest.approx(s.z) for s in pure.transform(world.rated))


def test_mass_is_the_size_of_the_signal_clipped_at_three(
    world: Any, fitted: ResidualCalibrator
) -> None:
    signals = fitted.transform(world.rated)
    assert all(s.mass == pytest.approx(min(abs(s.s), 3.0)) for s in signals)
    assert max(s.mass for s in signals) <= 3.0


def test_a_film_the_corpus_never_saw_predicts_the_users_mean(
    world: Any, fitted: ResidualCalibrator
) -> None:
    stranger = RatedFilm(999_999_999, 9)
    signal = fitted.transform([stranger])[0]
    assert signal.residual == pytest.approx(signal.z)


def test_a_stored_calibrator_gives_the_same_signals(world: Any, fitted: ResidualCalibrator) -> None:
    reloaded = ResidualCalibrator.from_row(json.loads(json.dumps(fitted.to_row())), world.store())
    before = fitted.transform(world.rated[:50])
    after = reloaded.transform(world.rated[:50])
    assert [s.s for s in after] == pytest.approx([s.s for s in before])
    assert reloaded.r2 == fitted.r2


def test_an_empty_history_is_refused(world: Any) -> None:
    with pytest.raises(ValueError, match="empty history"):
        ResidualCalibrator().fit([], world.store())


def test_the_sqlite_store_reads_votes_and_genres(db: Database) -> None:
    store = SqliteFilmStore(db.read())
    facts = store.facts([1398, 603, 999_999_999])
    assert set(facts) == {1398, 603}
    assert facts[1398].decade == 1970
    assert facts[1398].vote_count > 0
    assert facts[603].genres


class _Store:
    def __init__(self, facts: dict[int, FilmFacts]) -> None:
        self._facts = facts

    def facts(self, tmdb_ids: Any) -> dict[int, FilmFacts]:
        return {i: self._facts[i] for i in tmdb_ids if i in self._facts}


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=False)
    for name in ("movie_1398_stalker.json", "movie_603_matrix.json"):
        payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        with database.write() as conn:
            write_film(conn, normalize_movie(payload), fetched_at=STAMP)
    yield database
    database.close()
