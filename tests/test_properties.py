"""Properties that have to hold for every filter and every column, not just the ones I thought of."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fixtures.synth import pipeline
from fixtures.synth import sqlite as synth_sqlite
from fixtures.synth.world import GENRE_POOL
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from palate.db.connect import open_database
from palate.index import verify
from palate.index.vecstore import VecStore
from palate.paths import migrations_dir
from palate.retrieval.candidates import CandidateStore, HardFilters
from palate.retrieval.features import MIN_SUPPORT, scale_feature

LANGUAGES = ("en", "ru", "cn", "fr", "sv")
COUNTRIES = ("US", "SU", "HK", "FR", "SE")

SQL = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

years = st.integers(min_value=1930, max_value=2025)

filters = st.builds(
    HardFilters,
    year_min=st.none() | years,
    year_max=st.none() | years,
    runtime_min=st.none() | st.integers(min_value=70, max_value=200),
    runtime_max=st.none() | st.integers(min_value=70, max_value=200),
    include_languages=st.frozensets(st.sampled_from(LANGUAGES), max_size=2),
    exclude_languages=st.frozensets(st.sampled_from(LANGUAGES), max_size=2),
    include_genres=st.frozensets(st.sampled_from(GENRE_POOL), max_size=2),
    exclude_genres=st.frozensets(st.sampled_from(GENRE_POOL), max_size=2),
    include_countries=st.frozensets(st.sampled_from(COUNTRIES), max_size=2),
    exclude_countries=st.frozensets(st.sampled_from(COUNTRIES), max_size=2),
    min_vote_count=st.integers(min_value=0, max_value=20_000),
    exclude_watched=st.booleans(),
)


@st.composite
def columns(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """One raw column and the mask that says which of its entries were ever produced."""
    n = draw(st.integers(min_value=1, max_value=40))
    values = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return np.array(values, dtype=np.float64), np.array(mask, dtype=bool)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CandidateStore]:
    root = tmp_path_factory.mktemp("properties")
    db = open_database(root / "palate.db", migrations=migrations_dir())
    world = pipeline.small_world(
        per_cluster=10, background=20, rated_per_cluster=5, rated_background=5
    )
    synth_sqlite.install(db, world)
    record = verify.active(db)
    conn = db.read()
    yield CandidateStore(
        conn, vecs=VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
    )
    db.close()


@SQL
@given(left=filters, right=filters)
def test_merge_never_loosens(store: CandidateStore, left: HardFilters, right: HardFilters) -> None:
    merged = store.allow(left.merge(right)).ids
    assert merged <= store.allow(left).ids
    assert merged <= store.allow(right).ids


@SQL
@given(a=filters, b=filters, c=filters)
def test_merge_is_associative(
    store: CandidateStore, a: HardFilters, b: HardFilters, c: HardFilters
) -> None:
    left = store.allow(a.merge(b).merge(c)).ids
    right = store.allow(a.merge(b.merge(c))).ids
    assert left == right


@given(one=filters)
def test_merge_is_idempotent(one: HardFilters) -> None:
    assert one.merge(one) == one


@given(one=filters)
def test_merge_with_nothing_only_adds_the_watched_cut(one: HardFilters) -> None:
    merged = one.merge(HardFilters())
    assert merged.exclude_watched is True
    assert merged.year_min == one.year_min
    assert merged.exclude_genres == one.exclude_genres


@given(column=columns(), clip=st.floats(min_value=0.5, max_value=5.0))
def test_scaling_stays_finite_and_inside_the_clip(
    column: tuple[np.ndarray, np.ndarray], clip: float
) -> None:
    values, mask = column
    scaled, how = scale_feature(values, mask, clip=clip)
    assert scaled.shape == values.shape
    assert np.isfinite(scaled).all()
    assert float(np.abs(scaled).max(initial=0.0)) <= clip + 1e-9
    assert how in {"mad", "rank", "zeroed"}


@given(column=columns())
def test_absent_entries_are_always_exactly_zero(
    column: tuple[np.ndarray, np.ndarray],
) -> None:
    values, mask = column
    scaled, how = scale_feature(values, mask)
    assert np.all(scaled[~mask] == 0.0)
    if int(mask.sum()) < MIN_SUPPORT:
        assert how == "zeroed"
        assert not scaled.any()
