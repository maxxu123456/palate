"""Everything the taste model writes down: the profile, its staleness, and stated preferences."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.synth import LOVED, build_world
from fixtures.synth.sqlite import install, install_index

from palate.clock import frozen
from palate.db.connect import Database, open_database
from palate.errors import PreferenceRefused, StaleArtifact, ThinHistoryError
from palate.index import verify
from palate.paths import migrations_dir
from palate.providers.base import Message
from palate.taste import profile as taste
from palate.taste.memory import PreferenceDraft, PreferenceStore, ensure_session
from palate.taste.modes import mode_affinity

SESSION = "ses_test"
SAID = Message(role="user", content="I hate musicals, and keep it under two hours tonight please")


def small_world(**kwargs: Any) -> Any:
    return build_world(per_cluster=80, background=200, **kwargs)


@pytest.fixture
def blank(tmp_path: Path) -> Iterator[Database]:
    database = open_database(tmp_path / "palate.db", migrations=migrations_dir(), load_vec=True)
    yield database
    database.close()


@pytest.fixture
def world() -> Any:
    return small_world(rated_per_cluster=64, rated_background=40)


@pytest.fixture
def db(blank: Database, world: Any) -> Database:
    install(blank, world)
    return blank


def test_a_full_profile_carries_modes_a_direction_and_affinities(db: Database, world: Any) -> None:
    built = taste.build(db, seed=0)
    assert built.tier == "full"
    assert built.n_rated == len(world.rated)
    assert built.modes and built.anti_modes
    assert built.direction is not None
    assert built.direction.n_fit == built.n_rated
    assert len(built.direction.feature_names) == len(built.direction.w)
    assert built.direction.loocv_r2 > 0.1
    assert built.affinities["director"]


def test_the_top_director_is_one_the_user_actually_rates_highly(db: Database, world: Any) -> None:
    built = taste.build(db, seed=0)
    loved_directors = {
        str(world.by_id[i].director_id) for planted in LOVED for i in world.members(planted)
    }
    top = built.top("director", limit=3)
    assert any(row.entity_id in loved_directors for row in top)
    assert all(row.n >= 1 for row in top)


def test_exposure_and_affinity_are_two_different_questions(db: Database) -> None:
    built = taste.build(db, seed=0)
    decades = {row.entity_id: row for row in built.affinities["decade"]}
    assert decades
    assert any(row.exposure_logodds > 0 for row in decades.values())
    assert any(row.exposure_logodds < 0 for row in decades.values())


def test_a_profile_reloads_with_its_modes_and_its_direction(db: Database) -> None:
    built = taste.build(db, seed=0)
    back = taste.load(db, built.profile_id)
    assert back is not None
    assert [m.mode_id for m in back.modes] == [m.mode_id for m in built.modes]
    assert back.modes[0].members
    assert back.direction is not None and built.direction is not None
    assert back.direction.w == pytest.approx(built.direction.w, rel=1e-5)
    assert back.direction.svd_u is None
    newest = taste.latest(db)
    assert newest is not None and newest.profile_id == built.profile_id


def test_a_reloaded_mode_scores_a_candidate_the_same_way(db: Database, world: Any) -> None:
    built = taste.build(db, seed=0)
    back = taste.load(db, built.profile_id)
    point = world.centres[LOVED[0]][None, :]
    assert back is not None
    assert float(mode_affinity(point, back.modes)[0]) == pytest.approx(
        float(mode_affinity(point, built.modes)[0]), rel=1e-5
    )


def test_a_profile_goes_stale_when_the_active_index_flips(db: Database, world: Any) -> None:
    built = taste.build(db, seed=0)
    taste.assert_fresh(db, built)
    moved = install_index(db, world, revision="2")
    assert moved != built.index_id
    back = taste.load(db, built.profile_id)
    assert back is not None
    assert back.stale
    assert built.index_id in str(back.stale_reason)
    assert moved in str(back.stale_reason)
    with pytest.raises(StaleArtifact):
        taste.assert_fresh(db, back)
    assert taste.latest(db) is None


def test_activating_the_old_index_again_does_not_unstale_the_row(db: Database, world: Any) -> None:
    built = taste.build(db, seed=0)
    install_index(db, world, revision="2")
    verify.activate(db, built.index_id)
    back = taste.load(db, built.profile_id)
    assert back is not None
    assert back.stale, "a stale row is kept stale so old numbers stay readable"


def test_a_cutoff_leaves_out_everything_watched_after_it(db: Database, world: Any) -> None:
    middle = sorted(r.watched_at for r in world.rated if r.watched_at)[len(world.rated) // 2]
    built = taste.build(db, cutoff=middle, seed=0)
    assert built.cutoff_date == middle
    assert built.n_rated < len(world.rated)
    assert taste.latest(db, cutoff=middle) is not None
    assert taste.latest(db) is None


def test_a_thin_history_gets_modes_but_no_ridge(blank: Database) -> None:
    thin = small_world(rated_per_cluster=8, rated_background=0)
    install(blank, thin)
    built = taste.build(blank, seed=0)
    assert built.tier == "thin"
    assert built.direction is None
    assert built.affinities["genre"]


def test_a_cold_history_gets_neither(blank: Database) -> None:
    cold = small_world(rated_per_cluster=4, rated_background=0)
    install(blank, cold)
    built = taste.build(blank, seed=0)
    assert built.tier == "cold"
    assert built.modes == ()
    assert built.anti_modes == ()
    assert built.direction is None


def test_a_history_with_no_ratings_at_all_is_refused(blank: Database, world: Any) -> None:
    install_index(blank, world)
    with pytest.raises(ThinHistoryError):
        taste.build(blank, seed=0)


def test_the_metadata_features_are_named_so_a_candidate_can_be_scored(db: Database) -> None:
    built = taste.build(db, seed=0)
    assert built.direction is not None
    names = built.direction.feature_names
    assert sum(1 for n in names if n.startswith("emb:")) == 48
    assert any(n.startswith("meta:decade:") for n in names)
    assert any(n.startswith("meta:lang:") for n in names)
    assert names[-2:] == ("meta:log_popularity", "meta:log_vote_count")


def store(db: Database) -> PreferenceStore:
    ensure_session(db, SESSION)
    return PreferenceStore(db)


def draft(**kwargs: Any) -> PreferenceDraft:
    base: dict[str, Any] = {
        "target_kind": "genre",
        "target_id": "10402",
        "target_label": "musicals",
        "polarity": "dislike",
        "strength": 3,
        "hardness": "hard",
        "scope": "durable",
        "evidence_quote": "I hate musicals",
        "resolved_ids": ("10402",),
        "affected_films": 812,
    }
    return PreferenceDraft(**(base | kwargs))


def test_a_quote_nobody_said_is_refused(db: Database) -> None:
    with pytest.raises(PreferenceRefused, match="quote not found"):
        store(db).record(
            draft(evidence_quote="I adore musicals"), session_id=SESSION, user_messages=[SAID]
        )


def test_only_the_users_own_words_count_as_evidence(db: Database) -> None:
    echoed = Message(role="assistant", content="I hate musicals")
    with pytest.raises(PreferenceRefused):
        store(db).record(draft(), session_id=SESSION, user_messages=[echoed])


def test_a_hard_absolute_dislike_becomes_a_sql_exclusion(db: Database) -> None:
    prefs = store(db)
    saved = prefs.record(draft(), session_id=SESSION, user_messages=[SAID])
    assert saved.affected_films == 812
    compiled = prefs.as_filter(SESSION)
    assert compiled.exclude["genre"] == frozenset({"10402"})
    assert prefs.as_penalties(SESSION) == {}


def test_a_soft_preference_becomes_a_penalty_not_a_filter(db: Database) -> None:
    prefs = store(db)
    prefs.record(
        draft(
            target_kind="runtime",
            target_id="120",
            target_label="anything long",
            strength=1,
            hardness="soft",
            evidence_quote="keep it under two hours tonight",
            resolved_ids=(),
        ),
        session_id=SESSION,
        user_messages=[SAID],
    )
    assert not prefs.as_filter(SESSION)
    assert prefs.as_penalties(SESSION) == {"runtime:120": -0.5}


def test_a_hard_runtime_preference_becomes_a_cap(db: Database) -> None:
    prefs = store(db)
    prefs.record(
        draft(
            target_kind="runtime",
            target_id="120",
            target_label="anything over two hours",
            evidence_quote="keep it under two hours tonight",
            resolved_ids=(),
        ),
        session_id=SESSION,
        user_messages=[SAID],
    )
    assert prefs.as_filter(SESSION).max_runtime == 120


def test_a_contradiction_supersedes_and_the_newest_wins(db: Database) -> None:
    prefs = store(db)
    first = prefs.record(draft(), session_id=SESSION, user_messages=[SAID])
    second = prefs.record(
        draft(polarity="like", evidence_quote="I hate musicals"),
        session_id=SESSION,
        user_messages=[SAID],
    )
    live = prefs.active(SESSION)
    assert [p.pref_id for p in live] == [second.pref_id]
    assert prefs.get(first.pref_id) is not None
    assert prefs.as_filter(SESSION).require["genre"] == frozenset({"10402"})


def test_as_of_rebuilds_the_state_at_a_past_moment(db: Database) -> None:
    prefs = store(db)
    early = datetime(2026, 9, 1, 12, 0, 0)
    with frozen(early):
        first = prefs.record(draft(), session_id=SESSION, user_messages=[SAID])
    with frozen(early + timedelta(days=2)):
        prefs.record(
            draft(polarity="like", evidence_quote="I hate musicals"),
            session_id=SESSION,
            user_messages=[SAID],
        )
    was = prefs.active(SESSION, as_of=early + timedelta(days=1))
    assert [p.pref_id for p in was] == [first.pref_id]
    assert was[0].polarity == "dislike"
    assert prefs.active(SESSION, as_of=early - timedelta(days=1)) == []


def test_undo_supersedes_instead_of_deleting(db: Database) -> None:
    prefs = store(db)
    saved = prefs.record(draft(), session_id=SESSION, user_messages=[SAID])
    assert prefs.undo(saved.pref_id) is True
    assert prefs.active(SESSION) == []
    assert prefs.get(saved.pref_id) is not None
    assert prefs.undo(saved.pref_id) is False


def test_a_session_preference_does_not_leak_into_another_session(db: Database) -> None:
    prefs = store(db)
    ensure_session(db, "ses_other")
    prefs.record(draft(scope="session"), session_id=SESSION, user_messages=[SAID])
    assert prefs.active(SESSION)
    assert prefs.active("ses_other") == []
    assert prefs.active(None) == []


def test_an_empty_store_says_nothing_at_all(db: Database) -> None:
    assert PreferenceStore(db).as_prompt_block(SESSION) == ""


def test_the_prompt_block_lists_what_was_said_and_when(db: Database) -> None:
    prefs = store(db)
    prefs.record(draft(), session_id=SESSION, user_messages=[SAID])
    block = prefs.as_prompt_block(SESSION)
    assert "Saved preferences" in block
    assert "dislikes musicals" in block
    # A durable preference is the user's, not the session's, so it travels.
    assert prefs.as_prompt_block("ses_other") == block
    assert PreferenceStore(db).as_prompt_block(None) == block


def test_the_whole_profile_survives_a_reopen(tmp_path: Path, world: Any) -> None:
    path = tmp_path / "palate.db"
    first = open_database(path, migrations=migrations_dir(), load_vec=True)
    install(first, world)
    built = taste.build(first, seed=0)
    first.close()
    second = open_database(path, migrations=migrations_dir(), load_vec=True)
    try:
        back = taste.load(second, built.profile_id)
        assert back is not None
        assert back.n_rated == built.n_rated
        assert np.allclose(back.modes[0].centroid, built.modes[0].centroid, atol=1e-6)
        taste.assert_fresh(second, back)
    finally:
        second.close()


def test_a_cutoff_profile_and_a_live_one_can_coexist(db: Database, world: Any) -> None:
    live = taste.build(db, seed=0)
    middle = sorted(r.watched_at for r in world.rated if r.watched_at)[len(world.rated) // 2]
    sliced = taste.build(db, cutoff=middle, seed=0)
    newest = taste.latest(db)
    sliced_back = taste.latest(db, cutoff=middle)
    assert newest is not None and newest.profile_id == live.profile_id
    assert sliced_back is not None and sliced_back.profile_id == sliced.profile_id
    assert sliced.params_sha != live.params_sha


def test_the_rating_histogram_and_mean_come_back_from_the_fit(db: Database, world: Any) -> None:
    built = taste.build(db, seed=0)
    back = taste.load(db, built.profile_id)
    assert back is not None
    assert back.rating_histogram == built.rating_histogram
    assert sum(built.rating_histogram.values()) == len(world.rated)
    assert back.mean_rating == pytest.approx(
        sum(r.rating_half for r in world.rated) / len(world.rated) / 2.0
    )
    assert built.cutoff_date is None
    assert date.fromisoformat(built.built_at[:10]).year >= 2026
