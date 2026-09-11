"""The six baselines, the every-director join, and the query sets built from real reviews."""

from __future__ import annotations

from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
from fixtures.synth import pipeline

from palate.db.connect import Database
from palate.eval.baselines import (
    BASELINES,
    FoldInputs,
    baseline_director_affinity,
    baseline_people_affinity,
    baseline_popularity,
    baseline_popularity_era,
    baseline_random,
    baseline_single_centroid,
    credits_of,
    preferred_decades,
    shrunk,
)
from palate.eval.labels import labels_of
from palate.eval.metrics import ndcg_at_k
from palate.eval.queries import (
    QueryCase,
    build_review_queries,
    build_synthetic_queries,
    review_survival,
    strip_identity,
)
from palate.eval.split import Fold, Split, SplitSpec, build_split, load_corpus_ids
from palate.eval.systems import ABLATIONS, BY_NAME, SMOKE, blocked_reason, by_name, resolve
from palate.index import verify
from palate.index.vecstore import VecStore
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.taste import profile as taste
from palate.taste.profile import TasteProfile

SPEC = SplitSpec(name="t", min_reliable=50, cuts=(0.6, 0.8))


def cut(tmp_path: Path) -> tuple[pipeline.Fitted, Split]:
    """One fitted world and a split over it, with the floors lowered to fixture size."""
    fitted = pipeline.fit(tmp_path, docs=False)
    rated = taste.load_rated(fitted.db.read())
    split = build_split(rated, load_corpus_ids(fitted.db.read()), SPEC, relevance=labels_of)
    return fitted, split


def inputs(db: Database, fold: Fold, profile: TasteProfile) -> FoldInputs:
    """Candidates are the eligible corpus minus what the user had already seen by the cut."""
    conn = db.read()
    candidates = tuple(sorted(load_corpus_ids(conn) - set(fold.train)))
    record = verify.active(db)
    store = VecStore(conn, table=record.table_name, dim=record.fingerprint.dim)
    wanted = [*candidates, *fold.train]
    return FoldInputs(
        conn=conn,
        profile=profile,
        candidates=candidates,
        train=fold.train,
        vectors=store.vectors(wanted),
    )


def fold_profile(db: Database, fold: Fold) -> TasteProfile:
    return taste.build(db, cutoff=fold.t_split + timedelta(days=1))


def test_every_test_film_is_reachable_by_a_baseline(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    assert set(fold.test) <= set(ctx.candidates)
    assert not set(fold.train) & set(ctx.candidates)
    fitted.close()


def test_each_baseline_returns_the_candidate_set_exactly_once(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    for name, run in BASELINES.items():
        ranked = run(ctx, fold)
        assert len(ranked) == len(ctx.candidates), name
        assert set(ranked) == set(ctx.candidates), name
    fitted.close()


def test_random_is_seeded_and_moves_between_folds(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    assert baseline_random(ctx, fold, seed=3) == baseline_random(ctx, fold, seed=3)
    assert baseline_random(ctx, fold, seed=3) != baseline_random(ctx, fold, seed=4)
    assert baseline_random(ctx, fold) != baseline_random(ctx, split.folds[1])
    fitted.close()


def test_popularity_is_vote_count_descending(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    ranked = baseline_popularity(ctx, fold)
    votes = {f.tmdb_id: f.vote_count for f in fitted.world.films}
    head = [votes[i] for i in ranked[:20]]
    assert head == sorted(head, reverse=True)
    fitted.close()


def test_popularity_era_puts_the_favoured_decades_first(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    profile = fold_profile(fitted.db, fold)
    ctx = inputs(fitted.db, fold, profile)
    liked = preferred_decades(profile)
    assert liked
    ranked = baseline_popularity_era(ctx, fold)
    decade = {f.tmdb_id: (f.year // 10) * 10 for f in fitted.world.films}
    assert all(decade[i] in liked for i in ranked[:10])
    fitted.close()


def test_the_director_baseline_beats_random_and_popularity(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    metric = partial(ndcg_at_k, k=50)
    director = metric(baseline_director_affinity(ctx, fold), fold.rel)
    assert director > metric(baseline_random(ctx, fold), fold.rel)
    assert director > metric(baseline_popularity(ctx, fold), fold.rel)
    fitted.close()


def test_a_second_director_credit_counts_as_much_as_the_first(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    profile = fold_profile(fitted.db, fold)
    ctx = inputs(fitted.db, fold, profile)
    loved = max(
        shrunk(profile.affinities["director"], min_films=2, kappa=2.0).items(), key=lambda kv: kv[1]
    )[0]
    obscure = next(
        i for i in ctx.candidates if baseline_director_affinity(ctx, fold).index(i) > 200
    )
    before = baseline_director_affinity(ctx, fold).index(obscure)
    with fitted.db.write() as conn:
        conn.execute(
            "insert into credits (credit_id, tmdb_id, person_id, credit_kind, department, job, ord)"
            " values (?,?,?,'crew','Directing','Director',1)",
            (f"c{obscure}dir2", obscure, loved),
        )
    fresh = inputs(fitted.db, fold, profile)
    assert loved in credits_of(fresh, "director")[obscure]
    assert baseline_director_affinity(fresh, fold).index(obscure) < before
    fitted.close()


def test_the_people_baseline_reads_writers_and_billing_too(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    people = baseline_people_affinity(ctx, fold)
    assert people != baseline_director_affinity(ctx, fold)
    metric = partial(ndcg_at_k, k=50)
    assert metric(people, fold.rel) > metric(baseline_random(ctx, fold), fold.rel)
    fitted.close()


def test_the_single_centroid_is_similarity_and_nothing_else(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    ctx = inputs(fitted.db, fold, fold_profile(fitted.db, fold))
    ranked = baseline_single_centroid(ctx, fold)
    metric = partial(ndcg_at_k, k=50)
    assert metric(ranked, fold.rel) > metric(baseline_random(ctx, fold), fold.rel)
    empty = FoldInputs(conn=ctx.conn, profile=ctx.profile, candidates=ctx.candidates)
    assert baseline_single_centroid(empty, fold) == baseline_popularity(empty, fold)
    fitted.close()


def test_review_queries_lose_the_title_the_director_and_the_cast(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    target = fold.test[0]
    row = (
        fitted.db.read()
        .execute(
            "select f.title as title, p.name as name from films f join credits c "
            "on c.tmdb_id = f.tmdb_id and c.job = 'Director' join people p "
            "on p.person_id = c.person_id where f.tmdb_id = ?",
            (target,),
        )
        .fetchone()
    )
    surname = str(row["name"]).split()[-1]
    review = (
        f"{row['title']} is the coldest thing {surname} ever shot, all wet concrete and waiting, "
        "and the last twenty minutes just sit there daring you to look away from the puddle."
    )
    with fitted.db.write() as conn:
        conn.execute("update user_films set review_text = ? where tmdb_id = ?", (review, target))
        conn.execute(
            "update user_films set review_text = 'too short' where tmdb_id = ?", (fold.test[1],)
        )
    cases = build_review_queries(fitted.db, fold)
    assert [c.target_tmdb_id for c in cases] == [target]
    case = cases[0]
    assert surname.casefold() not in case.text.casefold()
    assert str(row["title"]).split()[-1].casefold() not in case.text.casefold()
    assert case.n_chars == len(case.text) >= 60
    assert case.source == "review"
    assert review_survival(fitted.db, fold) == (1, 2)
    fitted.close()


def test_strip_identity_only_removes_whole_words() -> None:
    assert strip_identity("a stalker stalking", ["stalker"]) == "a stalking"
    assert strip_identity("nothing to cut", []) == "nothing to cut"
    assert strip_identity("Tarr shot it", ["tarr"]) == "shot it"


async def test_synthetic_queries_come_back_labelled_circular(tmp_path: Path) -> None:
    fitted, split = cut(tmp_path)
    fold = split.folds[0]
    sentence = "Something slow and cold where very little happens and the weather does the talking."
    chat = FakeChatProvider([ScriptedTurn(text=sentence)], loop_last=True)
    cases = await build_synthetic_queries(fitted.db, fold, chat, n=3, model="fake-model")
    assert len(cases) == 3
    assert all(isinstance(c, QueryCase) and c.source == "synthetic" for c in cases)
    assert all(c.query_id.startswith("synth:fake-model:") for c in cases)
    assert all(c.text == sentence for c in cases)
    assert {c.target_tmdb_id for c in cases} <= set(fold.test)
    fitted.close()


def test_the_registry_names_every_arm_once() -> None:
    assert len(BY_NAME) == len(ABLATIONS)
    assert len({cfg.config_sha for cfg in ABLATIONS}) == len(ABLATIONS)
    assert set(SMOKE) <= set(BY_NAME)
    assert set(BASELINES) <= set(BY_NAME)
    assert all(by_name(name).is_baseline for name in BASELINES)


def test_an_arm_that_cannot_run_yet_says_why() -> None:
    assert blocked_reason(by_name("full")) is None
    assert blocked_reason(by_name("director_affinity")) is None
    assert "reranker" in str(blocked_reason(by_name("+cross_encoder(minilm)")))
    assert "credits" in str(blocked_reason(by_name("doc_no_credits")))


def test_resolve_takes_names_or_everything() -> None:
    assert resolve(None) == ABLATIONS
    assert [c.name for c in resolve(["full", "popularity"])] == ["full", "popularity"]
    with pytest.raises(KeyError):
        resolve(["nope"])
