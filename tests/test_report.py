"""The recorded matrix and the readme line budget."""

from __future__ import annotations

from pathlib import Path

from fixtures.eval import record

from palate.eval.systems import ABLATIONS, BY_NAME, blocked_reason

README = Path(__file__).resolve().parents[1] / "README.md"

# Arms whose knob cannot move an unconditioned run, so matching their base is the right answer.
INERT = {"+bm25": "dense_only", "full - bm25": "full"}

LIVE = {"+repulsion": "dense_only", "+ridge": "dense_only", "full - people_priors": "full"}


def test_the_readme_stays_inside_its_line_budget() -> None:
    assert len(README.read_text(encoding="utf-8").splitlines()) <= 80


def test_every_arm_the_registry_can_run_was_recorded() -> None:
    rows, _ = record.load()
    runnable = {cfg.name for cfg in ABLATIONS if blocked_reason(cfg) is None}
    assert {row.system for row in rows} == runnable
    assert runnable < set(BY_NAME)


def test_only_a_knob_that_cannot_bite_matches_its_base() -> None:
    rows, _ = record.load()
    point = {row.system: row.metrics["ndcg@10"].point for row in rows}
    # No query text in this suite means no lexical channel, so the bm25 arms cannot differ.
    for arm, base in INERT.items():
        assert point[arm] == point[base], arm
    for arm, base in LIVE.items():
        assert point[arm] != point[base], arm
