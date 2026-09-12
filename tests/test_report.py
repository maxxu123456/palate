"""The recorded matrix, and the readme block that has to keep matching it."""

from __future__ import annotations

from pathlib import Path

from fixtures.eval import record

from palate.eval import report as reporting
from palate.eval.systems import ABLATIONS, BY_NAME, blocked_reason

README = Path(__file__).resolve().parents[1] / "README.md"

# Arms whose knob cannot move an unconditioned run, so matching their base is the right answer.
INERT = {"+bm25": "dense_only", "full - bm25": "full"}

LIVE = {"+repulsion": "dense_only", "+ridge": "dense_only", "full - people_priors": "full"}


def block() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(reporting.START_MARKER) + len(reporting.START_MARKER)
    return text[start : text.index(reporting.END_MARKER)].strip()


def test_the_readme_table_is_what_the_recorded_run_renders() -> None:
    rows, deltas = record.load()
    assert block() == reporting.summary(rows, deltas).strip()


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
