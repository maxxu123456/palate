"""The table, printed from what the runs produced, whether or not it flatters the design."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from palate.db.connect import Database
from palate.eval.metrics import MetricCI
from palate.eval.split import Split
from palate.eval.systems import ABLATIONS, BY_NAME, blocked_reason
from palate.index import verify

START_MARKER = "<!-- eval-table:start -->"
END_MARKER = "<!-- eval-table:end -->"

REFERENCE = "full"
BASELINE = "director_affinity"

EMPTY = "no eval runs are stored for this split yet"


@dataclass(frozen=True, slots=True)
class Row:
    """One arm, folded across folds, as the table prints it."""

    system: str
    metrics: Mapping[str, MetricCI] = field(default_factory=dict)
    folds: int = 0
    elapsed_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class Delta:
    """One paired comparison as it was stored."""

    delta: float
    lo: float
    hi: float
    folds_positive: int
    folds_total: int

    @property
    def marker(self) -> str:
        """An interval spanning zero is no measurable effect, never a small improvement."""
        return " ns" if self.lo <= 0.0 <= self.hi else ""


def _pooled(values: Sequence[MetricCI]) -> MetricCI:
    """Folds averaged. A summary of several intervals, not one pooled interval."""
    return MetricCI(
        float(np.mean([v.point for v in values])),
        float(np.mean([v.lo for v in values])),
        float(np.mean([v.hi for v in values])),
        int(sum(v.n for v in values)),
    )


def load_rows(db: Database, split_name: str, *, condition: str = "unconditioned") -> list[Row]:
    """Every arm that has runs stored, folded across folds in registry order."""
    conn = db.read()
    runs = conn.execute(
        "select run_id, system, elapsed_ms, cost_usd from eval_run "
        "where split_name = ? and condition = ? order by system, fold",
        (split_name, condition),
    ).fetchall()
    if not runs:
        return []
    by_system: dict[str, list[sqlite3.Row]] = {}
    for run in runs:
        by_system.setdefault(str(run["system"]), []).append(run)
    metrics = _metrics_by_run(conn, split_name, condition)
    ordered = [cfg.name for cfg in ABLATIONS if cfg.name in by_system]
    ordered += sorted(name for name in by_system if name not in BY_NAME)
    out: list[Row] = []
    for system in ordered:
        rows = by_system[system]
        gathered: dict[str, list[MetricCI]] = {}
        for run in rows:
            for name, ci in metrics.get(str(run["run_id"]), {}).items():
                gathered.setdefault(name, []).append(ci)
        out.append(
            Row(
                system=system,
                metrics={name: _pooled(v) for name, v in gathered.items()},
                folds=len(rows),
                elapsed_ms=float(np.median([float(r["elapsed_ms"]) for r in rows])),
                cost_usd=float(sum(float(r["cost_usd"]) for r in rows)),
            )
        )
    return out


def _metrics_by_run(
    conn: sqlite3.Connection, split_name: str, condition: str
) -> dict[str, dict[str, MetricCI]]:
    rows = conn.execute(
        "select m.run_id as run_id, m.metric as metric, m.value as value, m.ci_lo as ci_lo, "
        "m.ci_hi as ci_hi, m.n as n from eval_metric m join eval_run r on r.run_id = m.run_id "
        "where r.split_name = ? and r.condition = ?",
        (split_name, condition),
    )
    out: dict[str, dict[str, MetricCI]] = {}
    for row in rows:
        out.setdefault(str(row["run_id"]), {})[str(row["metric"])] = MetricCI(
            float(row["value"]),
            float(row["ci_lo"] if row["ci_lo"] is not None else row["value"]),
            float(row["ci_hi"] if row["ci_hi"] is not None else row["value"]),
            int(row["n"] or 0),
        )
    return out


def load_deltas(
    db: Database, split_name: str, *, condition: str = "unconditioned"
) -> dict[tuple[str, str], Delta]:
    """Stored paired deltas, keyed by (system, reference)."""
    rows = db.read().execute(
        "select system, reference, delta, ci_lo, ci_hi, folds_positive, folds_total "
        "from eval_delta where split_name = ? and condition = ?",
        (split_name, condition),
    )
    return {
        (str(r["system"]), str(r["reference"])): Delta(
            float(r["delta"]),
            float(r["ci_lo"]),
            float(r["ci_hi"]),
            int(r["folds_positive"]),
            int(r["folds_total"]),
        )
        for r in rows
    }


def header_block(db: Database, split: Split, *, surviving: tuple[int, int] | None = None) -> str:
    """The caveats, first, because they decide how the table below is read."""
    lines: list[str] = []
    if split.degraded and split.degraded_reason:
        lines.extend(["```", split.degraded_reason, "```", ""])
    days = ", ".join(f"{day}: {n} films" for day, n in split.catalogue_days.items()) or "none"
    cuts = " ".join(f"p{int(c * 100)}" for c in split.spec.cuts)
    rated = split.n_reliable + split.n_dropped_unreliable
    coverage = " ".join(f"{k} {v:.2f}" for k, v in split.coverage_by_region.items())
    test = " / ".join(str(n) for n in split.test_per_fold)
    weak = [f.fold for f in split.folds if f.underpowered]
    fields = [
        ("split", f"{split.strategy}, {len(split.folds)} folds, cuts at {cuts}"),
        ("reliable dated", f"{split.n_reliable} of {rated} rated films"),
        ("catalogue days", f"{len(split.catalogue_days)} detected ({days})"),
        ("", f"those {split.n_dropped_unreliable} films are train-only in every fold"),
        ("test per fold", test + (f"   (underpowered: {weak})" if weak else "")),
        ("test coverage", f"{split.test_coverage:.2f} overall   {coverage}".rstrip()),
        (
            "dropped",
            f"{split.n_dropped_rewatch} rewatch, {split.n_dropped_not_in_corpus} off corpus",
        ),
    ]
    if surviving is not None:
        fields.append(("review queries", f"{surviving[0]} of {surviving[1]} survived stripping"))
    fields.append(("embedding", _index_line(db)))
    width = max(len(name) for name, _ in fields)
    lines.append("```")
    lines.extend(f"{name.ljust(width)}  {value}" for name, value in fields)
    lines.append("```")
    return "\n".join(lines)


def _index_line(db: Database) -> str:
    record = verify.active(db)
    mark = record.fingerprint
    return (
        f"{mark.model_id} via {mark.provider}, {mark.dim}d, doc {mark.doc_template_version}, "
        f"index {record.index_id[:8]}, {record.n_vectors} vectors"
    )


def _cell(ci: MetricCI | None) -> str:
    return "" if ci is None else f"{ci.point:.3f}"


def _interval(ci: MetricCI | None) -> str:
    return "" if ci is None else f"{ci.lo:.2f}-{ci.hi:.2f}"


def _delta_cell(found: Delta | None) -> str:
    if found is None:
        return ""
    return f"{found.delta:+.3f}{found.marker}"


def main_table(
    rows: Sequence[Row],
    deltas: Mapping[tuple[str, str], Delta],
    *,
    reference: str = REFERENCE,
    baseline: str = BASELINE,
) -> str:
    """One row per arm, with both deltas and the fold sign count beside every number."""
    if not rows:
        return EMPTY
    head = (
        f"| arm | ndcg@10 | 95% CI | ndcg@10_watch | recall@50 | d vs {reference} | "
        f"d vs {baseline} | folds | p50 ms |"
    )
    out = [head, "|" + "---|" * 9]
    for row in rows:
        against = deltas.get((row.system, reference))
        versus = deltas.get((row.system, baseline))
        signs = f"{against.folds_positive}/{against.folds_total}" if against else ""
        out.append(
            "| "
            + " | ".join(
                [
                    row.system,
                    _cell(row.metrics.get("ndcg@10")),
                    _interval(row.metrics.get("ndcg@10")),
                    _cell(row.metrics.get("ndcg@10_watch")),
                    _cell(row.metrics.get("recall@50")),
                    _delta_cell(against),
                    _delta_cell(versus),
                    signs,
                    f"{row.elapsed_ms:.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(out)


def query_table(db: Database, split: Split) -> str:
    """MRR and recall over the review and synthetic query sets, per condition."""
    blocks: list[str] = []
    for condition in ("query_review", "query_synth", "query_cold"):
        rows = load_rows(db, split.name, condition=condition)
        if not rows:
            continue
        blocks.append(f"**{condition}**\n")
        blocks.append("| arm | mrr@50 | 95% CI | recall@50 | folds |")
        blocks.append("|" + "---|" * 5)
        for row in rows:
            blocks.append(
                "| "
                + " | ".join(
                    [
                        row.system,
                        _cell(row.metrics.get("mrr@50")),
                        _interval(row.metrics.get("mrr@50")),
                        _cell(row.metrics.get("recall@50")),
                        str(row.folds),
                    ]
                )
                + " |"
            )
        blocks.append("")
    return "\n".join(blocks).strip()


def weights_table(db: Database, split: Split) -> str:
    """Per feature median, spread, sign agreement and how often L1 kept it."""
    rows = (
        db.read()
        .execute(
            "select condition, feature, mean_beta, sd_beta, sign_agree, n_active "
            "from eval_weight_stability where split_name = ? order by condition, feature",
            (split.name,),
        )
        .fetchall()
    )
    if not rows:
        return "fusion weights were never fitted for this split, so every arm ran the signed prior"
    out = ["| condition | feature | mean beta | sd | sign agree | folds active |", "|" + "---|" * 6]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    str(row["feature"]),
                    f"{float(row['mean_beta']):+.3f}",
                    f"{float(row['sd_beta']):.3f}",
                    str(int(row["sign_agree"])),
                    str(int(row["n_active"])),
                ]
            )
            + " |"
        )
    return "\n".join(out)


def not_run() -> str:
    """Arms in the registry that cannot produce a number yet, and why."""
    blocked = [(cfg.name, blocked_reason(cfg)) for cfg in ABLATIONS]
    live = [(name, why) for name, why in blocked if why is not None]
    if not live:
        return ""
    return "\n".join(f"- `{name}`: {why}" for name, why in live)


def honest_notes(split: Split, rows: Sequence[Row]) -> str:
    """The caveats in prose, including the ones that make the numbers look worse."""
    notes = [
        "`vote_average` and `popularity` are present-day values used as features for films "
        "watched years ago, which carries future popularity backward. It helps the popularity "
        "and director baselines at least as much as the full system.",
        "Anything outside the test window is unjudged and scored zero, so every absolute number "
        "here is a lower bound.",
        "The interval resamples the test item set against a fixed ranking. It measures how much "
        "the result depends on which films landed in this window, and nothing about the model "
        "fit or about taste drifting.",
        "A delta marked `ns` has an interval spanning zero. That is no measurable effect, not a "
        "small improvement. A result inside the interval but consistent across every fold is "
        "suggestive, never significant.",
        "No item-item collaborative filtering. One user, no co-rating matrix, nothing to "
        "collaborate with.",
    ]
    if split.degraded:
        notes.insert(0, "The temporal protocol was refused for this history, see the banner above.")
    if split.test_coverage < 0.85:
        notes.append(
            f"Test coverage is {split.test_coverage:.2f}, below 0.85. The crawl is the bug here, "
            "not the model."
        )
    missing = [r.system for r in rows if not r.metrics]
    if missing:
        notes.append(f"No metrics were stored for {', '.join(missing)}.")
    return "\n".join(f"- {n}" for n in notes)


def render(
    db: Database,
    split: Split,
    *,
    surviving: tuple[int, int] | None = None,
    reference: str = REFERENCE,
    baseline: str = BASELINE,
) -> str:
    """The whole report, header first, in the order a reader should meet it."""
    rows = load_rows(db, split.name)
    deltas = load_deltas(db, split.name)
    parts = [
        "## Evaluation",
        "",
        header_block(db, split, surviving=surviving),
        "",
        main_table(rows, deltas, reference=reference, baseline=baseline),
        "",
    ]
    queries = query_table(db, split)
    if queries:
        parts.extend(["### Query mode", "", queries, ""])
    parts.extend(["### Fusion weights", "", weights_table(db, split), ""])
    skipped = not_run()
    if skipped:
        parts.extend(["### Not run yet", "", skipped, ""])
    parts.extend(["### Honest notes", "", honest_notes(split, rows), ""])
    return "\n".join(parts)


def paste_into(path: Path, table: str) -> bool:
    """Replace whatever sits between the two markers. The README cannot drift from the code."""
    text = path.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        return False
    fresh = f"{START_MARKER}\n\n{table.strip()}\n\n{text[end:]}"
    path.write_text(text[:start] + fresh, encoding="utf-8")
    return True
