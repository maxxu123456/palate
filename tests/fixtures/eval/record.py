"""The recorded matrix behind the readme table. Regenerate with `python -m fixtures.eval.record`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import anyio
import orjson

from palate.eval import report as reporting
from palate.eval.harness import EvalContext, deltas, persist, publish_weights, run_matrix
from palate.eval.labels import labels_of
from palate.eval.metrics import MetricCI
from palate.eval.split import SplitSpec, build_split, freeze_split, load_corpus_ids, load_regions
from palate.eval.systems import ABLATIONS, BY_NAME, FULL
from palate.taste import profile as taste

RESULTS = Path(__file__).with_name("results.json")

SPEC = SplitSpec(name="rolling", min_reliable=50, cuts=(0.6, 0.8))

RESAMPLES = 400

# Only what the table prints. Timings are machine dependent and would not reproduce.
KEEP = ("ndcg@10", "ndcg@10_watch", "recall@50")


def load() -> tuple[list[reporting.Row], dict[tuple[str, str], reporting.Delta]]:
    """The recorded rows and deltas, in the shape the report renders from."""
    stored: dict[str, Any] = orjson.loads(RESULTS.read_bytes())
    rows = [
        reporting.Row(
            system=str(row["system"]),
            metrics={k: MetricCI(*v) for k, v in row["metrics"].items()},
            folds=int(row["folds"]),
        )
        for row in stored["rows"]
    ]
    found = {
        (str(d["system"]), str(d["reference"])): reporting.Delta(
            float(d["delta"]),
            float(d["lo"]),
            float(d["hi"]),
            int(d["folds_positive"]),
            int(d["folds_total"]),
        )
        for d in stored["deltas"]
    }
    return rows, found


def record() -> str:
    """Run the whole matrix over the planted history and write what the table needs."""
    from fixtures.synth import pipeline

    with tempfile.TemporaryDirectory() as tmp:
        built = pipeline.fit(Path(tmp))
        rated = taste.load_rated(built.db.read())
        split = build_split(
            rated,
            load_corpus_ids(built.db.read()),
            SPEC,
            region_of=load_regions(built.db.read()),
            relevance=labels_of,
        )
        freeze_split(split, built.db)
        ctx = EvalContext(db=built.db, split=split, resamples=RESAMPLES)
        publish_weights(ctx, FULL)
        results = anyio.run(run_matrix, ABLATIONS, ctx)
        persist(ctx, BY_NAME, results)
        for reference in (reporting.REFERENCE, reporting.BASELINE):
            deltas(ctx, results, reference=reference)
        payload = _payload(built.db, split.name)
        built.close()
    RESULTS.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    return reporting.summary(*load())


def _payload(db: Any, split_name: str) -> dict[str, Any]:
    rows = reporting.load_rows(db, split_name)
    found = reporting.load_deltas(db, split_name)
    return {
        "source": "the planted history in tests/fixtures/synth, seeded and offline",
        "split": split_name,
        "resamples": RESAMPLES,
        "rows": [
            {
                "system": row.system,
                "folds": row.folds,
                "metrics": {
                    name: [ci.point, ci.lo, ci.hi, ci.n]
                    for name, ci in sorted(row.metrics.items())
                    if name in KEEP
                },
            }
            for row in rows
        ],
        "deltas": [
            {
                "system": system,
                "reference": reference,
                "delta": d.delta,
                "lo": d.lo,
                "hi": d.hi,
                "folds_positive": d.folds_positive,
                "folds_total": d.folds_total,
            }
            for (system, reference), d in sorted(found.items())
        ],
    }


if __name__ == "__main__":
    print(record())
