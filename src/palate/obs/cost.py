"""One authority for pricing. pricing.toml is a seed, cost_rates is what cost_for reads."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from palate.clock import now_iso
from palate.db.connect import Database
from palate.providers.base import Usage

type CostSource = Literal["provider", "table", "unknown"]

# A rate row whose model is this one prices every model that provider serves.
ANY_MODEL = "*"

PER_MTOK = 1_000_000.0

_RATE = (
    "select input_per_mtok, output_per_mtok, cached_input_per_mtok from current_rates "
    "where provider = ? and model in (?, ?) order by model = ? desc limit 1"
)

_INSERT = (
    "insert or replace into cost_rates (provider, model, currency, input_per_mtok, "
    "output_per_mtok, cached_input_per_mtok, source, fetched_at) values (?,?,?,?,?,?,?,?)"
)


@dataclass(frozen=True, slots=True)
class CostResult:
    """What one call cost, and how confidently."""

    usd: float
    source: CostSource

    @property
    def known(self) -> bool:
        """Unknown is not zero, and a total that hides one is quietly short."""
        return self.source != "unknown"


def cost_for(
    db: Database,
    provider: str,
    model: str,
    usage: Usage,
    reported_usd: float | None,
) -> CostResult:
    """Provider-reported first, then the current_rates view, then unknown."""
    if reported_usd is not None:
        return CostResult(float(reported_usd), "provider")
    row = db.read().execute(_RATE, (provider, model, ANY_MODEL, model)).fetchone()
    if row is None:
        return CostResult(0.0, "unknown")
    cached_rate = row["cached_input_per_mtok"]
    fresh = max(usage.input_tokens - usage.cached_input_tokens, 0)
    usd = fresh * float(row["input_per_mtok"]) / PER_MTOK
    usd += usage.output_tokens * float(row["output_per_mtok"]) / PER_MTOK
    if cached_rate is not None:
        usd += usage.cached_input_tokens * float(cached_rate) / PER_MTOK
    else:
        usd += usage.cached_input_tokens * float(row["input_per_mtok"]) / PER_MTOK
    return CostResult(usd, "table")


def seed_rates(db: Database, toml_path: Path, *, source: str = "pricing.toml") -> int:
    """Load pricing.toml into cost_rates. Runs at startup, never read at pricing time."""
    if not toml_path.exists():
        return 0
    parsed = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    rates = parsed.get("rate", [])
    stamp = now_iso()
    rows = [
        (
            str(rate["provider"]),
            str(rate.get("model", ANY_MODEL)),
            str(rate.get("currency", "USD")),
            float(rate["input_per_mtok"]),
            float(rate["output_per_mtok"]),
            _optional(rate.get("cached_input_per_mtok")),
            source,
            stamp,
        )
        for rate in rates
    ]
    if not rows:
        return 0
    with db.write() as conn:
        conn.executemany(_INSERT, rows)
    return len(rows)


def _optional(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
