#!/usr/bin/env python
"""Run the Phase-1 gate: 12-1 momentum through the full engine.

    reports/status/phase1.json
    reports/figures/momentum_gate.png

Thin caller — all logic lives in src/master_us/backtest/momentum.py.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import polars as pl

from master_us.backtest.momentum import (
    FIGURES_DIR,
    GATE_MEMBERSHIP_CACHE,
    evaluate_gate,
    load_cost_config,
    load_wide_panel,
    plot_equity,
)
from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.reporting.status import phase_panel


def spy_returns(dates: pd.DatetimeIndex) -> np.ndarray:
    spy = pl.read_parquet(PRICE_CACHE_DIR / "SPY.parquet").sort("date").to_pandas()
    adj = spy.set_index("date")["adj_close"].reindex(dates)
    if adj.isna().any():
        missing = int(adj.isna().sum())
        raise ValueError(f"SPY is missing {missing} of the engine's dates — refetch it")
    r = np.array(adj.pct_change().to_numpy(), dtype=np.float64)
    r[0] = 0.0
    return r


def main() -> int:
    if not GATE_MEMBERSHIP_CACHE.exists():
        print(f"error: {GATE_MEMBERSHIP_CACHE} missing — run scripts/10_fetch_gate_data.py")
        return 1

    print("assembling wide panel (2006 -> 2025, extended gate window)")
    panel = load_wide_panel(GATE_MEMBERSHIP_CACHE, "2006-01-01", "2025-12-31")
    print(
        f"  {len(panel.dates)} dates x {len(panel.tickers)} tickers · "
        f"tradeable/day mean {panel.tradeable.sum(axis=1).mean():.0f}"
    )

    outcome = evaluate_gate(panel, load_cost_config(), None, eval_start="2008-01-01")
    # Benchmark joined after the engine window is known.
    bench = spy_returns(outcome.eval_dates)
    from master_us.backtest.metrics import excess_return, information_ratio

    outcome.metrics["excess_return"] = excess_return(outcome.result_baseline.series, bench)
    outcome.metrics["information_ratio"] = information_ratio(
        outcome.result_baseline.series, bench
    )

    figure = plot_equity(outcome, FIGURES_DIR / "momentum_gate.png")
    result = outcome.phase_result
    result = type(result)(
        **{**result.__dict__, "artifacts": [figure], "metrics": outcome.metrics}
    )
    path = result.save()

    phase_panel(result)
    print("gate checks:")
    for name, ok in outcome.checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"wrote {path}")
    print(f"wrote {figure}")
    return 0 if result.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
