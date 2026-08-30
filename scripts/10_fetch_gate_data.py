#!/usr/bin/env python
"""Extend the data window backward for the Phase-1 momentum gate.

    data/raw/sp500_membership_2007.parquet   membership reconstructed from 2007
    data/raw/sp500_changes_2007.parquet
    data/raw/prices/*.parquet                backfilled 2006 -> 2010 (append-only)
    data/raw/prices/SPY.parquet              benchmark, full window

The canonical 2010+ sample and its caches are untouched: the gate needs 2008-9
in view (the momentum crash), the model sample deliberately does not include
it. Idempotent: cached membership is reused; the price backfill re-merges with
keep-existing semantics, so re-running only fills holes.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from master_us.backtest.momentum import GATE_CHANGES_CACHE, GATE_MEMBERSHIP_CACHE
from master_us.data.loaders import YFinancePrices
from master_us.data.sources import RAW_ROOT, load_data_config, sec_user_agent
from master_us.data.universe import SP500Universe

BACKFILL_START = "2006-01-01"
BACKFILL_END = "2010-01-10"  # small overlap with the canonical pull


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    cfg = load_data_config()
    universe = SP500Universe(
        user_agent=sec_user_agent(cfg),
        cache_path=GATE_MEMBERSHIP_CACHE,
        changes_cache_path=GATE_CHANGES_CACHE,
    )
    membership = universe.get_membership("2007-01-01", args.end)
    tickers = sorted(t for t in membership.filter(pl.col("in_universe"))["ticker"].unique().to_list() if t)
    per_day = membership.filter(pl.col("in_universe")).group_by("date").len()
    print(f"gate membership: {len(tickers)} tickers ever, {per_day['len'].mean():.0f}/day")

    prices_cfg = cfg.get("prices", {})
    loader = YFinancePrices(
        batch_size=int(prices_cfg.get("batch_size", 30)),
        sleep_between_batches_sec=float(prices_cfg.get("sleep_between_batches_sec", 1.5)),
    )
    print(f"backfilling {len(tickers)} tickers + SPY, {BACKFILL_START} -> {BACKFILL_END}")
    loader.fetch_window(tickers, BACKFILL_START, BACKFILL_END)
    # SPY needs the whole window — it was never part of the canonical pull.
    loader.fetch_window(["SPY"], BACKFILL_START, args.end)
    loader.save_failures(RAW_ROOT / "failed_tickers_gate_backfill.json")

    ok = sum(1 for t in tickers if (loader.cache_dir / f"{t}.parquet").exists())
    print(f"tickers with any cached data: {ok}/{len(tickers)}")
    print(f"backfill failures recorded: {len(loader.failed_tickers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
