#!/usr/bin/env python
"""Fetch OHLCV for every ticker that ever appears in historical membership.

    data/raw/prices/{ticker}.parquet
    data/raw/failed_tickers.json

Cache-first and idempotent: a ticker with a cached parquet is skipped entirely,
so re-running after an interruption resumes where it stopped.
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from master_us.data.loaders import YFinancePrices
from master_us.data.sources import load_data_config
from master_us.data.universe import MEMBERSHIP_CACHE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None, help="fetch only the first N tickers")
    args = parser.parse_args()

    cfg = load_data_config()
    start = args.start or cfg["universe"]["start"]
    end = args.end or cfg["splits"]["test"][1]

    if not MEMBERSHIP_CACHE.exists():
        print(f"error: {MEMBERSHIP_CACHE} missing — run scripts/00_fetch_universe.py first")
        return 1

    membership = pl.read_parquet(MEMBERSHIP_CACHE)
    tickers = sorted(membership.filter(pl.col("in_universe"))["ticker"].unique().to_list())
    tickers = [t for t in tickers if t]
    if args.limit:
        tickers = tickers[: args.limit]

    prices_cfg = cfg.get("prices", {})
    loader = YFinancePrices(
        batch_size=int(prices_cfg.get("batch_size", 30)),
        sleep_between_batches_sec=float(prices_cfg.get("sleep_between_batches_sec", 1.5)),
        backoff_sec=tuple(float(x) for x in prices_cfg.get("backoff_sec", [2, 8, 32])),
        min_coverage_ratio=float(prices_cfg.get("min_coverage_ratio", 0.8)),
    )

    already = loader.cached_tickers()
    todo = [t for t in tickers if t not in already]
    print(f"tickers in historical membership : {len(tickers)}")
    print(f"already cached                   : {len(already & set(tickers))}")
    print(f"to fetch                         : {len(todo)}")
    print(f"window                           : {start} -> {end}")
    print(f"batches of {loader.batch_size}, {loader.sleep_between_batches_sec}s between", flush=True)

    t0 = time.monotonic()
    frame = loader.get_ohlcv(tickers, start, end)
    # Validation otherwise only covers tickers fetched on THIS run; a resumed run
    # would report nothing for the ones already on disk.
    loader.revalidate_cache(start, end, tickers)
    elapsed = time.monotonic() - t0

    path = loader.save_failures()
    retrieved = frame["ticker"].n_unique()
    print()
    print(f"elapsed            {elapsed / 60:.1f} min")
    print(f"rows               {len(frame):,}")
    print(f"tickers retrieved  {retrieved}/{len(tickers)} ({retrieved / len(tickers):.1%})")
    print(f"tickers failed     {len(loader.failed_tickers)}")
    print(f"failures written   {path}")

    if loader.failed_tickers:
        print("\nfirst 15 failures:")
        for ticker, reason in sorted(loader.failed_tickers.items())[:15]:
            print(f"  {ticker:<8} {reason[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
