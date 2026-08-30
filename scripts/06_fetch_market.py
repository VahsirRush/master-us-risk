#!/usr/bin/env python
"""Fetch index-level data for the market state vector (spec 4.4).

    data/raw/market/{symbol}.parquet

Symbols: ^GSPC (S&P 500), ^SP400 (S&P 400 MidCap), ^RUT (Russell 2000),
^VIX, and MDY (SPDR MidCap ETF - volume proxy; Yahoo reports zero volume for
^SP400 itself across its entire history, measured in Session 5).

Fetched from 2008 so every rolling window in the vector is warm before the
panel's 2010 start. Idempotent: cached symbols are skipped.
"""

from __future__ import annotations

import sys

import polars as pl

from master_us.data.loaders import YFinancePrices
from master_us.data.sources import RAW_ROOT

MARKET_DIR = RAW_ROOT / "market"
MARKET_SYMBOLS = ["^GSPC", "^SP400", "^RUT", "^VIX", "MDY"]
START, END = "2008-01-01", "2025-12-31"


def main() -> int:
    loader = YFinancePrices(cache_dir=MARKET_DIR)
    todo = [s for s in MARKET_SYMBOLS if not (MARKET_DIR / f"{s}.parquet").exists()]
    print(f"market symbols: {len(MARKET_SYMBOLS)}, cached {len(MARKET_SYMBOLS) - len(todo)}, fetching {len(todo)}")
    if todo:
        loader.fetch_window(todo, START, END)
    # ^VIX legitimately reports zero volume (it is an index of implied vols,
    # nothing trades at the index itself), and index "prices" can sit below
    # $5-style filters; the equity validation suite does not apply here. Only
    # existence and date coverage are checked.
    missing = []
    for sym in MARKET_SYMBOLS:
        path = MARKET_DIR / f"{sym}.parquet"
        if not path.exists():
            missing.append(sym)
            continue
        d = pl.read_parquet(path)
        print(f"  {sym:<7} {len(d):>5} rows  {str(d['date'].min())[:10]} -> {str(d['date'].max())[:10]}")
    if missing:
        print(f"FAILED to fetch: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
