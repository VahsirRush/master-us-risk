#!/usr/bin/env python
"""Assemble the aligned panel inputs from the three raw caches.

    data/interim/panel_inputs.parquet   [date, ticker, ohlcv, in_universe, label]
    data/interim/fundamentals.parquet   PIT facts, joined on `filed`

SCOPE: this script assembles everything a `Panel` needs EXCEPT `features` and
`market`, which are spec sections 4.3 and 4.4 and are not built yet. It does not
write `data/processed/panel.pkl` and does not pretend to — a Panel with an empty
feature array would satisfy the dataclass and fail every gate downstream for
reasons that would take a session to track down. Run this, then build the
feature bank, then the Panel.

Cache-first and idempotent: reads parquet, computes, writes. Nothing is fetched.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.data.sec import SECFundamentals
from master_us.data.sources import DATA_ROOT, load_data_config, sec_user_agent
from master_us.data.universe import MEMBERSHIP_CACHE

PANEL_INPUTS = DATA_ROOT / "interim" / "panel_inputs.parquet"
FUNDAMENTALS_INTERIM = DATA_ROOT / "interim" / "fundamentals.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()

    cfg = load_data_config()
    start = args.start or cfg["universe"]["start"]
    end = args.end or cfg["splits"]["test"][1]
    horizon = args.horizon or int(cfg["labels"]["horizon"])
    filters = cfg["universe"]["filters"]

    if not MEMBERSHIP_CACHE.exists():
        print(f"error: {MEMBERSHIP_CACHE} missing — run scripts/00_fetch_universe.py")
        return 1
    if not PRICE_CACHE_DIR.exists() or not any(PRICE_CACHE_DIR.glob("*.parquet")):
        print(f"error: no prices under {PRICE_CACHE_DIR} — run scripts/01_fetch_prices.py")
        return 1

    print(f"assembling {start} -> {end}, label horizon {horizon}d")

    membership = pl.read_parquet(MEMBERSHIP_CACHE).select(["date", "ticker", "in_universe"])
    prices = pl.concat([pl.read_parquet(p) for p in sorted(PRICE_CACHE_DIR.glob("*.parquet"))])
    prices = prices.filter(
        (pl.col("date") >= pl.lit(start).str.to_datetime())
        & (pl.col("date") <= pl.lit(end).str.to_datetime())
    )
    print(f"  prices      {len(prices):,} rows, {prices['ticker'].n_unique()} tickers")
    print(f"  membership  {len(membership):,} rows, {membership['ticker'].n_unique()} tickers")

    panel = (
        prices.join(membership, on=["date", "ticker"], how="left")
        .with_columns(pl.col("in_universe").fill_null(False))
        .sort(["ticker", "date"])
    )

    # Forward return over `horizon` trading days, computed from adj_close so that
    # splits and dividends are already handled. This is the LABEL: it is realized
    # strictly after `date`, which is the alignment convention panel.py fixes.
    panel = panel.with_columns(
        (pl.col("adj_close").shift(-horizon).over("ticker") / pl.col("adj_close") - 1.0).alias(
            "label"
        ),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
    )

    # Section 2.3 reconstitution filters, applied on data known AS OF each date.
    panel = panel.with_columns(
        pl.col("dollar_volume").rolling_mean(21, min_samples=21).over("ticker").alias("adv_21d"),
        pl.col("date").cum_count().over("ticker").alias("history_days"),
    ).with_columns(
        (
            pl.col("in_universe")
            & (pl.col("close") >= filters["min_price"])
            & (pl.col("adv_21d") >= filters["min_adv_usd"])
            & (pl.col("history_days") >= filters["min_history_days"])
        ).alias("tradeable")
    )

    PANEL_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(PANEL_INPUTS)

    per_day = panel.filter(pl.col("tradeable")).group_by("date").len().sort("date")
    print()
    print(f"  rows            {len(panel):,}")
    print(f"  dates           {panel['date'].n_unique():,}")
    print(f"  in_universe     {panel['in_universe'].sum():,}")
    print(f"  tradeable       {panel['tradeable'].sum():,} (after price/ADV/history filters)")
    print(
        f"  names/day       mean {per_day['len'].mean():.0f}  "
        f"min {per_day['len'].min()}  max {per_day['len'].max()}"
    )
    print(f"  label non-null  {panel['label'].is_not_null().sum():,}")
    print(f"  wrote {PANEL_INPUTS}")

    if not FUNDAMENTALS_INTERIM.exists():
        source = SECFundamentals(user_agent=sec_user_agent(cfg))
        facts = source.get_fundamentals(start, end)
        facts.write_parquet(FUNDAMENTALS_INTERIM)
        print(f"  wrote {FUNDAMENTALS_INTERIM} ({len(facts):,} facts)")

    print()
    print("NOT WRITTEN: data/processed/panel.pkl")
    print("  A Panel needs `features` (spec 4.3) and `market` (spec 4.4). Neither exists")
    print("  yet, and emitting a Panel with empty arrays would pass validation and fail")
    print("  every downstream gate for non-obvious reasons. Build the feature bank first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
