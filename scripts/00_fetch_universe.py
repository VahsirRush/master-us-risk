#!/usr/bin/env python
"""Scrape and reconstruct S&P 500 PIT membership.

    data/raw/sp500_membership.parquet
    data/raw/sp500_changes.parquet

Idempotent and cache-first: re-running is a no-op unless --force is passed or
the cache does not cover the requested window.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from master_us.data.sources import load_data_config, sec_user_agent
from master_us.data.universe import SP500Universe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true", help="re-scrape even if cached")
    args = parser.parse_args()

    cfg = load_data_config()
    start = args.start or cfg["universe"]["start"]
    end = args.end or cfg["splits"]["test"][1]

    # Reuse the SEC agent so every outbound request identifies the same way.
    universe = SP500Universe(user_agent=sec_user_agent(cfg))

    if universe.cache_path.exists() and not args.force:
        cached = pl.read_parquet(universe.cache_path)
        print(f"cache hit: {universe.cache_path}")
        print(f"  rows {len(cached):,}  dates {cached['date'].min()} -> {cached['date'].max()}")
        print(f"  unique tickers {cached['ticker'].n_unique()}")
        print("  (pass --force to re-scrape)")
        return 0

    print(f"scraping S&P 500 membership {start} -> {end}")
    scrape = universe.reconstruct(start, end)
    universe.save(scrape)

    per_day = scrape.membership.group_by("date").len()
    print(f"  scraped_at         {scrape.scraped_at.isoformat()}")
    print(f"  current members    {len(scrape.current_constituents)}")
    print(f"  changes applied    {scrape.n_changes_applied}")
    print(f"  unique tickers     {len(scrape.tickers)}")
    print(f"  names/day  mean {per_day['len'].mean():.1f}  min {per_day['len'].min()}  max {per_day['len'].max()}")
    print(f"  wrote {universe.cache_path}")
    print(f"  wrote {universe.changes_cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
