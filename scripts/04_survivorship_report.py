#!/usr/bin/env python
"""Measure survivorship bias and write reports/survivorship.md.

    reports/survivorship.md

Contract section 3.3. Idempotent: reads the membership and price caches, writes
the report. Nothing is fetched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from master_us.data.loaders import YFinancePrices
from master_us.data.sources import REPO_ROOT
from master_us.data.survivorship import build_report, render_markdown
from master_us.data.universe import CHANGES_CACHE, MEMBERSHIP_CACHE

REPORT_PATH = REPO_ROOT / "reports" / "survivorship.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    for path in (MEMBERSHIP_CACHE, CHANGES_CACHE):
        if not path.exists():
            print(f"error: {path} missing — run scripts/00_fetch_universe.py first")
            return 1

    membership = pl.read_parquet(MEMBERSHIP_CACHE)
    changes = pl.read_parquet(CHANGES_CACHE)
    retrieved = YFinancePrices().cached_tickers()
    if not retrieved:
        print("error: no cached prices — run scripts/01_fetch_prices.py first")
        return 1

    report = build_report(membership, changes, retrieved)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report))

    print(f"historical constituents : {report.n_total}")
    print(f"retrievable             : {report.n_retrieved} ({report.overall_rate:.1%})")
    print(f"absent                  : {report.n_total - report.n_retrieved}")
    print()
    print("retrieval rate by removal reason:")
    for row in report.by_reason.iter_rows(named=True):
        print(
            f"  {row['removal_reason']:<20} {row['n_retrieved']:>4}/{row['n']:<4} "
            f"{row['retrieval_rate']:>6.1%}"
        )
    print()
    print("retrieval rate by year:")
    for row in report.by_year.iter_rows(named=True):
        print(
            f"  {row['year']}  {row['n_retrieved']:>4}/{row['n_constituents']:<4} "
            f"{row['retrieval_rate']:>6.1%}"
        )
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
