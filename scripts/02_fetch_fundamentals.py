#!/usr/bin/env python
"""Download and parse SEC Financial Statement Data Sets, one quarter per file.

    data/raw/sec/{year}q{q}.parquet
    data/raw/sec_cik_ticker_map.parquet
    data/raw/failed_quarters.json

Cache-first and idempotent: a quarter with a cached parquet is skipped, so an
interrupted run resumes. Only the tags in TAG_FALLBACKS survive to the parquet,
which is what keeps ~4GB of archives down to tens of megabytes on disk.

Refuses to start if the configured SEC User-Agent is still a placeholder.
"""

from __future__ import annotations

import argparse
import sys
import time

from master_us.data.sec import SECFundamentals, quarters_between, save_failed_quarters
from master_us.data.sources import RAW_ROOT, FetchError, load_data_config, sec_user_agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-quarter", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_data_config()
    fundamentals_cfg = cfg.get("fundamentals", {})

    # The guard. Fails before the first request rather than after an IP block.
    user_agent = sec_user_agent(cfg)
    print(f"SEC User-Agent: {user_agent}")

    start_quarter = args.start_quarter or fundamentals_cfg.get("start_quarter", "2009q2")
    end = args.end or cfg["splits"]["test"][1]

    source = SECFundamentals(
        user_agent=user_agent,
        rate_limit_per_sec=float(fundamentals_cfg.get("rate_limit_per_sec", 8)),
    )

    print("fetching CIK -> ticker map")
    mapping = source.cik_ticker_map()
    print(f"  {len(mapping):,} rows, {mapping['cik'].n_unique():,} unique CIKs")

    quarters = quarters_between(start_quarter, end)
    if args.limit:
        quarters = quarters[: args.limit]
    todo = [q for q in quarters if args.force or not source._quarter_path(q).exists()]

    print(f"quarters in range : {len(quarters)}  ({quarters[0]} -> {quarters[-1]})")
    print(f"already cached    : {len(quarters) - len(todo)}")
    print(f"to fetch          : {len(todo)}", flush=True)

    t0 = time.monotonic()
    total_facts = 0
    for i, quarter in enumerate(todo, 1):
        try:
            frame = source.fetch_quarter(quarter, force=args.force)
        except FetchError as exc:
            source.failed_quarters[quarter] = exc.attempts[-1][:300]
            print(f"  [{i}/{len(todo)}] {quarter}  FAILED: {exc.attempts[-1][:120]}", flush=True)
            continue
        total_facts += len(frame)
        print(
            f"  [{i}/{len(todo)}] {quarter}  {len(frame):>8,} facts  "
            f"{frame['cik'].n_unique() if len(frame) else 0:>5} ciks",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    path = save_failed_quarters(source.failed_quarters, RAW_ROOT / "failed_quarters.json")

    print()
    print(f"elapsed        {elapsed / 60:.1f} min")
    print(f"facts parsed   {total_facts:,}")
    print(f"quarters ok    {len(todo) - len(source.failed_quarters)}/{len(todo)}")
    print(f"failures       {path}")
    if source.failed_quarters:
        for quarter, reason in sorted(source.failed_quarters.items()):
            print(f"  {quarter}: {reason[:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
