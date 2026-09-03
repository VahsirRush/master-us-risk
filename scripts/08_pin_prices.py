#!/usr/bin/env python
"""Pin (or verify) the price cache content hashes — contract section 3.2.

    data/raw/price_manifest.json

Default action verifies against the existing manifest; `--pin` (re)writes it.
Re-pinning is a deliberate act performed only after a drift has been
investigated and the affected tickers refetched on a single adjustment basis.
"""

from __future__ import annotations

import argparse
import sys

from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.data.pinning import PRICE_MANIFEST_PATH, PriceDataChangedError, pin, verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true", help="write/overwrite the manifest")
    args = parser.parse_args()

    if args.pin:
        hashes = pin(PRICE_CACHE_DIR)
        print(f"pinned {len(hashes)} tickers -> {PRICE_MANIFEST_PATH}")
        return 0

    try:
        result = verify(PRICE_CACHE_DIR)
    except (FileNotFoundError, PriceDataChangedError) as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"verified {result.n_checked} pinned tickers: no drift")
    if result.new:
        print(f"  {len(result.new)} unpinned newcomer(s): {list(result.new[:10])} — re-pin to adopt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
