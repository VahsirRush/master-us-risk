#!/usr/bin/env python
"""Assemble the real Panel(s) and write them to data/processed/.

    data/processed/panel.pkl       horizon 1 (the headline label)
    data/processed/panel_h5.pkl    horizon 5 (robustness, config labels.horizon note)

Thin caller — logic in src/master_us/data/assemble.py. This supersedes the
placeholder note in scripts/03_build_panel.py: the feature bank and market
vector now exist, so the Panel can finally be written.
"""

from __future__ import annotations

import sys
import time

from master_us.data.assemble import (
    PANEL_H5_PATH,
    PANEL_PATH,
    build_panel,
    group_nan_rates_by_year,
)


def main() -> int:
    for horizon, path in ((1, PANEL_PATH), (5, PANEL_H5_PATH)):
        print(f"building panel, horizon {horizon}d ...", flush=True)
        panel, stats = build_panel(horizon=horizon)
        panel.save(path)
        print(f"  {panel!r}")
        print(f"  built in {stats.build_seconds:.0f}s -> {path}")

        t0 = time.monotonic()
        type(panel).load(path)
        load_s = time.monotonic() - t0
        print(f"  reload from cache: {load_s:.1f}s (gate requires < 30s)")

        if horizon == 1:
            print("\ncoverage:")
            print(panel.coverage().to_string(float_format=lambda x: f"{x:.4f}"))
            print("\nfeature NaN rate by group and year (in-mask cells):")
            print(group_nan_rates_by_year(panel).to_string(float_format=lambda x: f"{x:.4f}"))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
