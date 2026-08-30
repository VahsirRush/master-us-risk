#!/usr/bin/env python
"""Run the Phase-0 verification and cache its PhaseResult for `make status`.

    reports/status/phase0.json

Runs the actual test suite (the gate) and reads the actual panel-input stats —
no number in the ladder is typed in by hand.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

import polars as pl

from master_us.data.sources import DATA_ROOT, REPO_ROOT
from master_us.reporting.results import MetricValue, PhaseResult
from master_us.reporting.status import phase_panel

PANEL_INPUTS = DATA_ROOT / "interim" / "panel_inputs.parquet"


def main() -> int:
    t0 = time.monotonic()
    # No explicit -q here: pyproject addopts already carries one, and doubling
    # it (-qq) suppresses the "N passed" summary this parses.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-header", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) passed", proc.stdout)
    n_passed = int(match.group(1)) if match else 0
    tests_green = proc.returncode == 0 and n_passed > 0

    metrics: dict[str, MetricValue] = {
        "tests": MetricValue(gross=float(n_passed), net=float(n_passed))
    }
    notes: list[str] = []
    if PANEL_INPUTS.exists():
        panel = pl.read_parquet(PANEL_INPUTS)
        per_day = panel.filter(pl.col("tradeable")).group_by("date").len()
        metrics["n_dates"] = MetricValue(
            gross=float(panel["date"].n_unique()), net=float(panel["date"].n_unique())
        )
        mean_names = float(per_day["len"].mean() or 0.0)
        metrics["names_per_day"] = MetricValue(gross=mean_names, net=mean_names)
        notes.append(
            f"panel inputs: {len(panel):,} rows · {panel['ticker'].n_unique()} tickers"
        )
    notes.append("features (spec 4.3) and market vector (spec 4.4) still pending — "
                 "panel.pkl deliberately not written")

    result = PhaseResult(
        phase=0,
        name="Data layer",
        status="pass" if tests_green else "fail",
        gate="leakage tests + data-contract tests pass on real caches",
        gate_passed=tests_green,
        metrics=metrics,
        duration_sec=time.monotonic() - t0,
        artifacts=[REPO_ROOT / "reports" / "survivorship.md"],
        notes=notes,
    )
    path = result.save()
    phase_panel(result)
    print(f"wrote {path}")
    return 0 if tests_green else 1


if __name__ == "__main__":
    sys.exit(main())
