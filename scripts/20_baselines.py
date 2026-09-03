#!/usr/bin/env python
"""Run Phase 2: four baselines, five seeds, one table.

    data/processed/phase2/*_scores.npy + seed_runs.json
    reports/status/phase2.json

Thin caller — logic in src/master_us/experiments/phase2.py.
"""

from __future__ import annotations

import argparse
import sys

from master_us.data.assemble import PANEL_PATH
from master_us.data.panel import Panel
from master_us.experiments.phase2 import render_baseline_table, run_phase2
from master_us.reporting.status import phase_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument(
        "--lgbm-config",
        default=None,
        help="name of a candidate from experiments.lgbm_tuning (validation-selected)",
    )
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()


    kwargs = {}
    if args.lgbm_config:
        from master_us.experiments.lgbm_tuning import candidates

        cand = next(c for c in candidates() if c.name == args.lgbm_config)
        kwargs["lgbm_params"] = cand.params
        kwargs["lgbm_subset"] = cand.feature_subset
        print(f"lgbm config: {cand.name} (selected on validation) subset={cand.feature_subset}")
    if args.models:
        kwargs["models"] = tuple(args.models)
    if args.seeds:
        kwargs["seeds"] = tuple(args.seeds)

    tables, result = run_phase2(
        Panel.load(PANEL_PATH),
        max_epochs=args.max_epochs,
        patience=args.patience,
        lookback=args.lookback,
        **kwargs,
    )
    render_baseline_table(tables)
    path = result.save()
    phase_panel(result)
    print(f"wrote {path}")
    return 0 if result.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
