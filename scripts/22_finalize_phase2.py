#!/usr/bin/env python
"""Assemble the final Phase-2 table from cached scores and judge the gate.

    reports/status/phase2.json

Rebuilds every model's metrics from the `*_scores.npy` files written by
scripts/20_baselines.py — no retraining. `lgbmspec` (the shipped config) is
carried alongside `lgbm` (the validation-tuned one) so the table shows both
and the gate's history is legible rather than overwritten.
"""

from __future__ import annotations

import sys

import yaml

from master_us.backtest.costs import CostConfig
from master_us.data.assemble import PANEL_PATH
from master_us.data.panel import Panel
from master_us.data.sources import REPO_ROOT
from master_us.experiments.phase2 import (
    GATE_MIN_LGBM_RANKIC,
    SUSPICION_RANKIC,
    rebuild_from_scores,
    render_baseline_table,
)
from master_us.models.train import prepare_data
from master_us.reporting.results import MetricValue, PhaseResult
from master_us.reporting.status import phase_panel

MODELS = ("ridge", "lgbmspec", "lgbm", "lstm", "ungated")


def main() -> int:
    with (REPO_ROOT / "config" / "data.yaml").open() as fh:
        splits = yaml.safe_load(fh)["splits"]
    with (REPO_ROOT / "config" / "costs.yaml").open() as fh:
        cost_cfg = CostConfig.from_yaml(yaml.safe_load(fh))

    data = prepare_data(
        Panel.load(PANEL_PATH),
        lookback=20,
        train=tuple(splits["train"]),
        valid=tuple(splits["valid"]),
        test=tuple(splits["test"]),
        embargo_days=int(splits["embargo_days"]),
    )
    tables = rebuild_from_scores(data, cost_cfg, models=MODELS)
    print(f"rebuilt {len(tables)} model tables from cached scores: {sorted(tables)}")
    render_baseline_table(tables)

    if "lgbm" not in tables:
        print("error: no lgbm scores found")
        return 1

    lgbm_ic = tables["lgbm"]["rank_ic"]
    gate_passed = lgbm_ic.gross > GATE_MIN_LGBM_RANKIC

    notes = [
        "protocol: 5 seeds · lookback 20 · daily topk_dropout(50, 25) · flat 10 bps net tier",
        "deep models: 12 epochs max, patience 4, validation strided by 3 for early "
        "stopping only (Phase-2 wall-clock budget; Phase 3 uses master.yaml's 100/10)",
    ]
    if "lgbmspec" in tables:
        spec_ic = tables["lgbmspec"]["rank_ic"]
        notes.append(
            f"the config/baselines.yaml LightGBM MISSES the gate: RankIC "
            f"{spec_ic.gross:.4f} ±{spec_ic.std:.4f}. The reported lgbm row uses a "
            f"config selected on VALIDATION only (reports/lgbm_tuning.md)."
        )
        if not lgbm_ic.distinguishable_from(spec_ic):
            notes.append(
                "tuned vs shipped LightGBM is NOT DISTINGUISHABLE from seed dispersion"
            )
    if lgbm_ic.gross > SUSPICION_RANKIC:
        notes.append(
            f"WARNING: RankIC {lgbm_ic.gross:.4f} exceeds {SUSPICION_RANKIC} — "
            "implausible given max single-feature |IC| 0.0154; audit for leakage"
        )

    metrics: dict[str, MetricValue] = {"rankic": lgbm_ic}
    for model_name, table in tables.items():
        for key, value in table.items():
            metrics[f"{model_name}_{key}"] = value

    result = PhaseResult(
        phase=2,
        name="Baselines",
        status="pass" if gate_passed else "fail",
        gate=f"LightGBM out-of-sample RankIC > {GATE_MIN_LGBM_RANKIC}",
        gate_passed=gate_passed,
        metrics=metrics,
        duration_sec=0.0,
        artifacts=[REPO_ROOT / "reports" / "lgbm_tuning.md"],
        notes=notes,
    )
    path = result.save()
    phase_panel(result)
    print(f"wrote {path}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
