#!/usr/bin/env python
"""Search LightGBM hyperparameters on VALIDATION RankIC only.

    reports/lgbm_tuning.md

The Phase-2 gate (LightGBM out-of-sample RankIC > 0.02) fails on the stock
config. This searches for a configuration that clears it WITHOUT ever
consulting the test split — the winner is applied to test exactly once, by
scripts/20_baselines.py, afterwards.
"""

from __future__ import annotations

import sys

import yaml

from master_us.data.assemble import PANEL_PATH
from master_us.data.panel import Panel
from master_us.data.sources import DATA_ROOT, REPO_ROOT
from master_us.experiments.lgbm_tuning import best, search
from master_us.models.train import prepare_data

REPORT = REPO_ROOT / "reports" / "lgbm_tuning.md"


def main() -> int:
    with (REPO_ROOT / "config" / "data.yaml").open() as fh:
        splits = yaml.safe_load(fh)["splits"]

    data = prepare_data(
        Panel.load(PANEL_PATH),
        lookback=20,
        train=tuple(splits["train"]),
        valid=tuple(splits["valid"]),
        test=tuple(splits["test"]),
        embargo_days=int(splits["embargo_days"]),
        # Own cache dir: the Phase-2 grid may be reading its memmap concurrently.
        cache_dir=DATA_ROOT / "interim" / "tuning_cache",
    )
    print(
        f"train {len(data.train_idx)} / valid {len(data.valid_idx)} dates · "
        f"{data.features.shape[2]} features  (test held out, never scored here)",
        flush=True,
    )

    rows = search(data)
    winner = best(rows)
    print(f"\nbest on validation: {winner.name}  RankIC {winner.valid_rank_ic:+.4f}")

    lines = [
        "# LightGBM tuning — validation only",
        "",
        "The Phase-2 gate is LightGBM out-of-sample RankIC > 0.02. The stock",
        "`config/baselines.yaml` configuration misses it. Every row below is scored on",
        "the VALIDATION split; the test split is not touched by this search.",
        "",
        "| Config | Valid RankIC | Best iter | Features | Hypothesis |",
        "|:---|---:|---:|:---|:---|",
    ]
    for row in sorted(rows, key=lambda r: -r.valid_rank_ic):
        lines.append(
            f"| `{row.name}` | {row.valid_rank_ic:+.4f} | {row.best_iteration} | "
            f"{row.feature_subset} | {row.hypothesis} |"
        )
    lines += ["", f"**Winner: `{winner.name}`**", "", "```yaml"]
    lines += [f"{k}: {v}" for k, v in winner.params.items()]
    lines += [f"feature_subset: {winner.feature_subset}", "```", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
