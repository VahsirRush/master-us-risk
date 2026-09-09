#!/usr/bin/env python
"""Phase 4: the ablation grid and β sweep — spec §8.1, §8.2.

    data/processed/phase2/{variant}_seed{seed}_scores.npy
    data/processed/phase4/metrics_bundle.npz   (small; lets analysis skip the panel)

Every cell runs at the SAME budget as the six-model table (12 epochs, patience
4, lookback 20) unless --full-budget is passed. See NOTES Session 9 for why.

Launch detached with a pidfile, per the Session 6-8 discipline:

    nohup caffeinate -dimsu python scripts/40_ablations.py \\
        --groups beta --pidfile /tmp/phase4.pid > /tmp/phase4.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from master_us.data.assemble import PANEL_PATH
from master_us.data.panel import Panel
from master_us.data.sources import DATA_ROOT, REPO_ROOT
from master_us.experiments.ablations import (
    Variant,
    beta_grid,
    grid,
    head_grid,
    lookback_grid,
    resolve_arch,
    with_shuffled_market,
)
from master_us.experiments.phase2 import PHASE2_DIR
from master_us.models.train import PreparedData, prepare_data, train_model

PHASE4_DIR = DATA_ROOT / "processed" / "phase4"
BUNDLE_PATH = PHASE4_DIR / "metrics_bundle.npz"

GROUPS = {
    "ablation": grid,
    "beta": beta_grid,
    "lookback": lookback_grid,
    "heads": head_grid,
}


def save_metrics_bundle(data: PreparedData, panel: Panel) -> Path:
    """Cache the small arrays every downstream analysis needs.

    Analysis otherwise has to load the 1.2 GB panel, which the training guard
    now (correctly) refuses to allow during a run. This bundle is ~40 MB, so
    the β sweep can be analysed while the next group is still training.
    """
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BUNDLE_PATH,
        dates=panel.dates.values.astype("datetime64[ns]"),
        labels=data.labels,
        mask=data.mask,
        raw_forward=data.raw_forward,
        train_idx=data.train_idx,
        valid_idx=data.valid_idx,
        test_idx=data.test_idx,
        tickers=np.asarray(panel.tickers, dtype=object),
        sectors=np.asarray(
            [
                panel.metadata.xs(t, level=1)["sector"].iloc[0]
                if panel.metadata is not None
                else "Unknown"
                for t in panel.tickers
            ],
            dtype=object,
        ),
        mcap=_mcap_matrix(panel),
    )
    return BUNDLE_PATH


def _mcap_matrix(panel: Panel) -> np.ndarray:
    """(T, N) market cap, for the §8.5 cap-tier stress test."""
    out = np.full((panel.n_dates, panel.n_tickers), np.nan, dtype=np.float32)
    if panel.metadata is None:
        return out
    frame = panel.metadata["mcap"].unstack()
    frame = frame.reindex(index=panel.dates, columns=panel.tickers)
    return frame.to_numpy(dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=["beta"], choices=[*GROUPS, "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--suffix", default="", help="tag appended to score filenames")
    parser.add_argument("--pidfile", type=Path, default=None)
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()

    if args.pidfile:
        args.pidfile.parent.mkdir(parents=True, exist_ok=True)
        args.pidfile.write_text(str(os.getpid()))

    with (REPO_ROOT / "config" / "master.yaml").open() as fh:
        arch = yaml.safe_load(fh)["architecture"]
    with (REPO_ROOT / "config" / "data.yaml").open() as fh:
        splits = yaml.safe_load(fh)["splits"]

    panel = Panel.load(PANEL_PATH)
    data = prepare_data(
        panel,
        lookback=args.lookback,
        train=tuple(splits["train"]),
        valid=tuple(splits["valid"]),
        test=tuple(splits["test"]),
        embargo_days=int(splits["embargo_days"]),
        cache_dir=PHASE2_DIR,
    )
    if not BUNDLE_PATH.exists() or args.bundle_only:
        print(f"writing metrics bundle -> {save_metrics_bundle(data, panel)}", flush=True)
    del panel
    if args.bundle_only:
        return 0

    names = [g for g in args.groups if g != "all"] or list(GROUPS)
    if "all" in args.groups:
        names = list(GROUPS)
    variants: list[Variant] = [v for name in names for v in GROUPS[name]()]
    print(
        f"phase 4: {len(variants)} variants x {len(args.seeds)} seeds "
        f"@ {args.max_epochs} epochs / patience {args.patience} / lookback {args.lookback}",
        flush=True,
    )

    for variant in variants:
        for seed in args.seeds:
            tag = f"{variant.name}{args.suffix}"
            out_path = PHASE2_DIR / f"{tag}_seed{seed}_scores.npy"
            if out_path.exists():
                print(f"  {tag} seed {seed}: cached, skipping", flush=True)
                continue

            run_data = with_shuffled_market(data, seed) if variant.shuffle_market else data
            if variant.lookback is not None and variant.lookback != args.lookback:
                print(f"  {tag}: needs lookback {variant.lookback}, re-preparing", flush=True)
                run_data = prepare_data(
                    Panel.load(PANEL_PATH),
                    lookback=variant.lookback,
                    train=tuple(splits["train"]),
                    valid=tuple(splits["valid"]),
                    test=tuple(splits["test"]),
                    embargo_days=int(splits["embargo_days"]),
                    cache_dir=PHASE2_DIR,
                )

            model = variant.build(
                run_data.features.shape[2], run_data.market.shape[1], resolve_arch(variant, arch)
            )
            t0 = time.monotonic()
            result = train_model(
                model,
                run_data,
                seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                log=lambda e, ic, best, _t=tag, _s=seed: print(
                    f"    {_t} seed {_s} epoch {e:>2}  valid RankIC {ic:+.4f}  best {best:+.4f}",
                    flush=True,
                ),
            )
            np.save(out_path, result.scores)
            print(
                f"  {tag} seed {seed}: valid RankIC {result.best_valid_rank_ic:+.4f} "
                f"(best epoch {result.best_epoch}) [{time.monotonic() - t0:.0f}s]",
                flush=True,
            )
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print("phase 4 group(s) complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
