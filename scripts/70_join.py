#!/usr/bin/env python
"""Phase 7 — the join (spec §10).

    data/processed/join/{books,attribution}.pkl
    reports/phase7.md
    reports/status/phase7.json

Thin caller. All logic is in `master_us.experiments.join`.

Long-running: the neutralized construction solves a 583-variable QP per date
per seed per arm. Launch detached per CLAUDE.md:

    nohup caffeinate -dimsu ~/.venvs/master-us/bin/python scripts/70_join.py \
        --pidfile /tmp/join.pid > /tmp/join.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from master_us.backtest.construct import decile_long_short
from master_us.backtest.costs import CostConfig
from master_us.backtest.engine import run_backtest
from master_us.data.sources import DATA_ROOT, REPO_ROOT
from master_us.experiments.join import (
    STYLE_FACTORS,
    align_to_risk_grid,
    net_of_costs,
    optimize_book,
    sharpe,
)
from master_us.experiments.phase2 import PHASE2_DIR
from master_us.risk.build import RISK_DIR
from master_us.risk.covariance import factor_covariance
from master_us.risk.forecast import build_forecast

JOIN_DIR = DATA_ROOT / "processed" / "join"
LAM_RISK = 5.0
LAM_TURNOVER = 5e-4


def heartbeat(path: Path, label: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'{{"label": "{label}", "pid": {os.getpid()}, "note": "{note}", '
        f'"stamp": "{pd.Timestamp.now(tz="UTC").isoformat()}"}}\n'
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="master", help="which variant's book to join")
    ap.add_argument("--dates", type=int, default=0, help="limit test dates (0 = all)")
    ap.add_argument("--seeds", type=int, default=0, help="limit seeds (0 = all)")
    ap.add_argument("--pidfile", default=None)
    ap.add_argument("--eigen-check", action="store_true",
                    help="also run the neutral arm with the eigenfactor adjustment OFF")
    args = ap.parse_args()

    if args.pidfile:
        Path(args.pidfile).write_text(f"{os.getpid()}\n")
    JOIN_DIR.mkdir(parents=True, exist_ok=True)
    hb = REPO_ROOT / "reports" / "status" / "join_heartbeat.json"
    t0 = time.time()

    z = np.load(DATA_ROOT / "processed" / "phase4" / "metrics_bundle.npz", allow_pickle=True)
    inputs = pickle.loads((RISK_DIR / "risk_inputs.pkl").read_bytes())
    model = pickle.loads((RISK_DIR / "risk_model_daily.pkl").read_bytes())
    fcast = build_forecast(inputs, model)

    panel_dates = pd.DatetimeIndex(inputs.panel.dates)
    drow, tcol = align_to_risk_grid(
        pd.DatetimeIndex(z["dates"]), np.asarray(z["tickers"]),
        panel_dates, np.asarray(inputs.panel.tickers),
    )

    ret_row = model.exposure_idx + 1
    keep = ret_row < len(panel_dates)
    ret_row, pidx = ret_row[keep], np.flatnonzero(keep)
    test_dates = set(pd.DatetimeIndex(z["dates"])[z["test_idx"]])
    sel = np.array([panel_dates[r] in test_dates for r in ret_row])
    pidx, ret_row = pidx[sel], ret_row[sel]
    if args.dates:
        pidx, ret_row = pidx[: args.dates], ret_row[: args.dates]

    covs = {"eigen": factor_covariance(fcast.factor_returns, fcast.factor_names).cov}
    if args.eigen_check:
        covs["no_eigen"] = factor_covariance(
            fcast.factor_returns, fcast.factor_names, eigen_adjust=False
        ).cov
    names = list(fcast.factor_names)
    style_idx = [names.index(f) for f in STYLE_FACTORS]

    with (REPO_ROOT / "config" / "costs.yaml").open() as fh:
        cost_cfg = CostConfig.from_yaml(yaml.safe_load(fh))
    bundle_dates = pd.DatetimeIndex(z["dates"])
    daily = np.full_like(z["raw_forward"], np.nan)
    daily[1:] = z["raw_forward"][:-1]
    test = z["test_idx"]

    seeds = sorted(
        int(p.stem.split("_seed")[1].split("_")[0])
        for p in PHASE2_DIR.glob(f"{args.tag}_seed*_scores.npy")
    )
    if args.seeds:
        seeds = seeds[: args.seeds]
    print(f"tag={args.tag} seeds={seeds} dates={len(pidx)} lam_risk={LAM_RISK}", flush=True)

    out: dict[str, dict] = {}
    wmap = {d: i for i, d in enumerate(bundle_dates[test])}
    rows = np.array([wmap[panel_dates[r]] for r in ret_row])

    for seed in seeds:
        raw = np.load(PHASE2_DIR / f"{args.tag}_seed{seed}_scores.npy")
        scores = raw[drow][:, tcol]

        res = run_backtest(
            dates=bundle_dates[test], returns=daily[test].astype(float),
            scores=raw[test].astype(float), mask=z["mask"][test],
            constructor=lambda s, m, p: decile_long_short(s, m, 10),
            rebalance_idx=np.arange(len(test)), cost_cfg=cost_cfg, tier="baseline",
        )
        entry: dict[str, object] = {
            "decile": {
                "gross": res.series.gross, "net": res.series.net,
                "turnover": res.series.turnover,
                "weights": res.weights[rows][:, tcol],
            }
        }
        for cov_tag, cov in covs.items():
            for neutral in (False, True):
                arm = f"{'neutral' if neutral else 'plain'}_{cov_tag}"
                if cov_tag == "no_eigen" and not neutral:
                    continue  # the eigen question is about the constrained book
                heartbeat(hb, "phase7-join", f"seed {seed} arm {arm}")
                t1 = time.time()
                book = optimize_book(
                    scores[pidx], fcast.exposures[pidx], cov[pidx],
                    fcast.specific_var[pidx], fcast.asset_returns[pidx],
                    style_idx, neutral, LAM_RISK, LAM_TURNOVER,
                )
                book["net"] = net_of_costs(book["gross"], book["turnover"])
                entry[arm] = book
                print(
                    f"  seed {seed} {arm:18s} gross {sharpe(book['gross']):+.3f} "
                    f"net {sharpe(book['net']):+.3f} turn {book['turnover'].mean() * 100:.0f}% "
                    f"[{time.time() - t1:.0f}s]",
                    flush=True,
                )
        out[str(seed)] = entry
        (JOIN_DIR / f"books_{args.tag}.pkl").write_bytes(pickle.dumps(out))
        print(f"  seed {seed} done, {time.time() - t0:.0f}s elapsed", flush=True)

    heartbeat(hb, "phase7-join", "complete")
    print(f"\ntotal {time.time() - t0:.0f}s -> {JOIN_DIR / f'books_{args.tag}.pkl'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
