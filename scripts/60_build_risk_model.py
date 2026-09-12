#!/usr/bin/env python
"""Phase 5 — descriptors, exposures, factor returns, and the sanity gate.

    data/processed/risk/{exposures,factor_returns,specific}_{freq}.parquet
    reports/phase5.md
    reports/status/phase5.json

Thin caller. All logic is in `master_us.risk`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

import numpy as np
import polars as pl

from master_us.data.sources import REPO_ROOT
from master_us.reporting.results import MetricValue, PhaseResult
from master_us.risk.build import RISK_DIR, build_risk_inputs, descriptor_coverage
from master_us.risk.descriptors import BARRA_CONCEPTS, coverage_table
from master_us.risk.pipeline import estimate
from master_us.risk.sanity import orthogonality, run_checks

CACHE = RISK_DIR / "risk_inputs.pkl"
REPORT = REPO_ROOT / "reports" / "phase5.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached inputs")
    args = ap.parse_args()

    RISK_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if CACHE.exists() and not args.rebuild:
        print(f"loading cached inputs from {CACHE.relative_to(REPO_ROOT)}")
        inputs = pickle.loads(CACHE.read_bytes())
    else:
        print("building risk inputs ...")
        inputs = build_risk_inputs()
        CACHE.write_bytes(pickle.dumps(inputs))

    panel = inputs.panel
    print(f"grid {len(panel.dates)} dates x {len(panel.tickers)} tickers")

    facts = pl.read_parquet(REPO_ROOT / "data" / "interim" / "fundamentals.parquet").filter(
        pl.col("ticker").is_in([str(t) for t in panel.tickers])
    )
    tag_cov = coverage_table(facts)
    desc_cov = descriptor_coverage(panel, inputs.observed)

    models = {}
    for freq in ("ME", "W-FRI"):
        rm = estimate(inputs, freq)
        models[freq] = rm
        tag = "monthly" if freq == "ME" else "weekly"
        fr = rm.factor_returns
        fr.to_frame().to_parquet(RISK_DIR / f"factor_returns_{tag}.parquet")
        pl.DataFrame(
            {"date": np.repeat(rm.period_dates, len(panel.tickers)),
             "ticker": np.tile(panel.tickers, len(rm.period_dates)),
             "specific": fr.specific.ravel()}
        ).write_parquet(RISK_DIR / f"specific_{tag}.parquet")
        print(f"{tag}: {len(fr.dates)} periods, mean n={fr.n_names[fr.n_names > 0].mean():.0f}, "
              f"R2={np.nanmean(fr.r2):.3f}")

    monthly = models["ME"]
    report = run_checks(monthly.factor_returns)
    orth = orthogonality(monthly.exposures, monthly.factor_returns.specific)
    print("\nPHASE 5 GATE\n" + report.render())
    print(f"  [{'PASS' if orth.passed else 'FAIL'}] {orth.name}: {orth.detail}")

    s = monthly.factor_returns.summary(12)
    passed = report.passed and orth.passed
    PhaseResult(
        phase=5,
        name="Barra descriptors and factor returns",
        status="pass" if passed else "fail",
        gate="factor returns reproduce known published patterns (momentum sign and "
        "crash, value cyclicality, equity-like market intercept)",
        gate_passed=passed,
        metrics={
            "momentum_ann": MetricValue(gross=float(s.loc["momentum", "ann_mean"])),
            "value_ann": MetricValue(gross=float(s.loc["value", "ann_mean"])),
            "market_ann": MetricValue(gross=float(s.loc["market", "ann_mean"])),
            "mean_r2": MetricValue(gross=float(np.nanmean(monthly.factor_returns.r2))),
        },
        duration_sec=time.time() - t0,
        artifacts=[RISK_DIR / "factor_returns_monthly.parquet", REPORT],
        notes=[c.detail for c in report.checks],
    ).save()

    lines = ["# Phase 5 — Barra descriptors and factor returns", ""]
    lines += ["## Tag resolution coverage (S&P-500 universe)", ""]
    lines += ["| year | " + " | ".join(BARRA_CONCEPTS) + " |",
              "|---:|" + "---:|" * len(BARRA_CONCEPTS)]
    for r in tag_cov.iter_rows(named=True):
        lines.append(f"| {r['year']} | " + " | ".join(
            f"{r[c] * 100:.1f}%" if r[c] is not None else "—" for c in BARRA_CONCEPTS) + " |")
    lines += ["", "## Descriptor coverage (conditional on the name trading)", "",
              "| descriptor | overall | worst year | rate |", "|---|---:|---:|---:|"]
    for r in desc_cov.iter_rows(named=True):
        lines.append(f"| {r['descriptor']} | {r['overall'] * 100:.1f}% | "
                     f"{r['worst_year']} | {r['worst_rate'] * 100:.1f}% |")
    lines += ["", "## Factor returns (monthly, annualized)", "",
              "| factor | ann mean | ann vol | t |", "|---|---:|---:|---:|"]
    for name in monthly.factor_returns.factor_names:
        r = s.loc[name]
        lines.append(f"| {name} | {r['ann_mean'] * 100:+.2f}% | "
                     f"{r['ann_vol'] * 100:.2f}% | {r['t_stat']:.2f} |")
    lines += ["", "## Gate", "", "```", report.render(), f"  [{'PASS' if orth.passed else 'FAIL'}] "
              f"{orth.name}: {orth.detail}", "```", ""]
    REPORT.write_text("\n".join(lines))
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
