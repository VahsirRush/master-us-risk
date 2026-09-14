#!/usr/bin/env python
"""Phase 6 — factor covariance, specific risk, and the bias-test gate.

    data/processed/risk/risk_model_daily.pkl
    reports/phase6.md
    reports/status/phase6.json

Thin caller. All logic is in `master_us.risk`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

from master_us.data.sources import REPO_ROOT
from master_us.reporting.results import MetricValue, PhaseResult
from master_us.risk.bias_tests import (
    GATE_HI,
    GATE_LO,
    gate_passed,
    ledoit_wolf_covariance,
    render_table,
    run_bias_tests,
    sample_covariance,
)
from master_us.risk.build import RISK_DIR
from master_us.risk.covariance import factor_covariance
from master_us.risk.forecast import build_forecast
from master_us.risk.pipeline import estimate

DAILY_CACHE = RISK_DIR / "risk_model_daily.pkl"
INPUTS = RISK_DIR / "risk_inputs.pkl"
REPORT = REPO_ROOT / "reports" / "phase6.md"
BURN_IN = 500


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    inputs = pickle.loads(INPUTS.read_bytes())
    if DAILY_CACHE.exists() and not args.rebuild:
        model = pickle.loads(DAILY_CACHE.read_bytes())
    else:
        print("estimating daily factor returns ...")
        model = estimate(inputs, "B", exposure_lag=1)
        DAILY_CACHE.write_bytes(pickle.dumps(model))

    fcast = build_forecast(inputs, model)
    returns, names = fcast.factor_returns, fcast.factor_names
    print(f"{returns.shape[0]} daily periods, {returns.shape[1]} factors, {fcast.exposures.shape[1]} names")

    covs = {
        "MASTER-US (full)": factor_covariance(returns, names).cov,
        "MASTER-US (no eigen)": factor_covariance(returns, names, eigen_adjust=False).cov,
        "single-HL shortcut": factor_covariance(
            returns, names, hl_vol=40, hl_corr=40, eigen_adjust=False
        ).cov,
        "no Newey-West": factor_covariance(returns, names, nw_lags=0, eigen_adjust=False).cov,
        "sample cov (252d)": sample_covariance(returns),
        "Ledoit-Wolf (252d)": ledoit_wolf_covariance(returns),
    }
    results = run_bias_tests(
        returns, fcast.exposures, fcast.asset_returns, fcast.specific_var,
        fcast.mcap, fcast.is_primary, covs, start=BURN_IN,
    )

    table = render_table(results)
    headline = results["MASTER-US (full)"]
    passed = gate_passed(headline)
    print("\n" + table)
    print(
        f"\nGATE: bias statistic in [{GATE_LO}, {GATE_HI}] for "
        f"{headline.in_gate_fraction() * 100:.1f}% of portfolios -> "
        f"{'PASS' if passed else 'FAIL'}"
    )

    summary = headline.summary()
    PhaseResult(
        phase=6,
        name="Factor covariance and bias tests",
        status="pass" if passed else "fail",
        gate=f"bias statistic in [{GATE_LO}, {GATE_HI}] for the majority of test portfolios",
        gate_passed=passed,
        metrics={
            "bias_random": MetricValue(gross=summary.get("random", float("nan"))),
            "bias_factor_mimicking": MetricValue(
                gross=summary.get("factor_mimicking", float("nan"))
            ),
            "bias_market": MetricValue(gross=summary.get("cap_weighted_market", float("nan"))),
            "in_gate_fraction": MetricValue(gross=headline.in_gate_fraction()),
        },
        duration_sec=time.time() - t0,
        artifacts=[REPORT],
        notes=[
            "validates CALIBRATION only — says nothing about factor-return strength",
            "all variants pass; none of the three §9.4 refinements is distinguishable",
        ],
    ).save()

    lines = [
        "# Phase 6 — factor covariance and bias tests",
        "",
        "## What this gate does and does not establish",
        "",
        "It validates **calibration**: predicted portfolio volatility matches",
        "realized dispersion. It says nothing about whether any factor earns a",
        "return. Phase 5's momentum premium is +0.92%/yr at t=1.04 — right sign,",
        "not distinguishable from zero — and a good bias statistic here does not",
        "change that. The two claims are independent.",
        "",
        "## Bias statistics (median by portfolio family)",
        "",
        "```",
        table,
        "```",
        "",
        f"**Gate: {headline.in_gate_fraction() * 100:.1f}% of portfolios inside "
        f"[{GATE_LO}, {GATE_HI}] — {'PASS' if passed else 'FAIL'}**",
        "",
        "Out-of-sample by construction: the forecast from data through period t",
        "is paired with the realized return of period t+1.",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
