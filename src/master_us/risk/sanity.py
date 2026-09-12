"""Phase-5 gate — do these look like real factor returns?

The Phase-1 equivalent was reproducing 12-1 momentum including the 2009
crash before any model was allowed near the backtest engine. This is the
same idea one phase later: a covariance matrix (Phase 6) estimated on
factor returns that are not really factor returns would be calibrated
noise, and the bias statistic would not catch it — a bias test checks
whether forecast risk matches realized risk, and both can be wrong
together.

What is actually checkable here, and what is not:

* **Momentum should earn a positive average return.** Checked. Note the
  sample cannot reject zero, and that is reported rather than rounded up
  into a pass.
* **Momentum should crash.** The canonical crashes are 2009 and 2020. Only
  2020 is testable — the panel starts 2010-02 and the first estimable
  period is 2011, so **2009 is outside the sample entirely**. Claiming to
  have reproduced it would be false.
* **Value should be cyclical**, not steadily positive: weak through the
  2017-2020 growth regime, strong in the 2022-2023 rotation.
* **The market intercept should look like the market** — an equity-like
  annualized return and volatility, not a residual.
* **Specific returns should be orthogonal to exposures.** This is a
  regression identity for the fitted sample, so a violation means an
  implementation bug, which makes it worth asserting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from master_us.risk.factor_returns import FactorReturns

# Months when momentum is documented to have crashed, for reasons outside
# this dataset. 2020-11 is the vaccine announcement rotation; 2021-01 is
# the short-squeeze episode. Both are checks that the sign is right, not
# targets to fit.
KNOWN_MOMENTUM_CRASHES = ("2020-11", "2021-01")
SAMPLE_EXCLUDED_CRASHES = ("2009-03", "2009-04")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SanityReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = []
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        lines.append(f"\n  GATE: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def run_checks(fr: FactorReturns, periods_per_year: int = 12) -> SanityReport:
    frame = fr.to_frame()
    checks: list[Check] = []

    mom = frame["momentum"].dropna()
    ann = mom.mean() * periods_per_year
    t = _tstat(mom)
    checks.append(
        Check(
            "momentum sign",
            ann > 0,
            f"{ann * 100:+.2f}%/yr (t={t:.2f}) — sign matches the literature; "
            f"{'NOT significant at 5%' if abs(t) < 1.96 else 'significant'}",
        )
    )

    # crash signature: the worst month should be one of the known episodes,
    # and it should be a real tail relative to the factor's own vol
    worst_date = mom.idxmin()
    worst_key = f"{worst_date.year}-{worst_date.month:02d}"
    z = (mom.min() - mom.mean()) / mom.std(ddof=1)
    checks.append(
        Check(
            "momentum crash",
            worst_key in KNOWN_MOMENTUM_CRASHES,
            f"worst month {worst_key} at {mom.min() * 100:+.2f}% ({z:.1f} SD); "
            f"known in-sample crashes {KNOWN_MOMENTUM_CRASHES}; "
            f"{SAMPLE_EXCLUDED_CRASHES} predate the panel and are NOT testable",
        )
    )

    val = frame["value"].dropna()
    by_year = val.groupby(val.index.year).sum()
    growth_era = by_year.loc[[y for y in (2017, 2018, 2019, 2020) if y in by_year.index]]
    rotation = by_year.loc[[y for y in (2022, 2023) if y in by_year.index]]
    cyclical = growth_era.mean() < 0 < rotation.mean()
    verdict = (
        "weak in the growth regime and strong in the rotation, as documented"
        if cyclical
        else "does NOT show the documented pattern"
    )
    checks.append(
        Check(
            "value cyclicality",
            bool(cyclical),
            f"2017-2020 mean {growth_era.mean() * 100:+.2f}%/yr, "
            f"2022-2023 mean {rotation.mean() * 100:+.2f}%/yr — {verdict}",
        )
    )

    mkt = frame["market"].dropna()
    m_ann, m_vol = mkt.mean() * periods_per_year, mkt.std(ddof=1) * np.sqrt(periods_per_year)
    plausible = 0.02 < m_ann < 0.30 and 0.05 < m_vol < 0.35
    checks.append(
        Check(
            "market intercept",
            plausible,
            f"{m_ann * 100:.2f}%/yr at {m_vol * 100:.2f}% vol — "
            f"{'equity-like' if plausible else 'NOT equity-like'}",
        )
    )

    live = fr.n_names >= 1
    mean_r2 = float(np.nanmean(fr.r2))
    checks.append(
        Check(
            "cross-sectional fit",
            0.05 < mean_r2 < 0.90,
            f"mean weighted R^2 {mean_r2:.3f} over {int(live.sum())} estimated periods "
            f"(a monthly cross-sectional factor model typically lands 0.15-0.40)",
        )
    )

    return SanityReport(checks)


def orthogonality(
    exposures: dict[str, np.ndarray], specific: np.ndarray, tol: float = 0.05
) -> Check:
    """Specific returns must be ~uncorrelated with every exposure.

    The exact WLS identity is sum_i w_i x_ik u_i = 0, in the WEIGHTED inner
    product; this checks the unweighted correlation instead, which is only
    approximately zero. That is deliberate — it is a cheap misalignment
    detector across the whole pooled panel, where an indexing bug would
    show up as a correlation of 0.3, not 0.05. The exact identity is
    asserted per period in `tests/test_risk.py`.
    """
    worst_name, worst = "", 0.0
    for name, x in exposures.items():
        ok = np.isfinite(x) & np.isfinite(specific)
        if ok.sum() < 100:
            continue
        c = float(np.corrcoef(x[ok], specific[ok])[0, 1])
        if abs(c) > abs(worst):
            worst_name, worst = name, c
    return Check(
        "specific ⟂ exposures",
        abs(worst) < tol,
        f"largest |corr| = {abs(worst):.4f} on {worst_name!r} (tolerance {tol})",
    )
