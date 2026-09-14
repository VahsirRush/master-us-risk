"""Phase-5 end-to-end: descriptors -> exposures -> factor returns.

Kept separate from `build.py` so the expensive descriptor construction can
be cached once and re-standardized cheaply at different frequencies. §9.3
asks for monthly with a weekly variant for robustness, and rebuilding
descriptors for each would be wasteful.

Exposures are standardized only on the dates a regression actually uses —
the first trading day of each period. Standardizing all 3,772 daily
cross-sections to use 190 of them would cost 20x the work for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from master_us.risk.build import RiskInputs
from master_us.risk.descriptors import FACTOR_DESCRIPTORS
from master_us.risk.factor_returns import FactorReturns, estimate_factor_returns, to_period_returns
from master_us.risk.standardize import standardize_factor

FREQ_PERIODS = {"ME": 12, "W-FRI": 52, "D": 252, "B": 252}


@dataclass(frozen=True)
class RiskModel:
    """Estimated factor returns plus the exposures that produced them."""

    factor_returns: FactorReturns
    exposures: dict[str, npt.NDArray[np.float64]]
    period_dates: pd.DatetimeIndex
    freq: str
    asset_returns: npt.NDArray[np.float64]
    """The (P x N) period returns the regressions actually explained.

    Carried rather than recomputed downstream: reproducing it needs the
    frequency, the lag, and the same drop-rows filter, and any one of those
    getting out of step silently turns the bias test in-sample."""

    exposure_idx: npt.NDArray[np.int64]
    """Row in the daily panel each period's exposures were read from.

    Carried so the covariance and bias-test work can line exposures up with
    the panel without re-deriving the lag and risking a different answer."""

    @property
    def periods_per_year(self) -> int:
        return FREQ_PERIODS.get(self.freq, 12)


def estimate(inputs: RiskInputs, freq: str = "ME", exposure_lag: int = 1) -> RiskModel:
    """Standardize exposures at each period start and run §9.3's WLS.

    THE ALIGNMENT, corrected in Session 14. `daily_returns[i]` is the return
    EARNED ON day i, and the period return compounds from `starts[row]`
    inclusive. Reading exposures at that same index puts day-`t` information
    on both sides: `vol_60d` and `beta` use windows ending at t, and
    `log_mcap` uses close_t. So `exposure_lag=1` reads exposures at the last
    trading day BEFORE the return window opens, which is Barra's convention
    — f_t is the cross-sectional regression of period-t returns on exposures
    known at the start of period t.

    At monthly frequency the old alignment overlapped by one day in 21 and
    the effect was small. At DAILY frequency, which §9.4's 40/90-day
    half-lives require, it would be total contamination: the exposure would
    contain the entire return it explains. `exposure_lag=0` is retained only
    to reproduce the Session 12 numbers for comparison.
    """
    panel = inputs.panel
    dates = pd.DatetimeIndex(panel.dates)
    period_rets, starts, period_ends = to_period_returns(inputs.daily_returns, dates, freq)

    exposure_idx = starts - exposure_lag
    usable = exposure_idx >= 0
    period_rets, starts, period_ends = (
        period_rets[usable], starts[usable], period_ends[usable],
    )
    exposure_idx = exposure_idx[usable]

    exposures: dict[str, npt.NDArray[np.float64]] = {
        f: np.full((len(starts), len(panel.tickers)), np.nan) for f in FACTOR_DESCRIPTORS
    }
    for row, t in enumerate(exposure_idx):
        for factor, names in FACTOR_DESCRIPTORS.items():
            comps = [panel.values[n][t] for n in names]
            if not any(np.isfinite(c).any() for c in comps):
                continue
            exposures[factor][row] = standardize_factor(
                comps, panel.industry, panel.mcap[t], panel.is_primary
            )

    mcap_at_start = panel.mcap[exposure_idx]
    fr = estimate_factor_returns(
        exposures=exposures,
        returns=period_rets,
        mcap=mcap_at_start,
        industry=panel.industry,
        dates=period_ends,
        is_primary=panel.is_primary,
    )
    return RiskModel(
        factor_returns=fr,
        exposures=exposures,
        period_dates=period_ends,
        freq=freq,
        asset_returns=period_rets,
        exposure_idx=exposure_idx,
    )
