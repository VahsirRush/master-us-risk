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

FREQ_PERIODS = {"ME": 12, "W-FRI": 52}


@dataclass(frozen=True)
class RiskModel:
    """Estimated factor returns plus the exposures that produced them."""

    factor_returns: FactorReturns
    exposures: dict[str, npt.NDArray[np.float64]]
    period_dates: pd.DatetimeIndex
    freq: str

    @property
    def periods_per_year(self) -> int:
        return FREQ_PERIODS.get(self.freq, 12)


def estimate(inputs: RiskInputs, freq: str = "ME") -> RiskModel:
    """Standardize exposures at each period start and run §9.3's WLS.

    The alignment that matters: exposures are read at the period's FIRST
    trading day, the return is compounded ACROSS the period. An exposure
    must predate the return it explains, or the factor returns are a
    look-ahead and every downstream attribution inherits it.
    """
    panel = inputs.panel
    dates = pd.DatetimeIndex(panel.dates)
    period_rets, starts, period_ends = to_period_returns(inputs.daily_returns, dates, freq)

    exposures: dict[str, npt.NDArray[np.float64]] = {
        f: np.full((len(starts), len(panel.tickers)), np.nan) for f in FACTOR_DESCRIPTORS
    }
    for row, t in enumerate(starts):
        for factor, names in FACTOR_DESCRIPTORS.items():
            comps = [panel.values[n][t] for n in names]
            if not any(np.isfinite(c).any() for c in comps):
                continue
            exposures[factor][row] = standardize_factor(
                comps, panel.industry, panel.mcap[t], panel.is_primary
            )

    mcap_at_start = panel.mcap[starts]
    fr = estimate_factor_returns(
        exposures=exposures,
        returns=period_rets,
        mcap=mcap_at_start,
        industry=panel.industry,
        dates=period_ends,
        is_primary=panel.is_primary,
    )
    return RiskModel(factor_returns=fr, exposures=exposures, period_dates=period_ends, freq=freq)
