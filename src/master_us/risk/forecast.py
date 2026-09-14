"""Assemble the full risk forecast: exposures, factor covariance, specific risk.

Sits between Phase 5's factor returns and Phase 6's bias tests. The job is
alignment, and alignment is the whole difficulty — the bias test is only
out-of-sample if every array agrees on what "period t" means.

The convention, fixed here once so nothing downstream re-derives it:

    exposure_idx[i]   the panel row exposures were read from  (= start - lag)
    asset_return[i]   the return the regression explained     (period i)
    cov[i]            the forecast built from data THROUGH i
    realized[i + 1]   what cov[i] is judged against

`cov[i]` paired with `realized[i]` would be in-sample and would produce a
bias statistic near 1.0 for the wrong reason. That pairing is the first
thing to check if the gate comes back suspiciously clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from master_us.risk.build import RiskInputs
from master_us.risk.covariance import specific_risk
from master_us.risk.descriptors import FACTOR_DESCRIPTORS
from master_us.risk.pipeline import RiskModel


@dataclass(frozen=True)
class RiskForecast:
    """Everything the bias tests need, on one aligned index."""

    exposures: npt.NDArray[np.float64]      # (T, N, K)
    factor_returns: npt.NDArray[np.float64]  # (T, K)
    asset_returns: npt.NDArray[np.float64]   # (T, N)
    specific_var: npt.NDArray[np.float64]    # (T, N)
    mcap: npt.NDArray[np.float64]            # (T, N)
    is_primary: npt.NDArray[np.bool_]        # (N,)
    factor_names: tuple[str, ...]
    tickers: npt.NDArray[np.object_]

    @property
    def n_factors(self) -> int:
        return int(self.exposures.shape[2])


def build_forecast(
    inputs: RiskInputs,
    model: RiskModel,
    hl_specific: float = 60.0,
    shrink: bool = True,
) -> RiskForecast:
    """Stack style + industry exposures and estimate specific risk.

    The exposure matrix carries a market column of ones, matching §9.3's
    intercept. Without it the portfolio's market risk would be attributed
    entirely to specific variance, and every long-only portfolio's risk
    would be badly under-forecast.
    """
    panel = inputs.panel
    fr = model.factor_returns
    idx = model.exposure_idx
    t, n = len(idx), len(panel.tickers)

    style_names = tuple(FACTOR_DESCRIPTORS)
    industries = tuple(sorted(set(panel.industry.tolist())))
    factor_names = ("market", *style_names, *(f"ind:{j}" for j in industries))
    k = len(factor_names)

    exposures = np.zeros((t, n, k))
    exposures[:, :, 0] = 1.0
    for j, name in enumerate(style_names, start=1):
        exposures[:, :, j] = np.nan_to_num(model.exposures[name], nan=0.0)
    for j, ind in enumerate(industries, start=1 + len(style_names)):
        exposures[:, :, j] = (panel.industry == ind).astype(float)[None, :]


    specific = specific_risk(
        fr.specific,
        size=np.nan_to_num(model.exposures["size"], nan=0.0),
        leverage=np.nan_to_num(model.exposures["leverage"], nan=0.0),
        industry=panel.industry,
        halflife=hl_specific,
        shrink=shrink,
    )

    return RiskForecast(
        exposures=exposures,
        factor_returns=np.hstack([fr.returns, fr.industry_returns]),
        asset_returns=model.asset_returns,
        specific_var=specific.variance,
        mcap=panel.mcap[idx],
        is_primary=panel.is_primary,
        factor_names=factor_names,
        tickers=panel.tickers,
    )
