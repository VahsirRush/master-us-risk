"""The backtest loop — ties construct, costs and metrics together.

One deliberately simple convention, stated once:

* Scores at date t use information through the close of t.
* The book is traded at the close of t (a rebalance date), so weights chosen
  at t earn returns from t+1 until the next rebalance.
* Costs are charged on the rebalance day.
* Weights are held CONSTANT between rebalances (no drift compounding). For a
  monthly-rebalanced, roughly equal-weight book this is the standard academic
  approximation; the error is second-order and it keeps the turnover
  attribution exact — every |Δw| the cost model sees is a real trade at a
  real rebalance, not drift noise.
* Equity is the cumulative SUM of daily returns (constant notional), matching
  `metrics.max_drawdown`.

The engine never invents trades: on non-rebalance days turnover is zero and
costs are zero. Names that leave the mask between rebalances keep their
weight until the next rebalance but earn 0 return on days their return is
missing (a delisting mid-hold is a real loss the day it happens, and shows up
in the last observed return).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from master_us.backtest.costs import CostConfig, flat_cost, realistic_cost
from master_us.backtest.metrics import BacktestSeries

FloatMat = npt.NDArray[np.float64]
FloatVec = npt.NDArray[np.float64]
BoolMat = npt.NDArray[np.bool_]

# constructor(scores_t, mask_t, prev_weights) -> weights
Constructor = Callable[[FloatVec, npt.NDArray[np.bool_], FloatVec], FloatVec]


@dataclass(frozen=True)
class BacktestResult:
    """Everything one engine run produces."""

    dates: pd.DatetimeIndex
    weights: FloatMat  # (T, N) book held INTO each day
    series: BacktestSeries
    rebalance_dates: pd.DatetimeIndex
    turnover_per_rebalance: FloatVec  # one-way, at each rebalance

    @property
    def equity(self) -> FloatVec:
        return np.cumsum(self.series.net)

    @property
    def equity_gross(self) -> FloatVec:
        return np.cumsum(self.series.gross)


def month_end_indices(dates: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
    """Index of the last trading day of each month in `dates`."""
    if len(dates) == 0:
        raise ValueError("dates is empty")
    period = dates.to_period("M")
    is_last = np.ones(len(dates), dtype=bool)
    is_last[:-1] = period[:-1] != period[1:]
    return np.flatnonzero(is_last)


def run_backtest(
    dates: pd.DatetimeIndex,
    returns: FloatMat,
    scores: FloatMat,
    mask: BoolMat,
    constructor: Constructor,
    rebalance_idx: npt.NDArray[np.int64],
    cost_cfg: CostConfig,
    tier: str = "baseline",
    spread: FloatMat | None = None,
    adv_dollars: FloatMat | None = None,
    portfolio_value: float = 1e8,
) -> BacktestResult:
    """Run one strategy through the engine.

    Parameters
    ----------
    returns:
        (T, N) simple daily returns. NaN where a name has no price — the
        engine treats a held name's NaN return as 0 for that day.
    scores:
        (T, N) signal, used only on rebalance dates. NaN = no opinion.
    rebalance_idx:
        Row indices of `dates` on which the book is re-formed.
    tier:
        "baseline" (flat bps) or "realistic" (Corwin-Schultz + impact —
        requires `spread` and `adv_dollars`).
    """
    t_len, n = returns.shape
    if scores.shape != (t_len, n) or mask.shape != (t_len, n):
        raise ValueError("returns, scores and mask must share a (T, N) shape")
    if len(dates) != t_len:
        raise ValueError(f"dates has {len(dates)} rows, returns has {t_len}")
    if tier not in ("baseline", "realistic"):
        raise ValueError(f"tier must be 'baseline' or 'realistic', got {tier!r}")
    if tier == "realistic" and (spread is None or adv_dollars is None):
        raise ValueError("realistic tier requires spread and adv_dollars matrices")
    if len(rebalance_idx) == 0:
        raise ValueError("no rebalance dates")

    weights = np.zeros((t_len, n))
    gross = np.zeros(t_len)
    costs = np.zeros(t_len)
    turnover = np.zeros(t_len)
    rebalance_set = set(int(i) for i in rebalance_idx)
    turnover_per_rebalance: list[float] = []

    held = np.zeros(n)
    for t in range(t_len):
        # The book held INTO day t earns day t's returns.
        weights[t] = held
        if held.any():
            r = returns[t]
            gross[t] = float(np.nansum(held * np.where(np.isfinite(r), r, 0.0)))

        if t in rebalance_set:
            target = constructor(scores[t], mask[t], held)
            traded = np.abs(target - held)
            turnover[t] = 0.5 * float(traded.sum())
            turnover_per_rebalance.append(turnover[t])

            if tier == "baseline":
                costs[t] = flat_cost(target, held, cost_cfg.baseline_bps)
            else:
                assert spread is not None and adv_dollars is not None
                costs[t] = realistic_cost(
                    target, held, spread[t], adv_dollars[t], portfolio_value, cost_cfg
                )
            held = target

    series = BacktestSeries(gross=gross, costs=costs, turnover=turnover)
    return BacktestResult(
        dates=dates,
        weights=weights,
        series=series,
        rebalance_dates=dates[np.asarray(sorted(rebalance_set))],
        turnover_per_rebalance=np.asarray(turnover_per_rebalance),
    )
