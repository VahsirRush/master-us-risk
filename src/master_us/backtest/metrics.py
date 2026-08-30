"""Backtest metrics — spec section 5.3.

Every metric function returns a `MetricValue` carrying the (gross, net) pair.
No function returns a single float. The two inputs everything derives from are
the GROSS daily return series and the COST series (return units, charged on
the day incurred); net is always gross - cost, computed here and nowhere else,
so the two columns cannot drift apart.

Convention notes, fixed once:

* Annualization uses 252 days.
* Sharpe is the plain rf=0 convention on daily returns — for a dollar-neutral
  long-short book the return is already an excess return, and for the
  long-only variants the same convention is used so numbers are comparable
  down a column.
* Max drawdown is on the cumulative-sum equity curve (arithmetic, matching a
  constant-notional book), reported as a NEGATIVE number.
* `std` on these MetricValues is None: a single backtest has no seed
  dispersion. Seed-level aggregation happens in Phase 2+, where model
  training introduces the randomness these fields exist to carry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

from master_us.reporting.results import MetricValue

FloatVec = npt.NDArray[np.float64]

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestSeries:
    """The raw material every metric consumes.

    gross:    (T,) daily portfolio return before costs
    costs:    (T,) cost in return units charged that day (0 off-rebalance)
    turnover: (T,) one-way turnover executed that day, 0.5·Σ|Δw|
    """

    gross: FloatVec
    costs: FloatVec
    turnover: FloatVec

    def __post_init__(self) -> None:
        t = self.gross.shape[0]
        if self.costs.shape[0] != t or self.turnover.shape[0] != t:
            raise ValueError("gross, costs and turnover must be equal length")
        for name in ("gross", "costs", "turnover"):
            arr = getattr(self, name)
            if np.isnan(arr).any():
                raise ValueError(f"{name} contains NaN — fix upstream, do not drop here")
        if (self.costs < 0).any():
            raise ValueError("costs must be >= 0")

    @property
    def net(self) -> FloatVec:
        return self.gross - self.costs

    @property
    def n_days(self) -> int:
        return int(self.gross.shape[0])


def _pair(fn: Callable[[FloatVec], float], series: BacktestSeries) -> MetricValue:
    """Apply one scalar statistic to gross and net, packaged together."""
    return MetricValue(gross=fn(series.gross), net=fn(series.net))


# --------------------------------------------------------------------- #
# Return / risk                                                          #
# --------------------------------------------------------------------- #


def annualized_return(series: BacktestSeries) -> MetricValue:
    """Arithmetic mean daily return x 252 — the constant-notional convention."""
    return _pair(lambda r: float(r.mean() * TRADING_DAYS), series)


def excess_return(series: BacktestSeries, benchmark: FloatVec) -> MetricValue:
    """Annualized mean of (portfolio - benchmark). Benchmark is SPY by contract."""
    b = np.asarray(benchmark, dtype=np.float64)
    if b.shape != series.gross.shape:
        raise ValueError(f"benchmark shape {b.shape} != returns {series.gross.shape}")
    if np.isnan(b).any():
        raise ValueError("benchmark contains NaN")
    return _pair(lambda r: float((r - b).mean() * TRADING_DAYS), series)


def sharpe(series: BacktestSeries) -> MetricValue:
    def _sharpe(r: FloatVec) -> float:
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    return _pair(_sharpe, series)


def information_ratio(series: BacktestSeries, benchmark: FloatVec) -> MetricValue:
    """Mean active return over tracking error, annualized."""
    b = np.asarray(benchmark, dtype=np.float64)
    if b.shape != series.gross.shape:
        raise ValueError(f"benchmark shape {b.shape} != returns {series.gross.shape}")

    def _ir(r: FloatVec) -> float:
        active = r - b
        te = active.std(ddof=1)
        return float(active.mean() / te * np.sqrt(TRADING_DAYS)) if te > 0 else 0.0

    return _pair(_ir, series)


def max_drawdown(series: BacktestSeries) -> MetricValue:
    """Deepest peak-to-trough fall of the cumulative-sum curve. Negative."""

    def _mdd(r: FloatVec) -> float:
        equity = np.concatenate([[0.0], np.cumsum(r)])
        peak = np.maximum.accumulate(equity)
        return float((equity - peak).min())

    return _pair(_mdd, series)


def calmar(series: BacktestSeries) -> MetricValue:
    """Annualized return over |max drawdown|."""

    def _calmar(r: FloatVec) -> float:
        equity = np.concatenate([[0.0], np.cumsum(r)])
        dd = float((equity - np.maximum.accumulate(equity)).min())
        if dd == 0.0:
            return 0.0
        return float(r.mean() * TRADING_DAYS / abs(dd))

    return _pair(_calmar, series)


# --------------------------------------------------------------------- #
# Trading                                                                #
# --------------------------------------------------------------------- #


def daily_turnover(series: BacktestSeries) -> MetricValue:
    """Mean one-way turnover per day, 0.5·Σ|Δw| — the permanent column.

    Turnover is a property of the trade schedule, not of the cost tier, so
    gross == net by construction. It is still a MetricValue because the
    contract says no function returns a bare float.
    """
    mean_to = float(series.turnover.mean())
    return MetricValue(gross=mean_to, net=mean_to)


def hit_rate(series: BacktestSeries) -> MetricValue:
    """Fraction of days with a positive return, counted over active days only.

    Flat days (weights all zero, e.g. before the first rebalance) carry no
    information about the signal and are excluded from the denominator.
    """

    def _hit(r: FloatVec) -> float:
        active = r != 0.0
        return float((r[active] > 0).mean()) if active.any() else 0.0

    return _pair(_hit, series)


def average_holding_period(turnover: FloatVec) -> MetricValue:
    """Mean holding period in days, as 1 / mean daily one-way turnover.

    A book that turns over 5% a day replaces itself every 20 trading days.
    Cost-free by nature; gross == net.
    """
    to = np.asarray(turnover, dtype=np.float64)
    mean_to = float(to.mean())
    period = float(1.0 / mean_to) if mean_to > 0 else float("inf")
    return MetricValue(gross=period, net=period)


# --------------------------------------------------------------------- #
# Signal quality — cost-free, so gross == net by construction            #
# --------------------------------------------------------------------- #


def _per_date_correlations(
    scores: npt.NDArray[np.floating],
    forward_returns: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    method: str,
    min_names: int = 10,
) -> FloatVec:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(forward_returns, dtype=np.float64)
    if s.shape != r.shape or s.shape != mask.shape:
        raise ValueError("scores, forward_returns and mask must share a (T, N) shape")

    out = []
    for t in range(s.shape[0]):
        ok = mask[t] & np.isfinite(s[t]) & np.isfinite(r[t])
        if ok.sum() < min_names:
            continue
        a, b = s[t, ok], r[t, ok]
        if a.std() == 0 or b.std() == 0:
            continue
        if method == "pearson":
            out.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            out.append(float(scipy_stats.spearmanr(a, b).statistic))
    return np.asarray(out, dtype=np.float64)


def information_coefficient(
    scores: npt.NDArray[np.floating],
    forward_returns: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    method: str = "pearson",
) -> tuple[MetricValue, MetricValue]:
    """(IC, ICIR) per-date, method 'pearson' or 'spearman' (RankIC).

    IC is the mean per-date cross-sectional correlation; ICIR is mean over
    std of the same series. Costs do not apply to a correlation, so
    gross == net; the pair convention is kept for renderer uniformity.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    series = _per_date_correlations(scores, forward_returns, mask, method)
    if series.size == 0:
        raise ValueError("no dates had enough names to compute a correlation")
    mean_ic = float(series.mean())
    icir = float(mean_ic / series.std(ddof=1)) if series.std(ddof=1) > 0 else 0.0
    return (
        MetricValue(gross=mean_ic, net=mean_ic),
        MetricValue(gross=icir, net=icir),
    )


# --------------------------------------------------------------------- #
# The full table                                                         #
# --------------------------------------------------------------------- #


def summarize(
    series: BacktestSeries,
    benchmark: FloatVec | None = None,
) -> dict[str, MetricValue]:
    """Every §5.3 return/trading metric in one dict, keyed for the renderers."""
    out: dict[str, MetricValue] = {
        "ann_return": annualized_return(series),
        "sharpe": sharpe(series),
        "max_drawdown": max_drawdown(series),
        "calmar": calmar(series),
        "turnover": daily_turnover(series),
        "hit_rate": hit_rate(series),
        "holding_period_days": average_holding_period(series.turnover),
    }
    if benchmark is not None:
        out["excess_return"] = excess_return(series, benchmark)
        out["information_ratio"] = information_ratio(series, benchmark)
    return out
