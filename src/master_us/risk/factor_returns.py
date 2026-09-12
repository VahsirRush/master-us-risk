"""Cross-sectional factor return estimation — spec §9.3.

    r_i = f_market + sum_k X_ik f_k + sum_j I_ij f_j + u_i

estimated per period by weighted least squares with weights sqrt(mcap),
subject to the cap-weighted industry returns summing to zero.

WHY THE CONSTRAINT. Industry dummies plus an intercept are collinear by
construction: every name is in exactly one industry, so the dummies sum to
the intercept column and the design matrix is rank-deficient. Something has
to pin it down. Dropping one industry works numerically but makes every
industry return a contrast against whichever one was dropped, and makes the
intercept that industry's return rather than the market's. The Barra
convention instead requires

    sum_j (cap-weight of industry j) * f_j = 0

so industries are returns relative to the cap-weighted market, and the
intercept is the market return itself. That is what makes the intercept
interpretable, and interpretability is the whole point of building this to
attribute a signal later.

IMPLEMENTATION. The constraint is imposed by reparameterization rather than
by a penalty. With J industries, the last industry's return is determined by
the other J-1 through the constraint, so the design matrix is rewritten in
J-1 free industry columns and the last is recovered afterwards. This is
exact — a Lagrange multiplier or a large penalty would only be approximate
and would leave the solution sensitive to the penalty weight.

WEIGHTS. sqrt(mcap), per the spec. The rationale is that residual variance
scales roughly inversely with size, so sqrt(mcap) weighting approximates GLS
without estimating a full residual covariance. Weights are capped at the
95th percentile of each date's cross-section: without a cap, the largest
handful of names dominate, and in 2024 the top name alone would carry more
weight than the bottom 200 combined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

MIN_NAMES = 30
WEIGHT_CAP_PCTILE = 95.0

# A name must have real values for at least this FRACTION of the style
# factors to enter the regression; the rest are set to their cap-weighted
# mean of zero. 0.75 is six of the eight real factors — enough that the
# imputation patches the weak fundamental factors rather than manufacturing
# a cross-section. Expressed as a fraction, not a count, so it keeps its
# meaning if the factor set changes size.
MIN_FACTOR_FRACTION = 0.75


def _min_factors(n_style: int) -> int:
    return max(1, int(np.ceil(MIN_FACTOR_FRACTION * n_style)))


@dataclass(frozen=True)
class FactorReturns:
    """Estimated factor returns and the residuals they leave behind."""

    dates: pd.DatetimeIndex
    factor_names: tuple[str, ...]
    industry_names: tuple[str, ...]
    returns: npt.NDArray[np.float64]        # (T x K) style + market
    industry_returns: npt.NDArray[np.float64]  # (T x J)
    specific: npt.NDArray[np.float64]       # (T x N)
    r2: npt.NDArray[np.float64]             # (T,) weighted R^2
    n_names: npt.NDArray[np.int64]          # (T,) names in each regression

    def to_frame(self) -> pd.DataFrame:
        """Factor returns as a DataFrame — market, styles, then industries."""
        return pd.DataFrame(
            np.hstack([self.returns, self.industry_returns]),
            index=self.dates,
            columns=list(self.factor_names) + list(self.industry_names),
        )

    def summary(self, periods_per_year: int = 12) -> pd.DataFrame:
        """Annualized mean, vol, t-stat and Sharpe for every factor."""
        frame = self.to_frame()
        mean = frame.mean() * periods_per_year
        vol = frame.std(ddof=1) * np.sqrt(periods_per_year)
        n = frame.notna().sum()
        tstat = frame.mean() / (frame.std(ddof=1) / np.sqrt(n))
        return pd.DataFrame(
            {"ann_mean": mean, "ann_vol": vol, "t_stat": tstat,
             "sharpe": mean / vol.replace(0, np.nan), "n_periods": n}
        )


def _weights(mcap: npt.NDArray[np.float64], cap_pctile: float = WEIGHT_CAP_PCTILE) -> npt.NDArray[np.float64]:
    """sqrt(mcap), capped at a percentile so mega-caps cannot dominate."""
    w = np.sqrt(np.where(np.isfinite(mcap) & (mcap > 0), mcap, 0.0))
    pos = w[w > 0]
    if pos.size:
        cap = np.percentile(pos, cap_pctile)
        if cap > 0:
            w = np.minimum(w, cap)
    return w


def _industry_design(
    industry: npt.NDArray[np.object_], names: tuple[str, ...]
) -> npt.NDArray[np.float64]:
    return np.column_stack([(industry == j).astype(np.float64) for j in names])


def estimate_period(
    exposures: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
    mcap: npt.NDArray[np.float64],
    industry: npt.NDArray[np.object_],
    industry_names: tuple[str, ...],
    is_primary: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float, int]:
    """One period's WLS. Returns (style+market, industry, specific, r2, n).

    The constraint sum_j c_j f_j = 0 (c = industry cap weights) is applied by
    substituting f_J = -(1/c_J) * sum_{j<J} c_j f_j into the design, solving
    for the J-1 free columns, then recovering f_J.
    """
    n_assets = returns.shape[0]
    n_style = exposures.shape[1]
    n_ind = len(industry_names)

    # A missing standardized exposure is set to zero, not dropped. Zero is
    # the cap-weighted mean by construction after §9.2's demean, so it says
    # "average on this factor" — the honest statement about a name whose
    # long-term-debt tag did not resolve. Requiring all eight factors
    # instead would gate the whole cross-section on the weakest one: it
    # silently produced NO regression at all before March 2013, because
    # `earnings_var` (57% coverage) is an intersection constraint on every
    # other factor.
    #
    # `MIN_FACTORS` stops this becoming imputation-all-the-way-down: a name
    # needs real values for most factors to enter at all.
    finite = np.isfinite(exposures)
    ok = (
        np.isfinite(returns)
        & np.isfinite(mcap)
        & (mcap > 0)
        & (finite.sum(axis=1) >= _min_factors(n_style))
    )
    n = int(ok.sum())
    specific = np.full(n_assets, np.nan)
    if n < MIN_NAMES:
        return np.full(n_style + 1, np.nan), np.full(n_ind, np.nan), specific, np.nan, n

    x_style = np.where(finite[ok], exposures[ok], 0.0)
    r = returns[ok]
    w = _weights(mcap[ok])
    ind = industry[ok]
    d = _industry_design(ind, industry_names)

    # cap weights per industry, from primary classes only
    cw_src = np.where(np.isfinite(mcap[ok]) & (mcap[ok] > 0), mcap[ok], 0.0)
    if is_primary is not None:
        cw_src = np.where(is_primary[ok], cw_src, 0.0)
    c = d.T @ cw_src
    total = c.sum()
    if total <= 0:
        return np.full(n_style + 1, np.nan), np.full(n_ind, np.nan), specific, np.nan, n
    c = c / total

    # The constrained-out industry must carry weight, or the substitution
    # divides by zero. Pick the largest.
    last = int(np.argmax(c))
    free = [j for j in range(n_ind) if j != last]
    if c[last] <= 0:
        return np.full(n_style + 1, np.nan), np.full(n_ind, np.nan), specific, np.nan, n

    # d_tilde_j = d_j - (c_j / c_last) * d_last  for j in free
    d_free = d[:, free] - np.outer(d[:, last], c[free] / c[last])
    design = np.column_stack([np.ones(n), x_style, d_free])

    sw = np.sqrt(w)
    aw = design * sw[:, None]
    rw = r * sw
    coef, *_ = np.linalg.lstsq(aw, rw, rcond=None)

    f_market_style = coef[: 1 + n_style]
    f_free = coef[1 + n_style :]
    f_ind = np.zeros(n_ind)
    f_ind[free] = f_free
    f_ind[last] = -float(c[free] @ f_free / c[last])

    fitted = design @ coef
    resid = r - fitted
    specific[np.flatnonzero(ok)] = resid

    ss_res = float((w * resid**2).sum())
    r_mean = float((w * r).sum() / w.sum())
    ss_tot = float((w * (r - r_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return f_market_style, f_ind, specific, r2, n


def estimate_factor_returns(
    exposures: dict[str, npt.NDArray[np.float64]],
    returns: npt.NDArray[np.float64],
    mcap: npt.NDArray[np.float64],
    industry: npt.NDArray[np.object_],
    dates: pd.DatetimeIndex,
    is_primary: npt.NDArray[np.bool_] | None = None,
) -> FactorReturns:
    """Cross-sectional WLS per period over the whole sample — spec §9.3.

    `returns` is (T x N) forward period returns aligned to `dates`, i.e.
    row t is the return EARNED over the period beginning at t, regressed on
    exposures known AT t. That alignment is the whole point: exposures must
    predate the return they explain or the factor returns are a look-ahead.
    """
    factor_names = tuple(exposures)
    industry_names = tuple(sorted(set(industry.tolist())))
    t = len(dates)
    x = np.stack([exposures[f] for f in factor_names], axis=-1)  # (T, N, K)

    f_style = np.full((t, len(factor_names) + 1), np.nan)
    f_ind = np.full((t, len(industry_names)), np.nan)
    specific = np.full(returns.shape, np.nan)
    r2 = np.full(t, np.nan)
    counts = np.zeros(t, dtype=np.int64)

    for i in range(t):
        fs, fi, sp, rr, n = estimate_period(
            x[i], returns[i], mcap[i], industry, industry_names, is_primary
        )
        f_style[i], f_ind[i], specific[i], r2[i], counts[i] = fs, fi, sp, rr, n

    return FactorReturns(
        dates=dates,
        factor_names=("market", *factor_names),
        industry_names=tuple(f"ind:{j}" for j in industry_names),
        returns=f_style,
        industry_returns=f_ind,
        specific=specific,
        r2=r2,
        n_names=counts,
    )


def to_period_returns(
    daily: npt.NDArray[np.float64], dates: pd.DatetimeIndex, freq: str = "ME"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], pd.DatetimeIndex]:
    """Compound daily simple returns into period returns.

    Returns (period_returns (P x N), index of each period's FIRST trading
    day, period end dates). The first-day index is what the caller uses to
    take exposures: exposures are read at the start of the period, the
    return is earned across it.
    """
    frame = pd.DataFrame(daily, index=dates)
    groups = frame.groupby(pd.Grouper(freq=freq))
    out, starts, ends = [], [], []
    pos = {d: i for i, d in enumerate(dates)}
    for end, block in groups:
        if block.empty:
            continue
        compounded = (1.0 + block.fillna(0.0)).prod() - 1.0
        all_nan = block.isna().all()
        compounded[all_nan] = np.nan
        out.append(compounded.to_numpy())
        starts.append(pos[block.index[0]])
        ends.append(end)
    return np.array(out), np.array(starts, dtype=np.int64), pd.DatetimeIndex(ends)
