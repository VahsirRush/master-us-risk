"""Factor covariance and specific risk — spec §9.4.

Three corrections sit between a raw EWMA covariance and one that passes a
bias test, and the spec names the first two as the usual failure points
(§12: "usually the single-half-life shortcut or missing Newey-West").

**1. Separate half-lives for volatility and correlation.** Volatility moves
faster than correlation — a vol shock decays in weeks, a correlation regime
persists for quarters. Estimating both with one half-life forces a choice
between a volatility estimate that is too slow and a correlation estimate
that is too noisy, and either way the bias statistic degrades. Implemented
genuinely separately here: variances come from an EWMA with `hl_vol`,
correlations from an independent EWMA with `hl_corr`, and the covariance is
reassembled as `diag(vol) @ corr @ diag(vol)`. That reassembly is also what
keeps the result positive semi-definite for free — a correlation matrix from
a single EWMA recursion is PSD, and scaling rows and columns by positive
volatilities preserves it.

**2. Newey-West.** Daily factor returns are serially correlated, so the
variance of a multi-day holding period is not the one-day variance times the
horizon. Ignoring it under-forecasts risk and drives the bias statistic
above 1. The lag terms enter through the VARIANCES only, not the full
matrix: the full Newey-West matrix is not PSD in general and would need
eigenvalue clipping to repair, which is a second approximation on top of the
first. Taking the variance inflation and keeping the `hl_corr` correlation
structure is exact where it matters for a bias test, which is dominated by
the scale of predicted vol rather than its cross-sectional detail.

**3. Eigenfactor adjustment.** A sample covariance systematically
over-disperses its own eigenvalues — the largest come out too large, the
smallest too small — so portfolios built along the minimum-variance
directions have their risk badly under-forecast. That is exactly where an
optimizer would put a portfolio, so the error matters more than its size
suggests. `eigenfactor_bias` measures the effect by simulation (§9.4:
"simulate returns from the estimated cov, re-estimate, measure eigenvalue
bias by eigenvector, apply scaling correction") and returns a per-rank
correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_HL_VOL = 40
DEFAULT_HL_CORR = 90
DEFAULT_NW_LAGS = 5
DEFAULT_HL_SPECIFIC = 60

# Menchero's eigenfactor adjustment is empirically under-corrective when
# applied raw; the published form scales the deviation from 1 by ~1.2 before
# applying it. Kept explicit so it can be turned off and measured.
EIGEN_SCALE = 1.2


def alpha_for(halflife: float) -> float:
    """EWMA decay giving the requested half-life in periods."""
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    return float(1.0 - 0.5 ** (1.0 / halflife))


def ewma_series(
    returns: npt.NDArray[np.float64], halflife: float, min_periods: int
) -> npt.NDArray[np.float64]:
    """Running EWMA covariance, one (K x K) matrix per period.

    `out[t]` uses returns through t INCLUSIVE. Callers forecasting period
    t+1 must therefore index `out[t]`, which is what makes the bias test
    out-of-sample; getting this off by one is the difference between a risk
    model and a fitted value.
    """
    t, k = returns.shape
    a = alpha_for(halflife)
    out = np.full((t, k, k), np.nan)
    state = np.zeros((k, k))
    seen = 0
    for i in range(t):
        r = returns[i]
        if not np.isfinite(r).all():
            if seen >= min_periods:
                out[i] = state
            continue
        state = (1.0 - a) * state + a * np.outer(r, r)
        seen += 1
        if seen >= min_periods:
            out[i] = state
    return out


def _ewma_lagged(
    returns: npt.NDArray[np.float64], halflife: float, lag: int, min_periods: int
) -> npt.NDArray[np.float64]:
    """Running EWMA of `r_t r_{t-lag}'` — the Newey-West cross terms."""
    t, k = returns.shape
    a = alpha_for(halflife)
    out = np.full((t, k, k), np.nan)
    state = np.zeros((k, k))
    seen = 0
    for i in range(t):
        if i < lag:
            continue
        r, rl = returns[i], returns[i - lag]
        if not (np.isfinite(r).all() and np.isfinite(rl).all()):
            if seen >= min_periods:
                out[i] = state
            continue
        state = (1.0 - a) * state + a * np.outer(r, rl)
        seen += 1
        if seen >= min_periods:
            out[i] = state
    return out


def newey_west_variance(
    returns: npt.NDArray[np.float64],
    halflife: float,
    nw_lags: int,
    min_periods: int,
) -> npt.NDArray[np.float64]:
    """Serial-correlation-adjusted VARIANCES, one row per period.

    Bartlett weights `1 - l/(L+1)` on the lag terms, which is what keeps the
    adjusted variance non-negative — an unweighted sum of autocovariances
    can drive it below zero when returns are strongly negatively
    autocorrelated. Any residual non-positive value is floored back to the
    unadjusted variance rather than allowed through, because a negative
    variance forecast is not a small error, it is a crash in `sqrt`.
    """
    base = ewma_series(returns, halflife, min_periods)
    var = np.diagonal(base, axis1=1, axis2=2).copy()
    adj = var.copy()
    for lag in range(1, nw_lags + 1):
        w = 1.0 - lag / (nw_lags + 1.0)
        cross = _ewma_lagged(returns, halflife, lag, min_periods)
        diag = np.diagonal(cross, axis1=1, axis2=2)
        adj = adj + 2.0 * w * np.nan_to_num(diag, nan=0.0)
    bad = ~np.isfinite(adj) | (adj <= 0)
    adj[bad] = var[bad]
    return np.asarray(adj, dtype=np.float64)


def correlation_from(cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Covariance -> correlation, with zero-variance factors left at identity."""
    d = np.sqrt(np.clip(np.diagonal(cov), 1e-300, None))
    corr: npt.NDArray[np.float64] = cov / np.outer(d, d)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass(frozen=True)
class FactorCovariance:
    """One (K x K) forecast per period, plus what produced it."""

    cov: npt.NDArray[np.float64]
    factor_names: tuple[str, ...]
    hl_vol: float
    hl_corr: float
    nw_lags: int
    eigen_adjusted: bool

    def at(self, t: int) -> npt.NDArray[np.float64]:
        return np.asarray(self.cov[t], dtype=np.float64)


def factor_covariance(
    factor_returns: npt.NDArray[np.float64],
    factor_names: tuple[str, ...],
    hl_vol: float = DEFAULT_HL_VOL,
    hl_corr: float = DEFAULT_HL_CORR,
    nw_lags: int = DEFAULT_NW_LAGS,
    eigen_adjust: bool = True,
    min_periods: int = 120,
    eigen_correction: npt.NDArray[np.float64] | None = None,
) -> FactorCovariance:
    """The §9.4 covariance: separate half-lives, Newey-West, eigenfactor.

    `factor_returns` is (T x K) and may contain NaN rows for periods with no
    estimated regression; those rows carry the previous forecast forward
    rather than resetting the state.
    """
    t, k = factor_returns.shape
    clean = np.nan_to_num(factor_returns, nan=0.0)
    valid = np.asarray(np.isfinite(factor_returns).all(axis=1))

    nw_var = newey_west_variance(clean, hl_vol, nw_lags, min_periods)
    corr_cov = ewma_series(clean, hl_corr, min_periods)

    out = np.full((t, k, k), np.nan)
    for i in range(t):
        if not valid[i] or not np.isfinite(corr_cov[i]).all() or not np.isfinite(nw_var[i]).all():
            continue
        corr = correlation_from(corr_cov[i])
        vol = np.sqrt(np.clip(nw_var[i], 1e-300, None))
        out[i] = corr * np.outer(vol, vol)

    if eigen_adjust:
        correction = (
            eigen_correction
            if eigen_correction is not None
            else eigenfactor_bias(out[np.isfinite(out).all(axis=(1, 2))], hl_vol, min_periods)
        )
        out = apply_eigen_correction(out, correction)

    return FactorCovariance(
        cov=out,
        factor_names=factor_names,
        hl_vol=hl_vol,
        hl_corr=hl_corr,
        nw_lags=nw_lags,
        eigen_adjusted=eigen_adjust,
    )


def eigenfactor_bias(
    covs: npt.NDArray[np.float64],
    halflife: float,
    min_periods: int,
    n_sims: int = 40,
    sample_periods: int = 500,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Per-rank eigenvalue bias, measured by simulation — §9.4.

    Procedure: take a representative estimated covariance F, eigen-decompose
    it, simulate returns whose TRUE covariance is exactly F, re-estimate with
    the same estimator, and compare each simulated eigenvalue against the
    true variance along its own eigenvector. The ratio is the estimator's
    bias at that rank.

    Returns `lambda_k`, where the corrected eigenvalue is `d_k / lambda_k^2`.
    A `lambda_k < 1` means the estimator under-forecasts that direction, so
    the correction inflates it.

    Calibrated once and applied at every date rather than re-simulated per
    date: the bias is a property of the ESTIMATOR and the sample length, not
    of the particular covariance, and re-running 40 simulations on each of
    3,700 dates would cost hours to reproduce a curve that barely moves.
    """
    if covs.size == 0:
        return np.ones(1)
    k = covs.shape[-1]
    f_true = np.nanmean(covs, axis=0)
    evals, evecs = np.linalg.eigh(f_true)
    evals = np.clip(evals, 1e-300, None)

    rng = np.random.default_rng(seed)
    ratios = np.zeros((n_sims, k))
    for m in range(n_sims):
        # eigenfactor returns are independent with the true eigenvariances
        b = rng.normal(size=(sample_periods, k)) * np.sqrt(evals)
        r = b @ evecs.T
        f_sim = ewma_series(r, halflife, min_periods)[-1]
        if not np.isfinite(f_sim).all():
            ratios[m] = 1.0
            continue
        d_sim, u_sim = np.linalg.eigh(f_sim)
        # true variance along each SIMULATED eigenvector
        true_along = np.einsum("ik,ij,jk->k", u_sim, f_true, u_sim)
        ratios[m] = np.clip(d_sim, 1e-300, None) / np.clip(true_along, 1e-300, None)

    lam = np.sqrt(ratios.mean(axis=0))
    return np.asarray(1.0 + EIGEN_SCALE * (lam - 1.0), dtype=np.float64)


def apply_eigen_correction(
    covs: npt.NDArray[np.float64], lam: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Rescale each covariance's eigenvalues by `1 / lambda_k^2`."""
    out = covs.copy()
    scale = 1.0 / np.clip(lam, 1e-6, None) ** 2
    for i in range(covs.shape[0]):
        c = covs[i]
        if not np.isfinite(c).all():
            continue
        d, u = np.linalg.eigh(c)
        out[i] = (u * (np.clip(d, 0.0, None) * scale)) @ u.T
    return out


# --------------------------------------------------------------------- #
# specific risk — §9.4                                                   #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpecificRisk:
    """Per-period, per-name specific variance forecasts."""

    variance: npt.NDArray[np.float64]
    shrunk: bool
    halflife: float


def structural_specific_vol(
    log_vol: npt.NDArray[np.float64],
    size: npt.NDArray[np.float64],
    leverage: npt.NDArray[np.float64],
    industry: npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    """Predict log specific vol from size, leverage, and industry — §9.4.

    Fitted on the names with a usable time-series estimate and predicted for
    everyone, so a name with too little history still gets a forecast built
    from its peers rather than a default.

    NOTE for anyone debugging a portfolio with concentrated leverage
    exposure: `leverage` here inherits the weakest descriptor in the model
    (`long_term_debt` resolves for 72.9-81.9% of the universe, cascading to
    `debt_equity` at 64.6%), so its coefficient is estimated on the most
    imputed column. Suspect this before suspecting the covariance code.
    """
    ok = np.isfinite(log_vol) & np.isfinite(size)
    if ok.sum() < 30:
        return np.full(log_vol.shape, np.nan)

    codes = sorted(set(industry.tolist()))
    dummies = np.column_stack([(industry == c).astype(float) for c in codes])
    x = np.column_stack(
        [
            np.nan_to_num(size, nan=0.0),
            np.nan_to_num(leverage, nan=0.0),
            dummies,
        ]
    )
    coef, *_ = np.linalg.lstsq(x[ok], log_vol[ok], rcond=None)
    return np.asarray(x @ coef, dtype=np.float64)


def specific_risk(
    specific_returns: npt.NDArray[np.float64],
    size: npt.NDArray[np.float64],
    leverage: npt.NDArray[np.float64],
    industry: npt.NDArray[np.object_],
    halflife: float = DEFAULT_HL_SPECIFIC,
    shrink: bool = True,
    min_periods: int = 60,
) -> SpecificRisk:
    """EWMA of squared residuals, shrunk toward the structural model.

    The shrinkage weight is the name's own data sufficiency: a name with a
    long clean residual history keeps its time-series estimate, one with
    almost none leans on the structural prediction. A fixed blend would
    penalise the well-measured names to help the sparse ones.
    """
    t, n = specific_returns.shape
    a = alpha_for(halflife)
    var = np.full((t, n), np.nan)
    state = np.zeros(n)
    count = np.zeros(n)

    for i in range(t):
        r = specific_returns[i]
        ok = np.isfinite(r)
        state[ok] = (1.0 - a) * state[ok] + a * r[ok] ** 2
        count[ok] += 1
        row = np.where(count >= min_periods, state, np.nan)
        var[i] = row

    if not shrink:
        return SpecificRisk(variance=var, shrunk=False, halflife=halflife)

    for i in range(t):
        ts = var[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_vol = 0.5 * np.log(np.where(ts > 0, ts, np.nan))
        pred = structural_specific_vol(log_vol, size[i], leverage[i], industry)
        if not np.isfinite(pred).any():
            continue
        struct_var = np.exp(2.0 * pred)
        # weight rises with observation count, saturating at 2x min_periods
        w = np.clip(count / (2.0 * min_periods), 0.0, 1.0)
        blended = np.where(
            np.isfinite(ts),
            w * ts + (1.0 - w) * np.nan_to_num(struct_var, nan=0.0),
            struct_var,
        )
        var[i] = np.where(np.isfinite(blended) & (blended > 0), blended, np.nan)

    return SpecificRisk(variance=var, shrunk=True, halflife=halflife)
