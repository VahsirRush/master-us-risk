"""Descriptor standardization — spec §9.2.

The order is fixed and the spec is explicit about why:

1. Winsorize each descriptor at +/-3 SD
2. **Z-score WITHIN INDUSTRY** — the Barra convention, not a global z-score
3. Multi-descriptor factors: equal-weight the component z-scores, re-z-score
4. Demean so the cap-weighted mean exposure is zero

Step 2 is the one that gets shortcut. A global z-score is easier and looks
almost identical on a scatter plot, but it leaves industry structure inside
the style exposures: in a cross-section where utilities are uniformly low-
beta, a global volatility z-score hands every utility the same negative
exposure, and the industry dummy and the style factor then fight over the
same variance. The spec flags this as degrading the eventual bias statistic,
which is the Phase 6 gate — so getting it wrong here fails a test two phases
later, in a way that would be very hard to trace back.

Step 4 makes the intercept in §9.3 interpretable as the market: if the
cap-weighted mean style exposure is zero, the cap-weighted portfolio has no
style tilt by construction, so whatever the intercept picks up is market
return rather than a residual size or value bet.

Everything here is per-date and cross-sectional. No function in this module
looks across time, which is what keeps it free of look-ahead by
construction rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

MIN_INDUSTRY_MEMBERS = 5


def winsorize(x: npt.NDArray[np.float64], n_sigma: float = 3.0) -> npt.NDArray[np.float64]:
    """Clip to ±`n_sigma` about the mean, ignoring NaN.

    Mean and sigma are computed on the unclipped cross-section, so a single
    extreme outlier inflates sigma and clips less aggressively than an
    iterated version would. That is deliberate: iterating to convergence
    would let the clip threshold depend on the outlier it is meant to
    contain.
    """
    out = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(out)
    if ok.sum() < 2:
        return out
    mu = out[ok].mean()
    sd = out[ok].std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return out
    np.clip(out, mu - n_sigma * sd, mu + n_sigma * sd, out=out, where=ok)
    return out


def zscore_within_industry(
    x: npt.NDArray[np.float64],
    industry: npt.NDArray[np.object_],
    min_members: int = MIN_INDUSTRY_MEMBERS,
) -> npt.NDArray[np.float64]:
    """Z-score each name against its own industry's cross-section.

    An industry with fewer than `min_members` valid observations on the date
    falls back to the global cross-section: a z-score against two peers is
    noise dressed as a standardized exposure, and the fallback is preferable
    to emitting one. An industry whose members are all identical (zero
    dispersion) gets zeros, not infinities.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return out

    g_mu = x[ok].mean()
    g_sd = x[ok].std(ddof=1)

    for code in np.unique(industry):
        members = (industry == code) & ok
        cnt = int(members.sum())
        if cnt == 0:
            continue
        if cnt < min_members:
            if g_sd > 0:
                out[members] = (x[members] - g_mu) / g_sd
            else:
                out[members] = 0.0
            continue
        mu = x[members].mean()
        sd = x[members].std(ddof=1)
        out[members] = (x[members] - mu) / sd if sd > 0 else 0.0
    return out


def combine_descriptors(
    components: list[npt.NDArray[np.float64]],
    industry: npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    """Equal-weight standardized components, then re-z-score the result.

    Equal weight, not a fit: the spec says equal-weight, and estimating
    descriptor weights from the same data the factor is then used to explain
    is circular. Names missing some components use the mean of the ones they
    have — dropping them instead would silently shrink the universe for
    every multi-descriptor factor, which on this data would cost the whole
    value factor for any name missing cash-flow.
    """
    if not components:
        raise ValueError("no components to combine")
    stacked = np.vstack([np.asarray(c, dtype=np.float64) for c in components])
    with np.errstate(invalid="ignore"):
        combined = np.nanmean(stacked, axis=0)
    combined[np.all(~np.isfinite(stacked), axis=0)] = np.nan
    return zscore_within_industry(combined, industry)


def demean_cap_weighted(
    x: npt.NDArray[np.float64],
    mcap: npt.NDArray[np.float64],
    is_primary: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float64]:
    """Shift so the cap-weighted mean exposure is exactly zero.

    `is_primary` excludes secondary share classes from the weighting. A
    multi-class issuer carries its entity-level market cap on every class,
    so including both would weight that company twice in the very average
    that defines "the market".

    Falls back to an equal-weighted demean when no usable caps exist on the
    date — a zero-mean exposure is still required for §9.3's intercept to
    mean anything, and refusing to demean would be worse than demeaning
    imperfectly.
    """
    out = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(out)
    if not ok.any():
        return out

    w = np.where(np.isfinite(mcap) & (mcap > 0), mcap, 0.0)
    if is_primary is not None:
        w = np.where(is_primary, w, 0.0)
    w = np.where(ok, w, 0.0)

    if w.sum() > 0:
        out[ok] -= float((w[ok] * out[ok]).sum() / w.sum())
    else:
        out[ok] -= float(out[ok].mean())
    return out


def standardize_factor(
    components: list[npt.NDArray[np.float64]],
    industry: npt.NDArray[np.object_],
    mcap: npt.NDArray[np.float64],
    is_primary: npt.NDArray[np.bool_] | None = None,
    n_sigma: float = 3.0,
) -> npt.NDArray[np.float64]:
    """The full §9.2 pipeline for one factor on one date.

    winsorize -> z-score within industry -> equal-weight combine and
    re-z-score -> cap-weighted demean. Single-descriptor factors skip the
    combine step's averaging but still re-z-score, so every factor leaves
    this function on the same scale.
    """
    standardized = [
        zscore_within_industry(winsorize(c, n_sigma), industry) for c in components
    ]
    combined = (
        standardized[0] if len(standardized) == 1
        else combine_descriptors(standardized, industry)
    )
    return demean_cap_weighted(combined, mcap, is_primary)


def build_exposures(
    descriptors: dict[str, npt.NDArray[np.float64]],
    factor_map: dict[str, tuple[str, ...]],
    industry: npt.NDArray[np.object_],
    mcap: npt.NDArray[np.float64],
    is_primary: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[np.float64]]:
    """Standardized exposures for every factor, on the full (T x N) grid.

    Loops dates because every step of §9.2 is cross-sectional; vectorizing
    across dates would require the industry partition to be constant in
    time, which it is here but should not be assumed by the code.
    """
    t = mcap.shape[0]
    out = {f: np.full(mcap.shape, np.nan) for f in factor_map}
    for f, names in factor_map.items():
        missing = [n for n in names if n not in descriptors]
        if missing:
            raise KeyError(f"factor {f!r} needs missing descriptors {missing}")
        for i in range(t):
            comps = [descriptors[n][i] for n in names]
            if not any(np.isfinite(c).any() for c in comps):
                continue
            out[f][i] = standardize_factor(comps, industry, mcap[i], is_primary)
    return out
