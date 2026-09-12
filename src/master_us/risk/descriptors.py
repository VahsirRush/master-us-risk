"""Style factor descriptors — spec §9.1.

Eight factors, each built from one or more descriptors:

    Size        log(market cap)
    Value       B/P, E/P, CF/P
    Momentum    12-1 return (skip the most recent month)
    Volatility  60d realized vol, CAPM residual vol, beta
    Liquidity   21/60/252d turnover ratios
    Leverage    debt/assets, debt/equity
    Growth      3yr sales growth, 3yr earnings growth
    Quality     ROE, accruals, earnings variability

POINT-IN-TIME DISCIPLINE. Every fundamental enters through an as-of backward
join on the SEC `filed` date, never `period_end`. This is rule 1 of the
project and the single most common way a factor model becomes a look-ahead
machine: a 2015-Q4 balance sheet is not knowable on 2015-12-31, it is
knowable when it is filed in February 2016. `fundamental_asof` is the only
path fundamentals take into this module, so there is one place to audit.

WHAT THE FREE DATA PATH CANNOT DO, stated rather than papered over:

* **No short-term debt.** Only `long_term_debt` is in the cached tag set, so
  leverage is long-term debt over assets and over equity, not total debt.
  Firms funding themselves on the front end look less levered than they are.
* **Book value is common equity**, not tangible book — no goodwill or
  intangibles tag is cached, so B/P is not adjusted for them.
* **Accruals use the cash-flow definition** (net income less operating cash
  flow, scaled by assets), not the balance-sheet working-capital build,
  which would need current assets and liabilities that are not cached.
* **`long_term_debt` resolves for only 72.5-87.5% of filers** (Phase 0's
  coverage table). Leverage is the weakest of the eight factors here and
  `coverage_table` reports it per year rather than letting it pass quietly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from master_us.data.sec import TAG_FALLBACKS

TRADING_DAYS = 252

# Descriptor -> the factor it belongs to. Multi-descriptor factors are
# equal-weighted in §9.2 after each component is standardized.
FACTOR_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "size": ("log_mcap",),
    "value": ("bp", "ep", "cfp"),
    "momentum": ("mom_12_1",),
    "volatility": ("vol_60d", "resid_vol", "beta"),
    "liquidity": ("turn_21d", "turn_60d", "turn_252d"),
    "leverage": ("debt_assets", "debt_equity"),
    "growth": ("sales_growth_3y", "earnings_growth_3y"),
    "quality": ("roe", "accruals", "earnings_var"),
}

ALL_DESCRIPTORS: tuple[str, ...] = tuple(
    d for group in FACTOR_DESCRIPTORS.values() for d in group
)

# The XBRL concepts the fundamentals-dependent descriptors need. Everything
# here is already in `TAG_FALLBACKS`; the list exists so `coverage_table` can
# report on exactly the concepts Phase 5 leans on rather than all of them.
BARRA_CONCEPTS: tuple[str, ...] = (
    "equity",
    "net_income",
    "operating_cash_flow",
    "assets",
    "long_term_debt",
    "revenue",
)


@dataclass(frozen=True)
class DescriptorPanel:
    """Descriptors on a (date x ticker) grid, aligned to `dates`/`tickers`."""

    dates: npt.NDArray[np.datetime64]
    tickers: npt.NDArray[np.object_]
    values: dict[str, npt.NDArray[np.float64]]
    industry: npt.NDArray[np.object_]
    mcap: npt.NDArray[np.float64]
    is_primary: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.tickers))
        for name, arr in self.values.items():
            if arr.shape != shape:
                raise ValueError(f"descriptor {name} has shape {arr.shape}, expected {shape}")
        if self.mcap.shape != shape:
            raise ValueError(f"mcap has shape {self.mcap.shape}, expected {shape}")
        if self.industry.shape != (len(self.tickers),):
            raise ValueError("industry must be one label per ticker")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.values)


# --------------------------------------------------------------------- #
# point-in-time fundamentals                                             #
# --------------------------------------------------------------------- #


def resolve_concept(facts: pl.DataFrame, concept: str) -> pl.DataFrame:
    """[ticker, filed, value] for one concept, best available tag per filing.

    Tag priority is `TAG_FALLBACKS[concept]` order — the same resolution the
    Phase 0 pipeline uses, so a concept means the same thing here as it did
    in the coverage table.
    """
    if concept not in TAG_FALLBACKS:
        raise KeyError(f"unknown concept {concept!r}; known: {sorted(TAG_FALLBACKS)}")
    priority = {tag: i for i, tag in enumerate(TAG_FALLBACKS[concept])}
    return (
        facts.filter(pl.col("tag").is_in(list(priority)))
        .with_columns(pl.col("tag").replace_strict(priority, return_dtype=pl.Int32).alias("_rank"))
        .sort("_rank")
        .group_by(["ticker", "filed"])
        .first()
        .select("ticker", pl.col("filed").cast(pl.Datetime("ns")), pl.col("value").alias(concept))
        .sort(["ticker", "filed"])
    )


def fundamental_asof(
    grid: pl.DataFrame,
    facts: pl.DataFrame,
    concept: str,
    tolerance_days: int = 400,
) -> pl.DataFrame:
    """As-of backward join of one concept onto (date, ticker).

    `tolerance_days` caps staleness: beyond it the value is null rather than
    a stale guess carried indefinitely. 400d covers an annual filer plus a
    late filing; measured sensitivity from 400d to 1100d moves coverage by
    ~0.5pp, so the cap is not what limits this data — early XBRL adoption is.
    """
    resolved = resolve_concept(facts, concept)
    return grid.join_asof(
        resolved,
        left_on="date",
        right_on="filed",
        by="ticker",
        strategy="backward",
        tolerance=f"{tolerance_days}d",
    ).drop("filed")


def fundamental_lagged(
    grid: pl.DataFrame,
    facts: pl.DataFrame,
    concept: str,
    years: int,
    tolerance_days: int = 400,
) -> pl.DataFrame:
    """The value of `concept` as known `years` before each date.

    Used by the growth descriptors. Implemented by shifting the *join key*
    back, not by shifting a filed series forward — so the comparison is
    "what was knowable then" against "what is knowable now", both PIT.
    """
    resolved = resolve_concept(facts, concept).rename({concept: f"{concept}_lag{years}y"})
    # The duration subtraction demotes the key to microseconds; join_asof
    # refuses a unit mismatch rather than casting silently, which is the
    # right behaviour and means the cast belongs here explicitly.
    shifted = grid.with_columns(
        (pl.col("date") - pl.duration(days=365 * years))
        .cast(pl.Datetime("ns"))
        .alias("_asof")
    ).sort(["ticker", "_asof"])
    return (
        shifted.join_asof(
            resolved,
            left_on="_asof",
            right_on="filed",
            by="ticker",
            strategy="backward",
            tolerance=f"{tolerance_days}d",
        )
        .drop("_asof", "filed")
    )


def coverage_table(
    facts: pl.DataFrame,
    concepts: tuple[str, ...] = BARRA_CONCEPTS,
    years: tuple[int, ...] = (2010, 2013, 2016, 2018, 2021, 2025),
) -> pl.DataFrame:
    """Share of filing CIKs resolving each concept, by year.

    Distinct CIKs, not filings — counting filings double-counts an issuer
    that files a 10-K and three 10-Qs, which is how a 2018 revenue figure of
    161.8% got produced once already (Phase 0, fixed).
    """
    rows: list[dict[str, object]] = []
    with_year = facts.with_columns(pl.col("filed").dt.year().alias("year"))
    for year in years:
        slice_ = with_year.filter(pl.col("year") == year)
        total = slice_["cik"].n_unique()
        row: dict[str, object] = {"year": year, "ciks": total}
        for concept in concepts:
            tags = TAG_FALLBACKS[concept]
            got = slice_.filter(pl.col("tag").is_in(list(tags)))["cik"].n_unique()
            row[concept] = (got / total) if total else None
        rows.append(row)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------- #
# price-based descriptors                                                #
# --------------------------------------------------------------------- #


def _rolling_std(returns: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Trailing std over `window`, NaN until the window is full.

    Uses the sum/sum-of-squares recursion rather than a stride trick because
    the array is (4000 x 584) and the strided version allocates a copy per
    window.
    """
    t, n = returns.shape
    out = np.full((t, n), np.nan)
    x = np.nan_to_num(returns, nan=0.0)
    valid = np.isfinite(returns).astype(np.float64)
    cs = np.cumsum(np.vstack([np.zeros((1, n)), x]), axis=0)
    cs2 = np.cumsum(np.vstack([np.zeros((1, n)), x * x]), axis=0)
    cv = np.cumsum(np.vstack([np.zeros((1, n)), valid]), axis=0)
    for i in range(window - 1, t):
        lo = i + 1 - window
        cnt = cv[i + 1] - cv[lo]
        s = cs[i + 1] - cs[lo]
        s2 = cs2[i + 1] - cs2[lo]
        ok = cnt >= window * 0.8  # tolerate a few missing days, not a mostly-empty window
        mean = np.where(ok, s / np.maximum(cnt, 1), np.nan)
        var = np.where(ok, s2 / np.maximum(cnt, 1) - mean**2, np.nan)
        out[i] = np.sqrt(np.maximum(var, 0.0))
    return out


def momentum_12_1(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cumulative return over t-252..t-21 — 12 months, skipping the last one.

    The skip is not optional decoration: without it the descriptor picks up
    short-horizon reversal, which is a different (and opposite-signed)
    effect from momentum.
    """
    t, n = returns.shape
    out = np.full((t, n), np.nan)
    log1p = np.log1p(np.where(np.isfinite(returns), returns, np.nan))
    filled = np.nan_to_num(log1p, nan=0.0)
    valid = np.isfinite(log1p).astype(np.float64)
    cs = np.cumsum(np.vstack([np.zeros((1, n)), filled]), axis=0)
    cv = np.cumsum(np.vstack([np.zeros((1, n)), valid]), axis=0)
    for i in range(TRADING_DAYS, t):
        lo, hi = i + 1 - TRADING_DAYS, i + 1 - 21
        cnt = cv[hi] - cv[lo]
        ok = cnt >= (TRADING_DAYS - 21) * 0.8
        out[i] = np.where(ok, np.expm1(cs[hi] - cs[lo]), np.nan)
    return out


def capm_beta_resid_vol(
    returns: npt.NDArray[np.float64],
    market: npt.NDArray[np.float64],
    window: int = 252,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rolling CAPM beta and residual vol against a market return series.

    Beta is cov(r, m)/var(m) over the window; residual vol is the std of
    r - beta*m. Both are computed from cumulative moments so the whole
    (4000 x 584) grid costs one pass per window step.
    """
    t, n = returns.shape
    beta = np.full((t, n), np.nan)
    resid = np.full((t, n), np.nan)
    m = np.nan_to_num(market, nan=0.0).reshape(-1, 1)
    r = np.nan_to_num(returns, nan=0.0)
    valid = (np.isfinite(returns) & np.isfinite(market).reshape(-1, 1)).astype(np.float64)

    def cum(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)

    c_r, c_m = cum(r * valid), cum(m * valid)
    c_rm, c_mm = cum(r * m * valid), cum(m * m * valid)
    c_rr, c_v = cum(r * r * valid), cum(valid)

    for i in range(window - 1, t):
        lo = i + 1 - window
        cnt = c_v[i + 1] - c_v[lo]
        ok = cnt >= window * 0.8
        cnt_safe = np.maximum(cnt, 1)
        mr = (c_r[i + 1] - c_r[lo]) / cnt_safe
        mm = (c_m[i + 1] - c_m[lo]) / cnt_safe
        cov = (c_rm[i + 1] - c_rm[lo]) / cnt_safe - mr * mm
        var_m = (c_mm[i + 1] - c_mm[lo]) / cnt_safe - mm * mm
        var_r = (c_rr[i + 1] - c_rr[lo]) / cnt_safe - mr * mr
        b = np.where(ok & (var_m > 0), cov / np.where(var_m > 0, var_m, 1.0), np.nan)
        beta[i] = b
        # var(resid) = var(r) - 2b*cov + b^2*var(m) = var(r) - b^2*var(m)
        rv = var_r - b**2 * var_m
        resid[i] = np.where(ok, np.sqrt(np.maximum(rv, 0.0)), np.nan)
    return beta, resid
