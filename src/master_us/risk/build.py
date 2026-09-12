"""Assemble the Phase-5 risk inputs from data already on disk.

Reads `data/interim/panel_inputs.parquet` and `data/interim/fundamentals.parquet`
directly rather than the 1.3 GB `panel.pkl`. Two reasons, both practical:
the panel does not fit comfortably beside anything else on this machine (a
concurrent load OOM-killed a training run in Session 6), and the risk model
needs a different grid anyway — every tradeable name, not the feature
tensor's padded universe.

The two files are the same inputs the panel was built from, so this is not a
second source of truth; it is the same source read at a different stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from master_us.data.metadata import SHARES_STALENESS_DAYS, fetch_sectors, shares_outstanding_asof
from master_us.data.sources import DATA_ROOT, sec_user_agent
from master_us.risk.descriptors import (
    FACTOR_DESCRIPTORS,
    DescriptorPanel,
    _rolling_std,
    capm_beta_resid_vol,
    fundamental_asof,
    fundamental_lagged,
    momentum_12_1,
)
from master_us.risk.industry import primary_class, resolve_industry

PANEL_INPUTS = DATA_ROOT / "interim" / "panel_inputs.parquet"
FUNDAMENTALS = DATA_ROOT / "interim" / "fundamentals.parquet"
CIK_MAP = DATA_ROOT / "raw" / "sec_cik_ticker_map.parquet"
SIC_CACHE = DATA_ROOT / "raw" / "sec_sic_by_cik.parquet"
RISK_DIR = DATA_ROOT / "processed" / "risk"


@dataclass(frozen=True)
class RiskInputs:
    """Everything §9.2 and §9.3 need, on one aligned grid."""

    panel: DescriptorPanel
    daily_returns: npt.NDArray[np.float64]
    market: npt.NDArray[np.float64]
    industry_source: npt.NDArray[np.object_]
    observed: npt.NDArray[np.bool_]
    """True where the name was tradeable that day.

    Coverage has to be measured against this, not against the full
    (dates x tickers) rectangle. A name that joined the index in 2020 has no
    2010 data because it was not there, which is correct behaviour, not a
    gap — counting it as missing understates every descriptor by roughly the
    universe turnover rate."""


def _grid(frame: pl.DataFrame, dates: list[object], tickers: list[str], col: str) -> npt.NDArray[np.float64]:
    """Pivot a long frame to a dense (T x N) float array."""
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: i for i, t in enumerate(tickers)}
    out = np.full((len(dates), len(tickers)), np.nan)
    d = frame["date"].to_list()
    t = frame["ticker"].to_list()
    v = frame[col].to_numpy()
    for k in range(len(v)):
        i, j = di.get(d[k]), ti.get(t[k])
        if i is not None and j is not None:
            out[i, j] = v[k]
    return out


def _safe_div(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """a / b with non-positive or non-finite denominators giving NaN.

    A negative book value is not a small positive one — B/P for a
    negative-equity firm is not meaningfully "very high value", it is
    undefined, and letting it through produces a huge positive exposure for
    the most distressed names in the cross-section.
    """
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out[ok] = a[ok] / b[ok]
    return out


def build_risk_inputs(
    start: str = "2010-01-01",
    panel_inputs: Path = PANEL_INPUTS,
    fundamentals: Path = FUNDAMENTALS,
) -> RiskInputs:
    """Descriptors, industry, and returns for the whole sample."""
    pi = (
        pl.read_parquet(panel_inputs)
        .filter(pl.col("date") >= pl.lit(start).str.to_datetime())
        .filter(pl.col("tradeable"))
        .select("date", "ticker", "close", "adj_close", "volume", "adv_21d")
        .sort(["ticker", "date"])
    )
    dates = sorted(pi["date"].unique().to_list())
    tickers = sorted(pi["ticker"].unique().to_list())
    shape = (len(dates), len(tickers))

    close = _grid(pi, dates, tickers, "close")
    adj = _grid(pi, dates, tickers, "adj_close")
    volume = _grid(pi, dates, tickers, "volume")

    # returns from the ADJUSTED close (splits and dividends), mcap from the
    # RAW close times as-filed shares — mixing the two bases is what put
    # AAPL's 2015 market cap at $188B instead of $731B in Session 5.
    rets = np.full(shape, np.nan)
    rets[1:] = adj[1:] / adj[:-1] - 1.0
    rets[~np.isfinite(rets)] = np.nan

    # ---- metadata: shares, mcap, industry, primary class ----------------
    shares_frame = shares_outstanding_asof(tickers)
    grid_frame = pi.select("date", "ticker")
    joined = grid_frame.join_asof(
        shares_frame,
        left_on="date",
        right_on="filed",
        by="ticker",
        strategy="backward",
        tolerance=f"{SHARES_STALENESS_DAYS}d",
    )
    shares = _grid(joined, dates, tickers, "shares")
    mcap = shares * close
    mcap[~np.isfinite(mcap) | (mcap <= 0)] = np.nan

    cik_map = pl.read_parquet(CIK_MAP)
    sic = pl.read_parquet(SIC_CACHE)
    gics = fetch_sectors(sec_user_agent())
    ind_frame = resolve_industry(tickers, cik_map, sic, gics)
    lookup = dict(zip(ind_frame["ticker"].to_list(), ind_frame["industry"].to_list(), strict=True))
    src = dict(zip(ind_frame["ticker"].to_list(), ind_frame["industry_source"].to_list(), strict=True))
    industry = np.array([lookup.get(t, "Unclassified") for t in tickers], dtype=object)
    industry_source = np.array([src.get(t, "none") for t in tickers], dtype=object)

    adv_mean = (
        pi.group_by("ticker").agg(pl.col("adv_21d").mean().alias("mean_adv"))
    )
    prim = primary_class(tickers, cik_map, adv_mean)
    pmap = dict(zip(prim["ticker"].to_list(), prim["is_primary_class"].to_list(), strict=True))
    is_primary = np.array([bool(pmap.get(t, True)) for t in tickers])

    # ---- price descriptors ----------------------------------------------
    # Equal-weighted cross-sectional mean as the market proxy for CAPM.
    with np.errstate(invalid="ignore"):
        market = np.nanmean(rets, axis=1)
    beta, resid_vol = capm_beta_resid_vol(rets, market)

    dollar_vol = close * volume
    turnover = _safe_div(dollar_vol, mcap)

    def roll_mean(a: npt.NDArray[np.float64], w: int) -> npt.NDArray[np.float64]:
        frame = pl.DataFrame(a, schema=[f"c{i}" for i in range(a.shape[1])])
        rolled = frame.select(
            [pl.col(c).rolling_mean(window_size=w, min_samples=int(w * 0.8)) for c in frame.columns]
        )
        return rolled.to_numpy()

    desc: dict[str, npt.NDArray[np.float64]] = {
        "log_mcap": np.log(np.where(np.isfinite(mcap) & (mcap > 0), mcap, np.nan)),
        "mom_12_1": momentum_12_1(rets),
        "vol_60d": _rolling_std(rets, 60),
        "resid_vol": resid_vol,
        "beta": beta,
        "turn_21d": roll_mean(turnover, 21),
        "turn_60d": roll_mean(turnover, 60),
        "turn_252d": roll_mean(turnover, 252),
    }

    # ---- fundamental descriptors ----------------------------------------
    facts = pl.read_parquet(fundamentals).filter(pl.col("ticker").is_in(tickers))

    def fund(concept: str) -> npt.NDArray[np.float64]:
        f = fundamental_asof(grid_frame, facts, concept)
        return _grid(f, dates, tickers, concept)

    def fund_lag(concept: str, years: int) -> npt.NDArray[np.float64]:
        f = fundamental_lagged(grid_frame, facts, concept, years)
        return _grid(f, dates, tickers, f"{concept}_lag{years}y")

    equity = fund("equity")
    net_income = fund("net_income")
    ocf = fund("operating_cash_flow")
    assets = fund("assets")
    ltd = fund("long_term_debt")
    revenue = fund("revenue")

    desc["bp"] = _safe_div(equity, mcap)
    desc["ep"] = _safe_div(net_income, mcap)
    desc["cfp"] = _safe_div(ocf, mcap)
    desc["debt_assets"] = _safe_div(ltd, assets)
    desc["debt_equity"] = _safe_div(ltd, equity)
    desc["roe"] = _safe_div(net_income, equity)
    desc["accruals"] = _safe_div(net_income - ocf, assets)

    rev3 = fund_lag("revenue", 3)
    ni3 = fund_lag("net_income", 3)
    desc["sales_growth_3y"] = _safe_div(revenue - rev3, np.abs(rev3))
    desc["earnings_growth_3y"] = _safe_div(net_income - ni3, np.abs(ni3))

    # Earnings variability: trailing dispersion of ROE, a slow-moving
    # quality signal. Computed on the PIT ROE series, so it inherits the
    # filed-date discipline rather than needing its own.
    desc["earnings_var"] = _rolling_std(desc["roe"], 252)

    panel = DescriptorPanel(
        dates=np.array(dates),
        tickers=np.array(tickers, dtype=object),
        values={k: desc[k] for group in FACTOR_DESCRIPTORS.values() for k in group},
        industry=industry,
        mcap=mcap,
        is_primary=is_primary,
    )
    return RiskInputs(
        panel=panel,
        daily_returns=rets,
        market=market,
        industry_source=industry_source,
        observed=np.isfinite(close),
    )


def descriptor_coverage(
    panel: DescriptorPanel,
    observed: npt.NDArray[np.bool_],
    warmup_years: int = 1,
) -> pl.DataFrame:
    """Non-NaN rate per descriptor among cells where the name was trading.

    `warmup_years` drops the leading period from the worst-year search. The
    252-day descriptors are NaN by construction until their window fills, so
    the first year is always the worst and reporting it as a coverage
    problem would be reporting the definition of a rolling window.
    """
    years = np.array([np.datetime64(d, "Y").astype(int) + 1970 for d in panel.dates])
    eligible = sorted(set(years.tolist()))[warmup_years:]
    rows = []
    for name, arr in panel.values.items():
        finite = np.isfinite(arr) & observed
        overall = float(finite[observed].mean())
        by_year = {}
        for y in eligible:
            sel = years == y
            denom = observed[sel].sum()
            if denom:
                by_year[int(y)] = float(finite[sel].sum() / denom)
        worst_year = min(by_year, key=lambda k: by_year[k]) if by_year else None
        rows.append(
            {
                "descriptor": name,
                "overall": overall,
                "worst_year": worst_year,
                "worst_rate": by_year.get(worst_year) if worst_year else None,
            }
        )
    return pl.DataFrame(rows).sort("overall")
