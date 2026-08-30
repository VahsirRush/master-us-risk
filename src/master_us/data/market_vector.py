"""The market state vector — spec section 4.4. The gate's input.

For each of S&P 500 (^GSPC), S&P 400 MidCap (^SP400), Russell 2000 (^RUT):
latest index return, and rolling mean/std of return and of dollar volume over
d in {5, 10, 20, 30, 60}. That is 3 x (1 + 20) = 63 columns. Appended: VIX
level, VIX 20d change, cross-sectional return dispersion of the universe, and
the fraction of the universe above its 200d moving average. M = 67.

THE NaN CONTRACT: the vector must never contain NaN — it gates every
prediction. Three mechanisms enforce it, in order:

1. **Warm-up, not fill.** Index data is fetched from 2008; every rolling
   window (max 60d) and the VIX 20d change are fully warm long before the
   panel's 2010 start. Head-NaN never reaches the panel.
2. **Bounded forward-fill for calendar mismatches.** The panel's trading
   calendar comes from equities; an index missing one of those dates (a
   half-day session Yahoo skipped for one symbol) carries its last value
   forward, at most `MAX_FFILL` (5) days — CLAUDE.md rule 6's ceiling. This
   is the declared gap policy: an index LEVEL is a state, and yesterday's
   state is the honest belief about an unobserved today; inventing returns
   by interpolation would manufacture information instead.
3. **Hard failure beyond that.** A gap longer than 5 days, or any NaN
   surviving to the end, raises with the date and column named. Silent
   degradation in the gate input would poison every prediction downstream.

DOLLAR VOLUME PROXY, measured not assumed: Yahoo reports zero volume for
^SP400 (and ^MID) across their entire history, so the midcap segment's
dollar-volume columns use MDY, the SPDR MidCap ETF — its own close x volume.
^GSPC and ^RUT carry real volume (zero on 0.0-0.1% of days, absorbed by the
rolling windows). The three segments' dollar-volume columns are therefore in
different units (index-level x aggregate volume vs. ETF dollar volume); that
is fine because each column is only ever compared with its own history, and
downstream normalization is per-column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from master_us.data.sources import RAW_ROOT

MARKET_DIR = RAW_ROOT / "market"

INDEX_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    # (column prefix, level symbol, dollar-volume symbol)
    ("sp500", "^GSPC", "^GSPC"),
    ("sp400", "^SP400", "MDY"),  # MDY volume proxy — ^SP400 reports zero volume
    ("rut", "^RUT", "^RUT"),
)
WINDOWS = (5, 10, 20, 30, 60)
MAX_FFILL = 5

FloatMat = npt.NDArray[np.float64]


def market_vector_names() -> list[str]:
    names: list[str] = []
    for prefix, _, _ in INDEX_SEGMENTS:
        names.append(f"{prefix}_ret")
        for w in WINDOWS:
            names += [
                f"{prefix}_ret_mean_{w}d",
                f"{prefix}_ret_std_{w}d",
                f"{prefix}_dv_mean_{w}d",
                f"{prefix}_dv_std_{w}d",
            ]
    names += ["vix_level", "vix_chg_20d", "xs_dispersion", "pct_above_200dma"]
    return names


def _aligned_series(
    symbol: str,
    column: str,
    dates: pd.DatetimeIndex,
    market_dir: Path,
) -> pd.Series:
    """One symbol's column on the panel calendar, forward-filled <= MAX_FFILL."""
    path = market_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/06_fetch_market.py")
    raw = pl.read_parquet(path).sort("date").to_pandas().set_index("date")[column]
    aligned = raw.reindex(dates)

    missing = aligned.isna()
    if missing.any():
        filled = aligned.ffill(limit=MAX_FFILL)
        still = filled.isna() & (dates >= raw.index.min())
        if still.any():
            first = dates[still][0]
            raise ValueError(
                f"{symbol}.{column} has a gap longer than {MAX_FFILL} days at "
                f"{first.date()} — refetch the symbol rather than filling further"
            )
        aligned = filled
    return aligned


def build_index_block(
    dates: pd.DatetimeIndex,
    market_dir: Path = MARKET_DIR,
) -> pd.DataFrame:
    """The 63 index columns plus the two VIX columns, on the panel calendar.

    Rolling windows include the current day (information as of the close of
    t). `dates` should extend before the panel start by at least 60 trading
    days of index history; with the 2008 fetch that is automatic.
    """
    out = pd.DataFrame(index=dates)
    for prefix, level_sym, dv_sym in INDEX_SEGMENTS:
        level = _aligned_series(level_sym, "adj_close", dates, market_dir)
        ret = level.pct_change()

        dv_close = _aligned_series(dv_sym, "close", dates, market_dir)
        dv_volume = _aligned_series(dv_sym, "volume", dates, market_dir)
        dollar_volume = dv_close * dv_volume

        out[f"{prefix}_ret"] = ret
        for w in WINDOWS:
            out[f"{prefix}_ret_mean_{w}d"] = ret.rolling(w, min_periods=w).mean()
            out[f"{prefix}_ret_std_{w}d"] = ret.rolling(w, min_periods=w).std()
            out[f"{prefix}_dv_mean_{w}d"] = dollar_volume.rolling(w, min_periods=w).mean()
            out[f"{prefix}_dv_std_{w}d"] = dollar_volume.rolling(w, min_periods=w).std()

    vix = _aligned_series("^VIX", "close", dates, market_dir)
    out["vix_level"] = vix
    out["vix_chg_20d"] = vix - vix.shift(20)
    return out


def breadth_columns(
    adj_close: FloatMat,
    returns: FloatMat,
    tradeable: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(xs_dispersion, pct_above_200dma) from the stock panel itself.

    Dispersion is the per-date cross-sectional std of daily returns over
    tradeable names. pct_above_200dma is the fraction of tradeable names
    whose adjusted close sits above their own 200d moving average (window
    ending at and including t). Both are pure functions of information at t.
    """
    t_len = adj_close.shape[0]
    dispersion = np.full(t_len, np.nan)
    for t in range(t_len):
        ok = tradeable[t] & np.isfinite(returns[t])
        if ok.sum() >= 10:
            dispersion[t] = float(returns[t, ok].std(ddof=1))

    ma200 = (
        pl.DataFrame(adj_close)
        .select(pl.all().rolling_mean(window_size=200, min_samples=200))
        .to_numpy()
    )
    above = np.full(t_len, np.nan)
    for t in range(t_len):
        ok = tradeable[t] & np.isfinite(adj_close[t]) & np.isfinite(ma200[t])
        if ok.sum() >= 10:
            above[t] = float((adj_close[t, ok] > ma200[t, ok]).mean())
    return dispersion, above


def assemble_market_vector(
    panel_dates: pd.DatetimeIndex,
    warmup_dates: pd.DatetimeIndex,
    adj_close: FloatMat,
    returns: FloatMat,
    tradeable: npt.NDArray[np.bool_],
    market_dir: Path = MARKET_DIR,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """The full (T, 67) market matrix for `panel_dates`, guaranteed NaN-free.

    `warmup_dates` is the full calendar the stock matrices are built on
    (superset of `panel_dates`, extending earlier); index rolling windows are
    computed over it, then sliced, so no head-NaN survives.
    """
    block = build_index_block(warmup_dates, market_dir)
    dispersion, above = breadth_columns(adj_close, returns, tradeable)
    block["xs_dispersion"] = dispersion
    block["pct_above_200dma"] = above

    sliced = block.loc[panel_dates, market_vector_names()]
    bad = sliced.isna()
    if bad.any().any():
        col = str(bad.any().idxmax())
        first = sliced.index[bad[col]][0]
        raise ValueError(
            f"market vector contains NaN after warm-up: first in {col!r} at "
            f"{first.date()} ({int(bad.to_numpy().sum())} total). The gate cannot "
            "consume NaN — extend the warm-up or fix the source"
        )
    return sliced.to_numpy(dtype=np.float32), market_vector_names()
