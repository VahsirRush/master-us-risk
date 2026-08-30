"""Assemble the real Panel: features + market vector + labels + mask.

The last mile of Phase 0. Everything upstream is cached long-format parquet;
this module turns it into the aligned (T, N, F) arrays `Panel` validates at
construction.

Alignment decisions, stated once:

* **The panel calendar starts where the universe does** (2010-02-01 — the
  canonical membership reconstruction's first daily snapshot), not where
  prices do. Features are COMPUTED over the extended price history (2006+,
  fetched for the Phase-1 gate) and then sliced, so a 60-day rolling window
  is already warm on the panel's first day and the >=252d history filter
  doesn't empty the early cross-sections.
* **Labels are the spec 4.5 processed labels**: forward return -> drop NaN ->
  drop the most extreme 5% (both tails, 2.5% each) -> cross-sectional z-score
  within each date, all via `normalize.process_labels`. Every step is
  per-date, so nothing crosses time. The RAW forward-return matrix rides
  along in `attrs["raw_forward_returns"]` — the backtest engine and IC-in-
  return-units both need it, and recomputing it downstream would invite a
  second, subtly different definition.
* **mask = the Session-3 `tradeable` flag**: PIT membership AND close >= $5
  AND 21d ADV >= $2M AND >= 252 observed days — evaluated on information
  through t only.
* **metadata is None for now.** The Panel contract wants mcap/sector/adv/
  price for the risk and cost models; mcap needs the SEC shares join and
  belongs with the Barra work. Recorded as an open item, not smuggled in
  half-built.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from master_us.data.features import (
    add_cross_sectional_ranks,
    all_feature_names,
    build_features,
    validate_feature_names,
)
from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.data.market_vector import assemble_market_vector
from master_us.data.normalize import process_labels
from master_us.data.panel import Panel
from master_us.data.sources import DATA_ROOT
from master_us.data.universe import MEMBERSHIP_CACHE

PANEL_PATH = DATA_ROOT / "processed" / "panel.pkl"
PANEL_H5_PATH = DATA_ROOT / "processed" / "panel_h5.pkl"

WARMUP_START = "2006-01-01"  # extended price history; features warm before 2010

FloatMat = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AssemblyStats:
    """What the build did, for NOTES.md and the phase result."""

    n_dates: int
    n_tickers: int
    n_features: int
    n_market: int
    build_seconds: float
    label_stats: dict[int, dict[str, int]]


def _load_long_inputs(end: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(ohlcv+tradeable long frame, spy frame, membership) over the warm-up window."""
    membership = pl.read_parquet(MEMBERSHIP_CACHE).select(["date", "ticker", "in_universe"])
    wanted = set(membership.filter(pl.col("in_universe"))["ticker"].unique().to_list())

    frames = [
        pl.read_parquet(p) for p in sorted(PRICE_CACHE_DIR.glob("*.parquet")) if p.stem in wanted
    ]
    if not frames:
        raise FileNotFoundError(f"no cached prices under {PRICE_CACHE_DIR} match membership")

    lo = pd.Timestamp(WARMUP_START).to_pydatetime()
    hi = pd.Timestamp(end).to_pydatetime()
    ohlcv = (
        pl.concat(frames)
        .filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        .join(membership, on=["date", "ticker"], how="left")
        .with_columns(pl.col("in_universe").fill_null(value=False))
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(21, min_samples=21)
            .over("ticker")
            .alias("_adv21"),
            pl.col("date").cum_count().over("ticker").alias("_history"),
        )
        .with_columns(
            (
                pl.col("in_universe")
                & (pl.col("close") >= 5.0)
                & (pl.col("_adv21") >= 2_000_000.0)
                & (pl.col("_history") >= 252)
            ).alias("tradeable")
        )
    )
    spy = pl.read_parquet(PRICE_CACHE_DIR / "SPY.parquet").select(["date", "adj_close"])
    return ohlcv, spy, membership


def _to_wide(
    frame: pl.DataFrame,
    value_col: str,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> FloatMat:
    """Long -> dense (T, N) float64 via positional scatter. NaN where absent.

    Rows whose ticker is not in `tickers` are ignored — the frame may still
    carry names that were dropped from the panel axis (never-tradeable).
    """
    date_pos = {d: i for i, d in enumerate(dates)}
    ticker_pos = {t: j for j, t in enumerate(tickers)}
    out = np.full((len(dates), len(tickers)), np.nan)
    sub = frame.select(["date", "ticker", value_col]).drop_nulls(subset=[value_col]).filter(
        pl.col("ticker").is_in(tickers)
    )
    rows = sub["date"].to_pandas().map(date_pos).to_numpy(dtype=np.int64)
    cols = sub["ticker"].to_pandas().map(ticker_pos).to_numpy(dtype=np.int64)
    out[rows, cols] = sub[value_col].to_numpy()
    return out


def build_panel(end: str = "2025-12-31", horizon: int = 1) -> tuple[Panel, AssemblyStats]:
    """The real Panel, validated at construction. Heavy: ~2-4 minutes."""
    t0 = time.monotonic()
    validate_feature_names()

    ohlcv, spy, _membership = _load_long_inputs(end)

    features_long = build_features(ohlcv, spy)
    features_long = add_cross_sectional_ranks(
        features_long, ohlcv.select(["date", "ticker", "tradeable"])
    )

    # Calendars: the full warm-up calendar, and the panel calendar (first
    # date on which the universe has any tradeable name).
    warmup_dates = pd.DatetimeIndex(
        sorted(ohlcv["date"].unique().to_list())
    )
    tickers = sorted(ohlcv["ticker"].unique().to_list())

    tradeable_w = (
        _to_wide(
            ohlcv.with_columns(pl.col("tradeable").cast(pl.Float64).alias("_t")),
            "_t",
            warmup_dates,
            tickers,
        )
        > 0.5
    )
    first_live = int(np.argmax(tradeable_w.any(axis=1)))
    if not tradeable_w.any():
        raise ValueError("no tradeable names anywhere — universe or filters are broken")
    panel_dates = warmup_dates[first_live:]

    # Drop tickers that are never tradeable anywhere in the sample. Measured on
    # the first real build: 42 of 626, almost all REUSED SYMBOLS — Yahoo's
    # series belongs to a newer company (DV: DeVry -> DoubleVerify, BEAM,
    # MMI, BTU post-bankruptcy...) whose data never overlaps the old company's
    # membership window, so the membership-and-filters conjunction correctly
    # never fires. Keeping them would carry all-NaN columns and read as
    # scrubbed history to any survivorship check. Their absence is already
    # counted in reports/survivorship.md.
    ever_tradeable = np.asarray(tradeable_w[first_live:].any(axis=0))
    keep_flags = [bool(k) for k in ever_tradeable]
    dropped_tickers = [t for t, keep in zip(tickers, keep_flags, strict=True) if not keep]
    if dropped_tickers:
        tickers = [t for t, keep in zip(tickers, keep_flags, strict=True) if keep]
        tradeable_w = tradeable_w[:, ever_tradeable]

    # Wide price matrices over the warm-up calendar (market breadth needs them).
    adj_w = _to_wide(ohlcv, "adj_close", warmup_dates, tickers)
    with np.errstate(invalid="ignore"):
        returns_w = adj_w[1:] / adj_w[:-1] - 1.0
    returns_w = np.vstack([np.full((1, len(tickers)), np.nan), returns_w])

    market, market_names = assemble_market_vector(
        panel_dates, warmup_dates, adj_w, returns_w, tradeable_w
    )

    # Features: scatter each column, slice to the panel window.
    names = all_feature_names()
    features = np.full(
        (len(panel_dates), len(tickers), len(names)), np.nan, dtype=np.float32
    )
    for k, name in enumerate(names):
        features[:, :, k] = _to_wide(features_long, name, warmup_dates, tickers)[
            first_live:
        ].astype(np.float32)

    mask = tradeable_w[first_live:]

    # Labels: raw forward return over `horizon`, then the 4.5 processing chain.
    with np.errstate(invalid="ignore"):
        fwd = adj_w[horizon:] / adj_w[:-horizon] - 1.0
    raw_forward = np.full_like(adj_w, np.nan)
    raw_forward[:-horizon] = fwd
    raw_forward = raw_forward[first_live:].astype(np.float32)

    processed = process_labels(raw_forward, mask, drop_pct=5.0)

    panel = Panel(
        dates=panel_dates,
        tickers=np.array(tickers),
        features=features,
        feature_names=names,
        market=market,
        market_names=market_names,
        labels=processed.values,
        mask=mask,
        label_horizon=horizon,
        metadata=None,
        attrs={
            "synthetic": False,
            "dropped_never_tradeable": dropped_tickers,
            "built_at": pd.Timestamp.utcnow().isoformat(),
            "warmup_start": WARMUP_START,
            "raw_forward_returns": raw_forward,
            "label_processing": {
                "n_input": processed.n_input,
                "n_missing": processed.n_missing,
                "n_trimmed": processed.n_trimmed,
                "n_output": processed.n_output,
                "n_dates_dropped": processed.n_dates_dropped,
            },
        },
    )

    stats = AssemblyStats(
        n_dates=panel.n_dates,
        n_tickers=panel.n_tickers,
        n_features=panel.n_features,
        n_market=panel.n_market,
        build_seconds=time.monotonic() - t0,
        label_stats={
            horizon: {
                "n_input": processed.n_input,
                "n_trimmed": processed.n_trimmed,
                "n_output": processed.n_output,
            }
        },
    )
    return panel, stats


def group_nan_rates_by_year(panel: Panel) -> pd.DataFrame:
    """Feature-NaN rate per registry group per year, over in-mask cells only."""
    from master_us.data.feature_groups import classify_features

    grouped = classify_features(panel.feature_names)
    years = panel.dates.year
    rows = []
    for year in sorted(set(years)):
        sel = years == year
        m = panel.mask[sel]
        row: dict[str, object] = {"year": int(year)}
        for group, members in grouped.items():
            idx = [panel.feature_names.index(name) for name in members]
            block = panel.features[sel][:, :, idx]
            row[str(group)] = float(np.isnan(block[m]).mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")
