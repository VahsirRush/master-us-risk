"""Tests for the market state vector — spec section 4.4.

The contract under test is the NaN guarantee: warm-up covers rolling heads,
calendar mismatches forward-fill at most 5 days, anything worse raises with
the column and date named.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from master_us.data.market_vector import (
    MARKET_DIR,
    MAX_FFILL,
    _aligned_series,
    breadth_columns,
    build_index_block,
    market_vector_names,
)


def test_vector_has_67_named_dimensions():
    names = market_vector_names()
    assert len(names) == 67
    assert len(set(names)) == 67
    for prefix in ("sp500", "sp400", "rut"):
        assert f"{prefix}_ret" in names
        assert f"{prefix}_dv_std_60d" in names
    assert {"vix_level", "vix_chg_20d", "xs_dispersion", "pct_above_200dma"} < set(names)


def test_real_index_block_is_nan_free_after_warmup():
    if not (MARKET_DIR / "^GSPC.parquet").exists():
        pytest.skip("no market cache — run scripts/06_fetch_market.py")
    raw = pl.read_parquet(MARKET_DIR / "^GSPC.parquet").sort("date")
    dates = pd.DatetimeIndex(raw["date"].to_list())
    block = build_index_block(dates)
    # After 60 rolling days + 20d VIX change, everything must be filled.
    warm = block.iloc[80:]
    assert not warm.isna().any().any(), (
        f"NaN after warm-up in: {list(warm.columns[warm.isna().any()])}"
    )
    # And the head is honestly NaN, not silently backfilled.
    assert block["sp500_ret_std_60d"].iloc[:59].isna().all()


def _write_symbol(tmp_path, symbol: str, dates, close):
    pl.DataFrame(
        {
            "date": list(dates),
            "ticker": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(close), 1e6),
        }
    ).write_parquet(tmp_path / f"{symbol}.parquet")


def test_short_gap_forward_fills_and_long_gap_raises(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=60)
    close = np.linspace(100, 120, 60)

    # Symbol missing 3 of the panel's dates: fills from the last value.
    keep = np.ones(60, dtype=bool)
    keep[30:33] = False
    _write_symbol(tmp_path, "SHORTGAP", dates[keep], close[keep])
    series = _aligned_series("SHORTGAP", "close", dates, tmp_path)
    assert not series.isna().any()
    assert series.iloc[31] == series.iloc[29]  # carried forward, not interpolated

    # Symbol missing MAX_FFILL+2 consecutive dates: must raise, not fill.
    keep2 = np.ones(60, dtype=bool)
    keep2[30 : 30 + MAX_FFILL + 2] = False
    _write_symbol(tmp_path, "LONGGAP", dates[keep2], close[keep2])
    with pytest.raises(ValueError, match="gap longer than"):
        _aligned_series("LONGGAP", "close", dates, tmp_path)


def test_missing_symbol_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="06_fetch_market"):
        _aligned_series("NOPE", "close", pd.bdate_range("2020-01-01", periods=5), tmp_path)


def test_breadth_columns_hand_computed():
    """15 of 20 names rising above their MA, 5 declining below: 0.75 exactly.

    Both breadth columns refuse to estimate from fewer than 10 names — a
    'universe breadth' read off a handful of names is noise — so the fixture
    uses 20. On the real panel (~425 names/day) the floor never binds.
    """
    t_len, n = 250, 20
    adj = np.tile(np.linspace(100, 150, t_len)[:, None], (1, n))
    adj[:, 15:] = np.linspace(100, 50, t_len)[:, None]  # five names in steady decline
    with np.errstate(invalid="ignore"):
        rets = np.vstack([np.full((1, n), np.nan), adj[1:] / adj[:-1] - 1.0])
    tradeable = np.ones((t_len, n), dtype=bool)

    dispersion, above = breadth_columns(adj, rets, tradeable)

    # Head: 200d MA unfilled -> NaN.
    assert np.isnan(above[:199]).all()
    # Tail: rising names above their rising MA, decliners below their falling MA.
    assert above[-1] == pytest.approx(0.75)
    assert np.isfinite(dispersion[1:]).all()


def test_breadth_refuses_thin_cross_sections():
    t_len, n = 250, 4
    adj = np.tile(np.linspace(100, 150, t_len)[:, None], (1, n))
    rets = np.vstack([np.full((1, n), np.nan), adj[1:] / adj[:-1] - 1.0])
    dispersion, above = breadth_columns(adj, rets, np.ones((t_len, n), dtype=bool))
    assert np.isnan(dispersion).all()
    assert np.isnan(above).all()


def test_breadth_dispersion_with_enough_names():
    rng = np.random.default_rng(0)
    t_len, n = 10, 50
    rets = rng.normal(0, 0.02, (t_len, n))
    adj = np.cumprod(1 + rets, axis=0) * 100
    tradeable = np.ones((t_len, n), dtype=bool)
    dispersion, _ = breadth_columns(adj, rets, tradeable)
    np.testing.assert_allclose(dispersion[3], rets[3].std(ddof=1))
