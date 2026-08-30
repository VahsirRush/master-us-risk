"""Tests for the feature bank — spec section 4.3.

The single most important test sweeps EVERY feature for backward-lookingness:
tamper the future, recompute, and demand bitwise-identical history. Everything
else is arithmetic spot-checks with hand-computable answers, plus the registry
wiring that keeps missingness masks grouped correctly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from master_us.data.feature_groups import FeatureGroup, classify_features
from master_us.data.features import (
    add_cross_sectional_ranks,
    all_feature_names,
    base_feature_names,
    build_features,
    validate_feature_names,
)


def make_ohlcv(n_days=200, tickers=("AAA", "BBB", "CCC"), seed=0) -> pl.DataFrame:
    """Synthetic OHLCV with valid bar geometry: L <= min(O,C) <= max(O,C) <= H."""
    rng = np.random.default_rng(seed)
    dates = pl.datetime_range(
        pl.datetime(2020, 1, 1), pl.datetime(2021, 12, 31), "1d", eager=True
    ).filter(pl.datetime_range(pl.datetime(2020, 1, 1), pl.datetime(2021, 12, 31), "1d", eager=True).dt.weekday() < 6)[:n_days]
    frames = []
    for tk in tickers:
        close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        open_ = close * np.exp(rng.normal(0, 0.005, n_days))
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n_days)))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n_days)))
        vol = rng.integers(1e5, 5e6, n_days).astype(float)
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": tk,
                    "open": open_,
                    "high": hi,
                    "low": lo,
                    "close": close,
                    "adj_close": close,
                    "volume": vol,
                }
            )
        )
    return pl.concat(frames)


def make_spy(ohlcv: pl.DataFrame, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = sorted(ohlcv["date"].unique().to_list())
    px = 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
    return pl.DataFrame({"date": dates, "adj_close": px})


@pytest.fixture(scope="module")
def bank():
    ohlcv = make_ohlcv()
    spy = make_spy(ohlcv)
    feats = build_features(ohlcv, spy)
    trade = ohlcv.select(["date", "ticker"]).with_columns(pl.lit(value=True).alias("tradeable"))
    full = add_cross_sectional_ranks(feats, trade)
    return ohlcv, spy, full


# ------------------------------------------------------------------ #
# The one test that matters most                                      #
# ------------------------------------------------------------------ #


def test_every_feature_is_backward_looking(bank):
    """Tamper everything after a cut date; history must be bitwise identical.

    This catches any accidental forward shift, any centered window, any
    rolling op that peeks — for all features at once, ranks included.
    """
    ohlcv, spy, full = bank
    dates = sorted(ohlcv["date"].unique().to_list())
    cut = dates[140]

    tampered = ohlcv.with_columns(
        [
            pl.when(pl.col("date") > cut).then(pl.col(c) * 7.7).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low", "close", "adj_close", "volume")
        ]
    )
    spy_tampered = spy.with_columns(
        pl.when(pl.col("date") > cut)
        .then(pl.col("adj_close") * 3.3)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    trade = ohlcv.select(["date", "ticker"]).with_columns(pl.lit(value=True).alias("tradeable"))
    full_tampered = add_cross_sectional_ranks(build_features(tampered, spy_tampered), trade)

    a = full.filter(pl.col("date") <= cut).sort(["ticker", "date"])
    b = full_tampered.filter(pl.col("date") <= cut).sort(["ticker", "date"])
    for name in all_feature_names():
        av, bv = a[name].to_numpy(), b[name].to_numpy()
        same = (av == bv) | (np.isnan(av) & np.isnan(bv))
        assert same.all(), f"{name} changed at {np.flatnonzero(~same)[:3]} when the FUTURE changed"


# ------------------------------------------------------------------ #
# Arithmetic spot-checks                                              #
# ------------------------------------------------------------------ #


def test_returns_are_exact(bank):
    ohlcv, _, full = bank
    sub = full.filter(pl.col("ticker") == "AAA").sort("date")
    px = ohlcv.filter(pl.col("ticker") == "AAA").sort("date")["adj_close"].to_numpy()
    got = sub["ret_5d"].to_numpy()
    expected = np.full_like(px, np.nan)
    expected[5:] = px[5:] / px[:-5] - 1.0
    np.testing.assert_allclose(got[5:], expected[5:], rtol=1e-10)
    assert np.isnan(got[:5]).all()


def test_close_position_is_bounded(bank):
    _, _, full = bank
    v = full["ps_close_pos_1d"].drop_nulls().to_numpy()
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_rsi_saturates_on_monotone_series():
    n = 60
    dates = pl.datetime_range(pl.datetime(2020, 1, 1), pl.datetime(2020, 3, 31), "1d", eager=True)[:n]
    up = np.linspace(100, 200, n)
    frame = pl.DataFrame(
        {
            "date": dates,
            "ticker": "UP",
            "open": up,
            "high": up * 1.01,
            "low": up * 0.99,
            "close": up,
            "adj_close": up,
            "volume": np.full(n, 1e6),
        }
    )
    spy = pl.DataFrame({"date": dates, "adj_close": np.linspace(300, 310, n)})
    rsi = build_features(frame, spy)["tech_rsi_14"].to_numpy()
    assert np.nanmax(rsi) > 99.9  # all gains, no losses


def test_parkinson_matches_constant_range():
    """Constant H/L ratio r => Parkinson vol = ln(r)/sqrt(4 ln 2), exactly."""
    n = 40
    dates = pl.datetime_range(pl.datetime(2020, 1, 1), pl.datetime(2020, 2, 28), "1d", eager=True)[:n]
    close = np.full(n, 100.0)
    ratio = 1.02
    frame = pl.DataFrame(
        {
            "date": dates,
            "ticker": "FLAT",
            "open": close,
            "high": close * np.sqrt(ratio),
            "low": close / np.sqrt(ratio),
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 1e6),
        }
    )
    spy = pl.DataFrame({"date": dates, "adj_close": np.full(n, 300.0)})
    got = build_features(frame, spy)["vol_parkinson_20d"].to_numpy()
    expected = np.log(ratio) / np.sqrt(4 * np.log(2))
    np.testing.assert_allclose(got[~np.isnan(got)], expected, rtol=1e-9)


def test_beta_of_a_levered_spy_clone(bank):
    """A synthetic name returning exactly 2x SPY must have beta ~= 2, corr ~= 1."""
    _, spy, _ = bank
    spy_px = spy["adj_close"].to_numpy()
    spy_ret = np.concatenate([[0.0], spy_px[1:] / spy_px[:-1] - 1.0])
    clone_px = 100 * np.cumprod(1 + 2 * spy_ret)
    frame = pl.DataFrame(
        {
            "date": spy["date"],
            "ticker": "CLONE",
            "open": clone_px,
            "high": clone_px * 1.001,
            "low": clone_px * 0.999,
            "close": clone_px,
            "adj_close": clone_px,
            "volume": np.full(len(clone_px), 1e6),
        }
    )
    feats = build_features(frame, spy)
    beta = feats["tech_beta_60d"].to_numpy()
    corr = feats["tech_corr_60d"].to_numpy()
    np.testing.assert_allclose(beta[np.isfinite(beta)], 2.0, atol=1e-6)
    np.testing.assert_allclose(corr[np.isfinite(corr)], 1.0, atol=1e-6)


def test_volume_ratio_excludes_today_from_reference(bank):
    """A volume spike today must not inflate its own reference mean."""
    ohlcv, spy, _ = bank
    spiked = ohlcv.with_columns(
        pl.when(
            (pl.col("ticker") == "AAA") & (pl.col("date") == pl.col("date").max())
        )
        .then(pl.col("volume") * 100)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    base = build_features(ohlcv, spy).filter(pl.col("ticker") == "AAA").sort("date")
    spike = build_features(spiked, spy).filter(pl.col("ticker") == "AAA").sort("date")
    r0 = base["volume_ratio_20d"].to_numpy()
    r1 = spike["volume_ratio_20d"].to_numpy()
    # today's ratio scales by ~100x because the reference (t-20..t-1) is untouched
    assert r1[-1] / r0[-1] == pytest.approx(100.0, rel=1e-9)


# ------------------------------------------------------------------ #
# Ranks                                                               #
# ------------------------------------------------------------------ #


def test_ranks_are_per_date_percentiles(bank):
    _, _, full = bank
    one_day = full.filter(pl.col("date") == full["date"].max())
    raw = one_day["ret_20d"].to_numpy()
    rk = one_day["xs_ret_20d"].to_numpy()
    ok = np.isfinite(raw) & np.isfinite(rk)
    assert rk[ok].min() > 0.0 and rk[ok].max() <= 1.0
    order_raw = np.argsort(raw[ok])
    assert (np.diff(rk[ok][order_raw]) >= 0).all(), "rank must be monotone in the raw value"


def test_ranks_exclude_untradeable_names(bank):
    ohlcv, _, _ = bank
    feats = build_features(ohlcv, make_spy(ohlcv))
    trade = ohlcv.select(["date", "ticker"]).with_columns(
        (pl.col("ticker") != "CCC").alias("tradeable")
    )
    ranked = add_cross_sectional_ranks(feats, trade)
    ccc = ranked.filter(pl.col("ticker") == "CCC")
    assert ccc["xs_ret_5d"].null_count() == len(ccc), "untradeable names must have no rank"
    # And with only two live names, ranks are {0.5, 1.0}
    last = ranked.filter(
        (pl.col("date") == ranked["date"].max()) & (pl.col("ticker") != "CCC")
    )["xs_ret_5d"].to_numpy()
    assert sorted(last) == pytest.approx([0.5, 1.0])


# ------------------------------------------------------------------ #
# Registry wiring                                                     #
# ------------------------------------------------------------------ #


def test_bank_size_and_registry_classification():
    names = all_feature_names()
    assert len(base_feature_names()) == 65
    assert len(names) == 130
    validate_feature_names()  # raises on any OTHER

    grouped = classify_features(names, strict=True)
    counts = {g: len(m) for g, m in grouped.items()}
    assert counts[FeatureGroup.RETURNS] == 8
    assert counts[FeatureGroup.VOLATILITY] == 21
    assert counts[FeatureGroup.VOLUME] == 10
    assert counts[FeatureGroup.PRICE_STRUCTURE] == 19
    assert counts[FeatureGroup.TECHNICAL] == 7
    assert counts[FeatureGroup.CROSS_SECTIONAL_RANK] == 65
    assert FeatureGroup.OTHER not in counts


def test_no_infinities_survive(bank):
    _, _, full = bank
    for name in base_feature_names():
        v = full[name].to_numpy()
        assert not np.isinf(v[np.isfinite(v) | np.isinf(v)]).any(), f"{name} emitted inf"
