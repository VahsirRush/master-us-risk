"""Panel contract tests.

Every assertion in Panel.__post_init__ exists because the corresponding bug is
one this project can plausibly produce. Each gets a test that proves the guard
fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_us.data.panel import Panel

from .fixtures import (
    corrupt_market_nan,
    make_synthetic_panel,
)

# ------------------------------------------------------------------ #
# Acceptance                                                          #
# ------------------------------------------------------------------ #


def test_clean_panel_constructs():
    p = make_synthetic_panel(n_dates=120, n_tickers=30, n_features=10)
    assert p.shape == (120, 30, 10)
    assert p.n_market == 8
    assert p.mask.any()


def test_signal_is_recoverable():
    """The injected signal must be present at roughly the requested strength.

    If this drifts, every downstream test built on synthetic data is measuring
    something other than what it thinks.
    """
    p = make_synthetic_panel(n_dates=1000, n_tickers=100, signal_strength=0.15, seed=1)
    x = p.features[:, :, p.attrs["signal_feature"]]
    ok = p.mask & ~np.isnan(p.labels) & ~np.isnan(x)
    corr = np.corrcoef(x[ok], p.labels[ok])[0, 1]
    assert 0.10 < corr < 0.22, f"injected signal measured at {corr:.3f}"


def test_repr_and_coverage():
    p = make_synthetic_panel(n_dates=300, n_tickers=20)
    assert "Panel(" in repr(p)
    cov = p.coverage()
    assert set(cov.columns) >= {"dates", "mean_names", "feature_nan_rate"}
    assert (cov["mean_names"] > 0).all()


# ------------------------------------------------------------------ #
# Rejection — shapes and ordering                                     #
# ------------------------------------------------------------------ #


def test_rejects_wrong_feature_shape():
    p = make_synthetic_panel(n_dates=50, n_tickers=10, n_features=5)
    with pytest.raises(ValueError, match="features has shape"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features[:, :, :3],
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


def test_rejects_label_mask_shape_mismatch():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    with pytest.raises(ValueError, match="mask has shape"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask[:, :5],
        )


def test_rejects_unsorted_dates():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    shuffled = pd.DatetimeIndex(p.dates.to_numpy()[::-1])
    with pytest.raises(ValueError, match="sorted ascending"):
        Panel(
            dates=shuffled,
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


def test_rejects_duplicate_dates():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    dupes = p.dates.to_numpy().copy()
    dupes[10] = dupes[9]
    with pytest.raises(ValueError, match="duplicate"):
        Panel(
            dates=pd.DatetimeIndex(dupes),
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


def test_rejects_duplicate_tickers():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    tickers = p.tickers.copy()
    tickers[3] = tickers[2]
    with pytest.raises(ValueError, match="tickers contains duplicates"):
        Panel(
            dates=p.dates,
            tickers=tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


def test_rejects_duplicate_feature_names():
    p = make_synthetic_panel(n_dates=50, n_tickers=10, n_features=5)
    names = list(p.feature_names)
    names[1] = names[0]
    with pytest.raises(ValueError, match="feature_names contains duplicates"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features,
            feature_names=names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


# ------------------------------------------------------------------ #
# Rejection — the market vector                                       #
# ------------------------------------------------------------------ #


def test_rejects_nan_in_market():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    with pytest.raises(ValueError, match=r"market contains .* NaN"):
        corrupt_market_nan(p)


def test_rejects_inf_in_market():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    market = p.market.copy()
    market[5, 1] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


# ------------------------------------------------------------------ #
# Rejection — dtypes and mask                                         #
# ------------------------------------------------------------------ #


def test_rejects_wrong_dtype():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    with pytest.raises(TypeError, match="features must be float32"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features.astype(np.float64),
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
        )


def test_rejects_empty_mask():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    with pytest.raises(ValueError, match="entirely False"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=np.zeros_like(p.mask),
        )


def test_rejects_date_with_no_tradeable_names():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    mask = p.mask.copy()
    mask[7, :] = False
    with pytest.raises(ValueError, match="no tradeable names"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=mask,
        )


def test_rejects_metadata_missing_columns():
    p = make_synthetic_panel(n_dates=50, n_tickers=10)
    meta = p.metadata.drop(columns=["adv"])
    with pytest.raises(ValueError, match="missing required columns"):
        Panel(
            dates=p.dates,
            tickers=p.tickers,
            features=p.features,
            feature_names=p.feature_names,
            market=p.market,
            market_names=p.market_names,
            labels=p.labels,
            mask=p.mask,
            metadata=meta,
        )


# ------------------------------------------------------------------ #
# Slicing and splits                                                  #
# ------------------------------------------------------------------ #


def test_date_slice_preserves_ticker_ordering():
    p = make_synthetic_panel(n_dates=400, n_tickers=25)
    mid = p.dates[200]
    sub = p.date_slice(p.dates[100], mid)
    assert np.array_equal(sub.tickers, p.tickers)
    assert sub.dates[0] >= p.dates[100]
    assert sub.dates[-1] <= mid


def test_split_applies_embargo():
    p = make_synthetic_panel(n_dates=1500, n_tickers=20)
    splits = p.split(
        train=("2015-01-01", "2017-12-31"),
        valid=("2018-01-01", "2018-12-31"),
        test=("2019-01-01", "2021-01-01"),
        embargo_days=21,
    )
    assert splits["train"].dates[-1] < pd.Timestamp("2017-12-31") - pd.Timedelta(days=20)
    assert splits["valid"].dates[0] >= pd.Timestamp("2018-01-01")
    assert splits["train"].dates[-1] < splits["valid"].dates[0]
    assert splits["valid"].dates[-1] < splits["test"].dates[0]


def test_split_rejects_overlapping_ranges():
    p = make_synthetic_panel(n_dates=800, n_tickers=20)
    with pytest.raises(ValueError, match="train must end before valid begins"):
        p.split(
            train=("2015-01-01", "2017-06-30"),
            valid=("2017-01-01", "2017-12-31"),
            test=("2018-01-01", "2018-06-30"),
        )


def test_roundtrip_save_load(tmp_path):
    p = make_synthetic_panel(n_dates=60, n_tickers=12)
    path = p.save(tmp_path / "panel.pkl")
    q = Panel.load(path)
    assert q.shape == p.shape
    assert np.array_equal(q.tickers, p.tickers)
    np.testing.assert_array_equal(q.mask, p.mask)
