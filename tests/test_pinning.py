"""Tests for price-data pinning — contract section 3.2.

The load-bearing test simulates the exact Session-4 LEG failure: a ticker's
adjusted closes silently re-based between pulls. The pin must catch it and
name the ticker.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.data.pinning import (
    PriceDataChangedError,
    pin,
    verify,
)


def _write(tmp_path, ticker: str, seed: int = 0, scale: float = 1.0, rows: int = 50):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    pl.DataFrame(
        {
            "date": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(rows)],
            "ticker": ticker,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close * 0.95 * scale,
            "volume": np.full(rows, 1e6),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ns"))).write_parquet(
        tmp_path / f"{ticker}.parquet"
    )


def test_pin_then_verify_clean(tmp_path):
    for tk in ("AAA", "BBB", "CCC"):
        _write(tmp_path, tk, seed=hash(tk) % 100)
    manifest = tmp_path / "manifest.json"
    hashes = pin(tmp_path, manifest)
    assert set(hashes) == {"AAA", "BBB", "CCC"}

    result = verify(tmp_path, manifest)
    assert result.ok
    assert result.n_checked == 3
    assert result.changed == () and result.missing == ()


def test_simulated_readjustment_is_caught_and_named(tmp_path):
    """The LEG failure mode, on purpose: adj_close re-based by 0.5%.

    This is the exact magnitude Yahoo's re-adjustment moved LEG by in
    Session 4 (-5.2e-3 at the seam). The pin must refuse, and must say WHICH
    ticker drifted.
    """
    for tk in ("AAA", "BBB", "CCC"):
        _write(tmp_path, tk, seed=hash(tk) % 100)
    manifest = tmp_path / "manifest.json"
    pin(tmp_path, manifest)

    _write(tmp_path, "BBB", seed=hash("BBB") % 100, scale=1.005)  # re-based adjustments

    with pytest.raises(PriceDataChangedError, match="BBB") as exc:
        verify(tmp_path, manifest)
    assert "re-based" in str(exc.value)
    assert "AAA" not in str(exc.value).split("gone from disk")[0].replace("['BBB']", "")

    result = verify(tmp_path, manifest, raise_on_drift=False)
    assert result.changed == ("BBB",)
    assert not result.ok


def test_one_ulp_change_is_caught(tmp_path):
    """The hash covers exact values — a single float nudge in one cell trips it."""
    _write(tmp_path, "AAA")
    manifest = tmp_path / "manifest.json"
    pin(tmp_path, manifest)

    frame = pl.read_parquet(tmp_path / "AAA.parquet")
    adj = frame["adj_close"].to_numpy().copy()
    adj[17] = np.nextafter(adj[17], np.inf)
    frame.with_columns(pl.Series("adj_close", adj)).write_parquet(tmp_path / "AAA.parquet")

    with pytest.raises(PriceDataChangedError, match="AAA"):
        verify(tmp_path, manifest)


def test_missing_ticker_is_an_error_new_ticker_is_not(tmp_path):
    for tk in ("AAA", "BBB"):
        _write(tmp_path, tk, seed=hash(tk) % 100)
    manifest = tmp_path / "manifest.json"
    pin(tmp_path, manifest)

    (tmp_path / "AAA.parquet").unlink()  # pinned data deleted -> error
    _write(tmp_path, "DDD", seed=7)  # append-only growth -> fine

    with pytest.raises(PriceDataChangedError, match="gone from disk"):
        verify(tmp_path, manifest)

    result = verify(tmp_path, manifest, raise_on_drift=False)
    assert result.missing == ("AAA",)
    assert result.new == ("DDD",)


def test_rewrite_with_identical_data_does_not_trip(tmp_path):
    """The hash is over content, not file bytes: a byte-different rewrite of
    the same values (different compression, row order) must verify clean."""
    _write(tmp_path, "AAA")
    manifest = tmp_path / "manifest.json"
    pin(tmp_path, manifest)

    frame = pl.read_parquet(tmp_path / "AAA.parquet").sample(fraction=1.0, shuffle=True, seed=1)
    frame.write_parquet(tmp_path / "AAA.parquet", compression="snappy")

    assert verify(tmp_path, manifest).ok


def test_missing_manifest_refuses_to_bless_anything(tmp_path):
    _write(tmp_path, "AAA")
    with pytest.raises(FileNotFoundError, match="unpinned"):
        verify(tmp_path, tmp_path / "nope.json")


def test_real_cache_is_pinned_and_clean():
    """The live guarantee: the actual price cache verifies against its manifest."""
    from master_us.data.pinning import PRICE_MANIFEST_PATH

    if not PRICE_MANIFEST_PATH.exists():
        pytest.skip("no manifest yet — run scripts/08_pin_prices.py --pin")
    result = verify(PRICE_CACHE_DIR)
    assert result.ok
    assert result.n_checked > 600
