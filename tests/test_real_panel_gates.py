"""The Phase-0 gates, run against the REAL assembled panel — spec section 4.7.

Sessions 1-2 built these gates and proved each fails on a corrupted synthetic
panel. This file is the other half of the bargain: the same properties checked
on `data/processed/panel.pkl`, the first time the gates see real data end to
end. Skips cleanly on a checkout without the panel.

One adaptation is substantive, not cosmetic. On synthetic data, delaying the
features collapses IC to zero because the injected signal has no memory. Real
features are autocorrelated — a 60-day volatility barely changes overnight —
so their delayed IC does NOT collapse, and demanding that it does would be
testing a property real data cannot have. The leakage property that survives
autocorrelation is one-sided: a feature may lose predictive power when
delivered late, it may keep most of it, but it must never IMPROVE. A feature
that predicts better with a one-day delay was reading tomorrow's information.
The same reasoning was already in Session 2's `shift_gate_violations`.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from master_us.data.assemble import PANEL_PATH
from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.data.normalize import AlreadyFittedError, RobustZScoreNorm
from master_us.data.panel import Panel

from .test_no_lookahead import IC_IMPLAUSIBLE, ic_by_date

# This module is the only one that loads the real ~1.2 GB panel. The marker
# lets `conftest.py` skip exactly these tests while a training job is live,
# instead of refusing the whole suite — see the guard's docstring.
pytestmark = pytest.mark.heavy

TRAIN_END = "2016-12-31"  # config/data.yaml splits.train[1]


@pytest.fixture(scope="module")
def panel() -> Panel:
    if not PANEL_PATH.exists():
        pytest.skip("no real panel — run scripts/07_assemble_panel.py")
    return Panel.load(PANEL_PATH)


@pytest.fixture(scope="module")
def feature_ics(panel: Panel):
    """|mean per-date IC| of every feature vs the labels, at lag 0 and delayed 1d.

    Computed once for the module (~40s for 130 features x 4,004 dates).
    """
    lag0 = np.empty(panel.n_features)
    lag1 = np.empty(panel.n_features)
    for k in range(panel.n_features):
        f = panel.features[:, :, k].astype(np.float64)
        series0 = ic_by_date(f, panel.labels, panel.mask, lag=0)
        series1 = ic_by_date(f, panel.labels, panel.mask, lag=1)
        lag0[k] = np.nanmean(series0)
        lag1[k] = np.nanmean(series1)
    return lag0, lag1


# ------------------------------------------------------------------ #
# Gate 1 — shifted features lose signal                               #
# ------------------------------------------------------------------ #


def test_no_feature_improves_when_delayed(feature_ics, panel):
    """|IC| delayed must not exceed |IC| now beyond estimation noise.

    The one-sided form of 'shifted features lose signal' that autocorrelated
    real features can honestly satisfy. Noise allowance: the per-date IC
    series has T~4,000 observations; the std error of a mean IC is ~0.004,
    so two estimates of the same underlying IC can differ by ~0.01. A delayed
    gain beyond that is future information, not noise.
    """
    lag0, lag1 = feature_ics
    tolerance = 0.01
    violations = [
        (panel.feature_names[k], float(lag0[k]), float(lag1[k]))
        for k in range(panel.n_features)
        if abs(lag1[k]) > abs(lag0[k]) + tolerance
    ]
    assert not violations, (
        f"{len(violations)} feature(s) predict BETTER when delivered a day late "
        f"(name, IC now, IC delayed): {violations[:5]}"
    )


def test_no_feature_has_an_implausible_ic(feature_ics, panel):
    """No honest daily feature correlates with tomorrow at anywhere near 0.5.

    On this panel the best features should sit in low single digits of IC;
    anything approaching the synthetic-gate bound of 0.5 is the label leaking
    into a feature.
    """
    lag0, _ = feature_ics
    worst = int(np.nanargmax(np.abs(lag0)))
    assert abs(lag0[worst]) < IC_IMPLAUSIBLE, (
        f"{panel.feature_names[worst]} has |IC| {abs(lag0[worst]):.3f}"
    )
    # And a tighter, real-data bound: max |IC| should be modest.
    assert abs(lag0[worst]) < 0.10, (
        f"{panel.feature_names[worst]} has |IC| {abs(lag0[worst]):.3f} — implausibly "
        "strong for a daily price/volume feature; check label alignment"
    )


# ------------------------------------------------------------------ #
# Gate 2 — normalization uses training statistics                     #
# ------------------------------------------------------------------ #


def test_normalization_uses_training_stats_on_real_features(panel):
    """Fit on train-span-masked full array == fit on the train slice, exactly.

    The first run of this property on real features. zero_mad stays 'raise':
    a constant real feature would be a genuine finding, and none fired.
    """
    import pandas as pd

    in_train = np.asarray(panel.dates <= pd.Timestamp(TRAIN_END))
    split = int(in_train.sum())
    obs_mask = in_train[:, None] & panel.mask

    full = RobustZScoreNorm(feature_names=panel.feature_names).fit(panel.features, obs_mask)
    sliced = RobustZScoreNorm(feature_names=panel.feature_names).fit(
        panel.features[:split], panel.mask[:split]
    )

    np.testing.assert_array_equal(full.stats.median, sliced.stats.median)
    np.testing.assert_array_equal(full.stats.scale, sliced.stats.scale)

    val_test = panel.features[split:]
    np.testing.assert_array_equal(full.transform(val_test), sliced.transform(val_test))

    with pytest.raises(AlreadyFittedError):
        full.fit(panel.features, panel.mask)


# ------------------------------------------------------------------ #
# Gate 3 — survivorship                                               #
# ------------------------------------------------------------------ #


def test_universe_no_survivorship_on_real_panel(panel):
    """Exits are present through their exit date and carry a terminal label.

    On this data path the panel's exits are index REMOVALS of names that kept
    trading (true delistings never reach the panel at all — Yahoo does not
    serve them; that absence is measured in reports/survivorship.md, not
    hidden here). What this gate can and does demand: no all-False ticker
    columns (the assembler drops reused-symbol ghosts), a healthy number of
    exits (departed names were NOT scrubbed from history), and a non-NaN
    forward return on every exit date.
    """
    ever = panel.mask.any(axis=0)
    assert ever.all(), (
        f"{int((~ever).sum())} never-tradeable ticker columns survived assembly"
    )

    last = panel.n_dates - 1 - np.argmax(panel.mask[::-1], axis=0)
    exiting = last < panel.n_dates - 1
    n_exits = int(exiting.sum())
    assert n_exits > 50, (
        f"only {n_exits} names exit before sample end over 15 years — departed "
        "names are being scrubbed"
    )

    raw = panel.attrs["raw_forward_returns"]
    missing_terminal = [
        str(panel.tickers[j])
        for j in np.flatnonzero(exiting)
        if not np.isfinite(raw[last[j], j])
    ]
    assert not missing_terminal, (
        f"{len(missing_terminal)} exiting names lack a terminal forward return: "
        f"{missing_terminal[:8]}"
    )


def test_universe_thins_toward_the_past(panel):
    """Retrieval loss concentrates in the early sample (Session 3's gradient).

    Not a pass/fail on the bias itself — it cannot be fixed here — but the
    signature must be present and pointing the right way; a FLAT profile
    would mean the panel quietly lost its early-sample names entirely.
    """
    years = panel.dates.year
    early = panel.mask[years == 2010].sum(axis=1).mean()
    late = panel.mask[years == 2025].sum(axis=1).mean()
    assert late > early > 250, f"universe 2010: {early:.0f}/day, 2025: {late:.0f}/day"


# ------------------------------------------------------------------ #
# Gate 4 — panel alignment                                            #
# ------------------------------------------------------------------ #


def test_panel_alignment_on_real_data(panel):
    """labels[t] is the FORWARD return after dates[t] — verified independently.

    The raw forward-return matrix is recomputed from the price cache for a
    handful of names and compared elementwise. An off-by-one anywhere in the
    assembly pipeline cannot survive this.
    """
    raw = panel.attrs["raw_forward_returns"]
    h = panel.label_horizon
    assert np.isnan(raw[-h:]).all(), "the last horizon rows cannot have a realized label"

    tickers = [t for t in ("AAPL", "MSFT", "KO", "JNJ") if t in panel.tickers]
    assert len(tickers) >= 3
    for tk in tickers:
        j = int(np.flatnonzero(panel.tickers == tk)[0])
        px = (
            pl.read_parquet(PRICE_CACHE_DIR / f"{tk}.parquet")
            .sort("date")
            .to_pandas()
            .set_index("date")["adj_close"]
            .reindex(panel.dates)
            .to_numpy()
        )
        expected = np.full_like(px, np.nan)
        expected[:-h] = px[h:] / px[:-h] - 1.0
        ok = np.isfinite(raw[:, j]) & np.isfinite(expected)
        assert ok.sum() > 3000
        np.testing.assert_allclose(raw[ok, j], expected[ok], rtol=1e-5, atol=1e-7)


def test_processed_labels_are_per_date_standardized(panel):
    """The 4.5 chain ran: per-date mean ~0, std ~1, ~5% trimmed."""
    lab = panel.labels
    live_dates = [t for t in range(0, panel.n_dates - 1, 97)]
    for t in live_dates:
        row = lab[t][~np.isnan(lab[t])]
        if row.size < 50:
            continue
        assert abs(float(row.mean())) < 1e-4
        assert float(row.std(ddof=0)) == pytest.approx(1.0, abs=1e-3)

    stats = panel.attrs["label_processing"]
    eligible = stats["n_input"] - stats["n_missing"]
    assert stats["n_trimmed"] / eligible == pytest.approx(0.05, abs=0.005)


def test_market_vector_is_nan_free_and_sane(panel):
    assert not np.isnan(panel.market).any()
    assert not np.isinf(panel.market).any()
    names = panel.market_names
    assert len(names) == 67
    vix = panel.market[:, names.index("vix_level")]
    assert vix.min() > 8 and vix.max() < 100, "VIX outside its historical range"
    breadth = panel.market[:, names.index("pct_above_200dma")]
    assert breadth.min() >= 0.0 and breadth.max() <= 1.0


def test_panel_loads_from_cache_fast_enough():
    """Spec 4.7: load time from cache < 30s."""
    if not PANEL_PATH.exists():
        pytest.skip("no real panel — run scripts/07_assemble_panel.py")
    t0 = time.monotonic()
    Panel.load(PANEL_PATH)
    elapsed = time.monotonic() - t0
    assert elapsed < 30.0, f"panel load took {elapsed:.1f}s"
