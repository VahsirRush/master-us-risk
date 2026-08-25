"""Tests for `data/normalize.py` — spec section 4.5.

Two things are under test here and they are not the same thing. One is that the
arithmetic is right: median/MAD, clip at [-3, 3], NaN to 0, a cross-sectionally
standardized label. The other is that the API cannot be talked into refitting on
data it should never see. The second set matters more, because the first fails
loudly and the second fails with a good-looking number.
"""

from __future__ import annotations

import numpy as np
import pytest

from master_us.data.normalize import (
    MAD_TO_SIGMA,
    AlreadyFittedError,
    NotFittedError,
    RobustZScoreNorm,
    infer_feature_groups,
    process_labels,
)

from .fixtures import corrupt_regime_shift, make_synthetic_panel


def _split_masks(panel, train_frac=0.6):
    """Observation-level train mask, and the index where val/test begins."""
    split = int(panel.n_dates * train_frac)
    train = np.zeros((panel.n_dates, panel.n_tickers), dtype=bool)
    train[:split] = panel.mask[:split]
    return train, split


# ------------------------------------------------------------------ #
# The leakage property — statistics come from train and only train    #
# ------------------------------------------------------------------ #


def test_transform_is_independent_of_whether_val_test_was_present_at_fit():
    """The core guarantee of spec section 4.5.

    Fitting on the full array with a train-only mask must produce exactly the
    same statistics — and exactly the same val/test output — as fitting on a
    train-only slice. If those diverge, val/test rows are reaching the estimator.

    The panel carries a deliberate regime break at the split so that a leak
    would be loud. On a stationary panel this test passes whether or not the
    code leaks, which is why the break is here.
    """
    panel = corrupt_regime_shift(
        make_synthetic_panel(n_dates=400, n_tickers=80, seed=11),
        from_idx=240,
        scale=4.0,
        shift=2.0,
    )
    train_mask, split = _split_masks(panel)

    full = RobustZScoreNorm().fit(panel.features, train_mask)
    sliced = RobustZScoreNorm().fit(panel.features[:split], panel.mask[:split])

    np.testing.assert_array_equal(full.stats.median, sliced.stats.median)
    np.testing.assert_array_equal(full.stats.scale, sliced.stats.scale)
    assert full.stats.n_observations == sliced.stats.n_observations

    for span in (slice(split, None), slice(None), slice(0, split)):
        np.testing.assert_array_equal(
            full.transform(panel.features[span]),
            sliced.transform(panel.features[span]),
        )


def test_leaked_fit_visibly_changes_val_test_output():
    """The negative control for the test above.

    Fit on everything instead of on train, and the val/test output must move.
    If it does not, the property being asserted is vacuous on this fixture.
    """
    panel = corrupt_regime_shift(
        make_synthetic_panel(n_dates=400, n_tickers=80, seed=11),
        from_idx=240,
        scale=4.0,
        shift=2.0,
    )
    train_mask, split = _split_masks(panel)

    honest = RobustZScoreNorm().fit(panel.features, train_mask)
    leaked = RobustZScoreNorm().fit(panel.features, panel.mask)

    assert abs(float(honest.stats.median[0]) - float(leaked.stats.median[0])) > 0.1
    assert float(leaked.stats.scale[0]) > 1.4 * float(honest.stats.scale[0])

    diff = np.abs(honest.transform(panel.features[split:]) - leaked.transform(panel.features[split:]))
    assert float(diff.mean()) > 0.2, "leaked statistics barely moved the output — weak fixture"
    assert float(diff.max()) > 1.0


def test_transform_before_fit_raises():
    panel = make_synthetic_panel(n_dates=60, n_tickers=20)
    norm = RobustZScoreNorm()

    assert not norm.is_fitted
    with pytest.raises(NotFittedError, match="call fit"):
        norm.transform(panel.features)
    with pytest.raises(NotFittedError):
        norm.transform_with_masks(panel.features)
    with pytest.raises(NotFittedError):
        _ = norm.stats


def test_refitting_raises():
    """The misuse this API is shaped to prevent.

    `for split in splits: norm.fit(split); norm.transform(split)` is a natural
    thing to write and is the exact bug. The second fit must not be reachable.
    """
    panel = make_synthetic_panel(n_dates=200, n_tickers=40, seed=2)
    train_mask, split = _split_masks(panel)
    norm = RobustZScoreNorm().fit(panel.features, train_mask)

    with pytest.raises(AlreadyFittedError, match="already fitted"):
        norm.fit(panel.features[split:], panel.mask[split:])

    # And the statistics survived the attempt untouched.
    fresh = RobustZScoreNorm().fit(panel.features, train_mask)
    np.testing.assert_array_equal(norm.stats.median, fresh.stats.median)


def test_fit_transform_does_not_exist_as_a_working_method():
    """sklearn muscle memory is a leakage vector, so the name is a tripwire."""
    norm = RobustZScoreNorm()
    with pytest.raises(AlreadyFittedError, match="no fit_transform"):
        norm.fit_transform(np.zeros((3, 4, 5)))


def test_fitted_statistics_are_read_only():
    panel = make_synthetic_panel(n_dates=100, n_tickers=25)
    norm = RobustZScoreNorm().fit(panel.features, panel.mask)

    assert not norm.stats.median.flags.writeable
    assert not norm.stats.scale.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        norm.stats.median[0] = 99.0


def test_fit_on_panel_matches_manual_mask():
    """The convenience path must be arithmetically identical to the explicit one."""
    panel = make_synthetic_panel(n_dates=300, n_tickers=50, seed=5)
    cutoff = panel.dates[179]

    manual_mask = np.asarray(panel.dates <= cutoff)[:, None] & panel.mask
    manual = RobustZScoreNorm().fit(panel.features, manual_mask)
    via_panel = RobustZScoreNorm().fit_on_panel(panel, train_end=cutoff)

    np.testing.assert_array_equal(manual.stats.median, via_panel.stats.median)
    np.testing.assert_array_equal(manual.stats.scale, via_panel.stats.scale)
    assert via_panel.stats.train_span == (panel.dates[0], panel.dates[179])
    assert via_panel.feature_names == tuple(panel.feature_names)


def test_fit_on_panel_excludes_out_of_universe_names():
    """`Panel.mask` is the source of truth: masked-out names must not set the scale."""
    panel = make_synthetic_panel(n_dates=200, n_tickers=60, missing_rate=0.0, seed=6)
    features = panel.features.copy()
    mask = panel.mask.copy()
    mask[:, 1:10] = False
    mask[:, 0] = True
    features[:, 1:10, :] = 500.0  # wild values, but out of universe

    from master_us.data.panel import Panel

    dirty = Panel(
        dates=panel.dates,
        tickers=panel.tickers,
        features=features,
        feature_names=panel.feature_names,
        market=panel.market,
        market_names=panel.market_names,
        labels=panel.labels,
        mask=mask,
        label_horizon=panel.label_horizon,
    )
    norm = RobustZScoreNorm().fit_on_panel(dirty, train_end=dirty.dates[-1])
    assert float(norm.stats.scale.max()) < 5.0, "out-of-universe outliers reached the statistics"


def test_persisted_normalizer_reproduces_output():
    """Inference must borrow train's numbers verbatim, including across processes."""
    panel = make_synthetic_panel(n_dates=200, n_tickers=40, seed=9)
    train_mask, split = _split_masks(panel)
    norm = RobustZScoreNorm().fit(panel.features, train_mask)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = norm.save(Path(tmp) / "norm.pkl")
        reloaded = RobustZScoreNorm.load(path)

    np.testing.assert_array_equal(
        norm.transform(panel.features[split:]), reloaded.transform(panel.features[split:])
    )
    with pytest.raises(AlreadyFittedError):
        reloaded.fit(panel.features, train_mask)


# ------------------------------------------------------------------ #
# The arithmetic                                                      #
# ------------------------------------------------------------------ #


def _known_panel_features():
    """A feature column with an exactly known median and MAD.

    values = -100..100 -> median 0, MAD 50, so scale = 1.4826 * 50 = 74.13.
    """
    values = np.arange(-100, 101, dtype=np.float32)
    x = np.zeros((201, 1, 1), dtype=np.float32)
    x[:, 0, 0] = values
    return x


def test_median_and_mad_are_computed_as_specified():
    x = _known_panel_features()
    norm = RobustZScoreNorm().fit(x, np.ones((201, 1), dtype=bool))

    assert float(norm.stats.median[0]) == pytest.approx(0.0)
    assert float(norm.stats.scale[0]) == pytest.approx(MAD_TO_SIGMA * 50.0)
    assert norm.stats.n_observations == 201


def test_clipping_bites_exactly_at_the_boundary():
    """Spec section 4.5 clips to [-3, 3]. Check the boundary, not just the tails."""
    x = _known_panel_features()
    norm = RobustZScoreNorm().fit(x, np.ones((201, 1), dtype=bool))
    scale = float(norm.stats.scale[0])

    probe = np.array(
        [[2.9], [3.0], [3.0001], [50.0], [-2.9], [-3.0], [-3.0001], [-50.0]], dtype=np.float32
    )
    z = norm.transform((probe * scale).reshape(8, 1, 1))[:, 0, 0]

    assert z[0] == pytest.approx(2.9, abs=1e-4)  # inside: untouched
    assert z[1] == pytest.approx(3.0, abs=1e-4)  # on the boundary
    assert z[2] == pytest.approx(3.0, abs=1e-6)  # just outside: clipped
    assert z[3] == 3.0  # far outside: clipped
    assert z[4] == pytest.approx(-2.9, abs=1e-4)
    assert z[6] == pytest.approx(-3.0, abs=1e-6)
    assert z[7] == -3.0

    assert float(z.max()) <= 3.0 and float(z.min()) >= -3.0


def test_output_is_bounded_and_nan_free_on_a_real_panel():
    panel = make_synthetic_panel(n_dates=300, n_tickers=60, seed=4)
    assert np.isnan(panel.features).any(), "fixture should contain NaN for this to mean anything"

    z = RobustZScoreNorm().fit(panel.features, panel.mask).transform(panel.features)

    assert not np.isnan(z).any()
    assert float(z.min()) >= -3.0
    assert float(z.max()) <= 3.0
    assert z.dtype == np.float32


def test_nan_maps_to_zero_and_only_nan_does():
    x = np.array([[[1.0, np.nan]], [[2.0, 5.0]], [[3.0, 6.0]], [[4.0, 7.0]]], dtype=np.float32)
    norm = RobustZScoreNorm().fit(x, np.ones((4, 1), dtype=bool))
    z = norm.transform(x)

    assert z[0, 0, 1] == 0.0  # the NaN
    # The median of feature 0 is 2.5, so nothing else here lands exactly on zero.
    assert not np.any(z[:, :, 0] == 0.0)


def test_transform_preserves_within_feature_ordering():
    """Median/MAD is affine and positive-scaled, so ranks must survive it."""
    panel = make_synthetic_panel(n_dates=120, n_tickers=40, missing_rate=0.0, seed=8)
    z = RobustZScoreNorm().fit(panel.features, panel.mask).transform(panel.features)

    x0, z0 = panel.features[:, :, 0].ravel(), z[:, :, 0].ravel()
    inside = np.abs(z0) < 3.0  # clipped values legitimately tie
    order_x = np.argsort(x0[inside], kind="stable")
    assert np.all(np.diff(z0[inside][order_x]) >= -1e-6)


# ------------------------------------------------------------------ #
# Missingness indicators                                              #
# ------------------------------------------------------------------ #


def test_indicator_columns_flag_the_original_nans():
    panel = make_synthetic_panel(n_dates=150, n_tickers=30, n_features=6, seed=12)
    norm = RobustZScoreNorm(feature_names=panel.feature_names).fit(panel.features, panel.mask)
    out = norm.transform_with_masks(panel.features)

    assert out.values.shape == (150, 30, 12)  # 6 features + 6 per-feature groups
    assert out.n_base_features == 6
    assert out.names[6:] == tuple(f"{n}__isnan" for n in panel.feature_names)

    np.testing.assert_array_equal(out.indicators.astype(bool), np.isnan(panel.features))
    np.testing.assert_array_equal(out.base, norm.transform(panel.features))
    assert not np.isnan(out.values).any()


def test_grouped_indicators_flag_any_member_missing():
    x = np.zeros((10, 3, 4), dtype=np.float32)
    x[:] = np.arange(10, dtype=np.float32)[:, None, None] + np.arange(4, dtype=np.float32)
    x[2, 1, 0] = np.nan  # group A, first member
    x[5, 0, 3] = np.nan  # group B, second member

    norm = RobustZScoreNorm(
        feature_names=["a1", "a2", "b1", "b2"],
        feature_groups={"A": ["a1", "a2"], "B": ["b1", "b2"]},
    ).fit(x, np.ones((10, 3), dtype=bool))
    out = norm.transform_with_masks(x)

    assert out.values.shape == (10, 3, 6)
    assert out.group_names == ("A", "B")
    assert out.values[2, 1, 4] == 1.0 and out.values[2, 1, 5] == 0.0
    assert out.values[5, 0, 5] == 1.0 and out.values[5, 0, 4] == 0.0
    assert out.indicators.sum() == 2


def test_unassigned_feature_in_group_spec_raises():
    x = np.ones((5, 2, 3), dtype=np.float32) * np.arange(3, dtype=np.float32)
    x += np.arange(5, dtype=np.float32)[:, None, None]
    norm = RobustZScoreNorm(feature_names=["a", "b", "c"], feature_groups={"g": ["a", "b"]})
    norm.fit(x, np.ones((5, 2), dtype=bool))

    with pytest.raises(ValueError, match="belong to no group"):
        norm.transform_with_masks(x)


def test_infer_feature_groups_uses_the_name_prefix():
    groups = infer_feature_groups(["ret_1d", "ret_5d", "vol_20d", "rsi14"])
    assert groups == {"ret": ["ret_1d", "ret_5d"], "vol": ["vol_20d"], "rsi14": ["rsi14"]}


# ------------------------------------------------------------------ #
# Input validation                                                    #
# ------------------------------------------------------------------ #


def test_date_level_and_observation_level_train_masks_agree():
    panel = make_synthetic_panel(n_dates=200, n_tickers=40, missing_rate=0.0, seed=13)
    by_date = np.zeros(200, dtype=bool)
    by_date[:120] = True

    a = RobustZScoreNorm().fit(panel.features, by_date)
    b = RobustZScoreNorm().fit(panel.features, np.broadcast_to(by_date[:, None], (200, 40)))
    np.testing.assert_array_equal(a.stats.median, b.stats.median)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (np.ones((5, 3), dtype=bool), "shape"),
        (np.ones((10, 4), dtype=np.int8), "bool"),
        (np.zeros((10, 4), dtype=bool), "selects no observations"),
    ],
)
def test_bad_train_mask_raises(mask, match):
    x = np.random.default_rng(0).standard_normal((10, 4, 3)).astype(np.float32)
    with pytest.raises((ValueError, TypeError), match=match):
        RobustZScoreNorm().fit(x, mask)


def test_feature_count_mismatch_on_transform_raises():
    x = np.random.default_rng(0).standard_normal((20, 5, 4)).astype(np.float32)
    norm = RobustZScoreNorm().fit(x, np.ones((20, 5), dtype=bool))
    with pytest.raises(ValueError, match="was fitted on 4"):
        norm.transform(x[:, :, :3])


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match=r"\(T, N, F\)"):
        RobustZScoreNorm().fit(np.zeros((10, 4)), np.ones((10,), dtype=bool))


def test_constant_feature_raises_by_default():
    """CLAUDE.md rule 6: fail loudly. A zero-MAD feature is a division by zero."""
    x = np.random.default_rng(0).standard_normal((30, 5, 3)).astype(np.float32)
    x[:, :, 1] = 7.0

    with pytest.raises(ValueError, match="zero or non-finite MAD"):
        RobustZScoreNorm(feature_names=["a", "b", "c"]).fit(x, np.ones((30, 5), dtype=bool))

    kept = RobustZScoreNorm(zero_mad="unit_scale").fit(x, np.ones((30, 5), dtype=bool))
    assert float(kept.stats.scale[1]) == 1.0
    assert np.all(kept.transform(x)[:, :, 1] == 0.0)


def test_all_nan_feature_over_training_span_raises():
    x = np.random.default_rng(0).standard_normal((30, 5, 3)).astype(np.float32)
    x[:20, :, 2] = np.nan
    train = np.zeros((30, 5), dtype=bool)
    train[:20] = True

    with pytest.raises(ValueError, match="entirely NaN"):
        RobustZScoreNorm(feature_names=["a", "b", "c"]).fit(x, train)


# ------------------------------------------------------------------ #
# Label processing — spec section 4.5                                 #
# ------------------------------------------------------------------ #


def test_label_processor_drops_exactly_the_specified_fraction():
    """5% total means 2.5% from each tail — see NOTES.md on the spec's wording.

    Sized so the per-tail count is exact (1000 names, 2.5% -> 25 per side) rather
    than a floor that rounds the answer into agreeing with itself.
    """
    panel = make_synthetic_panel(n_dates=120, n_tickers=1000, missing_rate=0.0, seed=3)
    out = process_labels(panel.labels, panel.mask, drop_pct=5.0)

    eligible = out.n_input - out.n_missing
    assert out.n_trimmed == 50 * 119  # 25 per tail, per date, on the 119 labelled dates
    assert out.trimmed_fraction == pytest.approx(0.05, abs=1e-9)
    assert out.n_output == eligible - out.n_trimmed
    assert out.n_dates_dropped == 0


def test_label_processor_drops_the_extremes_and_not_the_middle():
    """The right count discarded from the wrong place would still pass the count test."""
    panel = make_synthetic_panel(n_dates=40, n_tickers=400, missing_rate=0.0, seed=14)
    out = process_labels(panel.labels, panel.mask, drop_pct=10.0)

    for t in range(30):
        raw = panel.labels[t]
        kept = ~np.isnan(out.values[t])
        dropped = ~kept

        assert kept.sum() == 400 - 2 * 20
        # Every dropped value sits outside the range of every kept value.
        assert raw[dropped].max() >= raw[kept].max()
        assert raw[dropped].min() <= raw[kept].min()
        below = (raw[dropped] < raw[kept].min()).sum()
        above = (raw[dropped] > raw[kept].max()).sum()
        assert below == 20 and above == 20


def test_label_processor_is_cross_sectionally_standardized_per_date():
    panel = make_synthetic_panel(n_dates=200, n_tickers=500, seed=15)
    out = process_labels(panel.labels, panel.mask, drop_pct=5.0)

    live = ~np.all(np.isnan(out.values), axis=1)
    assert live.sum() >= 190

    for t in np.flatnonzero(live):
        row = out.values[t][~np.isnan(out.values[t])]
        assert abs(float(row.mean())) < 1e-5
        assert float(row.std(ddof=0)) == pytest.approx(1.0, abs=1e-5)


def test_label_processor_standardizes_each_date_independently():
    """A date scaled by 100x must come out looking like every other date.

    Pooled standardization would leave the inflated date inflated; this is the
    assertion that separates per-date from pooled.
    """
    panel = make_synthetic_panel(n_dates=60, n_tickers=300, missing_rate=0.0, seed=16)
    labels = panel.labels.copy()
    labels[10] *= 100.0
    labels[20] += 5.0

    out = process_labels(labels, panel.mask, drop_pct=5.0)
    stds = np.array([np.nanstd(out.values[t]) for t in (5, 10, 20)])
    means = np.array([np.nanmean(out.values[t]) for t in (5, 10, 20)])

    np.testing.assert_allclose(stds, 1.0, atol=1e-5)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)


def test_label_processor_preserves_shape_and_missingness():
    """Trimmed observations become NaN in place — the (T, N) grid never moves."""
    panel = make_synthetic_panel(n_dates=100, n_tickers=200, seed=17)
    out = process_labels(panel.labels, panel.mask, drop_pct=5.0)

    assert out.values.shape == panel.labels.shape
    assert out.values.dtype == np.float32
    # Anything absent in the input stays absent; nothing is invented.
    assert np.all(np.isnan(out.values[np.isnan(panel.labels)]))
    assert np.all(np.isnan(out.values[~panel.mask]))


def test_label_processor_respects_the_universe_mask():
    """Out-of-universe names must not influence the cross-sectional statistics."""
    panel = make_synthetic_panel(n_dates=50, n_tickers=200, missing_rate=0.0, seed=18)
    labels = panel.labels.copy()
    mask = panel.mask.copy()
    mask[:, 150:] = False
    labels[:, 150:] = 50.0  # extreme, but outside the universe

    out = process_labels(labels, mask, drop_pct=5.0)
    assert np.all(np.isnan(out.values[:, 150:]))

    reference = process_labels(labels[:, :150], mask[:, :150], drop_pct=5.0)
    np.testing.assert_allclose(out.values[:, :150], reference.values, rtol=0, atol=1e-6)


def test_per_tail_flag_doubles_the_trim():
    panel = make_synthetic_panel(n_dates=30, n_tickers=1000, missing_rate=0.0, seed=19)
    total = process_labels(panel.labels, panel.mask, drop_pct=5.0, per_tail=False)
    per_tail = process_labels(panel.labels, panel.mask, drop_pct=5.0, per_tail=True)

    assert per_tail.n_trimmed == 2 * total.n_trimmed
    assert per_tail.trimmed_fraction == pytest.approx(0.10, abs=1e-9)


def test_zero_drop_pct_trims_nothing_but_still_standardizes():
    panel = make_synthetic_panel(n_dates=40, n_tickers=200, missing_rate=0.0, seed=20)
    out = process_labels(panel.labels, panel.mask, drop_pct=0.0)

    assert out.n_trimmed == 0
    assert out.n_output == out.n_input - out.n_missing
    assert abs(float(np.nanmean(out.values[0]))) < 1e-5


def test_thin_and_degenerate_dates_are_voided():
    labels = np.full((3, 10), np.nan, dtype=np.float32)
    labels[0, :10] = np.arange(10, dtype=np.float32)  # healthy
    labels[1, :1] = 0.5  # a single name
    labels[2, :10] = 4.0  # no dispersion

    out = process_labels(labels, drop_pct=0.0, min_names=2)

    assert not np.all(np.isnan(out.values[0]))
    assert np.all(np.isnan(out.values[1]))
    assert np.all(np.isnan(out.values[2]))
    assert out.n_dates_dropped == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"drop_pct": -1.0}, "drop_pct"),
        ({"drop_pct": 100.0}, "drop_pct"),
        ({"drop_pct": 60.0, "per_tail": True}, "entire cross-section"),
        ({"min_names": 1}, "min_names"),
    ],
)
def test_label_processor_rejects_bad_arguments(kwargs, match):
    labels = np.zeros((5, 10), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        process_labels(labels, **kwargs)


def test_label_processor_rejects_mismatched_mask():
    with pytest.raises(ValueError, match="does not match"):
        process_labels(np.zeros((5, 10), dtype=np.float32), np.ones((5, 9), dtype=bool))
