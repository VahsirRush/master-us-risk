"""Phase 0 gates — spec section 4.7.

Four properties must hold before any model is trained. Each is asserted twice:
once on a clean panel, where it must hold, and once on a panel with the
corresponding defect deliberately introduced, where the assertion must fail. A
gate that has never been observed to fail is not a gate, it is decoration.

Where a corruption does NOT move the metric a gate measures, that is recorded
here in the test itself rather than papered over — see the note on
`corrupt_lookahead(lead=0)` under the shift test.

The IC helper below is local on purpose. `backtest/metrics.py` owns the
canonical (gross, net) implementation at Phase 1; duplicating a five-line
Pearson here keeps a blocking Phase-0 gate from depending on a module that does
not exist yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from master_us.data.normalize import AlreadyFittedError, RobustZScoreNorm
from master_us.data.panel import Panel

from .fixtures import (
    corrupt_lookahead,
    corrupt_misaligned_labels,
    corrupt_regime_shift,
    corrupt_survivorship,
    corrupt_survivorship_drop_delisted,
    make_synthetic_panel,
)

# Mean per-date IC below this is indistinguishable from zero for the panel sizes
# used here (750 dates x ~114 names gives a standard error near 0.0034, so this
# is roughly a six-sigma floor).
IC_NOISE_FLOOR = 0.02

# No honestly-constructed daily equity feature has a cross-sectional IC anywhere
# near this. A feature that does is holding the answer.
IC_IMPLAUSIBLE = 0.50


def ic_by_date(
    feature: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    lag: int = 0,
    min_names: int = 10,
) -> np.ndarray:
    """Per-date cross-sectional Pearson correlation of `feature[t - lag]` with `labels[t]`.

    `lag=+1` delays the feature by one day: the value formed on t-1 is used to
    predict the return following t. That is the "shift features forward one day"
    of spec section 4.7 — the feature arrives late.
    """
    n_dates = feature.shape[0]
    lo, hi = max(lag, 0), n_dates + min(lag, 0)
    rows = np.arange(lo, hi)

    a = feature[rows - lag].astype(np.float64)
    b = labels[rows].astype(np.float64)
    ok = mask[rows] & mask[rows - lag] & ~np.isnan(a) & ~np.isnan(b)

    cnt = ok.sum(axis=1)
    safe = np.maximum(cnt, 1)[:, None]
    ma = np.where(ok, a, 0.0).sum(axis=1, keepdims=True) / safe
    mb = np.where(ok, b, 0.0).sum(axis=1, keepdims=True) / safe

    da = np.where(ok, a - ma, 0.0)
    db = np.where(ok, b - mb, 0.0)
    sa = np.sqrt((da * da).sum(axis=1))
    sb = np.sqrt((db * db).sum(axis=1))

    denom = sa * sb
    out = np.divide((da * db).sum(axis=1), denom, out=np.full(rows.size, np.nan), where=denom > 0)
    out[cnt < min_names] = np.nan
    return out


def mean_ic(feature: np.ndarray, labels: np.ndarray, mask: np.ndarray, lag: int = 0) -> float:
    series = ic_by_date(feature, labels, mask, lag)
    return float(np.nanmean(series)) if np.isfinite(series).any() else np.nan


def all_feature_ics(panel: Panel, lag: int = 0) -> np.ndarray:
    """|mean IC| of every feature against the label, at the given lag."""
    return np.array(
        [
            abs(mean_ic(panel.features[:, :, j], panel.labels, panel.mask, lag))
            for j in range(panel.n_features)
        ]
    )


def gate_panel(**kwargs) -> Panel:
    return make_synthetic_panel(n_dates=750, n_tickers=120, n_features=20, seed=7, **kwargs)


# ------------------------------------------------------------------ #
# Gate 1 — shifted features lose signal                               #
# ------------------------------------------------------------------ #


def shift_gate_violations(panel: Panel) -> list[tuple[int, float, float]]:
    """Features whose IC survives — or improves under — a one-day delay.

    The reasoning: a feature built only from information available at t must get
    strictly worse when it is handed over a day late. If delaying it leaves the
    IC intact, or raises it, the value at t was already describing t+1.
    """
    now = all_feature_ics(panel, lag=0)
    delayed = all_feature_ics(panel, lag=1)
    return [
        (j, float(now[j]), float(delayed[j]))
        for j in range(panel.n_features)
        if delayed[j] > max(IC_NOISE_FLOOR, now[j])
    ]


def test_shifted_features_lose_signal():
    """Clean panel: delaying the features collapses IC toward zero."""
    panel = gate_panel()
    signal = panel.attrs["signal_feature"]

    contemporaneous = mean_ic(panel.features[:, :, signal], panel.labels, panel.mask, lag=0)
    delayed = mean_ic(panel.features[:, :, signal], panel.labels, panel.mask, lag=1)

    assert contemporaneous > 0.10, f"fixture signal is missing: IC={contemporaneous:.4f}"
    assert abs(delayed) < IC_NOISE_FLOOR, f"delayed IC={delayed:.4f} did not fall to zero"
    assert abs(delayed) < 0.25 * contemporaneous

    assert shift_gate_violations(panel) == []


def test_shift_gate_fires_on_a_feature_that_peeks_one_day_ahead():
    """Corrupted panel: feature[t] = label[t+1], the `.shift(-1)` bug.

    Its contemporaneous IC is near zero, so nothing at lag 0 notices it. Delaying
    the feature aligns it with the label it was copied from and the IC jumps to 1.
    """
    panel = corrupt_lookahead(gate_panel(), feature=5, lead=1)

    now = mean_ic(panel.features[:, :, 5], panel.labels, panel.mask, lag=0)
    delayed = mean_ic(panel.features[:, :, 5], panel.labels, panel.mask, lag=1)

    assert abs(now) < IC_NOISE_FLOOR, "the corruption should be invisible at lag 0"
    assert abs(delayed) > 0.9, f"corruption did not move the metric: delayed IC={delayed:.4f}"

    violations = shift_gate_violations(panel)
    assert [j for j, _, _ in violations] == [5]


def test_shift_gate_fires_on_labels_that_already_happened():
    """Corrupted panel: labels[t] is a return realized before t. The leaky direction."""
    panel = corrupt_misaligned_labels(gate_panel(), shift=-1)
    signal = panel.attrs["signal_feature"]

    now = mean_ic(panel.features[:, :, signal], panel.labels, panel.mask, lag=0)
    delayed = mean_ic(panel.features[:, :, signal], panel.labels, panel.mask, lag=1)

    assert abs(now) < IC_NOISE_FLOOR
    assert abs(delayed) > 0.10, f"corruption did not move the metric: delayed IC={delayed:.4f}"
    assert [j for j, _, _ in shift_gate_violations(panel)] == [signal]


def test_contemporaneous_lookahead_is_invisible_to_the_shift_gate():
    """Documented blind spot, asserted so it cannot quietly change.

    `corrupt_lookahead(lead=0)` sets feature[t] = label[t]. Delaying it gives
    label[t-1] against label[t], which is uncorrelated, so the shift gate sees a
    clean collapse and passes. That corruption is caught by the implausible-IC
    screen below instead. Two gates, two failure modes; neither covers both.
    """
    panel = corrupt_lookahead(gate_panel(), feature=5, lead=0)

    assert mean_ic(panel.features[:, :, 5], panel.labels, panel.mask, lag=0) == pytest.approx(
        1.0, abs=1e-6
    )
    assert shift_gate_violations(panel) == []  # the blind spot
    assert all_feature_ics(panel, lag=0).max() > IC_IMPLAUSIBLE  # but this catches it


def test_no_feature_has_an_implausible_information_coefficient():
    clean = gate_panel()
    assert clean.mask.sum(axis=1).mean() > 100
    assert all_feature_ics(clean, lag=0).max() < IC_IMPLAUSIBLE

    leaked = corrupt_lookahead(clean, feature=5, lead=0)
    ics = all_feature_ics(leaked, lag=0)
    assert ics.max() > IC_IMPLAUSIBLE
    assert int(ics.argmax()) == 5


# ------------------------------------------------------------------ #
# Gate 2 — normalization uses training statistics                     #
# ------------------------------------------------------------------ #


def _train_mask(panel: Panel, split: int) -> np.ndarray:
    out = np.zeros((panel.n_dates, panel.n_tickers), dtype=bool)
    out[:split] = panel.mask[:split]
    return out


def test_normalization_uses_training_stats():
    """Spec section 4.7: transform() on val/test is identical whether or not
    val/test data was present at fit() time.

    The panel has a regime break at the split. Without it the two fits would
    agree by coincidence and the gate would pass on leaking code.
    """
    split = 450
    panel = corrupt_regime_shift(gate_panel(), from_idx=split, scale=4.0, shift=2.0)

    with_val_test_present = RobustZScoreNorm().fit(panel.features, _train_mask(panel, split))
    train_only = RobustZScoreNorm().fit(panel.features[:split], panel.mask[:split])

    np.testing.assert_array_equal(
        with_val_test_present.stats.median, train_only.stats.median
    )
    np.testing.assert_array_equal(with_val_test_present.stats.scale, train_only.stats.scale)
    np.testing.assert_array_equal(
        with_val_test_present.transform(panel.features[split:]),
        train_only.transform(panel.features[split:]),
    )

    # And the guard that stops a later session from reintroducing the bug.
    with pytest.raises(AlreadyFittedError):
        train_only.fit(panel.features[split:], panel.mask[split:])


def test_normalization_gate_fires_when_statistics_are_fitted_on_everything():
    """Corrupted procedure: fit on the full sample, the canonical replication error."""
    split = 450
    panel = corrupt_regime_shift(gate_panel(), from_idx=split, scale=4.0, shift=2.0)

    honest = RobustZScoreNorm().fit(panel.features, _train_mask(panel, split))
    leaked = RobustZScoreNorm().fit(panel.features, panel.mask)

    assert abs(float(honest.stats.median[0]) - float(leaked.stats.median[0])) > 0.10
    assert float(leaked.stats.scale[0]) > 1.4 * float(honest.stats.scale[0])

    a = honest.transform(panel.features[split:])
    b = leaked.transform(panel.features[split:])
    assert float(np.abs(a - b).mean()) > 0.20
    assert float(np.abs(a - b).max()) > 1.0

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ #
# Gate 3 — no survivorship bias                                       #
# ------------------------------------------------------------------ #


def last_tradeable_index(panel: Panel) -> np.ndarray:
    """Index of each ticker's final in-universe date, or -1 if it never appears."""
    ever = panel.mask.any(axis=0)
    reversed_first = np.argmax(panel.mask[::-1], axis=0)
    return np.where(ever, panel.n_dates - 1 - reversed_first, -1)


def survivorship_violations(panel: Panel) -> dict[str, list[str]]:
    """Names scrubbed from the panel, and delistings missing a terminal return.

    On a real panel the delisting reference comes from the universe builder; here
    it is the synthetic factory's record. Either way the check is the same one:
    a name that left the universe must still be in the panel up to the date it
    left, carrying the return it left on.
    """
    order = {t: i for i, t in enumerate(panel.tickers.tolist())}
    last = last_tradeable_index(panel)

    never = [t for t in panel.tickers.tolist() if last[order[t]] < 0]
    no_terminal = [
        ticker
        for ticker, delist_idx in (panel.attrs.get("delisted") or {}).items()
        if not (
            panel.mask[delist_idx, order[ticker]]
            and not np.isnan(panel.labels[delist_idx, order[ticker]])
            and not panel.mask[delist_idx + 1 :, order[ticker]].any()
        )
    ]
    return {"never_tradeable": never, "missing_terminal_return": no_terminal}


def test_universe_no_survivorship():
    """Clean panel: delisted names are present to their delisting date, with a
    terminal return, and nothing has been scrubbed from the whole history."""
    panel = gate_panel(n_delistings=18)
    delisted = panel.attrs["delisted"]
    assert len(delisted) == 18

    violations = survivorship_violations(panel)
    assert violations == {"never_tradeable": [], "missing_terminal_return": []}

    # The delistings are real exits, not names that merely thin out at the end.
    last = last_tradeable_index(panel)
    exits = int(((last >= 0) & (last < panel.n_dates - 1)).sum())
    assert exits >= 18

    order = {t: i for i, t in enumerate(panel.tickers.tolist())}
    terminal = np.array([panel.labels[d, order[t]] for t, d in delisted.items()])
    assert not np.isnan(terminal).any()
    assert float(terminal.mean()) < -0.1, "terminal returns should be the delisting returns"


def test_survivorship_gate_fires_when_delisted_names_are_dropped():
    """Corrupted panel: keep only the names alive at the end of the sample."""
    clean = gate_panel(n_delistings=18)
    corrupted = corrupt_survivorship_drop_delisted(clean)

    violations = survivorship_violations(corrupted)
    assert len(violations["never_tradeable"]) == 18
    assert len(violations["missing_terminal_return"]) == 18

    clean_last = last_tradeable_index(clean)
    bad_last = last_tradeable_index(corrupted)
    clean_exits = int(((clean_last >= 0) & (clean_last < clean.n_dates - 1)).sum())
    bad_exits = int(((bad_last >= 0) & (bad_last < corrupted.n_dates - 1)).sum())

    assert clean_exits >= 18
    assert bad_exits <= 2, f"exits only fell from {clean_exits} to {bad_exits}"
    assert corrupted.mask.sum(axis=1).mean() < clean.mask.sum(axis=1).mean() - 5


def test_survivorship_gate_fires_on_an_arbitrary_scrub():
    """Corrupted panel: names removed from the whole history for no stated reason.

    Detected without any delisting reference at all — a ticker on the axis that
    is never tradeable anywhere is, by itself, evidence of a dropped history.
    """
    clean = gate_panel(n_delistings=18)
    corrupted = corrupt_survivorship(clean, drop_fraction=0.2, seed=0)

    assert survivorship_violations(clean)["never_tradeable"] == []
    assert len(survivorship_violations(corrupted)["never_tradeable"]) >= 20


# ------------------------------------------------------------------ #
# Gate 4 — panel alignment                                            #
# ------------------------------------------------------------------ #


def lead_lag_profile(panel: Panel, lags: range = range(-3, 4)) -> dict[int, float]:
    """Mean IC of the signal feature against the label at each lead/lag.

    A correctly aligned panel peaks at 0: features[t] predicts labels[t] and
    nothing else. A peak at a negative lag means the label reaches further
    forward than the feature was built for; a peak at a positive lag means the
    label describes something that had already happened by t.
    """
    signal = panel.attrs["signal_feature"]
    return {
        lag: mean_ic(panel.features[:, :, signal], panel.labels, panel.mask, lag) for lag in lags
    }


def test_panel_alignment():
    """Clean panel: features[t, n] and labels[t, n] are the same (date, ticker),
    and labels[t] is forward relative to features[t]."""
    panel = gate_panel()

    # Structural: one grid, shared by every array.
    assert panel.features.shape[:2] == panel.labels.shape == panel.mask.shape
    assert panel.labels.shape == (panel.n_dates, panel.n_tickers)

    # Forward: the last `label_horizon` dates cannot have a realized return yet.
    assert np.all(np.isnan(panel.labels[-panel.label_horizon :]))
    assert not np.all(np.isnan(panel.labels[: -panel.label_horizon]))

    profile = lead_lag_profile(panel)
    peak = max(profile, key=lambda k: abs(profile[k]))
    assert peak == 0, f"IC peaks at lag {peak:+d}, not 0: {profile}"
    assert abs(profile[0]) > 0.10
    off_peak = max(abs(v) for k, v in profile.items() if k != 0)
    assert off_peak < IC_NOISE_FLOOR
    assert abs(profile[0]) > 10 * off_peak


def test_alignment_gate_fires_when_labels_reach_too_far_forward():
    """Corrupted panel: labels[t] = labels[t+1]. The IC peak moves to lag -1."""
    panel = corrupt_misaligned_labels(gate_panel(), shift=+1)

    profile = lead_lag_profile(panel)
    peak = max(profile, key=lambda k: abs(profile[k]))
    assert peak == -1, f"expected the peak at -1, got {peak:+d}: {profile}"
    assert abs(profile[0]) < IC_NOISE_FLOOR
    assert abs(profile[-1]) > 0.10


def test_alignment_gate_fires_when_labels_describe_the_past():
    """Corrupted panel: labels[t] = labels[t-1], a return already realized by t.

    This is the direction that inflates a backtest, so the gate must locate the
    peak, not merely notice that it moved.
    """
    panel = corrupt_misaligned_labels(gate_panel(), shift=-1)

    profile = lead_lag_profile(panel)
    peak = max(profile, key=lambda k: abs(profile[k]))
    assert peak == +1, f"expected the peak at +1, got {peak:+d}: {profile}"
    assert abs(profile[0]) < IC_NOISE_FLOOR
    assert abs(profile[+1]) > 0.10


def test_alignment_gate_fires_when_the_ticker_axis_is_permuted():
    """Corrupted panel: labels reordered across names within each date.

    Same shapes, same marginal distributions, same dates — only the (date,
    ticker) correspondence is broken, which no shape assertion can see.
    """
    clean = gate_panel()
    rng = np.random.default_rng(0)
    labels = clean.labels.copy()
    for t in range(clean.n_dates):
        labels[t] = labels[t][rng.permutation(clean.n_tickers)]

    scrambled = Panel(
        dates=clean.dates,
        tickers=clean.tickers,
        features=clean.features,
        feature_names=clean.feature_names,
        market=clean.market,
        market_names=clean.market_names,
        labels=labels,
        mask=clean.mask,
        label_horizon=clean.label_horizon,
        attrs=clean.attrs,
    )

    assert scrambled.labels.shape == clean.labels.shape
    assert abs(mean_ic(clean.features[:, :, 0], clean.labels, clean.mask)) > 0.10
    assert (
        abs(mean_ic(scrambled.features[:, :, 0], scrambled.labels, scrambled.mask))
        < IC_NOISE_FLOOR
    )
    assert max(abs(v) for v in lead_lag_profile(scrambled).values()) < IC_NOISE_FLOOR
