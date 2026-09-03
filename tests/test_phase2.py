"""Tests for Phase-2 orchestration — spec section 6.

The protocol properties matter more than the numbers here: that the splits
carry an embargo, that every model is scored the same way, that seed
aggregation reports honest dispersion, and that the gate cannot be passed by
a model that was not run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_us.backtest.costs import CostConfig
from master_us.experiments.lgbm_tuning import candidates, subset_columns, with_subset
from master_us.experiments.phase2 import (
    GATE_MIN_LGBM_RANKIC,
    MODEL_ORDER,
    SUSPICION_RANKIC,
    SeedRun,
    aggregate,
    backtest_scores,
)
from master_us.models.train import PreparedData, prepare_data, rank_ic_by_date

CFG = CostConfig(
    baseline_bps=10.0,
    sweep_bps=(0.0, 10.0),
    spread_window=21,
    spread_floor_bps=1.0,
    impact_coefficient_bps=10.0,
    adv_lookback=21,
    max_participation=0.10,
)


def _prepared(t=200, n=60, f=8, m=5, lookback=5, seed=0) -> PreparedData:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((t, n, f)).astype(np.float32)
    labels = (0.2 * features[:, :, 0] + rng.standard_normal((t, n))).astype(np.float32)
    labels = (labels - labels.mean(axis=1, keepdims=True)) / labels.std(axis=1, keepdims=True)
    return PreparedData(
        dates=pd.bdate_range("2020-01-01", periods=t),
        features=features,
        market=rng.standard_normal((t, m)).astype(np.float32),
        labels=labels,
        mask=np.ones((t, n), dtype=bool),
        raw_forward=(labels * 0.01).astype(np.float32),
        lookback=lookback,
        train_idx=np.arange(lookback - 1, 120, dtype=np.int64),
        valid_idx=np.arange(120, 150, dtype=np.int64),
        test_idx=np.arange(150, t, dtype=np.int64),
        feature_names=tuple(f"xs_f{i}" if i % 2 else f"f{i}" for i in range(f)),
    )


# ------------------------------------------------------------------ #
# Protocol                                                            #
# ------------------------------------------------------------------ #


def test_splits_are_chronological_and_embargoed():
    """No index overlap, strictly increasing, with a real gap at each seam."""
    from master_us.data.assemble import PANEL_PATH
    from master_us.data.panel import Panel

    if not PANEL_PATH.exists():
        pytest.skip("no real panel")
    panel = Panel.load(PANEL_PATH)
    data = prepare_data(
        panel,
        lookback=20,
        train=("2010-01-01", "2016-12-31"),
        valid=("2017-01-01", "2018-12-31"),
        test=("2019-01-01", "2025-12-31"),
        embargo_days=21,
    )
    assert set(data.train_idx).isdisjoint(data.valid_idx)
    assert set(data.valid_idx).isdisjoint(data.test_idx)
    assert data.train_idx.max() < data.valid_idx.min() < data.test_idx.min()

    # The embargo is a real gap in DAYS, not just in index.
    train_end = panel.dates[data.train_idx.max()]
    valid_start = panel.dates[data.valid_idx.min()]
    assert (valid_start - train_end).days >= 21
    valid_end = panel.dates[data.valid_idx.max()]
    test_start = panel.dates[data.test_idx.min()]
    assert (test_start - valid_end).days >= 21

    # Every sample index can see a full lookback window.
    assert data.train_idx.min() >= 19


def test_prepare_data_normalizes_on_train_only():
    """Val/test rows must not influence the normalizer — the Session-2 property,
    now on the Phase-2 entry point."""
    from master_us.data.assemble import PANEL_PATH
    from master_us.data.panel import Panel

    if not PANEL_PATH.exists():
        pytest.skip("no real panel")
    panel = Panel.load(PANEL_PATH)
    full = prepare_data(
        panel, 20, ("2010-01-01", "2016-12-31"), ("2017-01-01", "2018-12-31"),
        ("2019-01-01", "2025-12-31"), 21,
    )
    # Truncating the panel after validation cannot change training-era features.
    cut = int(np.flatnonzero(np.asarray(panel.dates <= pd.Timestamp("2019-01-01")))[-1])
    trimmed = prepare_data(
        panel.date_slice("2010-01-01", "2019-01-01"),
        20, ("2010-01-01", "2016-12-31"), ("2017-01-01", "2018-12-31"),
        ("2018-06-01", "2019-01-01"), 21,
    )
    np.testing.assert_array_equal(full.features[:cut], trimmed.features[: cut])


def test_market_vector_is_normalized_not_raw():
    """Raw dollar volumes near 1e12 next to returns near 1e-3 would swamp the gate."""
    from master_us.data.assemble import PANEL_PATH
    from master_us.data.panel import Panel

    if not PANEL_PATH.exists():
        pytest.skip("no real panel")
    data = prepare_data(
        Panel.load(PANEL_PATH), 20, ("2010-01-01", "2016-12-31"),
        ("2017-01-01", "2018-12-31"), ("2019-01-01", "2025-12-31"), 21,
    )
    assert np.abs(data.market).max() <= 3.0 + 1e-5  # RobustZScoreNorm clip
    assert not np.isnan(data.market).any()


# ------------------------------------------------------------------ #
# Scoring and aggregation                                             #
# ------------------------------------------------------------------ #


def test_backtest_uses_next_day_returns_not_same_day():
    """A perfect-foresight score must earn, and a lagged one must not.

    Catches the classic off-by-one in the model-scoring path specifically:
    day t's realized return is the forward return recorded at t-1.
    """
    data = _prepared()
    perfect = np.full(data.labels.shape, np.nan, dtype=np.float32)
    perfect[:-1] = data.labels[1:]  # today's score = tomorrow's realized ranking
    got = backtest_scores(data, perfect, CFG, topk=10, buffer=5)
    assert got["sharpe_gross"] > 3.0  # foresight should be enormous

    lagged = np.full(data.labels.shape, np.nan, dtype=np.float32)
    lagged[1:] = data.labels[:-1]  # yesterday's ranking — stale
    got_lagged = backtest_scores(data, lagged, CFG, topk=10, buffer=5)
    assert got_lagged["sharpe_gross"] < got["sharpe_gross"]


def test_backtest_charges_costs_and_reports_turnover():
    data = _prepared()
    rng = np.random.default_rng(1)
    scores = rng.standard_normal(data.labels.shape).astype(np.float32)
    got = backtest_scores(data, scores, CFG)
    assert got["sharpe_net"] < got["sharpe_gross"]  # costs bite
    assert 0.0 < got["turnover"] <= 1.0
    assert got["max_drawdown"] <= 0.0


def test_backtest_reports_both_books():
    """Long-only carries beta; the dollar-neutral book is the alpha read.

    Both must be present, and they must be genuinely different numbers.
    """
    data = _prepared()
    perfect = np.full(data.labels.shape, np.nan, dtype=np.float32)
    perfect[:-1] = data.labels[1:]
    got = backtest_scores(data, perfect, CFG, topk=10, buffer=5)
    for key in ("sharpe_gross", "ls_sharpe_gross", "ls_sharpe_net", "ls_turnover"):
        assert key in got
    assert got["ls_sharpe_gross"] != got["sharpe_gross"]
    assert got["ls_sharpe_net"] < got["ls_sharpe_gross"]


def _run(model="lgbm", seed=0, ic=0.02, sharpe=1.0) -> SeedRun:
    return SeedRun(
        model=model, seed=seed, test_rank_ic=ic, test_icir=1.0,
        sharpe_gross=sharpe, sharpe_net=sharpe - 0.5,
        ann_return_gross=0.1, ann_return_net=0.05,
        turnover=0.5, max_drawdown=-0.2,
        ls_sharpe_gross=sharpe * 0.4, ls_sharpe_net=sharpe * 0.4 - 0.3,
        ls_turnover=0.8, detail={},
    )


def test_aggregate_reports_seed_dispersion():
    runs = [_run(ic=x) for x in (0.018, 0.020, 0.022, 0.019, 0.021)]
    table = aggregate(runs)
    assert table["rank_ic"].gross == pytest.approx(0.020)
    assert table["rank_ic"].std == pytest.approx(np.std([0.018, 0.020, 0.022, 0.019, 0.021], ddof=1))
    assert table["rank_ic"].n_seeds == 5


def test_aggregate_keeps_gross_net_distinct_for_sharpe():
    table = aggregate([_run(sharpe=1.0), _run(sharpe=1.2)])
    assert table["sharpe"].gross == pytest.approx(1.1)
    assert table["sharpe"].net == pytest.approx(0.6)


def test_deterministic_model_has_zero_dispersion():
    """Ridge is deterministic: five seeds must agree exactly, and the table
    must SAY zero rather than hide it."""
    table = aggregate([_run(model="ridge", seed=s, ic=0.0162) for s in range(5)])
    assert table["rank_ic"].std == 0.0
    assert table["rank_ic"].n_seeds == 5


def test_gate_and_suspicion_thresholds_are_ordered():
    assert GATE_MIN_LGBM_RANKIC == 0.02
    # The suspicion bar must sit above the gate but within reach of a real
    # leak — max single-feature |IC| on this panel is 0.0154.
    assert GATE_MIN_LGBM_RANKIC < SUSPICION_RANKIC < 0.5
    assert MODEL_ORDER == ("ridge", "lgbm", "lstm", "ungated")


# ------------------------------------------------------------------ #
# Tuning search                                                       #
# ------------------------------------------------------------------ #


def test_tuning_grid_is_reasoned_and_includes_the_spec_config():
    grid = candidates()
    names = {c.name for c in grid}
    assert "spec" in names, "the search must include the shipped config as a control"
    assert len(names) == len(grid), "duplicate candidate names"
    for cand in grid:
        assert cand.hypothesis, f"{cand.name} has no stated hypothesis"


def test_feature_subsets_split_ranks_from_raw():
    data = _prepared(f=8)
    ranks = subset_columns(data, "ranks")
    raw = subset_columns(data, "raw")
    assert len(ranks) + len(raw) == len(data.feature_names)
    assert all(data.feature_names[i].startswith("xs_") for i in ranks)
    assert not any(data.feature_names[i].startswith("xs_") for i in raw)

    view = with_subset(data, "ranks")
    assert view.features.shape[2] == len(ranks)
    assert view.train_idx is data.train_idx  # splits untouched


def test_subset_view_shares_labels_and_splits():
    data = _prepared()
    view = with_subset(data, "ranks")
    np.testing.assert_array_equal(view.labels, data.labels)
    np.testing.assert_array_equal(view.test_idx, data.test_idx)


def test_unknown_subset_raises():
    with pytest.raises(ValueError, match="unknown feature subset"):
        subset_columns(_prepared(), "vibes")


def test_rank_ic_by_date_matches_manual_spearman():
    from scipy import stats

    rng = np.random.default_rng(0)
    scores = rng.standard_normal((5, 40)).astype(np.float32)
    labels = (0.5 * scores + rng.standard_normal((5, 40))).astype(np.float32)
    valid = np.ones((5, 40), dtype=bool)
    got = rank_ic_by_date(scores, labels, valid, np.arange(5, dtype=np.int64))
    expected = [stats.spearmanr(scores[t], labels[t]).statistic for t in range(5)]
    np.testing.assert_allclose(got, expected, rtol=1e-10)
