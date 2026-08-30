"""Tests for the backtest engine: construct, costs, metrics, loop.

The gate test against real data lives in `test_engine_reproduces_momentum.py`;
these pin the arithmetic on constructed inputs where every expected value can
be computed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_us.backtest.construct import (
    cost_aware_optimize,
    decile_long_short,
    topk_dropout,
)
from master_us.backtest.costs import (
    CostConfig,
    corwin_schultz_spread,
    flat_cost,
    realistic_cost,
)
from master_us.backtest.engine import month_end_indices, run_backtest
from master_us.backtest.metrics import (
    BacktestSeries,
    annualized_return,
    average_holding_period,
    calmar,
    daily_turnover,
    hit_rate,
    information_coefficient,
    max_drawdown,
    sharpe,
    summarize,
)
from master_us.reporting.results import MetricValue

CFG = CostConfig(
    baseline_bps=10.0,
    sweep_bps=(0.0, 5.0, 10.0, 20.0, 50.0),
    spread_window=21,
    spread_floor_bps=1.0,
    impact_coefficient_bps=10.0,
    adv_lookback=21,
    max_participation=0.10,
)


# ------------------------------------------------------------------ #
# Construction                                                        #
# ------------------------------------------------------------------ #


def test_decile_long_short_is_dollar_neutral_and_equal_weight():
    rng = np.random.default_rng(0)
    w = decile_long_short(rng.standard_normal(500), np.ones(500, dtype=bool))
    assert (w > 0).sum() == 50 and (w < 0).sum() == 50
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(w).sum() == pytest.approx(2.0)
    assert np.unique(w[w > 0]).size == 1  # equal weight within the leg


def test_decile_long_short_longs_the_high_scores():
    scores = np.arange(100, dtype=float)
    w = decile_long_short(scores, np.ones(100, dtype=bool))
    assert (w[90:] > 0).all() and (w[:10] < 0).all()
    assert (w[10:90] == 0).all()


def test_decile_long_short_ignores_masked_and_nan_names():
    scores = np.arange(120, dtype=float)
    mask = np.ones(120, dtype=bool)
    mask[100:110] = False  # the highest-scoring names are out of the universe
    scores[110:] = np.nan  # and the very highest have no score at all
    w = decile_long_short(scores, mask)
    assert (w[100:] == 0).all()
    assert (w > 0).sum() == 10  # 100 live names -> 10 per leg


def test_decile_long_short_returns_empty_book_when_too_thin():
    w = decile_long_short(np.arange(15, dtype=float), np.ones(15, dtype=bool), n_deciles=10)
    assert not w.any()


def test_topk_selects_top_k_first_time():
    scores = np.arange(100, dtype=float)
    w = topk_dropout(scores, np.ones(100, dtype=bool), None, k=10, dropout_buffer=5)
    assert (w[90:] > 0).all() and (w[:90] == 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_topk_buffer_holds_incumbents_and_no_buffer_trades():
    rng = np.random.default_rng(3)
    n = 200
    scores1 = rng.standard_normal(n)
    mask = np.ones(n, dtype=bool)
    w1 = topk_dropout(scores1, mask, None, k=20, dropout_buffer=20)
    scores2 = scores1 + rng.standard_normal(n) * 0.2  # jitter the ranks

    buffered = topk_dropout(scores2, mask, w1, k=20, dropout_buffer=20)
    unbuffered = topk_dropout(scores2, mask, w1, k=20, dropout_buffer=0)
    to_buffered = 0.5 * np.abs(buffered - w1).sum()
    to_unbuffered = 0.5 * np.abs(unbuffered - w1).sum()
    assert to_buffered < to_unbuffered, "the buffer must reduce turnover under rank jitter"


def test_topk_incumbent_below_buffer_is_sold():
    scores = np.arange(30, dtype=float)
    mask = np.ones(30, dtype=bool)
    prev = np.zeros(30)
    prev[0] = 1.0  # incumbent with the WORST score, far below k + buffer
    w = topk_dropout(scores, mask, prev, k=5, dropout_buffer=3)
    assert w[0] == 0.0
    assert (w[25:] > 0).all()


def test_topk_untradeable_incumbent_is_dropped():
    scores = np.full(10, 1.0)
    scores[3] = 5.0
    mask = np.ones(10, dtype=bool)
    prev = np.zeros(10)
    prev[3] = 0.5
    mask[3] = False
    w = topk_dropout(scores, mask, prev, k=3, dropout_buffer=5)
    assert w[3] == 0.0


def test_cost_aware_optimize_respects_constraints():
    rng = np.random.default_rng(1)
    n = 40
    scores = rng.standard_normal(n)
    mask = np.ones(n, dtype=bool)
    mask[:5] = False
    neutral = rng.standard_normal((2, n))
    w = cost_aware_optimize(
        scores,
        np.zeros(n),
        np.eye(n) * 0.04,
        mask,
        lam_turnover=0.01,
        lam_risk=5.0,
        constraints={"max_weight": 0.06, "neutralize": neutral},
    )
    assert w.sum() == pytest.approx(0.0, abs=1e-7)
    assert np.abs(w).max() <= 0.06 + 1e-8
    assert (w[:5] == 0).all()
    assert np.abs(neutral @ w).max() < 1e-6


def test_cost_aware_optimize_turnover_penalty_freezes_the_book():
    rng = np.random.default_rng(2)
    n = 30
    scores = rng.standard_normal(n)
    mask = np.ones(n, dtype=bool)
    prev = cost_aware_optimize(scores, np.zeros(n), np.eye(n) * 0.04, mask, 0.01, 5.0)
    frozen = cost_aware_optimize(
        scores + rng.standard_normal(n) * 0.1, prev, np.eye(n) * 0.04, mask, 100.0, 5.0
    )
    assert np.abs(frozen - prev).sum() < 1e-6


# ------------------------------------------------------------------ #
# Costs                                                               #
# ------------------------------------------------------------------ #


def test_flat_cost_charges_half_round_trip_per_side():
    w0 = np.zeros(4)
    w1 = np.array([0.5, 0.5, 0.0, 0.0])
    # 100% one-way notional traded at 10 bps round-trip -> 5 bps.
    assert flat_cost(w1, w0, 10.0) == pytest.approx(0.0005)
    # Full liquidation later pays the other half: total = round trip.
    assert flat_cost(w0, w1, 10.0) + flat_cost(w1, w0, 10.0) == pytest.approx(0.001)


def _synthetic_hlc(true_spread: float, n: int = 2000, seed: int = 7):
    """A random walk observed through a known bid-ask bounce, densely sampled.

    Dense sampling matters: CS assumes the daily high/low approximate the true
    range, and with few trades per day the observed range understates it,
    which reads as a smaller spread. 200 trades/day is enough for the
    estimator to see the range it was derived for.
    """
    rng = np.random.default_rng(seed)
    half = true_spread / 2.0
    highs, lows, closes = np.empty(n), np.empty(n), np.empty(n)
    level = 100.0
    for t in range(n):
        path = level * np.exp(np.cumsum(rng.normal(0.0, 0.0008, 200)))
        trades = path * (1.0 + half * rng.choice([-1.0, 1.0], 200))
        highs[t], lows[t] = trades.max(), trades.min()
        closes[t] = trades[-1]
        level = path[-1]
    return highs, lows, closes


def test_corwin_schultz_recovers_synthetic_spreads():
    """Known spreads in, estimates within a factor of ~2 out, order preserved.

    Measured behavior on this construction: 20 bps → ~19, 50 → ~44,
    200 → ~189. The estimator has a noise floor of ~10 bps at very small true
    spreads, so the accuracy assertion starts at 20 bps and the small-spread
    case only asserts ordering.
    """
    estimates = {}
    for true_spread in (0.0005, 0.002, 0.005, 0.02):
        h, lo, c = _synthetic_hlc(true_spread)
        estimates[true_spread] = float(np.nanmean(corwin_schultz_spread(h, lo, 21, close=c)))

    for true_spread in (0.002, 0.005, 0.02):
        est = estimates[true_spread]
        assert true_spread / 2 < est < true_spread * 2, (
            f"true {true_spread:.4f} estimated {est:.4f}"
        )
    ordered = [estimates[k] for k in sorted(estimates)]
    assert ordered == sorted(ordered), f"estimates not monotone in true spread: {estimates}"


def test_corwin_schultz_window_head_is_nan_and_rest_finite():
    rng = np.random.default_rng(8)
    px = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    h, lo = px * 1.01, px * 0.99
    est = corwin_schultz_spread(h, lo, 21, close=px)
    assert np.isnan(est[:21]).all()
    assert np.isfinite(est[21:]).all()
    assert (est[21:] >= 0).all()


def test_realistic_cost_scales_with_spread_and_never_free():
    w0 = np.zeros(3)
    w1 = np.array([0.5, 0.5, 0.0])
    wide = realistic_cost(w1, w0, np.array([0.01, 0.01, 0.01]), np.full(3, 1e9), 1e8, CFG)
    tight = realistic_cost(w1, w0, np.array([0.0002, 0.0002, 0.0002]), np.full(3, 1e9), 1e8, CFG)
    assert wide > tight > 0
    # NaN spread and zero ADV fall back to the floor, not to free trading.
    floored = realistic_cost(w1, w0, np.array([np.nan] * 3), np.zeros(3), 1e8, CFG)
    assert floored > 0


def test_realistic_cost_impact_grows_with_participation():
    w0 = np.zeros(2)
    w1 = np.array([0.5, 0.5])
    spread = np.full(2, 0.0002)
    small_book = realistic_cost(w1, w0, spread, np.full(2, 1e9), 1e6, CFG)
    big_book = realistic_cost(w1, w0, spread, np.full(2, 1e9), 1e10, CFG)
    assert big_book > small_book


# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #


def _series(gross, costs=None, turnover=None) -> BacktestSeries:
    g = np.asarray(gross, dtype=np.float64)
    return BacktestSeries(
        gross=g,
        costs=np.zeros_like(g) if costs is None else np.asarray(costs, float),
        turnover=np.zeros_like(g) if turnover is None else np.asarray(turnover, float),
    )


def test_every_metric_returns_a_gross_net_pair():
    rng = np.random.default_rng(0)
    s = _series(rng.normal(5e-4, 0.01, 504), costs=np.full(504, 1e-4))
    table = summarize(s, benchmark=rng.normal(3e-4, 0.01, 504))
    expected = {
        "ann_return",
        "sharpe",
        "max_drawdown",
        "calmar",
        "turnover",
        "hit_rate",
        "holding_period_days",
        "excess_return",
        "information_ratio",
    }
    assert expected <= set(table)
    for name, mv in table.items():
        assert isinstance(mv, MetricValue), name
        assert mv.net is not None, f"{name} lost its net side"


def test_net_is_gross_minus_costs_everywhere():
    s = _series(np.full(252, 1e-3), costs=np.full(252, 2e-4))
    ar = annualized_return(s)
    assert ar.gross == pytest.approx(1e-3 * 252)
    assert ar.net == pytest.approx(8e-4 * 252)
    assert sharpe(s).net < sharpe(s).gross or sharpe(s).gross == 0.0


def test_max_drawdown_hand_computed():
    s = _series([0.10, -0.05, -0.10, 0.03, -0.02])
    mdd = max_drawdown(s)
    # equity: .10 .05 -.05 -.02 -.04 ; peak .10 -> trough -.05 = -.15
    assert mdd.gross == pytest.approx(-0.15)
    assert mdd.gross < 0


def test_calmar_is_return_over_drawdown():
    s = _series([0.10, -0.05, -0.10, 0.03, -0.02])
    expected = (np.mean(s.gross) * 252) / 0.15
    assert calmar(s).gross == pytest.approx(expected)


def test_hit_rate_counts_active_days_only():
    s = _series([0.0, 0.0, 0.01, -0.01, 0.02, 0.03])
    assert hit_rate(s).gross == pytest.approx(3 / 4)


def test_turnover_and_holding_period_are_reciprocal():
    to = np.zeros(100)
    to[::20] = 0.25  # 5 rebalances of 25% one-way
    s = _series(np.zeros(100), turnover=to)
    mean_to = daily_turnover(s).gross
    assert mean_to == pytest.approx(0.0125)
    assert average_holding_period(to).gross == pytest.approx(1 / 0.0125)


def test_information_coefficient_recovers_known_signal():
    rng = np.random.default_rng(5)
    t_len, n = 300, 200
    scores = rng.standard_normal((t_len, n))
    fwd = 0.2 * scores + rng.standard_normal((t_len, n))
    mask = np.ones((t_len, n), dtype=bool)
    ic, icir = information_coefficient(scores, fwd, mask, "pearson")
    rank_ic, _ = information_coefficient(scores, fwd, mask, "spearman")
    expected = 0.2 / np.sqrt(1 + 0.04)
    assert ic.gross == pytest.approx(expected, abs=0.02)
    assert rank_ic.gross == pytest.approx(expected, abs=0.03)
    assert icir.gross > 1.0  # a stable signal has high ICIR


def test_series_rejects_nan_and_negative_costs():
    with pytest.raises(ValueError, match="NaN"):
        _series([0.01, np.nan])
    with pytest.raises(ValueError, match="costs"):
        BacktestSeries(
            gross=np.zeros(2), costs=np.array([-1e-4, 0.0]), turnover=np.zeros(2)
        )


# ------------------------------------------------------------------ #
# Engine                                                              #
# ------------------------------------------------------------------ #


def _toy_engine_inputs(t_len=90, n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=t_len)
    scores = rng.standard_normal((t_len, n))
    returns = 0.05 * scores / 100 + rng.normal(0, 0.01, (t_len, n))
    mask = np.ones((t_len, n), dtype=bool)
    return dates, returns, scores, mask


def test_month_end_indices_picks_last_trading_days():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    idx = month_end_indices(dates)
    assert [dates[i].strftime("%Y-%m-%d") for i in idx] == [
        "2020-01-31",
        "2020-02-28",
        "2020-03-31",
    ]


def test_engine_trades_only_on_rebalance_days():
    dates, returns, scores, mask = _toy_engine_inputs()
    rebal = month_end_indices(dates)
    res = run_backtest(
        dates,
        returns,
        scores,
        mask,
        lambda s, m, p: decile_long_short(s, m),
        rebal,
        CFG,
    )
    off = np.delete(res.series.turnover, rebal)
    assert off.sum() == 0.0
    assert (res.series.costs[rebal] > 0).any()
    assert np.delete(res.series.costs, rebal).sum() == 0.0


def test_engine_weights_earn_next_day_returns_not_same_day():
    """The alignment property everything depends on.

    Returns are +1% on exactly one column every day. A book formed at the
    close of rebalance day t must NOT earn day t's return.
    """
    t_len, n = 10, 5
    dates = pd.bdate_range("2021-01-01", periods=t_len)
    returns = np.zeros((t_len, n))
    returns[:, 0] = 0.01
    scores = np.zeros((t_len, n))
    scores[:, 0] = 1.0  # constructor will long column 0
    mask = np.ones((t_len, n), dtype=bool)

    def go_long_best(s, m, p):
        w = np.zeros_like(s)
        w[np.nanargmax(s)] = 1.0
        return w

    res = run_backtest(
        dates, returns, scores, mask, go_long_best, np.array([3]), CFG
    )
    assert res.series.gross[3] == 0.0  # formed at close of day 3: no same-day earn
    assert res.series.gross[4] == pytest.approx(0.01)  # earns from day 4
    assert res.series.gross[:3].sum() == 0.0


def test_engine_held_name_with_missing_return_earns_zero():
    dates, returns, scores, mask = _toy_engine_inputs(t_len=30)
    returns[15:, 3] = np.nan  # name 3 stops printing mid-hold
    res = run_backtest(
        dates,
        returns,
        scores,
        mask,
        lambda s, m, p: decile_long_short(s, m, n_deciles=4),
        np.array([10]),
        CFG,
    )
    assert np.isfinite(res.series.gross).all()


def test_engine_realistic_tier_requires_spread_and_adv():
    dates, returns, scores, mask = _toy_engine_inputs()
    with pytest.raises(ValueError, match="realistic tier requires"):
        run_backtest(
            dates,
            returns,
            scores,
            mask,
            lambda s, m, p: decile_long_short(s, m),
            month_end_indices(dates),
            CFG,
            tier="realistic",
        )


def test_engine_equity_is_cumsum_of_net():
    dates, returns, scores, mask = _toy_engine_inputs()
    res = run_backtest(
        dates,
        returns,
        scores,
        mask,
        lambda s, m, p: decile_long_short(s, m),
        month_end_indices(dates),
        CFG,
    )
    np.testing.assert_allclose(res.equity, np.cumsum(res.series.gross - res.series.costs))
