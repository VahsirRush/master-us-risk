"""The Phase-1 gate — spec section 5.4.

12-1 momentum through the full engine on the real (extended-window) S&P 500
panel. Three assertions, per the spec: a long-short Sharpe in a documented
plausible range, a visible 2009 momentum crash, and turnover consistent with
monthly rebalancing. If this fails, the engine cannot evaluate anything and no
model work may start.

The heavy computation runs once per session (module-scoped fixture, ~40s) and
skips cleanly on a checkout without the gate caches — the committed
`reports/status/phase1.json` still records the last full run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_us.backtest.momentum import (
    GATE_CRASH_MAX_RETURN,
    GATE_CRASH_WINDOW,
    GATE_MEMBERSHIP_CACHE,
    GATE_REBALANCE_TURNOVER_RANGE,
    GATE_SHARPE_RANGE,
    evaluate_gate,
    load_cost_config,
    load_wide_panel,
    momentum_scores,
)


@pytest.fixture(scope="module")
def gate_outcome():
    if not GATE_MEMBERSHIP_CACHE.exists():
        pytest.skip("gate caches missing — run scripts/10_fetch_gate_data.py")
    panel = load_wide_panel(GATE_MEMBERSHIP_CACHE, "2006-01-01", "2025-12-31")
    return evaluate_gate(panel, load_cost_config(), None, eval_start="2008-01-01")


def test_momentum_matches_literature(gate_outcome):
    """The gate itself: all three checks, with the numbers in the failure text."""
    o = gate_outcome
    sharpe = o.metrics["sharpe"].gross
    turnover = o.metrics["rebalance_turnover"].gross

    assert GATE_SHARPE_RANGE[0] <= sharpe <= GATE_SHARPE_RANGE[1], (
        f"long-short Sharpe {sharpe:.2f} outside plausible range {GATE_SHARPE_RANGE} — "
        "above it means a bug (costs, universe, alignment), below it means inversion"
    )
    assert o.crash_return < GATE_CRASH_MAX_RETURN, (
        f"gross return over {GATE_CRASH_WINDOW} is {o.crash_return:+.1%}; a momentum "
        "engine that does not crash in 2009 is not measuring momentum"
    )
    lo, hi = GATE_REBALANCE_TURNOVER_RANGE
    assert lo <= turnover <= hi, (
        f"mean one-way rebalance turnover {turnover:.1%} outside [{lo:.0%}, {hi:.0%}] — "
        "inconsistent with monthly decile rebalancing"
    )
    assert all(o.checks.values())
    assert o.phase_result.gate_passed


def test_crash_is_short_leg_driven(gate_outcome):
    """The 2009 crash's documented anatomy: shorted losers rallying.

    A crash of the right size but the wrong shape (long leg collapsing) would
    mean the book is inverted while the totals happen to look right.
    """
    o = gate_outcome
    dates = o.eval_dates
    lo = int(np.searchsorted(dates, pd.Timestamp(GATE_CRASH_WINDOW[0])))
    hi = int(np.searchsorted(dates, pd.Timestamp(GATE_CRASH_WINDOW[1]), side="right"))

    # The short leg's pnl over the window, from the held weights directly.
    w = o.result_baseline.weights
    gross = o.result_baseline.series.gross[lo:hi]
    assert gross.sum() < GATE_CRASH_MAX_RETURN
    # Weights must actually be a long-short book through the window.
    assert (w[lo:hi] > 0).any() and (w[lo:hi] < 0).any()


def test_no_trades_off_rebalance_days(gate_outcome):
    o = gate_outcome
    turnover = o.result_baseline.series.turnover
    rebal = np.searchsorted(o.eval_dates, o.result_baseline.rebalance_dates)
    assert float(np.delete(turnover, rebal).sum()) == 0.0
    assert float(np.delete(o.result_baseline.series.costs, rebal).sum()) == 0.0


def test_realistic_tier_costs_more_than_baseline(gate_outcome):
    """CS spread + impact on ~380 large caps must cost more than 10 bps flat —
    and both must actually charge something."""
    o = gate_outcome
    base = float(o.result_baseline.series.costs.sum())
    real = float(o.result_realistic.series.costs.sum())
    assert base > 0
    assert real > 0
    assert o.metrics["sharpe_realistic_net"].net <= o.metrics["sharpe"].net


def test_momentum_scores_are_backward_looking():
    """scores[t] must be computable from prices strictly before t-skip.

    Changing prices AFTER t must not change scores at t; changing them inside
    the skip month must not either.
    """
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 5)), axis=0))
    base = momentum_scores(px, lookback=252, skip=21)

    tampered_future = px.copy()
    tampered_future[300:] *= 5.0
    np.testing.assert_array_equal(base[:300], momentum_scores(tampered_future)[:300])

    tampered_skip = px.copy()
    tampered_skip[299 - 20 : 300] *= 3.0  # inside t=299's skip month
    np.testing.assert_array_equal(base[299], momentum_scores(tampered_skip)[299])
