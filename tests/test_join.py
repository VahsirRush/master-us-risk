"""Phase 7 — the join.

Pure synthetic; nothing here loads the real panel, so no `pytestmark =
pytest.mark.heavy` and all of it runs beside a live job.

The load-bearing tests are the alignment ones. A misalignment between
weights, exposures and factor returns would not raise — it would quietly
attribute a book's P&L to factor returns it never earned, and every number
downstream would look plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_us.experiments.join import (
    aggregate,
    align_to_risk_grid,
    attribute,
    breakeven_bps,
    net_of_costs,
    sharpe,
    zero_metric,
)

# --------------------------------------------------------------------- #
# alignment                                                              #
# --------------------------------------------------------------------- #


def test_align_maps_onto_the_panel_grid() -> None:
    bundle_d = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    bundle_t = np.array(["AAA", "BBB", "CCC"], dtype=object)
    panel_d = pd.to_datetime(["2020-01-03", "2020-01-01"])
    panel_t = np.array(["CCC", "AAA"], dtype=object)

    rows, cols = align_to_risk_grid(bundle_d, bundle_t, panel_d, panel_t)
    data = np.arange(9).reshape(3, 3)
    assert data[rows][:, cols].tolist() == [[8, 6], [2, 0]]


def test_align_refuses_a_missing_date() -> None:
    """A silent drop would shrink the book and flatter every ratio from it."""
    with pytest.raises(KeyError, match="dates absent"):
        align_to_risk_grid(
            pd.to_datetime(["2020-01-01"]), np.array(["AAA"], dtype=object),
            pd.to_datetime(["2020-01-02"]), np.array(["AAA"], dtype=object),
        )


def test_align_refuses_a_missing_ticker() -> None:
    with pytest.raises(KeyError, match="tickers absent"):
        align_to_risk_grid(
            pd.to_datetime(["2020-01-01"]), np.array(["AAA"], dtype=object),
            pd.to_datetime(["2020-01-01"]), np.array(["ZZZ"], dtype=object),
        )


# --------------------------------------------------------------------- #
# attribution — §10.2, §10.3                                             #
# --------------------------------------------------------------------- #


def _setup(t: int = 200, n: int = 30, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    exposures = np.tile(rng.normal(size=(n, k)), (t, 1, 1))
    factor_returns = rng.normal(0, 0.01, (t, k))
    specific = rng.normal(0, 0.02, (t, n))
    asset_returns = np.einsum("tnk,tk->tn", exposures, factor_returns) + specific
    weights = np.tile(rng.normal(size=n) / n, (t, 1))
    cov = np.tile(np.eye(k) * 1e-4, (t, 1, 1))
    svar = np.full((t, n), 4e-4)
    dates = pd.date_range("2020-01-01", periods=t, freq="B")
    return weights, exposures, factor_returns, asset_returns, cov, svar, dates


def test_attribution_reconstructs_realized_pnl_exactly() -> None:
    """factor + specific must equal the book's realized return, to machine
    precision. They are not separately modelled — specific is the residual —
    so any drift here means an indexing error, not an estimation error."""
    w, x, f, r, cov, d, dates = _setup()
    at = attribute(w, x, f, r, cov, d, dates)
    ok = np.isfinite(at.total_pnl)
    resid = at.total_pnl[ok] - (at.factor_pnl[ok] + at.specific_pnl[ok])
    assert np.abs(resid).max() < 1e-12


def test_attribution_recovers_a_pure_factor_book() -> None:
    """A book built with zero idiosyncratic return must attribute ~entirely
    to factors."""
    rng = np.random.default_rng(1)
    t, n, k = 200, 30, 3
    exposures = np.tile(rng.normal(size=(n, k)), (t, 1, 1))
    f = rng.normal(0, 0.01, (t, k))
    asset_returns = np.einsum("tnk,tk->tn", exposures, f)  # no specific term
    w = np.tile(rng.normal(size=n) / n, (t, 1))
    cov = np.tile(np.eye(k) * 1e-4, (t, 1, 1))
    at = attribute(w, exposures, f, asset_returns, cov, np.full((t, n), 1e-8),
                   pd.date_range("2020-01-01", periods=t, freq="B"))
    assert np.abs(at.specific_pnl).max() < 1e-12
    assert np.corrcoef(at.factor_pnl, at.total_pnl)[0, 1] > 0.999


def test_book_exposure_is_the_weighted_sum_of_asset_exposures() -> None:
    w, x, f, r, cov, d, dates = _setup()
    at = attribute(w, x, f, r, cov, d, dates)
    assert at.exposures[0] == pytest.approx(w[0] @ x[0], rel=1e-10)


def test_misaligned_factor_returns_destroy_the_attribution() -> None:
    """The test that makes the alignment claim meaningful.

    Shifting the factor returns by one period must visibly degrade how well
    the factor component tracks the book. If this passed either way, the
    alignment would be unverified.
    """
    w, x, f, r, cov, d, dates = _setup(seed=3)
    good = attribute(w, x, f, r, cov, d, dates)
    bad = attribute(w, x, np.roll(f, 1, axis=0), r, cov, d, dates)
    ok = np.isfinite(good.factor_pnl) & np.isfinite(bad.factor_pnl)
    c_good = np.corrcoef(good.factor_pnl[ok], good.total_pnl[ok])[0, 1]
    c_bad = np.corrcoef(bad.factor_pnl[ok], bad.total_pnl[ok])[0, 1]
    assert c_good > c_bad + 0.3


def test_variance_shares_include_the_covariance_term() -> None:
    """var(f) + var(s) need not sum to var(total); the cross term is real.

    Reporting the two shares alone would leave a reader with numbers that
    do not add to 100% and no explanation.
    """
    w, x, f, r, cov, d, dates = _setup(seed=5)
    at = attribute(w, x, f, r, cov, d, dates)
    fs, ss = at.shares()
    ok = np.isfinite(at.factor_pnl) & np.isfinite(at.specific_pnl)
    # ddof=0 to match `shares()`, which uses np.var; the decomposition
    # identity var(f+s) = var(f) + var(s) + 2cov(f,s) only closes exactly
    # when both terms use the same convention.
    cross = 2 * np.cov(at.factor_pnl[ok], at.specific_pnl[ok], ddof=0)[0, 1] / np.var(
        at.factor_pnl[ok] + at.specific_pnl[ok]
    )
    assert fs + ss + cross == pytest.approx(1.0, abs=1e-9)


def test_risk_attribution_shares_are_bounded() -> None:
    w, x, f, r, cov, d, dates = _setup(seed=7)
    at = attribute(w, x, f, r, cov, d, dates)
    share = at.factor_var_share[np.isfinite(at.factor_var_share)]
    assert ((share >= 0) & (share <= 1)).all()


def test_zero_weight_book_produces_no_attribution() -> None:
    w, x, f, r, cov, d, dates = _setup()
    at = attribute(np.zeros_like(w), x, f, r, cov, d, dates)
    assert not np.isfinite(at.factor_pnl).any()


# --------------------------------------------------------------------- #
# costs and the MetricValue contract                                     #
# --------------------------------------------------------------------- #


def test_net_of_costs_matches_the_project_convention() -> None:
    """One side of a round trip per unit of one-way turnover — the same
    formula as `costs.flat_cost` and `ablations.cost_breakeven`."""
    gross = np.array([0.001, 0.002])
    turn = np.array([1.0, 0.5])
    net = net_of_costs(gross, turn, bps=10.0)
    assert net == pytest.approx(gross - turn * 5e-4)


def test_breakeven_is_zero_when_gross_alpha_is_negative() -> None:
    assert breakeven_bps(np.array([-0.001, -0.002]), np.array([1.0, 1.0])) == 0.0


def test_breakeven_matches_the_closed_form() -> None:
    gross = np.full(10, 0.0005)
    turn = np.full(10, 1.0)
    assert breakeven_bps(gross, turn) == pytest.approx(2e4 * 0.0005 / 1.0)


def test_sharpe_is_scale_invariant() -> None:
    """The neutralized book is rescaled to a gross target; that must not
    move its Sharpe, or the rescaling would be changing the answer."""
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 500)
    assert sharpe(r) == pytest.approx(sharpe(r * 7.3), rel=1e-12)


def test_net_sharpe_is_invariant_to_gross_rescaling() -> None:
    """Gross return, cost and volatility all scale together under a linear
    cost model, so rescaling the book cannot manufacture net Sharpe."""
    rng = np.random.default_rng(4)
    gross = rng.normal(0.0002, 0.01, 500)
    turn = np.abs(rng.normal(1.0, 0.1, 500))
    a = sharpe(net_of_costs(gross, turn))
    b = sharpe(net_of_costs(gross * 3, turn * 3))
    assert a == pytest.approx(b, rel=1e-10)


def test_aggregate_gives_net_its_own_dispersion() -> None:
    """Net std must come from the net series, never be derived from gross —
    the Session 6 contract correction that `net_distinguishable_from` needs."""
    mv = aggregate([1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
    assert mv.std > 0
    assert mv.net_std == pytest.approx(0.0)
    assert mv.n_seeds == 3


def test_distinguishable_from_zero_uses_own_dispersion() -> None:
    """A Sharpe of -0.5 +- 0.05 is distinguishable from zero; -0.5 +- 2.0 is not."""
    tight = aggregate([-0.5] * 5, [-0.5, -0.45, -0.55, -0.5, -0.5])
    loose = aggregate([-0.5] * 5, [-0.5, 2.0, -3.0, 1.5, -2.0])
    null = zero_metric(5)
    assert tight.net_distinguishable_from(null)
    assert not loose.net_distinguishable_from(null)
