"""Tests for Phase 4 — ablation variants, cost analysis, stress slices.

The variant definitions get the most scrutiny: an ablation grid is only worth
running if each cell differs from full MASTER by exactly one thing, and if the
thing it removes is actually removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from master_us.experiments.ablations import (
    BETA_VALUES,
    beta_grid,
    cost_breakeven,
    grid,
    head_kwargs,
    net_sharpe_at_bps,
    resolve_arch,
    shuffled_market,
    with_shuffled_market,
)
from master_us.experiments.stress import Bundle, decay_curve, rank_ic
from master_us.models.master import MASTER
from master_us.models.train import PreparedData

# ------------------------------------------------------------------ #
# Variants isolate exactly one thing                                  #
# ------------------------------------------------------------------ #


def test_every_variant_states_what_it_isolates():
    for variant in [*grid(), *beta_grid()]:
        assert variant.isolates, f"{variant.name} has no stated hypothesis"
        assert variant.name


def test_beta_grid_covers_the_spec_values_and_reuses_beta_one():
    """β=1.0 IS full MASTER — re-running it would waste 5 seeds and, worse,
    produce a second slightly-different 'full MASTER' row."""
    names = {v.name for v in beta_grid()}
    assert names == {f"beta_{b}" for b in BETA_VALUES if b != 1.0}
    assert "beta_1.0" not in names
    assert set(BETA_VALUES) == {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}


def test_no_inter_stock_variant_actually_removes_peer_mixing():
    """Perturbing one name must stop affecting the others."""
    torch.manual_seed(0)
    kw = dict(
        f_dim=6, m_dim=4, d_model=16, n_heads_temporal=2, n_heads_cross=2,
        n_layers_temporal=1, n_layers_cross=1, dropout=0.0,
    )
    ablated = MASTER(use_gate=True, use_inter_stock=False, **kw).eval()
    x = torch.randn(1, 5, 4, 6)
    m = torch.randn(1, 4)
    valid = torch.ones(1, 5, dtype=torch.bool)

    base = ablated(x, m, valid)
    tampered = x.clone()
    tampered[0, 0] = torch.randn(4, 6) * 3.0
    after = ablated(tampered, m, valid)

    torch.testing.assert_close(base[0, 1:], after[0, 1:])  # peers untouched
    assert not torch.allclose(base[0, 0], after[0, 0])  # the perturbed name moved

    # And the full model DOES mix, so the ablation is a real difference.
    torch.manual_seed(0)
    full = MASTER(use_gate=True, use_inter_stock=True, **kw).eval()
    assert not torch.allclose(full(x, m, valid)[0, 1:], full(tampered, m, valid)[0, 1:])


def test_time_aligned_variant_blocks_information_across_time():
    """The cross-time ablation: changing an earlier lookback position must no
    longer move the forecast-date output."""
    torch.manual_seed(0)
    kw = dict(
        f_dim=6, m_dim=4, d_model=16, n_heads_temporal=2, n_heads_cross=2,
        n_layers_temporal=1, n_layers_cross=1, dropout=0.0,
    )
    aligned = MASTER(use_gate=True, cross_time_mode="aligned", **kw).eval()
    x = torch.randn(1, 3, 6, 6)
    m = torch.randn(1, 4)
    valid = torch.ones(1, 3, dtype=torch.bool)

    base = aligned(x, m, valid)
    tampered = x.clone()
    tampered[:, :, :-1] = torch.randn_like(tampered[:, :, :-1]) * 4.0  # all but the last step
    torch.testing.assert_close(base, aligned(tampered, m, valid), atol=1e-5, rtol=1e-5)

    # The cross-time model, by contrast, must react.
    torch.manual_seed(0)
    cross = MASTER(use_gate=True, cross_time_mode="cross", **kw).eval()
    assert not torch.allclose(cross(x, m, valid), cross(tampered, m, valid))


def test_invalid_cross_time_mode_raises():
    with pytest.raises(ValueError, match="cross_time_mode"):
        MASTER(f_dim=4, m_dim=2, cross_time_mode="sideways")


# ------------------------------------------------------------------ #
# Market shuffle                                                      #
# ------------------------------------------------------------------ #


def _prepared(t=60, n=10, f=5, m=4) -> PreparedData:
    rng = np.random.default_rng(0)
    return PreparedData(
        dates=pd.bdate_range("2020-01-01", periods=t),
        features=rng.standard_normal((t, n, f)).astype(np.float32),
        market=rng.standard_normal((t, m)).astype(np.float32),
        labels=rng.standard_normal((t, n)).astype(np.float32),
        mask=np.ones((t, n), dtype=bool),
        raw_forward=rng.standard_normal((t, n)).astype(np.float32) * 0.01,
        lookback=5,
        train_idx=np.arange(4, 40, dtype=np.int64),
        valid_idx=np.arange(40, 50, dtype=np.int64),
        test_idx=np.arange(50, t, dtype=np.int64),
        feature_names=tuple(f"f{i}" for i in range(f)),
    )


def test_market_shuffle_preserves_marginals_and_destroys_alignment():
    """Same rows, different order: every column's distribution is identical,
    but no row lines up with the cross-section it should gate."""
    data = _prepared()
    shuffled = shuffled_market(data, seed=0)

    assert shuffled.shape == data.market.shape
    np.testing.assert_allclose(np.sort(shuffled, axis=0), np.sort(data.market, axis=0))
    assert not np.allclose(shuffled, data.market)


def test_market_shuffle_is_seeded_and_leaves_everything_else_alone():
    data = _prepared()
    np.testing.assert_array_equal(shuffled_market(data, 3), shuffled_market(data, 3))
    assert not np.allclose(shuffled_market(data, 3), shuffled_market(data, 4))

    view = with_shuffled_market(data, 0)
    np.testing.assert_array_equal(view.features, data.features)
    np.testing.assert_array_equal(view.labels, data.labels)
    np.testing.assert_array_equal(view.test_idx, data.test_idx)


def test_head_grid_names_parse_back_to_head_counts():
    assert head_kwargs("heads_16_4") == {"n_heads_temporal": 16, "n_heads_cross": 4}
    arch = resolve_arch(
        type(grid()[0])("heads_8_8", "x", {}), {"d_model": 128, "n_heads_temporal": 8}
    )
    assert arch["n_heads_cross"] == 8


# ------------------------------------------------------------------ #
# §8.4 cost analysis                                                  #
# ------------------------------------------------------------------ #


def test_cost_breakeven_is_the_bps_where_net_alpha_hits_zero():
    """Hand-computable: 10 bps of daily mean gross at 50% turnover.

    bps* = 2e4 * mean(gross) / mean(turnover) = 2e4 * 0.0010 / 0.50 = 40.

    The curve check needs a series with VARIANCE — a constant return series
    has zero std, which makes Sharpe degenerate rather than sign-flipping.
    """
    rng = np.random.default_rng(0)
    gross = rng.normal(0.0010, 0.008, 2520)
    gross += 0.0010 - gross.mean()  # pin the mean exactly
    turnover = np.full(2520, 0.50)

    assert cost_breakeven(gross, turnover) == pytest.approx(40.0, rel=1e-6)
    # The net-Sharpe curve crosses zero at the same place.
    assert net_sharpe_at_bps(gross, turnover, 30.0) > 0
    assert net_sharpe_at_bps(gross, turnover, 39.0) > 0
    assert net_sharpe_at_bps(gross, turnover, 41.0) < 0
    assert net_sharpe_at_bps(gross, turnover, 50.0) < 0


def test_cost_breakeven_edge_cases():
    assert cost_breakeven(np.full(10, -0.001), np.full(10, 0.5)) == 0.0  # no gross alpha
    assert cost_breakeven(np.full(10, 0.001), np.zeros(10)) == float("inf")  # trades nothing


def test_net_sharpe_declines_monotonically_with_cost():
    rng = np.random.default_rng(0)
    gross = rng.normal(5e-4, 0.01, 500)
    turnover = np.full(500, 0.6)
    curve = [net_sharpe_at_bps(gross, turnover, b) for b in (0, 5, 10, 20, 50)]
    assert curve == sorted(curve, reverse=True)


# ------------------------------------------------------------------ #
# §8.5 stress slices                                                  #
# ------------------------------------------------------------------ #


def _bundle(t=300, n=40) -> Bundle:
    rng = np.random.default_rng(1)
    labels = rng.standard_normal((t, n)).astype(np.float32)
    return Bundle(
        dates=pd.bdate_range("2019-01-01", periods=t),
        labels=labels,
        mask=np.ones((t, n), dtype=bool),
        raw_forward=(labels * 0.01).astype(np.float32),
        test_idx=np.arange(50, t, dtype=np.int64),
        tickers=np.array([f"T{i}" for i in range(n)], dtype=object),
        sectors=np.array(["Tech" if i % 2 else "Health" for i in range(n)], dtype=object),
        mcap=np.tile(np.linspace(1e9, 1e12, n).astype(np.float32), (t, 1)),
    )


def test_decay_curve_is_highest_at_the_horizon_the_signal_targets():
    """A score built to predict the 1-day return must decay as horizon grows."""
    bundle = _bundle()
    scores = bundle.raw_forward.copy()  # perfect 1-day foresight
    curve = decay_curve(bundle, scores)
    assert set(curve) == {1, 3, 5, 10, 21}
    assert curve[1] > curve[21], f"no decay: {curve}"
    assert curve[1] > 0.9


def test_rank_ic_skips_thin_dates():
    bundle = _bundle(t=60, n=40)
    scores = bundle.labels.copy()
    thin = bundle.mask.copy()
    thin[:, 5:] = False  # only 5 names -> below the floor
    assert np.isnan(rank_ic(scores, bundle.labels, thin, bundle.test_idx))
