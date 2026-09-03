"""Tests for the model stack: loss, layers, MASTER/ungated, trainer plumbing.

Kept on CPU with tiny dimensions — these pin correctness properties
(masking, gate wiring, backward-lookingness of the batcher), not performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from master_us.models.layers import CrossTimeAttention, InterStockAttention, MarketGate
from master_us.models.loss import ic_loss
from master_us.models.master import MASTER, PerStockLSTM

# ------------------------------------------------------------------ #
# ic_loss                                                             #
# ------------------------------------------------------------------ #


def test_ic_loss_perfect_correlation_is_minus_one():
    pred = torch.randn(4, 30)
    loss = ic_loss(pred, pred * 3.0 + 1.0, torch.ones(4, 30, dtype=torch.bool))
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_ic_loss_anticorrelation_is_plus_one():
    pred = torch.randn(4, 30)
    loss = ic_loss(pred, -pred, torch.ones(4, 30, dtype=torch.bool))
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_ic_loss_masked_names_cannot_move_the_loss():
    """Garbage in masked positions must not change the value OR the gradient."""
    torch.manual_seed(0)
    pred = torch.randn(3, 20, requires_grad=True)
    target = torch.randn(3, 20)
    mask = torch.rand(3, 20) > 0.3

    loss_a = ic_loss(pred, target, mask)
    loss_a.backward()
    grad_a = pred.grad.clone()

    pred2 = pred.detach().clone().requires_grad_(True)
    target2 = target.clone()
    target2[~mask] = 999.0  # corrupt everything outside the mask
    loss_b = ic_loss(pred2, target2, mask)
    loss_b.backward()

    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6)
    torch.testing.assert_close(grad_a, pred2.grad)
    assert pred.grad[~mask].abs().max().item() == 0.0  # masked entries get no gradient


def test_ic_loss_is_per_date_not_pooled():
    """Two dates with perfect within-date corr but opposite scales: loss -1.

    A pooled correlation would be dragged off -1 by the scale difference;
    per-date computation is exactly what makes this land at -1.
    """
    base = torch.randn(2, 40)
    target = base.clone()
    target[1] *= 100.0
    loss = ic_loss(base, target, torch.ones(2, 40, dtype=torch.bool))
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_ic_loss_thin_dates_contribute_nothing():
    pred = torch.randn(2, 10)
    target = pred.clone()
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0] = True
    mask[1, 0] = True  # one valid name — below the 2-name floor
    assert ic_loss(pred, target, mask).item() == pytest.approx(-1.0, abs=1e-5)


# ------------------------------------------------------------------ #
# Layers                                                              #
# ------------------------------------------------------------------ #


def test_cross_time_attention_shapes():
    layer = CrossTimeAttention(d_model=32, n_heads=4, n_layers=1, dropout=0.0)
    out = layer(torch.randn(6, 12, 32))
    assert out.shape == (6, 32)


def test_inter_stock_attention_padded_names_do_not_leak():
    """A padded name's value must not influence any real name's output."""
    torch.manual_seed(0)
    layer = InterStockAttention(d_model=16, n_heads=2, n_layers=1, dropout=0.0).eval()
    x = torch.randn(1, 8, 16)
    valid = torch.ones(1, 8, dtype=torch.bool)
    valid[0, 5:] = False

    out_a = layer(x, valid)
    x_b = x.clone()
    x_b[0, 5:] = 777.0  # corrupt the padded names
    out_b = layer(x_b, valid)
    torch.testing.assert_close(out_a[0, :5], out_b[0, :5])


def test_inter_stock_attention_mixes_real_names():
    """The opposite property: changing one REAL name changes the others.

    The perturbation must be non-uniform across dims: the pre-LN layers
    annihilate a constant shift exactly (LN(x + c) == LN(x)), which is a
    property of the normalization, not an absence of mixing.
    """
    torch.manual_seed(0)
    layer = InterStockAttention(d_model=16, n_heads=2, n_layers=1, dropout=0.0).eval()
    x = torch.randn(1, 6, 16)
    valid = torch.ones(1, 6, dtype=torch.bool)
    out_a = layer(x, valid)
    x_b = x.clone()
    x_b[0, 0] = torch.randn(16) * 3.0  # reshape the token, don't just shift it
    out_b = layer(x_b, valid)
    assert (out_a[0, 1:] - out_b[0, 1:]).abs().max().item() > 1e-4


def test_inter_stock_attention_rejects_empty_dates():
    layer = InterStockAttention(d_model=16, n_heads=2)
    with pytest.raises(ValueError, match="no valid names"):
        layer(torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool))


def test_market_gate_bounds_and_temperature():
    torch.manual_seed(0)
    gate = MarketGate(m_dim=10, f_dim=20, hidden=8, beta=1.0)
    g = gate(torch.randn(5, 10))
    assert g.shape == (5, 20)
    assert (g > 0).all() and (g < 1).all()

    # High beta flattens the gate toward 0.5 (uniform); low beta sharpens it.
    hot = MarketGate(m_dim=10, f_dim=20, hidden=8, beta=1000.0)
    hot.load_state_dict(gate.state_dict())
    g_hot = hot(torch.randn(5, 10))
    assert (g_hot - 0.5).abs().max().item() < 0.01

    with pytest.raises(ValueError, match="beta"):
        MarketGate(m_dim=10, f_dim=20, beta=0.0)


# ------------------------------------------------------------------ #
# MASTER / ungated / LSTM                                             #
# ------------------------------------------------------------------ #


def _toy_inputs(b=2, n=7, lookback=5, f=11, m=6, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(b, n, lookback, f)
    mkt = torch.randn(b, m)
    valid = torch.ones(b, n, dtype=torch.bool)
    return x, mkt, valid


def test_master_forward_shape_and_gate_wiring():
    x, m, valid = _toy_inputs()
    gated = MASTER(f_dim=11, m_dim=6, d_model=16, n_heads_temporal=2, n_heads_cross=2,
                   n_layers_temporal=1, n_layers_cross=1, dropout=0.0, use_gate=True)
    ungated = MASTER(f_dim=11, m_dim=6, d_model=16, n_heads_temporal=2, n_heads_cross=2,
                     n_layers_temporal=1, n_layers_cross=1, dropout=0.0, use_gate=False)
    assert gated(x, m, valid).shape == (2, 7)
    assert ungated(x, m, valid).shape == (2, 7)
    assert gated.gate is not None and ungated.gate is None


def test_ungated_ignores_the_market_vector_and_gated_does_not():
    """The defining difference between MASTER and its control."""
    x, m, valid = _toy_inputs()
    kwargs = dict(f_dim=11, m_dim=6, d_model=16, n_heads_temporal=2, n_heads_cross=2,
                  n_layers_temporal=1, n_layers_cross=1, dropout=0.0)
    torch.manual_seed(1)
    ungated = MASTER(use_gate=False, **kwargs).eval()
    torch.manual_seed(1)
    gated = MASTER(use_gate=True, **kwargs).eval()

    m2 = m + 3.0
    torch.testing.assert_close(ungated(x, m, valid), ungated(x, m2, valid))
    assert not torch.allclose(gated(x, m, valid), gated(x, m2, valid))


def test_lstm_ignores_market_and_cross_section():
    """Per-stock: no market input, no name-to-name mixing."""
    x, m, valid = _toy_inputs()
    torch.manual_seed(2)
    net = PerStockLSTM(f_dim=11, hidden=8, n_layers=2, dropout=0.0).eval()

    torch.testing.assert_close(net(x, m, valid), net(x, m * 100, valid))

    x_b = x.clone()
    x_b[:, 0] += 9.0  # perturb one name
    out_a, out_b = net(x, m, valid), net(x_b, m, valid)
    torch.testing.assert_close(out_a[:, 1:], out_b[:, 1:])  # others untouched
    assert not torch.allclose(out_a[:, 0], out_b[:, 0])


def test_master_from_config_matches_yaml_shape():
    import yaml

    from master_us.data.sources import REPO_ROOT

    with (REPO_ROOT / "config" / "master.yaml").open() as fh:
        arch = yaml.safe_load(fh)["architecture"]
    model = MASTER.from_config(f_dim=136, m_dim=67, arch=arch, use_gate=False)
    x = torch.randn(1, 4, 8, 136)
    out = model(x, torch.randn(1, 67), torch.ones(1, 4, dtype=torch.bool))
    assert out.shape == (1, 4)


# ------------------------------------------------------------------ #
# Trainer plumbing                                                    #
# ------------------------------------------------------------------ #


def _tiny_prepared(t=90, n=12, f=4, m=3, lookback=5, seed=0):
    from master_us.models.train import PreparedData

    rng = np.random.default_rng(seed)
    features = rng.standard_normal((t, n, f)).astype(np.float32)
    labels = (0.3 * features[:, :, 0] + rng.standard_normal((t, n))).astype(np.float32)
    labels = (labels - labels.mean(axis=1, keepdims=True)) / labels.std(axis=1, keepdims=True)
    mask = np.ones((t, n), dtype=bool)
    return PreparedData(
        dates=pd.bdate_range("2020-01-01", periods=t),
        features=features,
        market=rng.standard_normal((t, m)).astype(np.float32),
        labels=labels,
        mask=mask,
        raw_forward=labels.copy(),
        lookback=lookback,
        train_idx=np.arange(lookback - 1, 60, dtype=np.int64),
        valid_idx=np.arange(60, 75, dtype=np.int64),
        test_idx=np.arange(75, t, dtype=np.int64),
        feature_names=tuple(f"f{i}" for i in range(f)),
    )


def test_batcher_uses_only_backward_windows():
    """Sample at date t must see features [t-L+1 .. t] and nothing later."""
    from master_us.models.train import _make_batch

    data = _tiny_prepared()
    t_pick = 30
    x, _, _, _, union = _make_batch(
        data, np.array([t_pick], dtype=np.int64), torch.device("cpu")
    )
    expected = data.features[t_pick - data.lookback + 1 : t_pick + 1][:, union].transpose(1, 0, 2)
    np.testing.assert_array_equal(x.numpy()[0], expected)


def test_trainer_learns_a_planted_signal_and_stops_early():
    from master_us.models.train import rank_ic_by_date, train_model

    data = _tiny_prepared()
    net = PerStockLSTM(f_dim=4, hidden=8, n_layers=1, dropout=0.0)
    result = train_model(
        net, data, seed=0, max_epochs=12, patience=4, lr=1e-2,
        device=torch.device("cpu"), group_size=4,
    )
    assert result.best_valid_rank_ic > 0.15  # planted signal has corr ~0.29
    test_ic = float(np.nanmean(
        rank_ic_by_date(result.scores, data.labels, data.label_valid, data.test_idx)
    ))
    assert test_ic > 0.1
    # Scores exist for test dates and nowhere before valid.
    assert np.isnan(result.scores[: data.valid_idx[0]]).all()
    assert np.isfinite(result.scores[data.test_idx[-1]]).any()
