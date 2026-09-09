"""The MASTER model — spec section 7.2 — and its ungated control.

Forward path:
    x (B, N, L, F), m (B, M), valid (B, N)
    -> gate = MarketGate(m)                      (B, F)     [skipped if ungated]
    -> x = x * gate[:, None, None, :]            feature selection
    -> h = Linear(F -> d)(x)                     (B, N, L, d)
    -> h = CrossTimeAttention(h)                 (B, N, d)
    -> h = InterStockAttention(h, valid)         (B, N, d)
    -> y = Linear(d -> 1)(h).squeeze(-1)         (B, N)

`use_gate=False` is the Phase-2 "ungated transformer" baseline: MASTER minus
the gate, identical elsewhere — the critical control for whether the gating
mechanism itself, rather than the attention stack, carries any value.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from master_us.models.layers import CrossTimeAttention, InterStockAttention, MarketGate


class MASTER(nn.Module):
    def __init__(
        self,
        f_dim: int,
        m_dim: int,
        d_model: int = 128,
        n_heads_temporal: int = 8,
        n_heads_cross: int = 4,
        n_layers_temporal: int = 2,
        n_layers_cross: int = 1,
        dropout: float = 0.2,
        gate_hidden: int = 64,
        beta: float = 1.0,
        use_gate: bool = True,
        use_inter_stock: bool = True,
        cross_time_mode: str = "cross",
    ) -> None:
        super().__init__()
        if cross_time_mode not in ("cross", "aligned"):
            raise ValueError(f"cross_time_mode must be 'cross' or 'aligned', got {cross_time_mode!r}")
        self.use_gate = use_gate
        self.use_inter_stock = use_inter_stock
        self.cross_time_mode = cross_time_mode
        self.gate = MarketGate(m_dim, f_dim, hidden=gate_hidden, beta=beta) if use_gate else None
        self.embed = nn.Linear(f_dim, d_model)
        self.cross_time = CrossTimeAttention(
            d_model, n_heads_temporal, n_layers_temporal, dropout, mode=cross_time_mode
        )
        self.inter_stock = InterStockAttention(d_model, n_heads_cross, n_layers_cross, dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, m: Tensor, valid: Tensor) -> Tensor:
        b, n, lookback, _f = x.shape
        if self.gate is not None:
            x = x * self.gate(m)[:, None, None, :]
        h = self.embed(x)  # (B, N, L, d)
        h = self.cross_time(h.reshape(b * n, lookback, -1)).reshape(b, n, -1)
        if self.use_inter_stock:
            h = self.inter_stock(h, valid)
        out: Tensor = self.head(h).squeeze(-1)
        return out

    @classmethod
    def from_config(
        cls,
        f_dim: int,
        m_dim: int,
        arch: dict[str, Any],
        use_gate: bool = True,
        use_inter_stock: bool = True,
        cross_time_mode: str = "cross",
        beta: float | None = None,
        lookback_unused: int | None = None,
    ) -> MASTER:
        gate_cfg = arch.get("gate", {})
        return cls(
            f_dim=f_dim,
            m_dim=m_dim,
            d_model=int(arch.get("d_model", 128)),
            n_heads_temporal=int(arch.get("n_heads_temporal", 8)),
            n_heads_cross=int(arch.get("n_heads_cross", 4)),
            n_layers_temporal=int(arch.get("n_layers_temporal", 2)),
            n_layers_cross=int(arch.get("n_layers_cross", 1)),
            dropout=float(arch.get("dropout", 0.2)),
            gate_hidden=int(gate_cfg.get("hidden", 64)),
            beta=float(gate_cfg.get("beta", 1.0)) if beta is None else beta,
            use_gate=use_gate,
            use_inter_stock=use_inter_stock,
            cross_time_mode=cross_time_mode,
        )


class PerStockLSTM(nn.Module):
    """The temporal-only baseline: 2-layer LSTM per stock, no cross-section.

    Shares the (x, m, valid) interface so one trainer drives every deep
    model; `m` is accepted and ignored — that is the point of the baseline.
    """

    def __init__(
        self, f_dim: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=f_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, m: Tensor, valid: Tensor) -> Tensor:
        b, n, lookback, f = x.shape
        h, _ = self.lstm(x.reshape(b * n, lookback, f))
        out: Tensor = self.head(h[:, -1, :]).reshape(b, n)
        return out


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover — no CUDA on this machine
        return torch.device("cuda")
    return torch.device("cpu")
