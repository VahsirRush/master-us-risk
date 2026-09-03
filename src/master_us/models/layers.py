"""MASTER building blocks — spec section 7.1.

Built now, in Phase 2, because the ungated baseline must be a true preview of
Phase 3: MASTER minus the gate, identical elsewhere. Nothing here is
throwaway; Phase 3 adds only `use_gate=True` and the beta sweep.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """The standard fixed encoding, added to the lookback axis."""

    pe: Tensor

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # (S, L, d)
        return x + self.pe[: x.shape[1]]


class CrossTimeAttention(nn.Module):
    """Intra-stock temporal attention — the paper's first structural claim.

    Encoder layers mix information ACROSS lookback positions (cross-time, not
    time-aligned), then a single-query attention aggregates: the query is the
    token at the forecast date (the last position), keys/values span the whole
    lookback. Input (S, L, d) -> output (S, d), where S = B*N stocks.
    """

    def __init__(
        self, d_model: int = 128, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.positional = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.aggregate = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(self.positional(x))  # (S, L, d)
        query = h[:, -1:, :]  # the forecast-date token
        pooled, _ = self.aggregate(query, h, h, need_weights=False)
        out: Tensor = self.norm(pooled.squeeze(1) + h[:, -1, :])  # residual, last token
        return out


class InterStockAttention(nn.Module):
    """Attention ACROSS stocks on a single date — momentary peer correlation.

    This is what distinguishes the architecture from a stack of per-stock
    RNNs. Input (B, N, d) with a validity mask (True = real name); padded or
    invalid names neither attend nor are attended to, which is the spec's
    "-inf attention bias" realized as key_padding_mask.
    """

    def __init__(
        self, d_model: int = 128, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.2
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        # key_padding_mask semantics: True = IGNORE. A date with zero valid
        # names would make every key ignored and produce NaN; such dates are
        # excluded upstream, but guard anyway by leaving them untouched.
        if bool((~valid).all(dim=1).any()):
            raise ValueError("a batch row has no valid names — filter dates upstream")
        out: Tensor = self.encoder(x, src_key_padding_mask=~valid)
        return out


class MarketGate(nn.Module):
    """Market-guided feature gating — spec section 7.1.

    m_t -> MLP -> sigmoid(logits / beta) -> gate in [0,1]^F, multiplied into
    the RAW feature vector before embedding. beta is the temperature and the
    paper's key hyperparameter: beta -> 0 approaches hard selection,
    beta -> inf approaches uniform gating (i.e. no gating at all) — which is
    exactly why the ungated control multiplies by nothing rather than by a
    gate with huge beta: the control must remove the mechanism, not weaken it.
    """

    def __init__(self, m_dim: int, f_dim: int, hidden: int = 64, beta: float = 1.0) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta
        self.mlp = nn.Sequential(
            nn.Linear(m_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, f_dim),
        )

    def forward(self, m: Tensor) -> Tensor:  # (B, M) -> (B, F)
        return torch.sigmoid(self.mlp(m) / self.beta)
