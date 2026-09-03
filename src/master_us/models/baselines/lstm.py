"""Per-stock LSTM factory — isolates the value of temporal modeling alone."""

from __future__ import annotations

from typing import Any

from master_us.models.master import PerStockLSTM


def make_lstm(f_dim: int, cfg: dict[str, Any]) -> PerStockLSTM:
    return PerStockLSTM(
        f_dim=f_dim,
        hidden=int(cfg.get("hidden", 128)),
        n_layers=int(cfg.get("n_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
