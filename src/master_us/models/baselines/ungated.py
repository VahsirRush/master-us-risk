"""The ungated transformer — MASTER minus the gate, identical elsewhere.

The critical control: it inherits the full architecture config from
master.yaml so that Phase 3's only delta is `use_gate=True`.
"""

from __future__ import annotations

from typing import Any

from master_us.models.master import MASTER


def make_ungated(f_dim: int, m_dim: int, arch: dict[str, Any]) -> MASTER:
    return MASTER.from_config(f_dim=f_dim, m_dim=m_dim, arch=arch, use_gate=False)
