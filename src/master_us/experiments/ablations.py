"""Phase 4 — the ablation grid and β sweep (spec §8.1, §8.2).

READ `reports/framing.md` FIRST. This is not one table among several: Phase 2
showed the ungated transformer already matches the strongest baseline, and
Phase 3 could not distinguish full MASTER from ungated at 5 seeds (RankIC gap
0.001063 against a pooled threshold of 0.001121 — a 5% near-miss, with all
three measures same-signed). So the grid below exists to resolve one question:
**does the market gate do anything?** The β sweep against the no-gating
reference is the experiment that answers it.

TRAINING BUDGET — decided once, applied to every cell (see NOTES Session 9):
12 epochs, patience 4, lookback 20 — the same budget as every deep row already
in the six-model table. An ablation grid whose cells differ in training budget
measures budget, not mechanism. The cap is empirically not binding: MASTER's
best epochs across 5 seeds were 1, 1, 2, 3, 1, with runs self-terminating at
6-8 epochs. A separate, clearly-labelled confirmatory pair runs at the spec's
100/10 to check the null does not flip.

Every variant differs from full MASTER by exactly one thing. `market_shuffled`
is the exception that proves the rule: identical parameters, destroyed input —
it asks whether the gate learns real market structure or merely benefits from
having extra parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from master_us.models.master import MASTER
from master_us.models.train import PreparedData

# The reference cells already measured in Phases 2-3, reused rather than re-run.
ALREADY_MEASURED = {"master": "full MASTER", "ungated": "- market gating"}

BETA_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Variant:
    """One ablation cell: what it is, and what it isolates."""

    name: str
    isolates: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    shuffle_market: bool = False
    lookback: int | None = None  # None = the grid default

    def build(self, f_dim: int, m_dim: int, arch: dict[str, Any]) -> MASTER:
        return MASTER.from_config(f_dim=f_dim, m_dim=m_dim, arch=arch, **self.model_kwargs)


def grid() -> list[Variant]:
    """Spec §8.1, minus the two cells Phases 2-3 already measured."""
    return [
        Variant(
            "no_inter_stock",
            "cross-sectional modelling: is peer attention load-bearing?",
            {"use_gate": True, "use_inter_stock": False},
        ),
        Variant(
            "time_aligned",
            "the cross-time claim: each position attends only to itself",
            {"use_gate": True, "cross_time_mode": "aligned"},
        ),
        Variant(
            "market_shuffled",
            "whether the gate learns real market signal (same params, destroyed input)",
            {"use_gate": True},
            shuffle_market=True,
        ),
    ]


def beta_grid() -> list[Variant]:
    """§8.2. β=1.0 is full MASTER and is reused from Phase 3, not re-run."""
    return [
        Variant(
            f"beta_{beta}",
            f"gate temperature {beta}: 0 -> hard selection, inf -> no gating",
            {"use_gate": True, "beta": beta},
        )
        for beta in BETA_VALUES
        if beta != 1.0
    ]


def confirmatory_pair() -> list[Variant]:
    """The two rows the headline depends on, re-run at the spec's full budget.

    Phase 3-4 ran everything at 12 epochs / patience 4 so no cell differed
    from another in training budget. That is right for the grid and leaves one
    question open: does the gate null hold when both models train to the
    spec's 100/10? These two cells answer it. They are named separately so the
    short-budget rows they must be compared against are never overwritten —
    the comparison that matters is master_full vs ungated_full, NOT
    master_full vs the short-budget ungated.
    """
    return [
        Variant("ungated_full", "no gating, spec 100/10 budget", {"use_gate": False}),
        Variant("master_full", "full MASTER, spec 100/10 budget", {"use_gate": True}),
    ]


# §8.1 rows 6-7, exactly as config/master.yaml declares them. L=20 and
# (8, 4) are the grid defaults and are already measured as master/ungated,
# so only the remaining points are generated here.
LOOKBACK_VALUES = (20, 40, 60, 120)
HEAD_VALUES = ((4, 2), (8, 4), (8, 8), (16, 4))


def lookback_grid() -> list[Variant]:
    """§8.1 row 6, BOTH arms at each point.

    Each lookback gets a gated and an ungated cell, because the question is
    not only "does memory horizon matter" but "does the gate's (null) value
    depend on it". A gated-only sweep could not answer the second.
    """
    out: list[Variant] = []
    for lb in LOOKBACK_VALUES:
        if lb == 20:
            continue  # already measured as master_full / ungated_full
        out += [
            Variant(f"lb{lb}_gated", f"memory horizon {lb}d, gated", {"use_gate": True}, lookback=lb),
            Variant(f"lb{lb}_ungated", f"memory horizon {lb}d, no gate", {"use_gate": False}, lookback=lb),
        ]
    return out


def head_grid() -> list[Variant]:
    """§8.1 row 7, the paper's (N1, N2) sweep — BOTH arms at each point."""
    out: list[Variant] = []
    for n1, n2 in HEAD_VALUES:
        if (n1, n2) == (8, 4):
            continue  # the default, already measured
        out += [
            Variant(f"heads_{n1}_{n2}_gated", f"{n1}/{n2} heads, gated", {"use_gate": True}),
            Variant(f"heads_{n1}_{n2}_ungated", f"{n1}/{n2} heads, no gate", {"use_gate": False}),
        ]
    return out


def shuffled_market(
    data: PreparedData, seed: int
) -> npt.NDArray[np.float32]:
    """The market matrix with its DATE ordering permuted.

    Shuffling rows (not columns) keeps every marginal distribution and every
    cross-feature correlation intact while destroying the alignment between
    market state and the cross-section it is supposed to gate. If the gate
    still helps with this input, it was never reading market structure.

    Permuted within the training span only? No — the whole matrix, because the
    gate consumes it at every split. The permutation is seeded so a variant is
    reproducible.
    """
    rng = np.random.default_rng(10_000 + seed)
    return data.market[rng.permutation(data.market.shape[0])]


def with_shuffled_market(data: PreparedData, seed: int) -> PreparedData:
    return PreparedData(
        dates=data.dates,
        features=data.features,
        market=shuffled_market(data, seed),
        labels=data.labels,
        mask=data.mask,
        raw_forward=data.raw_forward,
        lookback=data.lookback,
        train_idx=data.train_idx,
        valid_idx=data.valid_idx,
        test_idx=data.test_idx,
        feature_names=data.feature_names,
    )


def head_kwargs(name: str) -> dict[str, Any]:
    """(n1, n2) parsed back out of a `heads_n1_n2_arm` variant name."""
    parts = name.split("_")
    return {"n_heads_temporal": int(parts[1]), "n_heads_cross": int(parts[2])}


def resolve_arch(variant: Variant, arch: dict[str, Any]) -> dict[str, Any]:
    """The architecture dict for a variant, with head counts applied."""
    out = dict(arch)
    if variant.name.startswith("heads_"):
        out.update(head_kwargs(variant.name))
    return out


def cost_breakeven(
    gross_daily: npt.NDArray[np.floating],
    turnover_daily: npt.NDArray[np.floating],
) -> float:
    """The bps level at which net alpha reaches zero — spec §8.4.

    net_return(bps) = mean(gross) - mean(turnover) * (bps/2) * 1e-4
    (one side of a round trip per unit of one-way turnover, matching
    `costs.flat_cost`). Solving for zero:

        bps* = 2e4 * mean(gross) / mean(turnover)

    Returns 0.0 when gross alpha is already non-positive — such a strategy has
    no cost budget at all — and inf when it trades nothing.
    """
    mean_gross = float(np.mean(gross_daily))
    mean_turnover = float(np.mean(turnover_daily))
    if mean_gross <= 0:
        return 0.0
    if mean_turnover <= 0:
        return float("inf")
    return 2e4 * mean_gross / mean_turnover


def net_sharpe_at_bps(
    gross_daily: npt.NDArray[np.floating],
    turnover_daily: npt.NDArray[np.floating],
    bps: float,
    trading_days: int = 252,
) -> float:
    """Annualized net Sharpe under an assumed round-trip cost — the §8.4 curve."""
    net = np.asarray(gross_daily, float) - np.asarray(turnover_daily, float) * (bps / 2.0) * 1e-4
    sd = net.std(ddof=1)
    return float(net.mean() / sd * np.sqrt(trading_days)) if sd > 0 else 0.0


VariantRunner = Callable[[Variant, int], npt.NDArray[np.float32]]
