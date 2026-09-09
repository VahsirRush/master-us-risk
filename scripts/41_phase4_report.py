#!/usr/bin/env python
"""Phase 4 report: ablation table, β sweep, cost breakeven, stress tests.

    reports/phase4.md
    reports/figures/beta_sweep.png
    reports/figures/cost_curve.png
    reports/status/phase4.json

Runs off `data/processed/phase4/metrics_bundle.npz` (~21 MB), never the panel,
so it is safe beside a live training job. Reports whatever variants have
cached scores; missing cells are shown as missing rather than assumed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import yaml

from master_us.backtest.construct import decile_long_short, topk_dropout
from master_us.backtest.costs import CostConfig
from master_us.backtest.engine import run_backtest
from master_us.backtest.metrics import summarize
from master_us.data.sources import REPO_ROOT
from master_us.experiments.ablations import BETA_VALUES, cost_breakeven, net_sharpe_at_bps
from master_us.experiments.phase2 import PHASE2_DIR
from master_us.experiments.stress import (
    Bundle,
    cap_tiers,
    decay_curve,
    load_bundle,
    rank_ic,
    regime_slices,
    sector_neutralized,
)
from master_us.reporting.results import MetricValue

REPORT = REPO_ROOT / "reports" / "phase4.md"
FIGURES = REPO_ROOT / "reports" / "figures"

REFERENCE = {"master": "Full MASTER", "ungated": "minus market gating"}


def seeds_for(tag: str) -> list[int]:
    return sorted(
        int(p.stem.split("_seed")[1].split("_")[0])
        for p in PHASE2_DIR.glob(f"{tag}_seed*_scores.npy")
    )


def evaluate(bundle: Bundle, tag: str, cost_cfg: CostConfig) -> dict[str, MetricValue] | None:
    """Per-seed RankIC and both books, aggregated across whatever seeds exist."""
    seeds = seeds_for(tag)
    if not seeds:
        return None

    rows: dict[str, list[float]] = {
        k: [] for k in ("rank_ic", "ls_gross", "ls_net", "lo_gross", "lo_net", "turnover",
                        "ls_turnover", "breakeven")
    }
    test = bundle.test_idx
    daily = np.full_like(bundle.raw_forward, np.nan)
    daily[1:] = bundle.raw_forward[:-1]

    for seed in seeds:
        scores = np.load(PHASE2_DIR / f"{tag}_seed{seed}_scores.npy")
        rows["rank_ic"].append(rank_ic(scores, bundle.labels, bundle.label_valid, test))

        common = {
            "dates": bundle.dates[test],
            "returns": daily[test].astype(np.float64),
            "scores": scores[test].astype(np.float64),
            "mask": bundle.mask[test],
            "rebalance_idx": np.arange(len(test), dtype=np.int64),
            "cost_cfg": cost_cfg,
            "tier": "baseline",
        }
        lo_res = run_backtest(
            constructor=lambda s, m, p: topk_dropout(s, m, p, k=50, dropout_buffer=25), **common
        )
        ls_res = run_backtest(
            constructor=lambda s, m, p: decile_long_short(s, m, n_deciles=10), **common
        )
        lo, ls = summarize(lo_res.series), summarize(ls_res.series)
        rows["lo_gross"].append(lo["sharpe"].gross)
        rows["lo_net"].append(lo["sharpe"].net or 0.0)
        rows["turnover"].append(lo["turnover"].gross)
        rows["ls_gross"].append(ls["sharpe"].gross)
        rows["ls_net"].append(ls["sharpe"].net or 0.0)
        rows["ls_turnover"].append(ls["turnover"].gross)
        rows["breakeven"].append(
            cost_breakeven(ls_res.series.gross, ls_res.series.turnover)
        )

    n = len(seeds)

    def mv(gross_key: str, net_key: str | None = None) -> MetricValue:
        g = np.asarray(rows[gross_key])
        net = np.asarray(rows[net_key]) if net_key else g
        return MetricValue(
            gross=float(g.mean()),
            net=float(net.mean()),
            std=float(g.std(ddof=1)) if n > 1 else 0.0,
            n_seeds=n,
            net_std=float(net.std(ddof=1)) if n > 1 else 0.0,
        )

    return {
        "rank_ic": mv("rank_ic"),
        "ls_sharpe": mv("ls_gross", "ls_net"),
        "lo_sharpe": mv("lo_gross", "lo_net"),
        "turnover": mv("turnover"),
        "ls_turnover": mv("ls_turnover"),
        "breakeven_bps": mv("breakeven"),
    }


def verdict(a: MetricValue, b: MetricValue, basis: str) -> str:
    real = a.net_distinguishable_from(b) if basis == "net" else a.distinguishable_from(b)
    return "distinguishable" if real else "NOT distinguishable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=None)
    args = parser.parse_args()

    with (REPO_ROOT / "config" / "costs.yaml").open() as fh:
        cost_cfg = CostConfig.from_yaml(yaml.safe_load(fh))
    bundle = load_bundle()

    beta_tags = [f"beta_{b}" for b in BETA_VALUES if b != 1.0]
    ablation_tags = ["no_inter_stock", "time_aligned", "market_shuffled"]
    tags = args.tags or ["master", "ungated", *ablation_tags, *beta_tags]

    tables = {t: e for t in tags if (e := evaluate(bundle, t, cost_cfg)) is not None}
    missing = [t for t in tags if t not in tables]
    print(f"evaluated {len(tables)} variants; missing {missing or 'none'}")

    lines: list[str] = [
        "# Phase 4 — ablation grid, β sweep, cost analysis, stress tests",
        "",
        "Read `reports/framing.md` first. The question this grid resolves is whether",
        "the market gate does anything over an ungated architecture already at parity",
        "with the strongest baseline.",
        "",
        "All cells: 5 seeds, 12 epochs / patience 4 / lookback 20 — the same budget as",
        "every deep row in the six-model table. See NOTES Session 9 for the decision.",
        "",
        "## Ablation table",
        "",
        "| Variant | n | RankIC | L/S Sharpe (gross → net) | L/S turn | Breakeven bps |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for tag, t in tables.items():
        ic, ls = t["rank_ic"], t["ls_sharpe"]
        lines.append(
            f"| {REFERENCE.get(tag, tag)} | {ic.n_seeds} | {ic.gross:+.4f} ±{ic.std:.4f} "
            f"| {ls.gross:+.2f} → {ls.net:+.2f} | {t['ls_turnover'].gross:.0%} "
            f"| {t['breakeven_bps'].gross:.1f} |"
        )

    if "master" in tables:
        lines += ["", "## Every variant vs full MASTER", "",
                  "| Variant | ΔRankIC | gross verdict | Δnet L/S Sharpe | net verdict |",
                  "|:---|---:|:---|---:|:---|"]
        ref = tables["master"]
        for tag, t in tables.items():
            if tag == "master":
                continue
            d_ic = t["rank_ic"].gross - ref["rank_ic"].gross
            d_net = (t["ls_sharpe"].net or 0) - (ref["ls_sharpe"].net or 0)
            lines.append(
                f"| {REFERENCE.get(tag, tag)} | {d_ic:+.4f} "
                f"| {verdict(t['rank_ic'], ref['rank_ic'], 'gross')} | {d_net:+.3f} "
                f"| {verdict(t['ls_sharpe'], ref['ls_sharpe'], 'net')} |"
            )

    # ---- β sweep -------------------------------------------------- #
    beta_rows = [(b, tables.get("master" if b == 1.0 else f"beta_{b}")) for b in BETA_VALUES]
    have_beta = [(b, t) for b, t in beta_rows if t is not None]
    if have_beta and "ungated" in tables:
        lines += ["", "## β sweep (§8.2)", "",
                  "Horizontal reference = the no-gating variant.", "",
                  "| β | RankIC | vs ungated | verdict |", "|---:|---:|---:|:---|"]
        u = tables["ungated"]["rank_ic"]
        for b, t in have_beta:
            ic = t["rank_ic"]
            lines.append(
                f"| {b} | {ic.gross:+.4f} ±{ic.std:.4f} | {ic.gross - u.gross:+.4f} "
                f"| {verdict(ic, u, 'gross')} |"
            )
        lines.append(f"| — (no gating) | {u.gross:+.4f} ±{u.std:.4f} | — | reference |")
        _plot_beta(have_beta, u)

    # ---- stress tests --------------------------------------------- #
    lines += ["", "## Stress tests (§8.5) — full MASTER vs ungated", ""]
    for tag in ("master", "ungated"):
        if tag not in tables:
            continue
        seeds = seeds_for(tag)
        stacked = np.nanmean(
            [np.load(PHASE2_DIR / f"{tag}_seed{s}_scores.npy") for s in seeds], axis=0
        )
        regimes = regime_slices(bundle, stacked)
        tiers = cap_tiers(bundle, stacked)
        decay = decay_curve(bundle, stacked)
        neutral = rank_ic(
            sector_neutralized(bundle, stacked), bundle.labels, bundle.label_valid,
            bundle.test_idx,
        )
        lines += [
            f"### {REFERENCE.get(tag, tag)} (seed-averaged scores)",
            "",
            "| slice | RankIC |", "|:---|---:|",
            *[
                f"| {k} | outside test split (0 dates) |"
                if n == 0
                else f"| {k} | {v:+.4f} ({n} dates) |"
                for k, (v, n) in regimes.items()
            ],
            *[f"| cap tier: {k} | {v:+.4f} |" for k, v in tiers.items()],
            f"| sector-neutralized | {neutral:+.4f} |",
            *[f"| decay h={h}d | {v:+.4f} |" for h, v in decay.items()],
            "",
        ]

    # ---- cost curve ------------------------------------------------ #
    if tables:
        lines += ["", "## Cost breakeven (§8.4)", "",
                  "bps at which net alpha reaches zero, dollar-neutral decile book.", "",
                  "| Variant | breakeven bps | net Sharpe @0 | @10 | @20 | @50 |",
                  "|:---|---:|---:|---:|---:|---:|"]
        for tag, t in tables.items():
            seeds = seeds_for(tag)
            curves = []
            for seed in seeds:
                scores = np.load(PHASE2_DIR / f"{tag}_seed{seed}_scores.npy")
                daily = np.full_like(bundle.raw_forward, np.nan)
                daily[1:] = bundle.raw_forward[:-1]
                test = bundle.test_idx
                res = run_backtest(
                    dates=bundle.dates[test],
                    returns=daily[test].astype(np.float64),
                    scores=scores[test].astype(np.float64),
                    mask=bundle.mask[test],
                    constructor=lambda s, m, p: decile_long_short(s, m, n_deciles=10),
                    rebalance_idx=np.arange(len(test), dtype=np.int64),
                    cost_cfg=cost_cfg,
                    tier="baseline",
                )
                curves.append(
                    [net_sharpe_at_bps(res.series.gross, res.series.turnover, b)
                     for b in (0.0, 10.0, 20.0, 50.0)]
                )
            mean_curve = np.mean(curves, axis=0)
            lines.append(
                f"| {REFERENCE.get(tag, tag)} | {t['breakeven_bps'].gross:.1f} | "
                + " | ".join(f"{v:+.2f}" for v in mean_curve) + " |"
            )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")
    return 0


def _plot_beta(rows: list, ungated: MetricValue) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = [b for b, _ in rows]
    means = [t["rank_ic"].gross for _, t in rows]
    stds = [t["rank_ic"].std or 0.0 for _, t in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(betas, means, yerr=stds, marker="o", capsize=4, label="MASTER (gated)")
    ax.axhline(ungated.gross, color="crimson", ls="--", label="no gating (reference)")
    ax.fill_between(
        [min(betas), max(betas)],
        ungated.gross - (ungated.std or 0),
        ungated.gross + (ungated.std or 0),
        color="crimson", alpha=0.15, label="no gating ±1 seed std",
    )
    ax.set_xscale("log")
    ax.set_xlabel("β (gate temperature; → ∞ approaches no gating)")
    ax.set_ylabel("test RankIC")
    ax.set_title("β sweep vs the no-gating reference — spec §8.2")
    ax.legend()
    ax.grid(alpha=0.3)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "beta_sweep.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
