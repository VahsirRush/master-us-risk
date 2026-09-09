"""Phase 2 orchestration — spec section 6.

Four baselines under one protocol: same panel, same splits, same 5 seeds,
same engine. Everything renders through MetricValue/PhaseResult; the
comparison lands in ONE table.

Backtest convention for the model rows (fixed here so every model is
measured identically): daily rebalance, flat 10 bps net tier, returns from
the panel's own raw forward-return matrix so the engine and the label share
one definition of "what happened next".

TWO books are run for every model, because one of them is not an alpha
measure. `topk_dropout(50, 25)` is long-only, so its Sharpe over a
2019-2025 test window is dominated by equity beta — SPY alone earns roughly
0.8 there, and a long-only book that merely holds large caps will print
something similar whatever its signal is worth. `decile_long_short` is
dollar-neutral and strips that out. The long-short Sharpe is the honest
read on the signal; the long-only one is reported beside it because the
paper's convention is top-k and dropping it would hide the comparison.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml

from master_us.backtest.construct import decile_long_short, topk_dropout
from master_us.backtest.costs import CostConfig
from master_us.backtest.engine import run_backtest
from master_us.backtest.metrics import summarize
from master_us.data.panel import Panel
from master_us.data.sources import DATA_ROOT, REPO_ROOT
from master_us.experiments.lgbm_tuning import with_subset
from master_us.models.baselines.lgbm import run_lgbm
from master_us.models.baselines.lstm import make_lstm
from master_us.models.baselines.ridge import run_ridge
from master_us.models.master import MASTER
from master_us.models.train import PreparedData, prepare_data, rank_ic_by_date, train_model
from master_us.reporting.results import MetricValue, PhaseResult

PHASE2_DIR = DATA_ROOT / "processed" / "phase2"

GATE_MIN_LGBM_RANKIC = 0.02
# Session-5 measurement: the single best raw feature has |IC| 0.0154. A model
# pooling 136 features clearing ~4x that is not plausible on this panel; treat
# anything above as a leak until proven otherwise (spec section 12 discipline).
SUSPICION_RANKIC = 0.06

MODEL_ORDER = ("ridge", "lgbm", "lstm", "ungated", "master")

# Per-model device. PyTorch's MPS LSTM kernel crashes this machine with
# SIGSEGV partway through the second epoch — reproduced three times in
# Session 6, and isolated by running the identical trainer with the
# transformer (stable on MPS) and the LSTM (stable on CPU). The transformer
# stays on MPS because CPU would cost 6.5h for the grid versus 2h.
# Revisit when PyTorch's MPS RNN support improves; nothing about the models
# themselves depends on this.
MODEL_DEVICE = {"lstm": "cpu", "ungated": None, "master": None}  # None = pick_device()


def _epoch_logger(
    log: Callable[[str], None], model: str, seed: int
) -> Callable[[int, float, float], None]:
    """Per-epoch progress. A multi-hour run that prints nothing cannot be
    diagnosed when it dies — measured the hard way in Session 6."""

    def emit(epoch: int, valid_ic: float, best_ic: float) -> None:
        log(
            f"    {model} seed {seed} epoch {epoch:>2}  valid RankIC {valid_ic:+.4f}"
            f"  best {best_ic:+.4f}"
        )

    return emit


@dataclass(frozen=True)
class SeedRun:
    """One (model, seed) outcome."""

    model: str
    seed: int
    test_rank_ic: float
    test_icir: float
    sharpe_gross: float
    sharpe_net: float
    ann_return_gross: float
    ann_return_net: float
    turnover: float
    max_drawdown: float
    ls_sharpe_gross: float
    ls_sharpe_net: float
    ls_turnover: float
    detail: dict[str, Any]


def backtest_scores(
    data: PreparedData,
    scores: npt.NDArray[np.float32],
    cost_cfg: CostConfig,
    topk: int = 50,
    buffer: int = 25,
) -> dict[str, Any]:
    """Run one seed's test-period scores through the Phase-1 engine."""
    test = data.test_idx
    # Day t's realized return is the forward return recorded at t-1.
    daily = np.full_like(data.raw_forward, np.nan)
    daily[1:] = data.raw_forward[:-1]

    def constructor(
        s: npt.NDArray[np.float64],
        m: npt.NDArray[np.bool_],
        prev: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return topk_dropout(s, m, prev, k=topk, dropout_buffer=buffer)

    def neutral(
        s: npt.NDArray[np.float64],
        m: npt.NDArray[np.bool_],
        prev: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return decile_long_short(s, m, n_deciles=10)

    common = {
        "dates": data.dates[test],
        "returns": daily[test].astype(np.float64),
        "scores": scores[test].astype(np.float64),
        "mask": data.mask[test],
        "rebalance_idx": np.arange(len(test), dtype=np.int64),
        "cost_cfg": cost_cfg,
        "tier": "baseline",
    }
    long_only = summarize(run_backtest(constructor=constructor, **common).series)
    long_short = summarize(run_backtest(constructor=neutral, **common).series)
    return {
        "sharpe_gross": long_only["sharpe"].gross,
        "sharpe_net": long_only["sharpe"].net,
        "ann_return_gross": long_only["ann_return"].gross,
        "ann_return_net": long_only["ann_return"].net,
        "turnover": long_only["turnover"].gross,
        "max_drawdown": long_only["max_drawdown"].gross,
        "ls_sharpe_gross": long_short["sharpe"].gross,
        "ls_sharpe_net": long_short["sharpe"].net,
        "ls_turnover": long_short["turnover"].gross,
    }


def evaluate_seed(
    data: PreparedData,
    model: str,
    seed: int,
    scores: npt.NDArray[np.float32],
    cost_cfg: CostConfig,
    detail: dict[str, Any],
) -> SeedRun:
    ic_series = rank_ic_by_date(scores, data.labels, data.label_valid, data.test_idx)
    rank_ic = float(np.nanmean(ic_series))
    icir = float(rank_ic / ic_series.std(ddof=1)) if ic_series.std(ddof=1) > 0 else 0.0
    bt = backtest_scores(data, scores, cost_cfg)
    return SeedRun(
        model=model,
        seed=seed,
        test_rank_ic=rank_ic,
        test_icir=icir,
        turnover=bt["turnover"],
        max_drawdown=bt["max_drawdown"],
        sharpe_gross=bt["sharpe_gross"],
        sharpe_net=bt["sharpe_net"],
        ann_return_gross=bt["ann_return_gross"],
        ann_return_net=bt["ann_return_net"],
        ls_sharpe_gross=bt["ls_sharpe_gross"],
        ls_sharpe_net=bt["ls_sharpe_net"],
        ls_turnover=bt["ls_turnover"],
        detail=detail,
    )


def aggregate(runs: list[SeedRun]) -> dict[str, MetricValue]:
    """Across-seed mean/std for one model, as MetricValues."""
    n = len(runs)

    def mv(gross_attr: str, net_attr: str | None = None) -> MetricValue:
        gross = np.array([getattr(r, gross_attr) for r in runs])
        net = np.array([getattr(r, net_attr) for r in runs]) if net_attr else gross
        # net_std comes from the NET array, never from the gross one: costs
        # scale with each seed's own turnover, so the two dispersions are
        # genuinely different quantities (Session 7 contract fix).
        return MetricValue(
            gross=float(gross.mean()),
            net=float(net.mean()),
            std=float(gross.std(ddof=1)) if n > 1 else 0.0,
            n_seeds=n,
            net_std=float(net.std(ddof=1)) if n > 1 else 0.0,
        )

    return {
        "rank_ic": mv("test_rank_ic"),
        "icir": mv("test_icir"),
        "sharpe": mv("sharpe_gross", "sharpe_net"),
        "ann_return": mv("ann_return_gross", "ann_return_net"),
        "turnover": mv("turnover"),
        "max_drawdown": mv("max_drawdown"),
        "ls_sharpe": mv("ls_sharpe_gross", "ls_sharpe_net"),
        "ls_turnover": mv("ls_turnover"),
    }


def run_phase2(
    panel: Panel,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    models: tuple[str, ...] = MODEL_ORDER,
    max_epochs: int = 12,
    patience: int = 4,
    lookback: int = 20,
    lgbm_params: dict[str, Any] | None = None,
    lgbm_subset: str = "all",
    log: Callable[[str], None] = print,
) -> tuple[dict[str, dict[str, MetricValue]], PhaseResult]:
    t0 = time.monotonic()
    with (REPO_ROOT / "config" / "baselines.yaml").open() as fh:
        base_cfg = yaml.safe_load(fh)
    with (REPO_ROOT / "config" / "master.yaml").open() as fh:
        master_cfg = yaml.safe_load(fh)
    with (REPO_ROOT / "config" / "costs.yaml").open() as fh:
        cost_cfg = CostConfig.from_yaml(yaml.safe_load(fh))
    with (REPO_ROOT / "config" / "data.yaml").open() as fh:
        data_cfg = yaml.safe_load(fh)

    arch = master_cfg["architecture"]
    splits = data_cfg["splits"]
    # Phase-2 lookback is 20, NOT master.yaml's 60: this machine has 8 GB of
    # RAM and the L=60 batch geometry provably swaps/OOMs on MPS (measured in
    # Session 6; L=20 trains at ~1.3 min/epoch). 20 is the smallest value in
    # master.yaml's own lookback sweep, so Phase 3's sweep covers the gap.

    log(f"preparing data (lookback {lookback}, embargo {splits['embargo_days']}d)")
    data = prepare_data(
        panel,
        lookback=lookback,
        train=tuple(splits["train"]),
        valid=tuple(splits["valid"]),
        test=tuple(splits["test"]),
        embargo_days=int(splits["embargo_days"]),
        cache_dir=PHASE2_DIR,
    )
    del panel  # 8 GB machine: PreparedData carries everything the run needs
    log(
        f"  train {len(data.train_idx)} / valid {len(data.valid_idx)} / "
        f"test {len(data.test_idx)} dates · {data.features.shape[2]} features"
    )

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    all_runs: dict[str, list[SeedRun]] = {m: [] for m in models}

    for model_name in models:
        for seed in seeds:
            t_seed = time.monotonic()
            if model_name == "ridge":
                r = run_ridge(data, seed, tuple(base_cfg["ridge"]["alphas"]))
                scores, detail = r.scores, {"alpha": r.alpha, "valid_rank_ic": r.valid_rank_ic}
            elif model_name == "lgbm":
                view = with_subset(data, lgbm_subset)
                g = run_lgbm(view, seed, lgbm_params or base_cfg["lgbm"])
                scores, detail = g.scores, {
                    "best_iteration": g.best_iteration,
                    "valid_rank_ic": g.valid_rank_ic,
                }
            elif model_name == "lstm":
                lstm_net = make_lstm(data.features.shape[2], base_cfg["lstm"])
                tr = train_model(
                    lstm_net, data, seed, max_epochs=max_epochs, patience=patience,
                    device=torch.device("cpu"),
                    log=_epoch_logger(log, model_name, seed),
                )
                scores, detail = tr.scores, {
                    "valid_rank_ic": tr.best_valid_rank_ic,
                    "best_epoch": tr.best_epoch,
                    "epochs_run": tr.n_epochs_run,
                }
            elif model_name in ("ungated", "master"):
                # `master` differs from `ungated` by exactly one thing: use_gate.
                # Everything else — layers, widths, optimizer, schedule, seeds —
                # is shared, which is what makes the Phase-4 gate ablation exact
                # rather than a comparison of two loosely similar models.
                ungated_net = MASTER.from_config(
                    f_dim=data.features.shape[2],
                    m_dim=data.market.shape[1],
                    arch=arch,
                    use_gate=(model_name == "master"),
                )
                tr = train_model(
                    ungated_net, data, seed, max_epochs=max_epochs, patience=patience,
                    log=_epoch_logger(log, model_name, seed),
                )
                scores, detail = tr.scores, {
                    "valid_rank_ic": tr.best_valid_rank_ic,
                    "best_epoch": tr.best_epoch,
                    "epochs_run": tr.n_epochs_run,
                }
            else:
                raise ValueError(f"unknown model {model_name!r}")

            np.save(PHASE2_DIR / f"{model_name}_seed{seed}_scores.npy", scores)
            run = evaluate_seed(data, model_name, seed, scores, cost_cfg, detail)
            all_runs[model_name].append(run)
            del scores
            log(
                f"  {model_name:<8} seed {seed}  RankIC {run.test_rank_ic:+.4f}  "
                f"Sharpe {run.sharpe_gross:+.2f} (net {run.sharpe_net:+.2f})  "
                f"turnover {run.turnover:.1%}  [{time.monotonic() - t_seed:.0f}s]"
            )

    tables = {m: aggregate(all_runs[m]) for m in models}

    # ---- the gate ------------------------------------------------------ #
    notes = [
        f"protocol: {len(seeds)} seeds · daily topk_dropout(50, 25) · flat "
        f"{cost_cfg.baseline_bps:.0f} bps net tier · deep models {max_epochs} epochs "
        f"max, patience {patience} (Phase-2 schedule; Phase 3 uses 100/10) · lookback {lookback}",
    ]
    if lgbm_params is not None:
        notes.append(
            f"lgbm uses a VALIDATION-tuned config (subset={lgbm_subset}); the shipped config missed the gate — see reports/lgbm_tuning.md"
        )
    if "lgbm" in tables:
        lgbm_ic = tables["lgbm"]["rank_ic"]
        gate_passed = lgbm_ic.gross > GATE_MIN_LGBM_RANKIC
        if lgbm_ic.gross > SUSPICION_RANKIC:
            notes.append(
                f"WARNING: LGBM RankIC {lgbm_ic.gross:.4f} clears the gate implausibly "
                f"wide (>{SUSPICION_RANKIC}) given max single-feature |IC| 0.0154 — "
                "audit for leakage before celebrating (spec section 12)"
            )
    else:
        # Partial run (smoke or debugging): the gate cannot be evaluated.
        lgbm_ic = MetricValue(gross=float("nan"), net=float("nan"))
        gate_passed = False
        notes.append("lgbm was not part of this run — gate NOT evaluated")

    metrics: dict[str, MetricValue] = {"rankic": lgbm_ic}
    for model_name in models:
        for key, value in tables[model_name].items():
            metrics[f"{model_name}_{key}"] = value

    result = PhaseResult(
        phase=2,
        name="Baselines",
        status="pass" if gate_passed else "fail",
        gate=f"LightGBM out-of-sample RankIC > {GATE_MIN_LGBM_RANKIC}",
        gate_passed=gate_passed,
        metrics=metrics,
        duration_sec=time.monotonic() - t0,
        artifacts=[PHASE2_DIR],
        notes=notes,
    )

    with (PHASE2_DIR / "seed_runs.json").open("w") as fh:
        json.dump(
            {
                m: [
                    {k: v for k, v in run.__dict__.items() if k != "detail"} | run.detail
                    for run in runs
                ]
                for m, runs in all_runs.items()
            },
            fh,
            indent=2,
        )
    return tables, result


def render_baseline_table(tables: dict[str, dict[str, MetricValue]]) -> None:
    """One table, all models — output-layer 3.3 conventions.

    `n` is printed per row because a partial grid must never be mistaken for
    a complete one, and turnover is a permanent column (CLAUDE.md rule 5).
    """
    from rich.console import Console
    from rich.table import Table

    from master_us.reporting.theme import THEME

    console = Console(theme=THEME, width=150)
    seeds = sorted({t["rank_ic"].n_seeds or 0 for t in tables.values()})
    seed_note = f"{seeds[0]}-{seeds[-1]} seeds" if len(seeds) > 1 else f"{seeds[0]} seeds"
    table = Table(
        title=(
            f"Baselines · S&P 500 · test 2019-2025 · {seed_note} · net = flat 10 bps\n"
            "L/S = dollar-neutral deciles (the alpha read) · "
            "long-only = topk_dropout(50,25), carries market beta"
        ),
        title_justify="left",
    )
    table.add_column("Model", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("RankIC", justify="right", no_wrap=True)
    table.add_column("ICIR", justify="right")
    table.add_column("L/S Sharpe", justify="right", no_wrap=True)
    table.add_column("L/S turn", justify="right")
    table.add_column("Long-only Sharpe", justify="right", no_wrap=True)
    table.add_column("LO turn", justify="right")

    display = {
        "ridge": "Ridge",
        "lgbmspec": "LightGBM (shipped)",
        "lgbm": "LightGBM (tuned)",
        "lstm": "LSTM",
        "ungated": "Ungated transformer",
        "master": "MASTER (gated)",
    }
    for model_name in ("ridge", "lgbmspec", "lgbm", "lstm", "ungated", "master"):
        if model_name not in tables:
            continue
        t = tables[model_name]
        ic, ls, lo = t["rank_ic"], t["ls_sharpe"], t["sharpe"]
        table.add_row(
            display.get(model_name, model_name),
            str(ic.n_seeds),
            f"{ic.gross:+.4f} ±{ic.std:.4f}",
            f"{t['icir'].gross:+.3f}",
            f"{ls.gross:+.2f} → {ls.net:+.2f}",
            f"{t['ls_turnover'].gross:.0%}",
            f"{lo.gross:+.2f} → {lo.net:+.2f}",
            f"{t['turnover'].gross:.0%}",
            style="degraded" if ls.degraded else "metric",
        )
    console.print(table)


def compare_models(
    tables: dict[str, dict[str, MetricValue]],
    metric: str = "rank_ic",
    basis: str = "gross",
) -> list[tuple[str, str, float, bool]]:
    """Every pairwise gap on `metric`, with the distinguishability verdict.

    The verdict comes from `MetricValue.distinguishable_from` (gross) or
    `net_distinguishable_from` (net) — pooled seed dispersion on the basis
    being asked about, not eyeballing. CLAUDE.md rule 4 requires gaps inside
    that envelope to be reported as "not distinguishable" in those words.
    """
    if basis not in ("gross", "net"):
        raise ValueError(f"basis must be 'gross' or 'net', got {basis!r}")
    names = [m for m in ("ridge", "lgbmspec", "lgbm", "lstm", "ungated", "master") if m in tables]
    out: list[tuple[str, str, float, bool]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            mv_a, mv_b = tables[a][metric], tables[b][metric]
            if basis == "net":
                gap = (mv_a.net or 0.0) - (mv_b.net or 0.0)
                real = mv_a.net_distinguishable_from(mv_b)
            else:
                gap = mv_a.gross - mv_b.gross
                real = mv_a.distinguishable_from(mv_b)
            out.append((a, b, gap, real))
    return out


def render_comparison(
    tables: dict[str, dict[str, MetricValue]],
    metric: str = "rank_ic",
    basis: str = "gross",
) -> None:
    """Print the pairwise verdicts, real gaps first."""
    from rich.console import Console

    from master_us.reporting.theme import THEME

    console = Console(theme=THEME, width=150)
    rows = compare_models(tables, metric, basis)
    console.print(
        f"\nPairwise {metric} ({basis}): gap vs pooled {basis} seed dispersion", style="bold"
    )
    for a, b, gap, real in sorted(rows, key=lambda r: (not r[3], -abs(r[2]))):
        verdict = "distinguishable" if real else "NOT distinguishable"
        console.print(
            f"  {a:<10} vs {b:<10} {gap:+.4f}   {verdict}",
            style="metric" if real else "noise",
        )


def prepare_metrics_only(
    panel: Panel,
    train: tuple[str, str],
    valid: tuple[str, str],
    test: tuple[str, str],
    lookback: int = 20,
    embargo_days: int = 21,
) -> PreparedData:
    """A PreparedData carrying everything the METRICS need and no features.

    Scoring cached predictions needs labels, mask, raw forward returns, dates
    and the split indices — never the feature tensor. Building it through
    `prepare_data` would normalize 1.3 GB of features for nothing, and worse,
    would reopen the shared feature memmap in write mode while a training run
    has it mapped. This path allocates a zero-width feature array instead, so
    finalization is safe to run beside a live grid.
    """

    def span(lo: str | pd.Timestamp, hi: str | pd.Timestamp) -> npt.NDArray[np.int64]:
        sel = (panel.dates >= pd.Timestamp(lo)) & (panel.dates <= pd.Timestamp(hi))
        idx = np.flatnonzero(np.asarray(sel))
        return idx[idx >= lookback - 1]

    gap = pd.Timedelta(days=embargo_days)
    return PreparedData(
        dates=panel.dates,
        features=np.empty((panel.n_dates, panel.n_tickers, 0), dtype=np.float32),
        market=np.empty((panel.n_dates, 0), dtype=np.float32),
        labels=panel.labels,
        mask=panel.mask,
        raw_forward=panel.attrs["raw_forward_returns"],
        lookback=lookback,
        train_idx=span(train[0], pd.Timestamp(train[1]) - gap),
        valid_idx=span(valid[0], pd.Timestamp(valid[1]) - gap),
        test_idx=span(test[0], test[1]),
        feature_names=(),
    )


def rebuild_from_scores(
    data: PreparedData,
    cost_cfg: CostConfig,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    models: tuple[str, ...] = MODEL_ORDER,
    scores_dir: Path = PHASE2_DIR,
) -> dict[str, dict[str, MetricValue]]:
    """Rebuild the comparison table from cached score arrays, no retraining.

    Every run writes `{model}_seed{seed}_scores.npy`, so the table can be
    reassembled after a partial re-run — or while a grid is still going —
    without spending hours reproducing the deep models. Models with no cached
    seeds are simply absent from the result rather than faked.
    """
    tables: dict[str, dict[str, MetricValue]] = {}
    for model_name in models:
        runs: list[SeedRun] = []
        for seed in seeds:
            path = scores_dir / f"{model_name}_seed{seed}_scores.npy"
            if not path.exists():
                continue
            runs.append(
                evaluate_seed(data, model_name, seed, np.load(path), cost_cfg, {})
            )
        if runs:
            tables[model_name] = aggregate(runs)
    return tables
