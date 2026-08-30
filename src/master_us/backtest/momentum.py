"""The Phase-1 gate: 12-1 momentum through the full engine — spec section 5.4.

The engine is not trusted until it reproduces a factor whose behavior is
documented: 12-1 momentum (skip the most recent month), monthly rebalanced,
decile long-short. The reproduction must show the 2009 momentum crash — the
loser-stock rally of March-September 2009 that destroyed WML portfolios — a
plausible long-short Sharpe, and turnover consistent with monthly
rebalancing. If any of those is missing, the engine cannot evaluate anything
and no model work may start.

DATA WINDOW DEVIATION, flagged loudly: the project's canonical sample starts
2010 (data contract section 2.2), but a 2010 start cannot contain a 2009
crash. The gate therefore runs on an EXTENDED window — membership
reconstructed from 2007 and prices backfilled to 2006 — kept in separate
append-only caches so the canonical 2010+ sample and its splits are
untouched. Survivorship is worse pre-2010 (fewer of that era's names still
resolve on Yahoo), which ATTENUATES the measured crash: the missing names are
disproportionately the crushed-then-rallying losers that drove it. The gate
demands the crash be visible, not that its full published magnitude be
recovered. Measured numbers in NOTES.md, Session 4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from master_us.backtest.construct import decile_long_short
from master_us.backtest.costs import CostConfig, corwin_schultz_spread
from master_us.backtest.engine import BacktestResult, month_end_indices, run_backtest
from master_us.backtest.metrics import summarize
from master_us.data.loaders import PRICE_CACHE_DIR
from master_us.data.sources import RAW_ROOT, REPO_ROOT
from master_us.reporting.results import MetricValue, PhaseResult

FloatMat = npt.NDArray[np.float64]
BoolMat = npt.NDArray[np.bool_]

# Gate-specific caches — separate from the canonical 2010+ files on purpose.
GATE_MEMBERSHIP_CACHE = RAW_ROOT / "sp500_membership_2007.parquet"
GATE_CHANGES_CACHE = RAW_ROOT / "sp500_changes_2007.parquet"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

MOMENTUM_LOOKBACK = 252  # ~12 months
MOMENTUM_SKIP = 21  # ~1 month

# ---- Gate thresholds, with their reasoning ----------------------------- #
# Sharpe: large-cap 12-1 momentum through a sample containing the 2009 crash
# is a weak-to-moderate factor. Published long-sample UMD Sharpe is ~0.5 with
# post-2008 large-cap implementations well below that. Above 1.0 on this
# sample means a bug (spec section 12); below -0.5 means inverted alignment.
GATE_SHARPE_RANGE = (-0.5, 1.0)
# Crash: WML lost 30%+ even in conservative large-cap implementations over
# Mar-Sep 2009. Survivorship attenuation on this data path is real, so the
# bar is "unambiguously visible", not "full published magnitude".
GATE_CRASH_WINDOW = ("2009-03-01", "2009-09-30")
GATE_CRASH_MAX_RETURN = -0.15  # the window's gross return must be below this
# Turnover: monthly-rebalanced deciles turn over some 15-50% one-way per
# rebalance in the literature; outside that band the schedule or the
# constructor is wrong.
GATE_REBALANCE_TURNOVER_RANGE = (0.10, 0.60)


@dataclass(frozen=True)
class WidePanel:
    """Wide (T, N) matrices for the engine, assembled from the price cache."""

    dates: pd.DatetimeIndex
    tickers: list[str]
    returns: FloatMat  # simple daily, from adj_close; NaN where absent
    adj_close: FloatMat
    close: FloatMat
    high: FloatMat
    low: FloatMat
    volume: FloatMat
    tradeable: BoolMat


def load_wide_panel(
    membership_path: Path,
    start: str,
    end: str,
    min_price: float = 5.0,
    min_adv_usd: float = 2_000_000.0,
    min_history_days: int = 252,
    price_dir: Path = PRICE_CACHE_DIR,
) -> WidePanel:
    """Assemble wide matrices from the per-ticker price cache plus membership.

    `tradeable` applies the section 2.3 reconstitution filters on information
    known as of each date: membership AND raw close >= min_price AND 21d ADV
    >= min_adv_usd AND >= min_history_days of observed history.
    """
    membership = pl.read_parquet(membership_path).select(["date", "ticker", "in_universe"])
    wanted = set(membership.filter(pl.col("in_universe"))["ticker"].unique().to_list())

    frames = [
        pl.read_parquet(p)
        for p in sorted(price_dir.glob("*.parquet"))
        if p.stem in wanted
    ]
    if not frames:
        raise FileNotFoundError(f"no cached prices under {price_dir} match the membership")

    lo, hi = pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()
    prices = (
        pl.concat(frames)
        .filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        .join(membership, on=["date", "ticker"], how="left")
        .with_columns(pl.col("in_universe").fill_null(value=False))
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(21, min_samples=21)
            .over("ticker")
            .alias("adv_21d"),
            pl.col("date").cum_count().over("ticker").alias("history_days"),
        )
        .with_columns(
            (
                pl.col("in_universe")
                & (pl.col("close") >= min_price)
                & (pl.col("adv_21d") >= min_adv_usd)
                & (pl.col("history_days") >= min_history_days)
            ).alias("tradeable")
        )
    )

    def pivot(col: str) -> pd.DataFrame:
        return (
            prices.pivot(index="date", on="ticker", values=col)
            .sort("date")
            .to_pandas()
            .set_index("date")
            .sort_index(axis=1)
        )

    adj = pivot("adj_close")
    dates = pd.DatetimeIndex(adj.index)
    tickers = list(adj.columns)

    def mat(col: str) -> FloatMat:
        out: FloatMat = pivot(col).to_numpy(dtype=np.float64)
        return out

    adj_np = adj.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        returns = adj_np[1:] / adj_np[:-1] - 1.0
    returns = np.vstack([np.full((1, len(tickers)), np.nan), returns])

    tradeable = pivot("tradeable").to_numpy()
    tradeable = np.where(np.isnan(tradeable.astype(np.float64)), 0, tradeable).astype(bool)

    return WidePanel(
        dates=dates,
        tickers=tickers,
        returns=returns,
        adj_close=adj_np,
        close=mat("close"),
        high=mat("high"),
        low=mat("low"),
        volume=mat("volume"),
        tradeable=tradeable,
    )


def momentum_scores(
    adj_close: FloatMat,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> FloatMat:
    """12-1 momentum: return from t-lookback to t-skip. NaN where insufficient.

    Skipping the most recent month sidesteps short-term reversal — the
    standard Jegadeesh-Titman construction.
    """
    t_len, n = adj_close.shape
    out = np.full((t_len, n), np.nan)
    if t_len <= lookback:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        out[lookback:] = adj_close[lookback - skip : t_len - skip] / adj_close[: t_len - lookback] - 1.0
    return out


def spread_matrix(panel: WidePanel, window: int) -> FloatMat:
    """Corwin-Schultz spread per (date, ticker), for the realistic cost tier."""
    t_len, n = panel.high.shape
    out = np.full((t_len, n), np.nan)
    for j in range(n):
        h, lo, c = panel.high[:, j], panel.low[:, j], panel.close[:, j]
        ok = np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
        if ok.sum() < window + 2:
            continue
        idx = np.flatnonzero(ok)
        out[idx, j] = corwin_schultz_spread(h[idx], lo[idx], window, close=c[idx])
    return out


def adv_matrix(panel: WidePanel, lookback: int) -> FloatMat:
    """Trailing mean dollar volume per (date, ticker)."""
    dollar = panel.close * panel.volume
    frame = pl.DataFrame(dollar)
    return (
        frame.select(
            pl.all().rolling_mean(window_size=lookback, min_samples=lookback)
        )
        .to_numpy()
        .astype(np.float64)
    )


@dataclass(frozen=True)
class GateOutcome:
    """Everything the gate run produces, for the script and the test to share."""

    result_baseline: BacktestResult
    result_realistic: BacktestResult
    metrics: dict[str, MetricValue]
    checks: dict[str, bool]
    crash_return: float  # gross return over GATE_CRASH_WINDOW
    crash_drawdown: float  # deepest drawdown within the crash window
    phase_result: PhaseResult
    eval_dates: pd.DatetimeIndex


def evaluate_gate(
    panel: WidePanel,
    cost_cfg: CostConfig,
    benchmark_returns: npt.NDArray[np.float64] | None,
    eval_start: str = "2008-01-01",
) -> GateOutcome:
    """Run 12-1 momentum through the engine on both cost tiers and judge it."""
    t0 = time.monotonic()
    scores = momentum_scores(panel.adj_close)

    # Evaluation starts once signals exist; before eval_start everything is
    # warm-up. Slicing AFTER signal construction keeps lookbacks intact.
    start_idx = int(np.searchsorted(panel.dates, pd.Timestamp(eval_start)))
    dates = panel.dates[start_idx:]
    returns = panel.returns[start_idx:]
    scores_v = scores[start_idx:]
    mask = panel.tradeable[start_idx:]

    rebalance_idx = month_end_indices(dates)

    def constructor(
        s: npt.NDArray[np.float64],
        m: npt.NDArray[np.bool_],
        prev: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return decile_long_short(s, m, n_deciles=10, weighting="equal")

    baseline = run_backtest(
        dates, returns, scores_v, mask, constructor, rebalance_idx, cost_cfg, tier="baseline"
    )
    spread = spread_matrix(panel, cost_cfg.spread_window)[start_idx:]
    adv = adv_matrix(panel, cost_cfg.adv_lookback)[start_idx:]
    realistic = run_backtest(
        dates,
        returns,
        scores_v,
        mask,
        constructor,
        rebalance_idx,
        cost_cfg,
        tier="realistic",
        spread=spread,
        adv_dollars=adv,
    )

    bench = None
    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns, dtype=np.float64)
        if bench.shape[0] != len(dates):
            raise ValueError(f"benchmark has {bench.shape[0]} rows, engine window {len(dates)}")

    metrics = summarize(baseline.series, benchmark=bench)
    metrics_real = summarize(realistic.series)
    # One MetricValue per headline stat where gross = baseline-net-free gross
    # and net = the REALISTIC tier's net — the pair the project reports.
    metrics["sharpe_realistic_net"] = MetricValue(
        gross=metrics["sharpe"].gross, net=metrics_real["sharpe"].net
    )

    # ---- The three gate checks ---------------------------------------- #
    crash_lo = int(np.searchsorted(dates, pd.Timestamp(GATE_CRASH_WINDOW[0])))
    crash_hi = int(np.searchsorted(dates, pd.Timestamp(GATE_CRASH_WINDOW[1]), side="right"))
    crash_slice = baseline.series.gross[crash_lo:crash_hi]
    crash_return = float(crash_slice.sum())
    crash_equity = np.concatenate([[0.0], np.cumsum(crash_slice)])
    crash_drawdown = float((crash_equity - np.maximum.accumulate(crash_equity)).min())

    sharpe_gross = metrics["sharpe"].gross
    mean_rebalance_to = float(baseline.turnover_per_rebalance.mean())

    checks = {
        "sharpe_in_plausible_range": GATE_SHARPE_RANGE[0]
        <= sharpe_gross
        <= GATE_SHARPE_RANGE[1],
        "crash_2009_visible": crash_return < GATE_CRASH_MAX_RETURN,
        "turnover_monthly_consistent": (
            GATE_REBALANCE_TURNOVER_RANGE[0]
            <= mean_rebalance_to
            <= GATE_REBALANCE_TURNOVER_RANGE[1]
        )
        # off-rebalance days must be genuinely tradeless
        and float(np.delete(baseline.series.turnover, rebalance_idx).sum()) == 0.0,
    }

    metrics["crash_2009_return"] = MetricValue(gross=crash_return, net=crash_return)
    metrics["rebalance_turnover"] = MetricValue(
        gross=mean_rebalance_to, net=mean_rebalance_to
    )

    gate_passed = all(checks.values())
    phase_result = PhaseResult(
        phase=1,
        name="Backtest engine",
        status="pass" if gate_passed else "fail",
        gate="12-1 momentum reproduces: plausible Sharpe, visible 2009 crash, monthly turnover",
        gate_passed=gate_passed,
        metrics=metrics,
        duration_sec=time.monotonic() - t0,
        artifacts=[],
        notes=[
            f"eval window {dates[0].date()} → {dates[-1].date()}, "
            f"{len(baseline.rebalance_dates)} rebalances",
            f"2009 crash window gross return {crash_return:+.1%}, "
            f"within-window drawdown {crash_drawdown:+.1%}",
            "gate ran on the extended 2007+ window; survivorship attenuates the crash "
            "(see NOTES.md Session 4)",
        ],
    )

    return GateOutcome(
        result_baseline=baseline,
        result_realistic=realistic,
        metrics=metrics,
        checks=checks,
        crash_return=crash_return,
        crash_drawdown=crash_drawdown,
        phase_result=phase_result,
        eval_dates=dates,
    )


def plot_equity(outcome: GateOutcome, path: Path) -> Path:
    """Equity curve with the crash window shaded — the gate's visual artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = outcome.eval_dates
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(dates, outcome.result_baseline.equity_gross, label="gross", lw=1.2)
    ax1.plot(dates, outcome.result_baseline.equity, label="net (10 bps flat)", lw=1.2)
    ax1.plot(
        dates, outcome.result_realistic.equity, label="net (CS spread + impact)", lw=1.0
    )
    ax1.axvspan(*(pd.Timestamp(d) for d in GATE_CRASH_WINDOW), alpha=0.15, color="red")
    ax1.set_title("12-1 momentum, decile long-short, monthly — Phase 1 gate")
    ax1.set_ylabel("cumulative return")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    equity = outcome.result_baseline.equity_gross
    drawdown = equity - np.maximum.accumulate(equity)
    ax2.fill_between(dates, drawdown, 0, alpha=0.6)
    ax2.axvspan(*(pd.Timestamp(d) for d in GATE_CRASH_WINDOW), alpha=0.15, color="red")
    ax2.set_ylabel("drawdown (gross)")
    ax2.grid(alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def load_cost_config() -> CostConfig:
    import yaml

    with (REPO_ROOT / "config" / "costs.yaml").open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return CostConfig.from_yaml(raw)
