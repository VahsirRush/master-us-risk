"""Phase 7 — the join (spec §10). Where the two halves become one result.

MASTER's long-short book is run through the Barra risk model built in Phases
5-6, and its P&L and risk are split into what the factor model already
explains and what is left over as specific alpha.

THE ALIGNMENT, which is the whole difficulty. Three separate index
conventions have to agree:

    engine        `weights[t]` is the book held INTO day t, formed from
                  `scores[t-1]`, earning `returns[t]`
    risk model    period i has exposures at panel row `exposure_idx[i]`
                  and its return lands at `exposure_idx[i] + 1`
    bundle        MASTER's scores live on a different (date x ticker) grid
                  from the risk panel — 4004x584 against 3772x583

They line up on one statement: the book held into panel date
`exposure_idx[i] + 1` was formed at `exposure_idx[i]`, which is exactly where
the risk model read its exposures. So weights and exposures are
contemporaneous by construction, and the factor return they are paired with
is the one earned over the following day. Getting this wrong would not throw
— it would quietly produce an attribution that explains the book's P&L using
factor returns it never earned.

WHAT A GOOD RESULT LOOKS LIKE HERE. The raw book has been net-negative in
every measurement this project has made. A neutralized book that comes back
strongly positive is a bug signal before it is a finding, and is treated as
one: §12 already flags "Sharpe > 3" as a bug, and the same reasoning applies
to any large positive number appearing where every prior measurement found a
negative one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from master_us.reporting.results import MetricValue
from master_us.risk.descriptors import FACTOR_DESCRIPTORS

STYLE_FACTORS: tuple[str, ...] = tuple(FACTOR_DESCRIPTORS)

# The four books §10 compares, in reporting order. `plain_eigen` is a CONTROL,
# not a candidate strategy: without it, comparing the decile book against the
# neutralized one would confound style-neutralization with the separate change
# from a decile rule to an optimizer.
JOIN_ARMS: tuple[str, ...] = ("decile", "plain_eigen", "neutral_eigen", "neutral_no_eigen")
ARM_LABELS: dict[str, str] = {
    "decile": "MASTER decile L/S",
    "plain_eigen": "optimized, unconstrained",
    "neutral_eigen": "optimized, STYLE-NEUTRAL",
    "neutral_no_eigen": "same, eigenfactor OFF",
}
CONTROL_ARMS: frozenset[str] = frozenset({"plain_eigen", "neutral_no_eigen"})

# The t-statistic magnitude above which a timing correlation is called real.
TIMING_SIGNIFICANCE_T = 1.96

# --------------------------------------------------------------------- #
# Disclosure text — single source of truth                               #
# --------------------------------------------------------------------- #
#
# These live here rather than in the report script because the same words have
# to appear in `reports/phase7.md` AND in the terminal payload. A caveat that
# exists in the markdown but not on the dashboard is not a caveat; it is a
# number published without one. Keeping them as constants means the two cannot
# drift apart.

SPECIFIC_ALPHA_CAVEAT = (
    "This is NOT an alpha number and must not be quoted as one. It is the return left "
    "after factor exposure is removed arithmetically, which assumes factor hedging is "
    "free. It is not achievable. Constructing the hedge and paying for it is the "
    "style-neutral book, which is WORSE than the unhedged book. The gap between the two "
    "is the difference between an attribution and a portfolio."
)

TIMING_METHOD_NOTE = (
    "§10.5 specifies regressing the learned gate activations on the market state vector "
    "and on contemporaneous factor returns. No Phase-3 checkpoint was saved and no "
    "activations were cached, so the activations are not available. What is measured "
    "here instead is the gate's CONSEQUENCE: whether the gated book times factors "
    "better than the ungated one. The two models share seeds, data and protocol and "
    "differ only in the gate, so the difference is attributable to it. This is a "
    "SUBSTITUTE for the specified activation regression, not an implementation of it."
)

EIGEN_CAVEAT = (
    "Phase 6 found the eigenfactor adjustment slightly hurt the bias statistic and "
    "argued it should help once a real optimizer ran against the covariance, because "
    "the adjustment corrects minimum-variance directions while Phase 6's test "
    "portfolios were random and factor-mimicking, neither optimized. Phase 7 ran that "
    "optimizer. The prediction is NOT confirmed, and is recorded as a failed prediction "
    "rather than explained away."
)

CONTROL_NOTE = (
    "A control, not a candidate strategy. Both optimized arms run through the same "
    "function with identical lambdas, so the neutral-vs-unconstrained comparison "
    "isolates neutralization rather than the switch from a decile rule to an optimizer."
)


@dataclass(frozen=True)
class BookAttribution:
    """One seed's attribution of a book's realized P&L and predicted risk."""

    dates: pd.DatetimeIndex
    weights: npt.NDArray[np.float64]          # (T, N) held into each date
    exposures: npt.NDArray[np.float64]        # (T, K) book factor exposure
    factor_pnl: npt.NDArray[np.float64]       # (T,) from factor returns
    specific_pnl: npt.NDArray[np.float64]     # (T,) residual
    total_pnl: npt.NDArray[np.float64]        # (T,) realized gross
    factor_var_share: npt.NDArray[np.float64]  # (T,) predicted variance share
    mctr: npt.NDArray[np.float64]             # (T, K) marginal contribution

    def shares(self) -> tuple[float, float]:
        """Fraction of realized P&L VARIANCE from factors vs specific.

        Variance rather than mean: the means are small and noisy enough that
        a ratio of them is unstable, while the variance decomposition is
        exactly what the risk model claims to predict and so is the honest
        thing to check it against.
        """
        ok = np.isfinite(self.factor_pnl) & np.isfinite(self.specific_pnl)
        f, s = self.factor_pnl[ok], self.specific_pnl[ok]
        tot = np.var(f + s)
        return (float(np.var(f) / tot), float(np.var(s) / tot)) if tot > 0 else (np.nan, np.nan)


def align_to_risk_grid(
    bundle_dates: pd.DatetimeIndex,
    bundle_tickers: npt.NDArray[np.object_],
    panel_dates: pd.DatetimeIndex,
    panel_tickers: npt.NDArray[np.object_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Index maps from the bundle grid onto the risk panel grid.

    Returns (date_rows, ticker_cols) such that `bundle[date_rows][:, ticker_cols]`
    is on the panel's grid. Raises if any panel date or ticker is absent
    rather than silently dropping — a quiet drop here shrinks the book and
    flatters every ratio computed from it.
    """
    d_pos = {d: i for i, d in enumerate(bundle_dates)}
    t_pos = {t: i for i, t in enumerate(bundle_tickers)}
    missing_d = [d for d in panel_dates if d not in d_pos]
    if missing_d:
        raise KeyError(f"{len(missing_d)} panel dates absent from the bundle, first {missing_d[0]}")
    missing_t = [t for t in panel_tickers if t not in t_pos]
    if missing_t:
        raise KeyError(f"{len(missing_t)} panel tickers absent from the bundle: {missing_t[:5]}")
    return (
        np.array([d_pos[d] for d in panel_dates], dtype=np.int64),
        np.array([t_pos[t] for t in panel_tickers], dtype=np.int64),
    )


def attribute(
    weights: npt.NDArray[np.float64],
    exposures: npt.NDArray[np.float64],
    factor_returns: npt.NDArray[np.float64],
    asset_returns: npt.NDArray[np.float64],
    factor_cov: npt.NDArray[np.float64],
    specific_var: npt.NDArray[np.float64],
    dates: pd.DatetimeIndex,
) -> BookAttribution:
    """Split realized P&L and predicted risk — spec §10.2, §10.3.

    `weights[i]` is the book held into period i, `exposures[i]` its factor
    loadings as of the start of that period, and `factor_returns[i]` /
    `asset_returns[i]` what was earned across it.

    Specific P&L is the residual of realized total against the factor
    contribution, NOT a separately modelled quantity — so the two parts sum
    to the realized book return exactly, by construction, and any modelling
    error lands in the specific term where it is visible rather than being
    split silently across both.
    """
    t, k = factor_returns.shape
    book_exp = np.full((t, k), np.nan)
    factor_pnl = np.full(t, np.nan)
    total_pnl = np.full(t, np.nan)
    fvar_share = np.full(t, np.nan)
    mctr = np.full((t, k), np.nan)

    for i in range(t):
        w = weights[i]
        if not np.isfinite(w).any() or not np.abs(w).sum() > 0:
            continue
        x = np.nan_to_num(exposures[i], nan=0.0)
        b = w @ x
        book_exp[i] = b

        f = factor_returns[i]
        if np.isfinite(f).all():
            factor_pnl[i] = float(b @ f)
        r = asset_returns[i]
        total_pnl[i] = float(np.nansum(w * np.where(np.isfinite(r), r, 0.0)))

        cov, d = factor_cov[i], specific_var[i]
        if np.isfinite(cov).all() and np.isfinite(d).any():
            dd = np.nan_to_num(d, nan=0.0)
            fv = float(b @ cov @ b)
            sv = float((w**2) @ dd)
            if fv + sv > 0:
                fvar_share[i] = fv / (fv + sv)
                total_vol = np.sqrt(fv + sv)
                # d(sigma)/d(b_k) — marginal contribution to risk per factor
                mctr[i] = (cov @ b) / total_vol if total_vol > 0 else np.nan

    specific_pnl = total_pnl - factor_pnl
    return BookAttribution(
        dates=dates,
        weights=weights,
        exposures=book_exp,
        factor_pnl=factor_pnl,
        specific_pnl=specific_pnl,
        total_pnl=total_pnl,
        factor_var_share=fvar_share,
        mctr=mctr,
    )


def sharpe(x: npt.NDArray[np.float64], periods: int = 252) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    sd = a.std(ddof=1)
    return float(a.mean() / sd * np.sqrt(periods)) if sd > 0 and a.size > 1 else 0.0


def aggregate(values_gross: list[float], values_net: list[float]) -> MetricValue:
    """Per-seed results -> the project's MetricValue contract.

    Net carries its own dispersion computed from the net series, never
    derived from gross — the correction made in Session 6 and the reason
    `net_distinguishable_from` exists at all.
    """
    g, n = np.asarray(values_gross, float), np.asarray(values_net, float)
    return MetricValue(
        gross=float(g.mean()),
        net=float(n.mean()),
        std=float(g.std(ddof=1)) if g.size > 1 else 0.0,
        n_seeds=int(g.size),
        net_std=float(n.std(ddof=1)) if n.size > 1 else 0.0,
    )


def zero_metric(n_seeds: int) -> MetricValue:
    """A literal zero with no dispersion, for 'distinguishable from zero'.

    `net_distinguishable_from` compares a gap against pooled dispersion
    sqrt(a^2 + b^2); giving the null zero std means the comparison is
    against the estimate's own seed dispersion alone, which is the intended
    question — is this Sharpe separable from nothing at all.
    """
    return MetricValue(gross=0.0, net=0.0, std=0.0, n_seeds=n_seeds, net_std=0.0)


# --------------------------------------------------------------------- #
# §10.4 the neutralized portfolio                                        #
# --------------------------------------------------------------------- #


def _standardize(scores: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Cross-sectionally z-score the alpha so the lambdas mean the same thing daily.

    Without this the risk and turnover penalties are calibrated against
    whatever scale the network happened to emit on a given day, and the
    optimizer's effective aggressiveness drifts with it.
    """
    out = np.zeros_like(scores, dtype=np.float64)
    ok = mask & np.isfinite(scores)
    if ok.sum() < 2:
        return out
    v = scores[ok]
    sd = v.std(ddof=1)
    out[ok] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out


def optimize_book(
    scores: npt.NDArray[np.float64],
    exposures: npt.NDArray[np.float64],
    factor_cov: npt.NDArray[np.float64],
    specific_var: npt.NDArray[np.float64],
    asset_returns: npt.NDArray[np.float64],
    style_idx: list[int],
    neutralize: bool,
    lam_risk: float,
    lam_turnover: float,
    max_weight: float = 0.02,
    gross_target: float = 1.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """Roll `cost_aware_optimize` across the test period — spec §10.4.

    Both arms (neutral and not) run through THIS function with identical
    lambdas, so the comparison isolates the neutrality constraint rather
    than confounding it with the switch from a decile book to an optimizer.

    Weights are rescaled to `gross_target` after each solve. Net Sharpe is
    invariant to that rescaling — gross return, cost and volatility all
    scale together under a linear cost model — so it only makes turnover and
    exposure directly comparable with the decile book, without touching the
    headline number.
    """
    from master_us.backtest.construct import cost_aware_optimize

    t, n = asset_returns.shape
    weights = np.zeros((t, n))
    gross = np.zeros(t)
    turnover = np.zeros(t)
    prev = np.zeros(n)

    for i in range(t):
        tradeable = (
            np.isfinite(specific_var[i])
            & (specific_var[i] > 0)
            & np.isfinite(scores[i])
            & np.isfinite(exposures[i]).all(axis=1)
        )
        if tradeable.sum() < 50 or not np.isfinite(factor_cov[i]).all():
            weights[i] = prev
            continue

        x = np.nan_to_num(exposures[i], nan=0.0)
        d = np.nan_to_num(specific_var[i], nan=0.0)
        sigma = x @ factor_cov[i] @ x.T + np.diag(d)

        cons: dict[str, object] = {"max_weight": max_weight}
        if neutralize:
            cons["neutralize"] = x[:, style_idx].T

        target = cost_aware_optimize(
            _standardize(scores[i], tradeable), prev, sigma, tradeable,
            lam_turnover=lam_turnover, lam_risk=lam_risk, constraints=cons,
        )
        g = np.abs(target).sum()
        if g > 0:
            target = target * (gross_target / g)

        turnover[i] = 0.5 * float(np.abs(target - prev).sum())
        # the book chosen at i is held INTO i+1, matching the engine
        if i + 1 < t:
            weights[i + 1] = target
        prev = target

    for i in range(t):
        r = asset_returns[i]
        gross[i] = float(np.nansum(weights[i] * np.where(np.isfinite(r), r, 0.0)))

    return {"weights": weights, "gross": gross, "turnover": turnover}


def net_of_costs(
    gross: npt.NDArray[np.float64], turnover: npt.NDArray[np.float64], bps: float = 10.0
) -> npt.NDArray[np.float64]:
    """Charge one side of a round trip per unit of one-way turnover.

    Same convention as `costs.flat_cost` and `ablations.cost_breakeven`, so
    a net Sharpe here is comparable with every other net number in the
    project rather than being a differently-defined quantity.
    """
    return np.asarray(gross, float) - np.asarray(turnover, float) * (bps / 2.0) * 1e-4


def breakeven_bps(
    gross: npt.NDArray[np.float64], turnover: npt.NDArray[np.float64]
) -> float:
    """bps at which net alpha reaches zero — the §10 headline's B."""
    mg, mt = float(np.nanmean(gross)), float(np.nanmean(turnover))
    if mg <= 0:
        return 0.0
    return 2e4 * mg / mt if mt > 0 else float("inf")


# --------------------------------------------------------------------- #
# §10.5 gate interpretation                                              #
# --------------------------------------------------------------------- #


def factor_timing(
    exposures: npt.NDArray[np.float64],
    factor_returns: npt.NDArray[np.float64],
    style_idx: list[int],
) -> dict[str, float]:
    """Does a book tilt toward factors that then pay? — spec §10.5.

    Factor timing, stated in the vocabulary §10.5 asks for: the correlation
    over time between the book's exposure to factor k at the start of a
    period and the return factor k earns across it. Positive means the book
    leans into factors before they pay.

    Both series are contemporaneous by the alignment fixed at the top of
    this module — exposure at the start of period i, factor return earned
    over period i — so this is a genuine timing measure, not a
    same-instant identity.
    """
    out: dict[str, float] = {}
    for name, k in zip(STYLE_FACTORS, style_idx, strict=True):
        e, f = exposures[:, k], factor_returns[:, k]
        ok = np.isfinite(e) & np.isfinite(f)
        out[name] = float(np.corrcoef(e[ok], f[ok])[0, 1]) if ok.sum() > 30 else float("nan")
    return out


def gate_timing_delta(
    gated_exposures: npt.NDArray[np.float64],
    ungated_exposures: npt.NDArray[np.float64],
    factor_returns: npt.NDArray[np.float64],
    style_idx: list[int],
) -> dict[str, dict[str, float]]:
    """What the GATE adds to factor timing, isolated.

    WHAT THIS IS AND IS NOT. §10.5 asks to regress the learned gate
    activations on the market state vector and on contemporaneous factor
    returns. No checkpoint from Phase 3 was saved and no activations were
    cached, so the activations themselves are not available without
    retraining. This measures the gate's CONSEQUENCE instead: the difference
    between the gated and ungated books' factor timing, which is the thing
    the activation regression was meant to detect. It answers "does the gate
    produce regime-conditional factor timing", not "what do the activations
    look like".

    The gated and ungated models share seeds, data and protocol and differ
    only in the gate, so the difference in their timing is attributable to
    it — the same logic the Phase 4 ablation grid rests on.
    """
    gated = factor_timing(gated_exposures, factor_returns, style_idx)
    ungated = factor_timing(ungated_exposures, factor_returns, style_idx)
    return {
        name: {
            "gated": gated[name],
            "ungated": ungated[name],
            "delta": gated[name] - ungated[name],
        }
        for name in STYLE_FACTORS
    }


def build_join_tables(
    metrics: Mapping[str, MetricValue],
    turnover_breakeven: Mapping[str, tuple[float, float]],
    timing: Sequence[tuple[str, float, float, float, float, float]],
    cross_factor_dispersion: float,
    attribution: Mapping[str, float],
    n_seeds: int,
) -> dict[str, Any]:
    """The §10 tables as machine-readable values — spec §10.2-10.5.

    Phase 7's detailed tables otherwise exist only as prose in
    `reports/phase7.md`, which would force any downstream consumer (the
    terminal export, notably) to parse markdown to recover a number. Parsing
    prose for figures is how a rendering silently disagrees with its source, so
    the same values are emitted structurally here.

    `metrics` and `turnover_breakeven` are keyed by arm; `timing` is the
    per-factor tuple `(factor, gated, ungated, delta, t_stat, delta_sd)` where
    `delta_sd` is the delta's own across-seed dispersion. Every value is passed
    in already computed — this function derives no statistic, it only shapes
    what the caller measured, and attaches the disclosure text that must travel
    with two of the numbers.
    """
    null = zero_metric(n_seeds)
    books = []
    for arm in JOIN_ARMS:
        if arm not in metrics:
            continue
        m = metrics[arm]
        turn, be = turnover_breakeven[arm]
        books.append(
            {
                "key": arm,
                "label": ARM_LABELS.get(arm, arm),
                "is_control": arm in CONTROL_ARMS,
                "gross": m.gross,
                "gross_sd": m.std,
                "net": m.net,
                "net_sd": m.net_std,
                "turnover": turn,
                "breakeven_bps": be,
                "net_distinguishable_from_zero": m.net_distinguishable_from(null),
                "n_seeds": m.n_seeds,
            }
        )

    sig_rows: list[dict[str, Any]] = [
        {
            "factor": f,
            "gated": g,
            "ungated": u,
            "delta": d,
            "delta_sd": sd,
            "t_stat": t,
            "significant": bool(abs(t) > TIMING_SIGNIFICANCE_T),
            # Rule 4's shape: a gap smaller than its own dispersion is not a
            # result. Reported per factor so the table can show both tests.
            "exceeds_seed_dispersion": bool(abs(d) > sd) if sd > 0 else None,
        }
        for f, g, u, d, t, sd in timing
    ]
    # Taken from the typed tuples rather than the dicts above, so the delta stays
    # a float instead of widening to object.
    largest = max(timing, key=lambda r: abs(r[3]), default=None)
    largest_delta = abs(largest[3]) if largest is not None else None
    largest_factor = largest[0] if largest is not None else None
    largest_delta_sd = largest[5] if largest is not None else None

    eigen_gap = None
    eigen_real = None
    if "neutral_eigen" in metrics and "neutral_no_eigen" in metrics:
        a, b = metrics["neutral_eigen"], metrics["neutral_no_eigen"]
        eigen_gap = _net_of(a) - _net_of(b)
        eigen_real = a.net_distinguishable_from(b)

    neutral_gap = None
    neutral_real = None
    if "neutral_eigen" in metrics and "plain_eigen" in metrics:
        a, b = metrics["neutral_eigen"], metrics["plain_eigen"]
        neutral_gap = _net_of(a) - _net_of(b)
        neutral_real = a.net_distinguishable_from(b)

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_seeds": n_seeds,
        "control_note": CONTROL_NOTE,
        "books": books,
        "neutral_vs_unconstrained": {"gap": neutral_gap, "distinguishable": neutral_real},
        "attribution": {
            **{k: float(v) for k, v in attribution.items()},
            # The caveat is a sibling of the number it qualifies, so a consumer
            # cannot pick up one without seeing the other.
            "specific_gross_sharpe_caveat": SPECIFIC_ALPHA_CAVEAT,
        },
        "timing": sig_rows,
        "timing_summary": {
            "n_significant": sum(1 for r in sig_rows if r["significant"]),
            "n_factors": len(sig_rows),
            "n_exceeding_dispersion": sum(1 for r in sig_rows if r["exceeds_seed_dispersion"]),
            "largest_delta": largest_delta,
            "largest_delta_factor": largest_factor,
            "largest_delta_sd": largest_delta_sd,
            "cross_factor_dispersion": cross_factor_dispersion,
            "significance_t": TIMING_SIGNIFICANCE_T,
            "method_note": TIMING_METHOD_NOTE,
        },
        "eigen": {
            "gap": eigen_gap,
            "distinguishable": eigen_real,
            "prediction_confirmed": False if eigen_real is False else eigen_real,
            "caveat": EIGEN_CAVEAT,
        },
    }


def _net_of(m: MetricValue) -> float:
    """The net figure, asserted present — every §10 book is built from a net series."""
    if m.net is None:
        raise ValueError("MetricValue.net is None; expected a net series for a §10 book")
    return m.net


def timing_tstat(
    exposures: npt.NDArray[np.float64],
    factor_returns: npt.NDArray[np.float64],
    k: int,
) -> float:
    """t-statistic on the timing correlation, so it can be called null.

    A correlation of 0.03 over 1,759 days is not the same claim as 0.03 over
    30, and this project reports distinguishability rather than point
    estimates.
    """
    e, f = exposures[:, k], factor_returns[:, k]
    ok = np.isfinite(e) & np.isfinite(f)
    n = int(ok.sum())
    if n < 30:
        return float("nan")
    r = float(np.corrcoef(e[ok], f[ok])[0, 1])
    if abs(r) >= 1.0:
        return float("inf")
    return float(r * np.sqrt((n - 2) / (1 - r**2)))
