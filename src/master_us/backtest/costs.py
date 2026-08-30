"""Transaction cost model — spec section 5.2 and config/costs.yaml.

Two tiers, and every strategy metric downstream is reported under both:

* **baseline** — flat proportional bps on traded value. The convention readers
  expect; the headline comparison runs here.
* **realistic** — half-spread from the Corwin-Schultz high-low estimator plus
  square-root market impact. Free data has no quoted spreads, so CS is a proxy,
  and section 5 of the data contract gives the sanity check this module must
  pass: mega-cap mean estimated spread in single-digit bps.

Cost accounting convention, used everywhere: a cost is returned in RETURN
units for the rebalance (a 10 bps cost on 40% one-way turnover is
0.4 x 0.0010 = 4 bps of portfolio return). Costs are charged at the rebalance
that incurs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

FloatVec = npt.NDArray[np.float64]

TRADING_DAYS = 252


# --------------------------------------------------------------------- #
# Tier 1 — flat proportional                                             #
# --------------------------------------------------------------------- #


def flat_cost(
    weights_t: npt.NDArray[np.floating],
    weights_tm1: npt.NDArray[np.floating],
    round_trip_bps: float,
) -> float:
    """Flat proportional cost of the rebalance, in return units.

    `round_trip_bps` is the config convention (`baseline_bps: 10` means a full
    buy-and-later-sell costs 10 bps), so ONE side of a trade pays half. Charging
    the full round-trip on every one-way trade double-counts — a book that
    turns over 100% would pay the round-trip twice.
    """
    if round_trip_bps < 0:
        raise ValueError(f"round_trip_bps must be >= 0, got {round_trip_bps}")
    traded = np.abs(np.asarray(weights_t, float) - np.asarray(weights_tm1, float)).sum()
    return float(traded * (round_trip_bps / 2.0) * 1e-4)


# --------------------------------------------------------------------- #
# Tier 2 — Corwin-Schultz spread + sqrt impact                           #
# --------------------------------------------------------------------- #


def corwin_schultz_spread(
    high: npt.NDArray[np.floating],
    low: npt.NDArray[np.floating],
    window: int = 21,
    close: npt.NDArray[np.floating] | None = None,
) -> FloatVec:
    """Two-day high-low spread estimator (Corwin & Schultz 2012), per day.

    beta  = sum over the two days of ln(H/L)²
    gamma = ln(H₂/L₂)² over the two-day joint range
    alpha = (√(2beta) - √beta)/(3 - 2√2) - √(gamma/(3 - 2√2))
    S     = 2(e^alpha - 1)/(1 + e^alpha)

    **Pass `close`.** With it, day t's range is shifted by the overnight gap
    before the two-day quantities are formed, exactly as the original paper
    prescribes (their Section II.C): if L_t > C_{t-1} (gap up), subtract
    (L_t - C_{t-1}) from both H_t and L_t; if H_t < C_{t-1} (gap down), add
    (C_{t-1} - H_t). Without the adjustment, overnight moves land in gamma and
    read as spread — measured on this project's own data, that inflates
    mega-cap estimates from ~2-4 bps to ~13-21 bps, which fails the data
    contract's sanity check (mega-caps must be single-digit bps). The
    unadjusted path is kept only because the contract's signature has no
    close argument; every caller in this repository passes close.

    Negative daily estimates are a known property of the estimator.
    AGGREGATION DEVIATES from the data contract's wording ("floor at zero,
    then rolling median") because that ordering fails the contract's own
    sanity check: measured on this project's data it yields 15-22 bps mean for
    mega-caps, where the contract demands single digits. The cause is
    mechanical — flooring before aggregating turns symmetric estimation noise
    into a one-sided positive bias. Retaining the signed daily estimates,
    taking the rolling MEAN over `window`, and flooring the aggregate (the
    variant Corwin & Schultz themselves note reduces small-spread bias) gives
    AAPL 1.9 / MSFT 5.1 / JNJ 6.5 / KO 6.5 / XOM 3.9 bps, with smaller names
    meaningfully higher (ALK 9.4, NWSA 14.2) — exactly the structure the
    check requires. See NOTES.md, Session 4. Output is the FULL spread as a
    fraction of price; callers wanting the half-spread divide by 2.

    The first `window` values are NaN — a rolling window has no lookahead-free
    value before it fills. Backfilling them would be quiet leakage.
    """
    h = np.asarray(high, dtype=np.float64).copy()
    lo = np.asarray(low, dtype=np.float64).copy()
    if h.shape != lo.shape or h.ndim != 1:
        raise ValueError(f"high {h.shape} and low {lo.shape} must be equal-length 1-D")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    n = h.shape[0]
    if n < 2:
        return np.full(n, np.nan)

    if close is not None:
        c = np.asarray(close, dtype=np.float64)
        if c.shape != h.shape:
            raise ValueError(f"close shape {c.shape} != high {h.shape}")
        prev_close = c[:-1]
        gap_up = np.maximum(lo[1:] - prev_close, 0.0)  # L_t above yesterday's close
        gap_down = np.maximum(prev_close - h[1:], 0.0)  # H_t below yesterday's close
        shift = gap_down - gap_up
        h[1:] += shift
        lo[1:] += shift

    with np.errstate(invalid="ignore", divide="ignore"):
        log_hl_sq = np.log(h / lo) ** 2
        beta = log_hl_sq[:-1] + log_hl_sq[1:]

        h2 = np.maximum(h[:-1], h[1:])
        l2 = np.minimum(lo[:-1], lo[1:])
        gamma = np.log(h2 / l2) ** 2

        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    daily = np.full(n, np.nan)
    daily[1:] = spread  # signed — negatives are retained into the mean

    smoothed = (
        pl.Series(daily)
        .rolling_mean(window_size=window, min_samples=window)
        .to_numpy()
        .astype(np.float64)
    )
    floored: FloatVec = np.maximum(smoothed, 0.0)  # floor the AGGREGATE, not the dailies
    return floored


@dataclass(frozen=True)
class CostConfig:
    """Parsed view of config/costs.yaml. One object, both tiers."""

    baseline_bps: float
    sweep_bps: tuple[float, ...]
    spread_window: int
    spread_floor_bps: float
    impact_coefficient_bps: float
    adv_lookback: int
    max_participation: float

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> CostConfig:
        realistic = cfg.get("realistic", {})
        impact = realistic.get("impact", {})
        estimator = realistic.get("spread_estimator", "corwin_schultz")
        if estimator != "corwin_schultz":
            raise NotImplementedError(f"spread_estimator={estimator!r} is not built")
        if impact.get("model", "sqrt") != "sqrt":
            raise NotImplementedError(f"impact model={impact.get('model')!r} is not built")
        return cls(
            baseline_bps=float(cfg["baseline_bps"]),
            sweep_bps=tuple(float(x) for x in cfg.get("sweep_bps", [])),
            spread_window=int(realistic.get("spread_window", 21)),
            spread_floor_bps=float(realistic.get("spread_floor_bps", 1.0)),
            impact_coefficient_bps=float(impact.get("coefficient_bps", 10.0)),
            adv_lookback=int(impact.get("adv_lookback", 21)),
            max_participation=float(impact.get("max_participation", 0.10)),
        )


def realistic_cost(
    weights_t: npt.NDArray[np.floating],
    weights_tm1: npt.NDArray[np.floating],
    spread: npt.NDArray[np.floating],
    adv_dollars: npt.NDArray[np.floating],
    portfolio_value: float,
    cfg: CostConfig,
) -> float:
    """Half-spread plus square-root impact for one rebalance, in return units.

    Per name: |Δw| · (spread/2 + coef_bps · √participation), where
    participation = min(|Δw| · portfolio_value / ADV$, max_participation).

    A name with a NaN spread (estimator window unfilled) or NaN/zero ADV uses
    the spread floor and zero participation rather than silently free trading —
    the floor exists precisely so unknowns cost something.
    """
    w1 = np.asarray(weights_t, dtype=np.float64)
    w0 = np.asarray(weights_tm1, dtype=np.float64)
    dw = np.abs(w1 - w0)
    if not dw.any():
        return 0.0
    if portfolio_value <= 0:
        raise ValueError(f"portfolio_value must be positive, got {portfolio_value}")

    sp = np.asarray(spread, dtype=np.float64).copy()
    floor = cfg.spread_floor_bps * 1e-4
    sp = np.where(np.isfinite(sp), np.maximum(sp, floor), floor)

    adv = np.asarray(adv_dollars, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        participation = np.where(adv > 0, dw * portfolio_value / adv, 0.0)
    participation = np.minimum(np.nan_to_num(participation, nan=0.0), cfg.max_participation)

    per_name = dw * (sp / 2.0 + cfg.impact_coefficient_bps * 1e-4 * np.sqrt(participation))
    return float(per_name.sum())
