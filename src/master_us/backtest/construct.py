"""Portfolio construction — spec section 5.1.

All three constructors share one convention:

    scores:  (N,) float, higher = better. NaN allowed; a NaN score is treated
             as untradeable regardless of the mask.
    mask:    (N,) bool, True = in the tradeable universe today.
    return:  (N,) float weights. Long-short books are dollar-neutral with
             gross exposure 2 (long leg sums to +1, short leg to -1); long-only
             books sum to +1. Weights are 0 outside the mask, always.

The constructors are deliberately stateless: turnover control lives either in
the buffer rule (`topk_dropout`) or in the optimizer penalty
(`cost_aware_optimize`), never in hidden instance state that a backtest loop
could desynchronize from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatVec = npt.NDArray[np.float64]
BoolVec = npt.NDArray[np.bool_]


def _validated(
    scores: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
) -> tuple[FloatVec, BoolVec]:
    s = np.asarray(scores, dtype=np.float64)
    m = np.asarray(mask)
    if s.ndim != 1 or m.ndim != 1 or s.shape != m.shape:
        raise ValueError(f"scores {s.shape} and mask {m.shape} must be equal-length 1-D")
    if m.dtype != np.bool_:
        raise TypeError(f"mask must be bool, got {m.dtype}")
    # NaN score = no opinion = untradeable. Silently ranking NaN would put it
    # wherever the sort implementation feels like, which is worse than a crash.
    return s, m & ~np.isnan(s)


def topk_dropout(
    scores: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    prev_weights: npt.NDArray[np.floating] | None = None,
    k: int = 50,
    dropout_buffer: int = 25,
) -> FloatVec:
    """Original paper's convention: hold top-k, sell only below rank k + buffer.

    An existing holding keeps its seat while it ranks within k + dropout_buffer;
    a new name enters only through the top k. The buffer is the cheapest
    turnover control available — it converts rank jitter around the k-th seat
    from a trade into a hold — and it materially changes net performance.

    `prev_weights` carries yesterday's book. None (or all-zero) means first
    rebalance: plain top-k. Long-only, equal-weight, sums to 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if dropout_buffer < 0:
        raise ValueError(f"dropout_buffer must be >= 0, got {dropout_buffer}")

    s, tradeable = _validated(scores, mask)
    n_live = int(tradeable.sum())
    if n_live == 0:
        return np.zeros_like(s)

    # Dense ranks over the tradeable set only: rank 1 = best.
    order = np.argsort(-s[tradeable], kind="stable")
    live_idx = np.flatnonzero(tradeable)[order]
    rank = np.full(s.shape, np.iinfo(np.int64).max, dtype=np.int64)
    rank[live_idx] = np.arange(1, n_live + 1)

    held_before = (
        np.zeros(s.shape, dtype=bool)
        if prev_weights is None
        else np.asarray(prev_weights, dtype=np.float64) > 0.0
    )
    if held_before.shape != s.shape:
        raise ValueError(f"prev_weights shape {held_before.shape} != scores {s.shape}")

    # Keep an incumbent while it survives the buffer AND is still tradeable.
    keep = held_before & tradeable & (rank <= k + dropout_buffer)

    # Fill remaining seats with the best-ranked non-holdings.
    seats_left = k - int(keep.sum())
    holdings = keep.copy()
    if seats_left > 0:
        for idx in live_idx:  # already in rank order
            if not holdings[idx]:
                holdings[idx] = True
                seats_left -= 1
                if seats_left == 0:
                    break
    elif seats_left < 0:
        # More incumbents survive the buffer than seats. Spec's rule is "sold
        # only when it falls below k + buffer", so the book is allowed to be
        # temporarily oversized; equal-weighting below renormalizes.
        pass

    weights = np.zeros_like(s)
    n_held = int(holdings.sum())
    if n_held:
        weights[holdings] = 1.0 / n_held
    return weights


def decile_long_short(
    scores: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_deciles: int = 10,
    weighting: str = "equal",
) -> FloatVec:
    """Academic standard: long the top decile, short the bottom decile.

    Dollar-neutral, equal-weight within each leg: long leg sums to +1, short
    leg to -1. Decile membership is by rank over today's tradeable set, so leg
    sizes track the universe (~50 names per leg at N≈500).
    """
    if n_deciles < 2:
        raise ValueError(f"n_deciles must be >= 2, got {n_deciles}")
    if weighting != "equal":
        raise NotImplementedError(
            f"weighting={weighting!r} — only 'equal' exists; score-weighting is a "
            "deliberate non-feature until something needs it"
        )

    s, tradeable = _validated(scores, mask)
    n_live = int(tradeable.sum())
    if n_live < 2 * n_deciles:
        # Fewer than two names per leg is a coin flip wearing a portfolio's
        # clothes. An empty book is the honest output.
        return np.zeros_like(s)

    leg = n_live // n_deciles
    order = np.argsort(-s[tradeable], kind="stable")
    live_idx = np.flatnonzero(tradeable)[order]

    weights = np.zeros_like(s)
    weights[live_idx[:leg]] = 1.0 / leg
    weights[live_idx[-leg:]] = -1.0 / leg
    return weights


def cost_aware_optimize(
    scores: npt.NDArray[np.floating],
    prev_w: npt.NDArray[np.floating],
    cov: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    lam_turnover: float,
    lam_risk: float,
    constraints: dict[str, Any] | None = None,
) -> FloatVec:
    """max  s'w - lam_risk·w'Σw - lam_turnover·||w - prev_w||₁

    s.t. sum(w) = 0 (dollar-neutral, the default) or 1 (`budget: 1`),
         |w_i| <= cap (`max_weight`), optional factor-neutrality rows
         (`neutralize`: (F, N) matrix A with A @ w = 0).

    Used in Phase 7 for the neutralized variant; built and tested now so the
    engine's interface is complete before any model exists.
    """
    import cvxpy as cp

    cfg = dict(constraints or {})
    s, tradeable = _validated(scores, mask)
    n = s.shape[0]

    prev = np.asarray(prev_w, dtype=np.float64)
    if prev.shape != s.shape:
        raise ValueError(f"prev_w shape {prev.shape} != scores {s.shape}")
    sigma = np.asarray(cov, dtype=np.float64)
    if sigma.shape != (n, n):
        raise ValueError(f"cov shape {sigma.shape}, expected {(n, n)}")
    if lam_turnover < 0 or lam_risk < 0:
        raise ValueError("lam_turnover and lam_risk must be >= 0")

    budget = float(cfg.pop("budget", 0.0))
    max_weight = float(cfg.pop("max_weight", 0.05))
    neutralize = cfg.pop("neutralize", None)
    if cfg:
        raise ValueError(f"unknown constraint keys: {sorted(cfg)}")

    # Scores are the only place NaN can hide; the optimizer must never see one.
    s_clean = np.where(tradeable, np.nan_to_num(s, nan=0.0), 0.0)

    w = cp.Variable(n)
    objective = cp.Maximize(
        s_clean @ w
        - lam_risk * cp.quad_form(w, cp.psd_wrap(sigma))
        - lam_turnover * cp.norm1(w - prev)
    )
    cons = [
        cp.sum(w) == budget,
        cp.abs(w) <= max_weight,
        w[~tradeable] == 0,
    ]
    if neutralize is not None:
        a = np.asarray(neutralize, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != n:
            raise ValueError(f"neutralize must be (F, {n}), got {a.shape}")
        cons.append(a @ w == 0)

    problem = cp.Problem(objective, cons)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"cost_aware_optimize did not solve: status={problem.status!r}. "
            "Loosen max_weight or check the covariance for non-PSD garbage."
        )
    if w.value is None:  # pragma: no cover — guarded by status check
        raise RuntimeError("solver reported success but returned no solution")

    out = np.asarray(w.value, dtype=np.float64)
    out[~tradeable] = 0.0
    out[np.abs(out) < 1e-10] = 0.0  # solver dust, not positions
    return out
