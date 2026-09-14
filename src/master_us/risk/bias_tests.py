"""Phase-6 gate — bias tests (spec §9.5).

    b = std(realized_return / predicted_vol),  target ~ 1.0
    b > 1  =>  under-forecasting risk
    b < 1  =>  over-forecasting risk

WHAT THIS GATE DOES AND DOES NOT ESTABLISH. It validates CALIBRATION: that
the predicted volatility of a portfolio matches its realized dispersion. It
says nothing about whether any factor earns a return. A well-calibrated
covariance built on factor returns that are weak but correctly signed is a
perfectly valid risk model, and Phase 5's momentum premium (+0.92%/yr,
t=1.04) is exactly that — right sign, not distinguishable from zero. A good
bias statistic must never be read, reported, or quoted as evidence that
momentum or any other factor is a strong signal. The two claims are
independent and are kept separate everywhere they appear.

OUT-OF-SAMPLE BY CONSTRUCTION. The forecast for period t+1 uses the
covariance estimated through period t. `predicted[t]` is paired with
`realized[t+1]`, never with `realized[t]`. An in-sample bias statistic sits
suspiciously close to 1.0 for the same reason an in-sample R^2 looks good,
so the alignment is the single thing most worth checking if the number comes
back too clean.

BENCHMARKS. §9.5 requires comparison against a sample covariance and against
Ledoit-Wolf shrinkage. The model has to win or match; a bias statistic near 1
is not impressive if a plain sample covariance does the same on this data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

GATE_LO, GATE_HI = 0.9, 1.1
N_RANDOM_PORTFOLIOS = 1000


@dataclass(frozen=True)
class BiasResult:
    """One estimator's bias statistics, by portfolio family."""

    label: str
    by_family: dict[str, npt.NDArray[np.float64]]

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for family, stats in self.by_family.items():
            good = stats[np.isfinite(stats)]
            out[family] = float(np.median(good)) if good.size else float("nan")
        return out

    @property
    def all_stats(self) -> npt.NDArray[np.float64]:
        if not self.by_family:
            return np.array([])
        joined = np.concatenate([v.ravel() for v in self.by_family.values()])
        return np.asarray(joined[np.isfinite(joined)], dtype=np.float64)

    def in_gate_fraction(self, lo: float = GATE_LO, hi: float = GATE_HI) -> float:
        s = self.all_stats
        return float(((s >= lo) & (s <= hi)).mean()) if s.size else float("nan")


def bias_statistic(
    realized: npt.NDArray[np.float64], predicted_vol: npt.NDArray[np.float64]
) -> float:
    """std of standardized returns — §9.5.

    Uses the standard deviation about zero rather than about the sample
    mean: the quantity being validated is whether predicted vol matches the
    SCALE of realized returns, and a portfolio with genuine drift should not
    have that drift quietly removed before its risk is judged.
    """
    ok = np.isfinite(realized) & np.isfinite(predicted_vol) & (predicted_vol > 0)
    if ok.sum() < 20:
        return float("nan")
    z = realized[ok] / predicted_vol[ok]
    return float(np.sqrt(np.mean(z**2)))


def random_portfolios(
    n_names: int, n_portfolios: int = N_RANDOM_PORTFOLIOS, seed: int = 0, n_held: int = 50
) -> npt.NDArray[np.float64]:
    """Long-short, dollar-neutral, unit-gross random portfolios.

    Dollar-neutral because a long-only random portfolio is ~95% market
    factor, and its bias statistic would then measure how well the market
    factor's variance is forecast, 1000 times over, rather than exercising
    the rest of the covariance.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros((n_portfolios, n_names))
    for i in range(n_portfolios):
        pick = rng.choice(n_names, size=min(n_held, n_names), replace=False)
        half = len(pick) // 2
        w[i, pick[:half]] = 1.0 / (2 * half)
        w[i, pick[half : 2 * half]] = -1.0 / (2 * half)
    return w


def factor_mimicking_portfolios(
    exposures: npt.NDArray[np.float64], mcap: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """One unit-exposure portfolio per factor, from a cap-weighted regression.

    w_k = e_k' (X' W X)^-1 X' W — the weights that give unit exposure to
    factor k and zero to the others, which is the portfolio whose return IS
    the factor return.
    """
    ok = np.isfinite(exposures).all(axis=1) & np.isfinite(mcap) & (mcap > 0)
    n, k = exposures.shape
    out = np.zeros((k, n))
    if ok.sum() < k + 10:
        return out
    x = exposures[ok]
    w = np.sqrt(mcap[ok])
    xtw = x.T * w
    try:
        inv = np.linalg.pinv(xtw @ x)
    except np.linalg.LinAlgError:
        return out
    out[:, np.flatnonzero(ok)] = inv @ xtw
    return out


def cap_weighted_market(
    mcap: npt.NDArray[np.float64], is_primary: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    w = np.where(np.isfinite(mcap) & (mcap > 0) & is_primary, mcap, 0.0)
    total = w.sum()
    return w / total if total > 0 else w


def portfolio_vol(
    weights: npt.NDArray[np.float64],
    exposures: npt.NDArray[np.float64],
    factor_cov: npt.NDArray[np.float64],
    specific_var: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """sqrt(w'(X F X' + D)w) for a stack of portfolios.

    Never forms X F X' — that is (N x N) and N is ~580, so the stacked form
    costs a 580x580 matrix per date per estimator. The portfolio's factor
    exposure `b = w' X` is (P x K) and the whole thing collapses to
    `b F b'` plus the specific term.
    """
    x = np.nan_to_num(exposures, nan=0.0)
    d = np.nan_to_num(specific_var, nan=0.0)
    b = weights @ x
    factor_var = np.einsum("pk,kl,pl->p", b, factor_cov, b)
    spec_var = (weights**2) @ d
    return np.asarray(np.sqrt(np.clip(factor_var + spec_var, 0.0, None)), dtype=np.float64)


def sample_covariance(
    returns: npt.NDArray[np.float64], window: int = 252
) -> npt.NDArray[np.float64]:
    """Rolling sample covariance — the §9.5 naive benchmark."""
    t, k = returns.shape
    out = np.full((t, k, k), np.nan)
    for i in range(window, t):
        block = returns[i - window : i]
        block = block[np.isfinite(block).all(axis=1)]
        if len(block) > k:
            out[i] = np.cov(block, rowvar=False)
    return out


def ledoit_wolf_covariance(
    returns: npt.NDArray[np.float64], window: int = 252
) -> npt.NDArray[np.float64]:
    """Rolling Ledoit-Wolf shrinkage toward a scaled identity — §9.5.

    Implemented directly rather than via sklearn so the rolling window and
    the NaN handling match the other estimators exactly; a benchmark that
    differs from the model in its sample handling is not a benchmark.
    """
    t, k = returns.shape
    out = np.full((t, k, k), np.nan)
    for i in range(window, t):
        block = returns[i - window : i]
        block = block[np.isfinite(block).all(axis=1)]
        n = len(block)
        if n <= k:
            continue
        x = block - block.mean(axis=0)
        s = (x.T @ x) / n
        mu = np.trace(s) / k
        target = mu * np.eye(k)
        d2 = np.sum((s - target) ** 2) / k
        b_bar2 = sum(np.sum((np.outer(x[j], x[j]) - s) ** 2) / k for j in range(n)) / n**2
        shrink = float(np.clip(b_bar2 / d2, 0.0, 1.0)) if d2 > 0 else 1.0
        out[i] = shrink * target + (1.0 - shrink) * s
    return out


def run_bias_tests(
    factor_returns: npt.NDArray[np.float64],
    exposures_by_date: npt.NDArray[np.float64],
    asset_returns: npt.NDArray[np.float64],
    specific_var: npt.NDArray[np.float64],
    mcap: npt.NDArray[np.float64],
    is_primary: npt.NDArray[np.bool_],
    factor_covs: dict[str, npt.NDArray[np.float64]],
    start: int,
    seed: int = 0,
) -> dict[str, BiasResult]:
    """Bias statistics per estimator, across the three §9.5 portfolio families.

    `factor_covs[label][t]` must be the forecast made FROM data through t.
    It is paired with `asset_returns[t + 1]`, so every statistic here is out
    of sample.
    """
    t, n = asset_returns.shape
    rand_w = random_portfolios(n, seed=seed)

    results: dict[str, BiasResult] = {}
    for label, covs in factor_covs.items():
        rand_real, rand_pred = [], []
        fm_real, fm_pred = [], []
        mkt_real, mkt_pred = [], []

        for i in range(start, t - 1):
            cov = covs[i]
            if cov is None or not np.isfinite(cov).all():
                continue
            x = exposures_by_date[i]
            d = specific_var[i]
            if not np.isfinite(d).any():
                continue
            nxt = asset_returns[i + 1]
            # A portfolio can only be judged on the names it actually holds.
            # Masking the WEIGHTS (rather than zeroing the missing returns)
            # is what keeps predicted and realized referring to the same
            # portfolio: only ~64% of names are in the regression on a given
            # day, so charging predicted variance for a name that
            # contributes no realized return inflated predicted variance by
            # ~1/0.64 and drove the whole gate. The bias statistic is scale
            # invariant per portfolio, so the masked weights need no
            # renormalization — both sides scale together.
            held = np.isfinite(nxt) & np.isfinite(d) & (d > 0)
            if held.sum() < 30:
                continue

            safe_next = np.where(held, nxt, 0.0)
            hold_mask = held.astype(float)

            rw = rand_w * hold_mask
            keep = np.abs(rw).sum(axis=1) > 0
            if keep.any():
                pred = np.full(len(rand_w), np.nan)
                real = np.full(len(rand_w), np.nan)
                pred[keep] = portfolio_vol(rw[keep], x, cov, d)
                real[keep] = rw[keep] @ safe_next
                rand_pred.append(pred)
                rand_real.append(real)

            fm_w = factor_mimicking_portfolios(x, mcap[i]) * hold_mask
            if np.abs(fm_w).sum() > 0:
                fm_pred.append(portfolio_vol(fm_w, x, cov, d))
                fm_real.append(fm_w @ safe_next)

            mk = cap_weighted_market(mcap[i], is_primary) * hold_mask
            if mk.sum() > 0:
                mk = mk / mk.sum()
                mkt_pred.append(portfolio_vol(mk[None, :], x, cov, d)[0])
                mkt_real.append(float(mk @ safe_next))

        by_family: dict[str, npt.NDArray[np.float64]] = {}
        if rand_real:
            rr, rp = np.array(rand_real), np.array(rand_pred)
            by_family["random"] = np.array(
                [bias_statistic(rr[:, p], rp[:, p]) for p in range(rr.shape[1])]
            )
        if fm_real:
            fr_, fp = np.array(fm_real), np.array(fm_pred)
            by_family["factor_mimicking"] = np.array(
                [bias_statistic(fr_[:, p], fp[:, p]) for p in range(fr_.shape[1])]
            )
        if mkt_real:
            by_family["cap_weighted_market"] = np.array(
                [bias_statistic(np.array(mkt_real), np.array(mkt_pred))]
            )
        results[label] = BiasResult(label=label, by_family=by_family)
    return results


def render_table(results: dict[str, BiasResult]) -> str:
    families = ["random", "factor_mimicking", "cap_weighted_market"]
    head = f"{'estimator':<22}" + "".join(f"{f:>20}" for f in families) + f"{'in [0.9,1.1]':>14}"
    lines = [head, "-" * len(head)]
    for label, res in results.items():
        s = res.summary()
        row = f"{label:<22}"
        for f in families:
            v = s.get(f, float("nan"))
            row += f"{v:>20.3f}" if np.isfinite(v) else f"{'—':>20}"
        row += f"{res.in_gate_fraction() * 100:>13.1f}%"
        lines.append(row)
    return "\n".join(lines)


def gate_passed(result: BiasResult, lo: float = GATE_LO, hi: float = GATE_HI) -> bool:
    """§9.5: bias statistic in [0.9, 1.1] for the MAJORITY of portfolios."""
    return result.in_gate_fraction(lo, hi) > 0.5
