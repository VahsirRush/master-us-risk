"""Stress tests and cost analysis — spec §8.4, §8.5.

All of this runs off `data/processed/phase4/metrics_bundle.npz` (~21 MB), not
the 1.2 GB panel, so it is safe to run beside a live training job. That is not
a convenience: the training guard added in Session 9 refuses panel loads while
a run is live, and analysis that needed the panel would have been blocked
exactly when it was most useful.

The §8.5 slices exist because a single test-period number hides regime
dependence. A signal that works only in 2019 and only in mega-caps is a
different object from one that works throughout, and the two are
indistinguishable in the headline table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as scipy_stats

from master_us.data.sources import DATA_ROOT

BUNDLE_PATH = DATA_ROOT / "processed" / "phase4" / "metrics_bundle.npz"

# §8.5 regime slices. Each is a period whose character is documented and
# different: a volatility shock, a crash, a bear year, and a recovery.
REGIMES: tuple[tuple[str, str, str], ...] = (
    ("2018Q4 selloff", "2018-10-01", "2018-12-31"),
    ("COVID crash", "2020-02-01", "2020-04-30"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
    ("2023-2025 recovery", "2023-01-01", "2025-12-31"),
)

DECAY_HORIZONS = (1, 3, 5, 10, 21)


@dataclass(frozen=True)
class Bundle:
    """The small arrays every §8.4/§8.5 analysis needs."""

    dates: pd.DatetimeIndex
    labels: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    raw_forward: npt.NDArray[np.float32]
    test_idx: npt.NDArray[np.int64]
    tickers: npt.NDArray[np.object_]
    sectors: npt.NDArray[np.object_]
    mcap: npt.NDArray[np.float32]

    @property
    def label_valid(self) -> npt.NDArray[np.bool_]:
        return self.mask & np.isfinite(self.labels)


def load_bundle(path: Path = BUNDLE_PATH) -> Bundle:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `scripts/40_ablations.py --bundle-only` first"
        )
    z = np.load(path, allow_pickle=True)
    return Bundle(
        dates=pd.DatetimeIndex(z["dates"]),
        labels=z["labels"],
        mask=z["mask"],
        raw_forward=z["raw_forward"],
        test_idx=z["test_idx"],
        tickers=z["tickers"],
        sectors=z["sectors"],
        mcap=z["mcap"],
    )


def rank_ic(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.floating],
    valid: npt.NDArray[np.bool_],
    rows: npt.NDArray[np.int64],
    min_names: int = 10,
) -> float:
    """Mean per-date Spearman over the given rows. NaN if no date qualifies."""
    out = []
    for t in rows:
        ok = valid[t] & np.isfinite(scores[t]) & np.isfinite(labels[t])
        if ok.sum() < min_names:
            continue
        out.append(float(scipy_stats.spearmanr(scores[t, ok], labels[t, ok]).statistic))
    return float(np.mean(out)) if out else float("nan")


# --------------------------------------------------------------------- #
# §8.5 slices                                                            #
# --------------------------------------------------------------------- #


def regime_slices(
    bundle: Bundle, scores: npt.NDArray[np.floating]
) -> dict[str, tuple[float, int]]:
    """RankIC within each documented regime window, with its date count.

    The count is returned, not just the IC, because a window can fall wholly
    outside the test split — 2018Q4 does, since test starts 2019-01-01 — and
    "0 test dates" is a different statement from "IC could not be estimated".
    Reporting a bare NaN would blur the two.
    """
    out: dict[str, tuple[float, int]] = {}
    for name, lo, hi in REGIMES:
        sel = (bundle.dates >= pd.Timestamp(lo)) & (bundle.dates <= pd.Timestamp(hi))
        rows = np.intersect1d(np.flatnonzero(np.asarray(sel)), bundle.test_idx)
        value = rank_ic(scores, bundle.labels, bundle.label_valid, rows) if len(rows) else float("nan")
        out[name] = (value, len(rows))
    return out


def sector_neutralized(
    bundle: Bundle, scores: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
    """Residualize each date's scores on GICS L1 — i.e. demean within sector.

    A signal that survives this is picking stocks; one that does not was
    largely making a sector bet, which a risk model would strip out anyway.
    """
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    sectors = bundle.sectors
    codes = {s: i for i, s in enumerate(sorted(set(sectors.tolist())))}
    code_arr = np.array([codes[s] for s in sectors])

    for t in bundle.test_idx:
        ok = bundle.mask[t] & np.isfinite(scores[t])
        if ok.sum() < 10:
            continue
        row = scores[t].astype(np.float64)
        resid = np.full(scores.shape[1], np.nan)
        for c in np.unique(code_arr[ok]):
            members = ok & (code_arr == c)
            if members.sum() >= 2:
                resid[members] = row[members] - row[members].mean()
            elif members.sum() == 1:
                resid[members] = 0.0  # a lone name carries no within-sector signal
        out[t] = resid
    return out


def cap_tiers(
    bundle: Bundle, scores: npt.NDArray[np.floating]
) -> dict[str, float]:
    """RankIC within large / mid / small tiers, split per date by market cap."""
    tiers: dict[str, list[float]] = {"large": [], "mid": [], "small": []}
    for t in bundle.test_idx:
        ok = bundle.label_valid[t] & np.isfinite(scores[t]) & np.isfinite(bundle.mcap[t])
        if ok.sum() < 30:
            continue
        caps = bundle.mcap[t][ok]
        lo, hi = np.quantile(caps, [1 / 3, 2 / 3])
        idx = np.flatnonzero(ok)
        groups = {
            "small": idx[caps <= lo],
            "mid": idx[(caps > lo) & (caps <= hi)],
            "large": idx[caps > hi],
        }
        for name, cols in groups.items():
            if len(cols) >= 10:
                r = scipy_stats.spearmanr(scores[t, cols], bundle.labels[t, cols]).statistic
                if np.isfinite(r):
                    tiers[name].append(float(r))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in tiers.items()}


def decay_curve(
    bundle: Bundle,
    scores: npt.NDArray[np.floating],
    adj_forward: npt.NDArray[np.floating] | None = None,
) -> dict[int, float]:
    """IC of today's score against forward returns at 1, 3, 5, 10, 21 days.

    The bundle carries the 1-day forward return; longer horizons are compounded
    from it, which is exact for the log-free product of consecutive simple
    returns.
    """
    fwd = bundle.raw_forward if adj_forward is None else adj_forward
    out: dict[int, float] = {}
    t_len = fwd.shape[0]
    for h in DECAY_HORIZONS:
        target = np.full(fwd.shape, np.nan, dtype=np.float64)
        acc = np.ones(fwd.shape, dtype=np.float64)
        for k in range(h):
            shifted = np.full(fwd.shape, np.nan)
            shifted[: t_len - k] = fwd[k:]
            acc *= 1.0 + shifted
        target = acc - 1.0
        out[h] = rank_ic(scores, target, bundle.mask, bundle.test_idx)
    return out


# --------------------------------------------------------------------- #
# §8.4 cost analysis                                                     #
# --------------------------------------------------------------------- #


def breakeven_table(
    gross: npt.NDArray[np.floating], turnover: npt.NDArray[np.floating]
) -> dict[str, float]:
    """Breakeven bps plus the net-Sharpe curve at the config's sweep levels."""
    from master_us.experiments.ablations import cost_breakeven, net_sharpe_at_bps

    out = {"breakeven_bps": cost_breakeven(gross, turnover)}
    for bps in (0.0, 5.0, 10.0, 20.0, 50.0):
        out[f"net_sharpe_{bps:.0f}bps"] = net_sharpe_at_bps(gross, turnover, bps)
    return out
