"""LightGBM hyperparameter search — VALIDATION ONLY.

The spec's stock configuration (500 trees, lr 0.05, num_leaves 64) misses the
Phase-2 gate on this panel: test RankIC 0.0185 +/- 0.0005 against a bar of
0.02 (Session 6, measured). This module does the tuning the gate deserves.

THE RULE THAT SHAPES EVERYTHING HERE: every candidate is scored on VALIDATION
RankIC and nothing else. CLAUDE.md rule 3 says test is touched once per model,
at the end. Searching against test would manufacture a passing gate out of
selection bias — the exact failure this project exists to avoid. The search
reports validation numbers; the winner is run once on test afterwards.

The hypotheses being tested, each with a reason:

* **capacity** — num_leaves 64 with 500 trees is enormous for a signal whose
  best single feature has |IC| 0.0154. Weak-signal financial ML is usually
  regularization-bound, not capacity-bound.
* **feature subsampling** — the bank is 130 features of which 65 are
  cross-sectional RANKS of the other 65. Columns are near-duplicates in
  pairs, so a low colsample decorrelates trees more than it costs.
* **feature subset** — rank features are per-date scale-free and match the
  per-date z-scored target's geometry; the raw features are normalized with
  GLOBAL train statistics and may just add noise for a tree.
* **min_child_samples** — with ~700k training rows, leaves of 100 are small
  enough to fit date-specific noise.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from master_us.models.baselines.lgbm import run_lgbm
from master_us.models.train import PreparedData, rank_ic_by_date


@dataclass(frozen=True)
class Candidate:
    """One configuration and why it is being tried."""

    name: str
    hypothesis: str
    params: dict[str, Any] = field(default_factory=dict)
    feature_subset: str = "all"  # "all" | "ranks" | "raw"


@dataclass(frozen=True)
class SearchRow:
    name: str
    hypothesis: str
    valid_rank_ic: float
    best_iteration: int
    seconds: float
    params: dict[str, Any]
    feature_subset: str


BASE = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
}


def candidates() -> list[Candidate]:
    """The search grid. Small and reasoned, not a random sweep."""
    return [
        Candidate("spec", "the configuration config/baselines.yaml ships with", dict(BASE)),
        Candidate(
            "leaves_31",
            "halve capacity: 64 leaves overfits a max-|IC|-0.0154 signal",
            BASE | {"num_leaves": 31},
        ),
        Candidate(
            "leaves_15",
            "quarter capacity",
            BASE | {"num_leaves": 15},
        ),
        Candidate(
            "leaves_7",
            "shallow trees; near-additive model over weak features",
            BASE | {"num_leaves": 7},
        ),
        Candidate(
            "leaves_15_slow",
            "shallow + slow: more trees, smaller steps",
            BASE | {"num_leaves": 15, "learning_rate": 0.02, "n_estimators": 1500},
        ),
        Candidate(
            "leaves_15_child1000",
            "shallow + large leaves: forbid fitting date-specific noise",
            BASE | {"num_leaves": 15, "min_child_samples": 1000},
        ),
        Candidate(
            "leaves_15_colsample04",
            "decorrelate trees across the raw/rank near-duplicate pairs",
            BASE | {"num_leaves": 15, "colsample_bytree": 0.4},
        ),
        Candidate(
            "ranks_only_leaves_15",
            "rank features match the per-date z-scored target's geometry",
            BASE | {"num_leaves": 15},
            feature_subset="ranks",
        ),
        Candidate(
            "ranks_only_tuned",
            "best-guess combination of the above, ranks only",
            BASE
            | {
                "num_leaves": 15,
                "learning_rate": 0.02,
                "n_estimators": 2000,
                "min_child_samples": 500,
                "colsample_bytree": 0.5,
                "early_stopping_rounds": 100,
            },
            feature_subset="ranks",
        ),
        Candidate(
            "all_tuned",
            "same combination, full feature bank",
            BASE
            | {
                "num_leaves": 15,
                "learning_rate": 0.02,
                "n_estimators": 2000,
                "min_child_samples": 500,
                "colsample_bytree": 0.5,
                "early_stopping_rounds": 100,
            },
        ),
    ]


def subset_columns(data: PreparedData, subset: str) -> np.ndarray:
    """Column indices for a feature subset.

    Missingness indicators (the `__isnan` group columns) ride along in every
    subset — they are group-level flags, not per-feature duplicates.
    """
    names = data.feature_names
    if subset == "all":
        return np.arange(len(names))
    if subset == "ranks":
        keep = [i for i, n in enumerate(names) if n.startswith("xs_") or n.endswith("__isnan")]
    elif subset == "raw":
        keep = [
            i for i, n in enumerate(names) if not n.startswith("xs_") or n.endswith("__isnan")
        ]
    else:
        raise ValueError(f"unknown feature subset {subset!r}")
    return np.asarray(keep, dtype=np.int64)


def with_subset(data: PreparedData, subset: str) -> PreparedData:
    """A view of `data` restricted to a feature subset."""
    if subset == "all":
        return data
    cols = subset_columns(data, subset)
    return PreparedData(
        dates=data.dates,
        features=data.features[:, :, cols],
        market=data.market,
        labels=data.labels,
        mask=data.mask,
        raw_forward=data.raw_forward,
        lookback=data.lookback,
        train_idx=data.train_idx,
        valid_idx=data.valid_idx,
        test_idx=data.test_idx,
        feature_names=tuple(data.feature_names[i] for i in cols),
    )


def search(
    data: PreparedData,
    seed: int = 0,
    grid: list[Candidate] | None = None,
    log: Callable[[str], None] = print,
) -> list[SearchRow]:
    """Score every candidate on VALIDATION RankIC. Never touches test."""
    rows: list[SearchRow] = []
    for cand in grid or candidates():
        t0 = time.monotonic()
        view = with_subset(data, cand.feature_subset)
        result = run_lgbm(view, seed, cand.params)
        # run_lgbm's own valid_rank_ic is computed on the validation split.
        valid_ic = float(
            np.nanmean(
                rank_ic_by_date(result.scores, view.labels, view.label_valid, view.valid_idx)
            )
        )
        row = SearchRow(
            name=cand.name,
            hypothesis=cand.hypothesis,
            valid_rank_ic=valid_ic,
            best_iteration=result.best_iteration,
            seconds=time.monotonic() - t0,
            params=cand.params,
            feature_subset=cand.feature_subset,
        )
        rows.append(row)
        log(
            f"  {cand.name:<24} valid RankIC {valid_ic:+.4f}  "
            f"best_iter {result.best_iteration:>4}  [{row.seconds:.0f}s]  {cand.feature_subset}"
        )
    return rows


def best(rows: list[SearchRow]) -> SearchRow:
    return max(rows, key=lambda r: r.valid_rank_ic)
