"""LightGBM — the real bar. Gate: out-of-sample RankIC > 0.02 (spec section 6).

Early stopping is on VALIDATION RankIC, not l2: the task is per-date ranking,
and an l2-stopped model routinely stops at the wrong place for it. The custom
eval groups validation rows by date and averages per-date Spearman.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

from master_us.models.train import PreparedData, rank_ic_by_date


@dataclass(frozen=True)
class LgbmResult:
    scores: npt.NDArray[np.float32]
    valid_rank_ic: float
    best_iteration: int


def _pooled_with_dates(
    data: PreparedData, date_idx: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    lv = data.label_valid
    xs, ys, ds = [], [], []
    for t in date_idx:
        ok = lv[t]
        if ok.sum() < 10:
            continue
        xs.append(data.features[t, ok])
        ys.append(data.labels[t, ok])
        ds.append(np.full(int(ok.sum()), t, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)


def run_lgbm(data: PreparedData, seed: int, cfg: dict[str, Any]) -> LgbmResult:
    x_train, y_train, _ = _pooled_with_dates(data, data.train_idx)
    x_valid, y_valid, d_valid = _pooled_with_dates(data, data.valid_idx)

    # Precompute per-date row slices for the eval metric.
    order = np.argsort(d_valid, kind="stable")
    x_valid, y_valid, d_valid = x_valid[order], y_valid[order], d_valid[order]
    boundaries = np.flatnonzero(np.diff(d_valid)) + 1
    slices = np.split(np.arange(len(d_valid)), boundaries)

    def rank_ic_eval(
        y_true: np.ndarray | None, y_pred: np.ndarray
    ) -> tuple[str, float, bool]:
        truth = y_valid if y_true is None else y_true
        ics = [
            float(scipy_stats.spearmanr(y_pred[rows], truth[rows]).statistic)
            for rows in slices
            if len(rows) >= 10
        ]
        return "rank_ic", float(np.mean(ics)), True  # higher is better

    model = lgb.LGBMRegressor(
        n_estimators=int(cfg.get("n_estimators", 500)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        num_leaves=int(cfg.get("num_leaves", 64)),
        min_child_samples=int(cfg.get("min_child_samples", 100)),
        subsample=float(cfg.get("subsample", 0.8)),
        subsample_freq=1,
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=rank_ic_eval,
        callbacks=[
            lgb.early_stopping(int(cfg.get("early_stopping_rounds", 50)), first_metric_only=True, verbose=False)
        ],
    )

    scores = np.full(data.labels.shape, np.nan, dtype=np.float32)
    for t in np.concatenate([data.valid_idx, data.test_idx]):
        ok = data.mask[t]
        if ok.any():
            scores[t, ok] = model.predict(
                data.features[t, ok], num_iteration=model.best_iteration_
            )

    valid_ic = float(
        np.nanmean(rank_ic_by_date(scores, data.labels, data.label_valid, data.valid_idx))
    )
    return LgbmResult(
        scores=scores,
        valid_rank_ic=valid_ic,
        best_iteration=int(model.best_iteration_ or 0),
    )
