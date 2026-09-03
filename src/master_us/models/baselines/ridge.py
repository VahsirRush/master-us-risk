"""Cross-sectional ridge — the floor. Below this, debug rather than continue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge

from master_us.models.train import PreparedData, rank_ic_by_date


def _pooled(
    data: PreparedData, date_idx: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Flatten (date, name) samples where tradeable and labelled."""
    lv = data.label_valid
    xs, ys = [], []
    for t in date_idx:
        ok = lv[t]
        if ok.sum() < 10:
            continue
        xs.append(data.features[t, ok])
        ys.append(data.labels[t, ok])
    return np.concatenate(xs), np.concatenate(ys)


@dataclass(frozen=True)
class RidgeResult:
    scores: npt.NDArray[np.float32]
    alpha: float
    valid_rank_ic: float


def run_ridge(data: PreparedData, seed: int, alphas: tuple[float, ...]) -> RidgeResult:
    """Fit on train, choose alpha by validation RankIC, score valid+test.

    Deterministic — the seed exists only to honor the shared 5-seed protocol,
    and the resulting seed dispersion is legitimately zero.
    """
    x_train, y_train = _pooled(data, data.train_idx)

    best: tuple[float, float, Ridge] | None = None
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(x_train, y_train)
        scores = _score_dates(model, data, data.valid_idx)
        ic = float(
            np.nanmean(rank_ic_by_date(scores, data.labels, data.label_valid, data.valid_idx))
        )
        if best is None or ic > best[1]:
            best = (alpha, ic, model)

    assert best is not None
    alpha, valid_ic, model = best
    scores = _score_dates(model, data, np.concatenate([data.valid_idx, data.test_idx]))
    return RidgeResult(scores=scores, alpha=alpha, valid_rank_ic=valid_ic)


def _score_dates(
    model: Ridge, data: PreparedData, date_idx: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    scores = np.full(data.labels.shape, np.nan, dtype=np.float32)
    for t in date_idx:
        ok = data.mask[t]
        if ok.any():
            scores[t, ok] = model.predict(data.features[t, ok])
    return scores
