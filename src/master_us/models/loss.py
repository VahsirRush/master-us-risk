"""Training loss — spec section 7.3.

Negative Pearson correlation computed WITHIN each date, averaged over dates.
The task is cross-sectional ranking, not level prediction; plain MSE
materially underperforms because it spends capacity matching the magnitude of
a per-date z-scored target whose magnitude carries no signal.
"""

from __future__ import annotations

from torch import Tensor

EPS = 1e-8


def ic_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean over dates of -corr(pred, target) within the date.

    pred, target: (B, N); mask: (B, N) bool — True where the name is valid
    (tradeable AND has a label). Masked names are excluded from BOTH moments.
    Dates with fewer than 2 valid names contribute nothing.
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)}, target {tuple(target.shape)}, "
            f"mask {tuple(mask.shape)} must share a (B, N) shape"
        )
    m = mask.float()
    count = m.sum(dim=1)  # (B,)
    valid_date = count >= 2

    count_safe = count.clamp(min=1.0)
    pred_mean = (pred * m).sum(dim=1) / count_safe
    target_mean = (target * m).sum(dim=1) / count_safe

    pred_centered = (pred - pred_mean[:, None]) * m
    target_centered = (target - target_mean[:, None]) * m

    cov = (pred_centered * target_centered).sum(dim=1)
    pred_var = pred_centered.pow(2).sum(dim=1)
    target_var = target_centered.pow(2).sum(dim=1)

    corr = cov / (pred_var.sqrt() * target_var.sqrt() + EPS)
    if not valid_date.any():
        # No usable date in the batch: a zero WITH a graph, so backward is a no-op
        # rather than a crash.
        return (pred * 0.0).sum()
    return -(corr[valid_date]).mean()
