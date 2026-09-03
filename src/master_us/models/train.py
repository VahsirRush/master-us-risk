"""Data preparation and the shared trainer for all deep models — spec 7.4.

One trainer drives the LSTM, the ungated transformer, and (in Phase 3) MASTER
itself: same batching, same loss, same early stopping. Differences between
models live entirely in the module passed in, which is what makes the Phase-2
comparison a controlled experiment rather than four bespoke pipelines.

Batching is BY DATE (spec: "one batch = one trading date, all names present
that date" — the inter-stock attention requires the full cross-section).
Several dates are grouped per optimizer step for throughput; names are
gathered to the union of valid names in the group and padded, with the
validity mask carrying which are real.

Normalization: features through `RobustZScoreNorm` (train statistics only,
median/MAD, clip 3, NaN -> 0 + group missingness indicators). The market
vector goes through the same class reshaped to (T, 1, M) — same train-only
statistics discipline; without it the gate would see raw dollar volumes of
1e12 next to returns of 1e-3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy import stats as scipy_stats
from torch import nn

from master_us.data.normalize import RobustZScoreNorm
from master_us.data.panel import Panel
from master_us.models.loss import ic_loss
from master_us.models.master import pick_device
from master_us.utils.validation import set_determinism

FloatMat = npt.NDArray[np.float32]


@dataclass(frozen=True)
class PreparedData:
    """Everything a model run needs, normalized once and shared."""

    dates: pd.DatetimeIndex
    features: FloatMat  # (T, N, F') — normalized, NaN-free
    market: FloatMat  # (T, M) — normalized
    labels: npt.NDArray[np.float32]  # (T, N) processed labels, NaN where absent
    mask: npt.NDArray[np.bool_]  # (T, N) tradeable
    raw_forward: npt.NDArray[np.float32]  # (T, N) raw forward returns
    lookback: int
    train_idx: npt.NDArray[np.int64]  # date indices, embargo applied
    valid_idx: npt.NDArray[np.int64]
    test_idx: npt.NDArray[np.int64]
    feature_names: tuple[str, ...]

    @property
    def label_valid(self) -> npt.NDArray[np.bool_]:
        return self.mask & np.isfinite(self.labels)


def prepare_data(
    panel: Panel,
    lookback: int,
    train: tuple[str, str],
    valid: tuple[str, str],
    test: tuple[str, str],
    embargo_days: int = 21,
    cache_dir: Path | None = None,
) -> PreparedData:
    """Normalize and split. Same embargo convention as `Panel.split`: the END
    of each earlier segment is trimmed by `embargo_days` calendar days.

    Every date index also requires `lookback` days of history in the panel,
    so a sample can always see its full window (windows may cross split
    boundaries backward — features are public information; only labels leak,
    and the embargo handles those).
    """
    in_train = np.asarray(panel.dates <= pd.Timestamp(train[1]) - pd.Timedelta(days=embargo_days))
    norm = RobustZScoreNorm(feature_names=panel.feature_names, strict_groups=True).fit(
        panel.features, in_train[:, None] & panel.mask
    )
    # MEMORY, on an 8 GB machine. The normalized tensor is (T, N, F') float32
    # ~= 1.3 GB, and the naive path holds three copies at once (the Panel's own
    # features, the per-chunk outputs, and the concatenated result), which the
    # macOS OOM killer ends without a traceback — measured in Session 6.
    #
    # Two defences: transform in date chunks into a PREALLOCATED output (no
    # concat copy), and, when `cache_dir` is given, back that output with a
    # memory-mapped file so steady-state resident memory is page cache the OS
    # can evict rather than heap it cannot. Chunking is numerically identical:
    # every operation in the transform is per-cell.
    probe = norm.transform_with_masks(panel.features[:1])
    names = probe.names
    shape = (panel.n_dates, panel.n_tickers, probe.values.shape[2])
    del probe

    values: FloatMat
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"features_norm_L{lookback}.npy"
        values = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32, shape=shape
        )
    else:
        values = np.empty(shape, dtype=np.float32)

    for i in range(0, panel.n_dates, 256):
        values[i : i + 256] = norm.transform_with_masks(panel.features[i : i + 256]).values
    if cache_dir is not None:
        values.flush()  # type: ignore[union-attr]
        values = np.load(cache_path, mmap_mode="r")  # reopen read-only

    market_norm = (
        RobustZScoreNorm(zero_mad="unit_scale")
        .fit(panel.market[:, None, :].astype(np.float32), in_train)
        .transform(panel.market[:, None, :].astype(np.float32))[:, 0, :]
    )

    def span(lo: str | pd.Timestamp, hi: str | pd.Timestamp) -> npt.NDArray[np.int64]:
        sel = (panel.dates >= pd.Timestamp(lo)) & (panel.dates <= pd.Timestamp(hi))
        idx = np.flatnonzero(np.asarray(sel))
        return idx[idx >= lookback - 1]

    gap = pd.Timedelta(days=embargo_days)
    train_idx = span(train[0], pd.Timestamp(train[1]) - gap)
    valid_idx = span(valid[0], pd.Timestamp(valid[1]) - gap)
    test_idx = span(test[0], test[1])
    for name, idx in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
        if len(idx) == 0:
            raise ValueError(f"{name} split is empty")

    return PreparedData(
        dates=panel.dates,
        features=values,
        market=market_norm,
        labels=panel.labels,
        mask=panel.mask,
        raw_forward=panel.attrs["raw_forward_returns"],
        lookback=lookback,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        feature_names=names,
    )


def rank_ic_by_date(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.floating],
    valid: npt.NDArray[np.bool_],
    date_idx: npt.NDArray[np.int64],
    min_names: int = 10,
) -> npt.NDArray[np.float64]:
    """Per-date Spearman correlation over the given dates."""
    out = []
    for t in date_idx:
        ok = valid[t] & np.isfinite(scores[t]) & np.isfinite(labels[t])
        if ok.sum() < min_names:
            continue
        out.append(float(scipy_stats.spearmanr(scores[t, ok], labels[t, ok]).statistic))
    return np.asarray(out)


# --------------------------------------------------------------------- #
# Batching                                                               #
# --------------------------------------------------------------------- #


def _gather(
    data: PreparedData,
    date_group: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    n_slots: int,
) -> npt.NDArray[np.float32]:
    """(B, n_slots, L, F) feature block, zero-padded past `len(cols)`."""
    lookback = data.lookback
    out = np.zeros((len(date_group), n_slots, lookback, data.features.shape[2]), np.float32)
    for i, t in enumerate(date_group):
        window = data.features[t - lookback + 1 : t + 1][:, cols]  # (L, n, F)
        out[i, : len(cols)] = window.transpose(1, 0, 2)
    return out


def batch_width(data: PreparedData, group_size: int) -> int:
    """Largest union of names over any contiguous date group — the pad width."""
    lv = data.label_valid
    widest = 0
    for idx in (data.train_idx, data.valid_idx, data.test_idx):
        for start in range(0, len(idx), group_size):
            group = idx[start : start + group_size]
            widest = max(widest, int(lv[group].any(axis=0).sum()))
            widest = max(widest, int(data.mask[group].any(axis=0).sum()))
    return widest


def _make_batch(
    data: PreparedData,
    date_group: npt.NDArray[np.int64],
    device: torch.device,
    n_slots: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, npt.NDArray[np.int64]]:
    """(x, m, valid_mask, target, union_cols) for a group of dates.

    Names are gathered to the union of valid names across the group.

    FIXED WIDTH, and it is not cosmetic. The union size drifts by a name or
    two between consecutive date groups (332, 331, 333, ...), and on MPS a
    fresh tensor shape every step fragments the caching allocator until the
    OS kills the process with no traceback — three times in Session 6 before
    this was found. Padding every batch to `n_slots` keeps the shape constant
    so the allocator reuses its blocks. The padding is inert: `valid` is
    False there, which excludes those slots from attention and from both
    moments of the loss.
    """
    lv = data.label_valid
    union = np.flatnonzero(lv[date_group].any(axis=0))
    width = n_slots or len(union)
    if len(union) > width:
        raise ValueError(f"{len(union)} valid names exceeds n_slots={width}")

    x = _gather(data, date_group, union, width)
    target = np.zeros((len(date_group), width), np.float32)
    target[:, : len(union)] = np.nan_to_num(data.labels[date_group][:, union], nan=0.0)
    valid = np.zeros((len(date_group), width), bool)
    valid[:, : len(union)] = lv[date_group][:, union]

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(data.market[date_group]).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(target).to(device),
        union,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    data: PreparedData,
    date_idx: npt.NDArray[np.int64],
    device: torch.device,
    group_size: int = 8,
    n_slots: int | None = None,
) -> npt.NDArray[np.float32]:
    """Scores over `mask` for the given dates. (T, N) full-panel shape, NaN
    outside the mask."""
    model.eval()
    scores = np.full(data.labels.shape, np.nan, dtype=np.float32)
    width = n_slots or batch_width(data, group_size)
    for start in range(0, len(date_idx), group_size):
        group = date_idx[start : start + group_size]
        union = np.flatnonzero(data.mask[group].any(axis=0))
        if len(union) == 0:
            continue
        x = _gather(data, group, union, width)
        valid = np.zeros((len(group), width), bool)
        valid[:, : len(union)] = data.mask[group][:, union]
        pred = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(data.market[group]).to(device),
            torch.from_numpy(valid).to(device),
        )
        pred_np = pred.float().cpu().numpy()
        for i, t in enumerate(group):
            live = valid[i, : len(union)]
            scores[t, union[live]] = pred_np[i, : len(union)][live]
    return scores


@dataclass(frozen=True)
class TrainResult:
    scores: npt.NDArray[np.float32]  # (T, N) over valid+test dates
    best_valid_rank_ic: float
    best_epoch: int
    n_epochs_run: int


def train_model(
    model: nn.Module,
    data: PreparedData,
    seed: int,
    max_epochs: int = 25,
    patience: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    group_size: int = 4,
    valid_stride: int = 3,
    device: torch.device | None = None,
    log: Callable[[int, float, float], None] | None = None,
) -> TrainResult:
    """AdamW + cosine + grad clip, early stop on validation RankIC — spec 7.4.

    Two Phase-2 wall-clock concessions, both recorded in NOTES.md and both
    lifted for Phase 3:

    * max_epochs/patience default to 25/5 rather than master.yaml's 100/10.
    * early stopping scores every `valid_stride`-th validation date rather
      than all of them. RankIC over ~160 dates estimates the stopping signal
      to well inside its own seed dispersion, and it cuts per-epoch cost by
      roughly a third. The FINAL scores are always produced over every
      validation and test date — the stride is a training-loop economy, never
      a reporting one.
    """
    set_determinism(seed)
    device = device or pick_device()
    model = model.to(device)

    n_slots = batch_width(data, group_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    rng = np.random.default_rng(seed)

    best_ic = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(data.train_idx))
        for start in range(0, len(order), group_size):
            group = data.train_idx[order[start : start + group_size]]
            x, m, valid, target, _ = _make_batch(data, group, device, n_slots)
            if not bool(valid.any()):
                continue
            opt.zero_grad(set_to_none=True)
            loss = ic_loss(model(x, m, valid), target, valid)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step()
        epochs_run = epoch + 1
        if device.type == "mps":
            torch.mps.empty_cache()  # 8 GB machine: return freed blocks eagerly

        watch_idx = data.valid_idx[::valid_stride]
        scores = predict(model, data, watch_idx, device, group_size, n_slots)
        valid_ic = float(
            np.nanmean(rank_ic_by_date(scores, data.labels, data.label_valid, watch_idx))
        )
        if log is not None:
            log(epoch, valid_ic, best_ic)
        if valid_ic > best_ic:
            best_ic = valid_ic
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    eval_dates = np.concatenate([data.valid_idx, data.test_idx])
    scores = predict(model, data, eval_dates, device, group_size, n_slots)
    return TrainResult(
        scores=scores,
        best_valid_rank_ic=best_ic,
        best_epoch=best_epoch,
        n_epochs_run=epochs_run,
    )
