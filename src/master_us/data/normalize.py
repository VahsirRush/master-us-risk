"""Feature normalization and label processing — spec section 4.5.

This is the leakage-critical component. The failure mode it exists to prevent
is not exotic: you fit a scaler on the whole panel, transform every split with
it, and every number downstream is quietly inflated because the validation set
helped choose its own scale. The original MASTER authors shipped exactly this
bug; see docs/project-conventions.md, "What this is NOT".

The defences here are structural rather than advisory:

* `fit()` refuses to run twice on the same instance. The most common way to
  leak is a loop — `for split in splits: norm.fit(split); norm.transform(split)`
  — which reads perfectly naturally and is completely wrong. It now raises.
* There is no `fit_transform()`. The sklearn habit is strong enough that the
  name is defined solely to raise and explain itself.
* Fitted statistics are exposed only as read-only copies, so a caller cannot
  quietly nudge them after the fact.
* `fit_on_panel()` takes the train split boundary as an explicit date, which
  makes the correct call shorter to write than the incorrect one.

Alignment convention is inherited from `panel.py` and not restated.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
import numpy.typing as npt
import pandas as pd

from master_us.data.feature_groups import group_names
from master_us.data.panel import FeatureArray, MaskArray, Panel

# Inputs arrive as float32 from `Panel` but statistics are accumulated in
# float64, so the two are named separately rather than collapsed into one alias.
FloatArray = npt.NDArray[np.floating[Any]]
StatArray = npt.NDArray[np.float64]
CountArray = npt.NDArray[np.int64]

# Scaling MAD by this constant makes it a consistent estimator of the standard
# deviation for normally distributed data, which is what puts the clip bounds
# at [-3, 3] rather than at an arbitrary multiple of a raw median deviation.
MAD_TO_SIGMA = 1.4826

ZeroMadPolicy = Literal["raise", "unit_scale"]

# Either one of the two strategy literals, or an explicit group -> members map.
GroupingSpec = Literal["auto", "per_feature"] | Mapping[str, Sequence[int | str]]


class NotFittedError(RuntimeError):
    """Raised when `transform()` is called before `fit()`."""


class AlreadyFittedError(RuntimeError):
    """Raised when `fit()` is called on an instance that already holds statistics.

    This is the leakage guard. A normalizer is fitted exactly once, on train.
    """


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature location and scale, estimated on the training span only.

    `n_observations` and `train_span` are provenance, not decoration — when a
    result looks too good six phases from now, the first question is which rows
    these numbers came from, and this is the only place that answer is recorded.
    """

    median: StatArray  # (F,)
    scale: StatArray  # (F,) — 1.4826 * MAD
    n_observations: int
    n_valid_per_feature: CountArray  # (F,)
    feature_names: tuple[str, ...] | None
    clip: float
    train_span: tuple[pd.Timestamp, pd.Timestamp] | None = None

    @property
    def n_features(self) -> int:
        return int(self.median.shape[0])


@dataclass(frozen=True)
class NormalizedFeatures:
    """Transformed features plus the companion missingness indicators.

    `values` is the base features followed by one binary column per feature
    group. Base features have already had NaN replaced by 0; the indicator
    columns are the only surviving record that the value was ever absent, which
    is why they are emitted rather than left to the caller to reconstruct.
    """

    values: FeatureArray  # (T, N, F + G)
    names: tuple[str, ...]  # len F + G
    n_base_features: int
    group_names: tuple[str, ...]  # len G

    @property
    def base(self) -> FeatureArray:
        return self.values[:, :, : self.n_base_features]

    @property
    def indicators(self) -> FeatureArray:
        return self.values[:, :, self.n_base_features :]


@dataclass(frozen=True)
class ProcessedLabels:
    """Output of `process_labels`, with the audit trail of what was discarded.

    Trimmed observations become NaN rather than disappearing: the (T, N) shape
    is load-bearing everywhere downstream, so an observation can be excluded
    but never removed.
    """

    values: FeatureArray  # (T, N), NaN where absent or trimmed
    n_input: int  # observations considered (mask-eligible)
    n_missing: int  # of those, NaN before trimming
    n_trimmed: int  # dropped as extreme
    n_output: int  # surviving, z-scored
    n_dates_dropped: int  # dates voided for having too few names
    per_tail_fraction: float

    @property
    def trimmed_fraction(self) -> float:
        """Fraction of non-missing observations dropped as extreme."""
        eligible = self.n_input - self.n_missing
        return self.n_trimmed / eligible if eligible else 0.0


# --------------------------------------------------------------------- #
# Feature normalization                                                  #
# --------------------------------------------------------------------- #


class RobustZScoreNorm:
    """Median/MAD normalization with training-only statistics.

    fit()       computes per-feature median and MAD over the TRAINING span only.
    transform() applies stored training statistics to any split, then clips to
                [-3, 3], then maps NaN to 0.

    Validation and test data are transformed with BORROWED training statistics.
    Refitting on val/test is leakage and is the single most common replication
    error. `fit()` therefore raises if called a second time — to fit a
    different span, construct a new instance, which makes the intent explicit
    at the call site.

    Parameters
    ----------
    clip:
        Symmetric bound applied after scaling. Spec section 4.5 fixes this at 3.0.
    feature_names:
        Optional, length F. Recorded for provenance and used to name the
        emitted indicator columns.
    feature_groups:
        How to group features for missingness indicators. One binary indicator
        column is emitted per group, True where ANY feature in the group was NaN
        before transform.

        - `"auto"` (the default) classifies `feature_names` through
          `data.feature_groups`, giving one indicator per spec section 4.3 group.
          Features in a group go missing together, so a per-group indicator
          carries nearly all the information of a per-feature one at a fraction
          of the width.
        - `"per_feature"` gives every feature its own indicator. Correct but
          wide; at the target F of ~150 it doubles the column count.
        - An explicit mapping of group name -> feature indices or names
          overrides both.

        With `"auto"` and no `feature_names`, there is nothing to classify and
        this falls back to `"per_feature"`.
    strict_groups:
        Under `"auto"`, raise if any feature name matches no known group prefix
        instead of pooling it into `other`. Every path that builds a real panel
        should set this — a misnamed feature otherwise has its missingness
        pooled with unrelated features, degrading the indicator without failing.
    zero_mad:
        What to do with a feature whose training MAD is zero (constant, or
        constant across the surviving majority). "raise" is the default and
        matches docs/project-conventions.md rule 6; "unit_scale" divides by 1.0 instead and is
        an explicit, logged decision rather than a silent one.
    """

    def __init__(
        self,
        clip: float = 3.0,
        feature_names: Sequence[str] | None = None,
        feature_groups: GroupingSpec = "auto",
        strict_groups: bool = False,
        zero_mad: ZeroMadPolicy = "raise",
    ) -> None:
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
        if zero_mad not in ("raise", "unit_scale"):
            raise ValueError(f"zero_mad must be 'raise' or 'unit_scale', got {zero_mad!r}")
        if isinstance(feature_groups, str) and feature_groups not in ("auto", "per_feature"):
            raise ValueError(
                f"feature_groups must be 'auto', 'per_feature', or a mapping, "
                f"got {feature_groups!r}"
            )

        self.clip = float(clip)
        self.feature_names: tuple[str, ...] | None = (
            tuple(feature_names) if feature_names is not None else None
        )
        self.zero_mad: ZeroMadPolicy = zero_mad
        self.strict_groups = strict_groups
        self._feature_groups_spec = feature_groups
        self._stats: NormalizationStats | None = None

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        return self._stats is not None

    @property
    def stats(self) -> NormalizationStats:
        """The fitted statistics. Arrays are read-only copies by construction."""
        if self._stats is None:
            raise NotFittedError(
                "RobustZScoreNorm has no statistics — call fit(x, train_mask) on the "
                "TRAINING span first."
            )
        return self._stats

    @property
    def n_features(self) -> int:
        return self.stats.n_features

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        x: FloatArray,
        train_mask: MaskArray,
        train_span: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> RobustZScoreNorm:
        """Estimate median and MAD over the observations selected by `train_mask`.

        Parameters
        ----------
        x:
            (T, N, F) feature array. May span the whole sample — only the rows
            selected by `train_mask` contribute to the statistics, which is the
            property `test_normalization_uses_training_stats` pins down.
        train_mask:
            (T,) date-level or (T, N) observation-level boolean. Observation
            level is preferred: it lets the caller intersect the training span
            with `Panel.mask` so that names outside the tradeable universe never
            influence the scale.
        train_span:
            Optional (first, last) training date, recorded as provenance.

        Raises
        ------
        AlreadyFittedError:
            If this instance has already been fitted. Refitting is the leakage
            path this class exists to close.
        """
        if self._stats is not None:
            raise AlreadyFittedError(
                "This RobustZScoreNorm is already fitted"
                + (
                    f" on {self._stats.train_span[0].date()}..{self._stats.train_span[1].date()}"
                    if self._stats.train_span is not None
                    else ""
                )
                + f" ({self._stats.n_observations} observations). Refitting would replace "
                "training statistics with statistics from whatever span you just passed, "
                "which is exactly the val/test leakage this class prevents. To normalize "
                "another split, call transform(). To fit a genuinely different training "
                "span, construct a new RobustZScoreNorm."
            )

        x = self._validate_feature_array(x)
        obs = self._resolve_train_mask(train_mask, x.shape)

        n_obs = int(obs.sum())
        if n_obs == 0:
            raise ValueError("train_mask selects no observations")

        selected = x[obs]  # (K, F)
        n_valid = np.sum(~np.isnan(selected), axis=0).astype(np.int64)

        all_nan = n_valid == 0
        if all_nan.any():
            raise ValueError(
                f"{int(all_nan.sum())} feature(s) are entirely NaN over the training span: "
                f"{self._describe_features(np.flatnonzero(all_nan))}. "
                "A feature with no training observations cannot be normalized."
            )

        median = np.nanmedian(selected, axis=0)
        mad = np.nanmedian(np.abs(selected - median), axis=0)
        scale = MAD_TO_SIGMA * mad

        degenerate = ~np.isfinite(scale) | (scale <= 0.0)
        if degenerate.any():
            if self.zero_mad == "raise":
                raise ValueError(
                    f"{int(degenerate.sum())} feature(s) have zero or non-finite MAD over the "
                    f"training span: {self._describe_features(np.flatnonzero(degenerate))}. "
                    "These are constant in training and carry no cross-sectional information. "
                    "Drop them upstream, or pass zero_mad='unit_scale' to keep them at unit "
                    "scale as a deliberate, recorded choice."
                )
            scale = np.where(degenerate, 1.0, scale)

        self._stats = NormalizationStats(
            median=_frozen(median.astype(np.float64)),
            scale=_frozen(scale.astype(np.float64)),
            n_observations=n_obs,
            n_valid_per_feature=_frozen(n_valid),
            feature_names=self.feature_names,
            clip=self.clip,
            train_span=train_span,
        )
        return self

    def fit_on_panel(
        self,
        panel: Panel,
        train_end: str | pd.Timestamp,
        train_start: str | pd.Timestamp | None = None,
        use_universe_mask: bool = True,
    ) -> RobustZScoreNorm:
        """Fit on `panel` restricted to dates <= `train_end`.

        The preferred entry point. Naming the training boundary explicitly makes
        the correct call shorter than the incorrect one, and intersecting with
        `Panel.mask` keeps names outside the tradeable universe from setting the
        scale for names inside it.
        """
        if self.feature_names is None:
            self.feature_names = tuple(panel.feature_names)
        elif tuple(panel.feature_names) != self.feature_names:
            raise ValueError(
                "panel.feature_names does not match the feature_names this normalizer "
                "was constructed with"
            )

        hi = pd.Timestamp(train_end)
        in_span = panel.dates <= hi
        if train_start is not None:
            in_span &= panel.dates >= pd.Timestamp(train_start)
        if not in_span.any():
            raise ValueError(f"no panel dates at or before {hi.date()}")

        obs = np.asarray(in_span)[:, None] & (
            panel.mask if use_universe_mask else np.ones_like(panel.mask)
        )
        span = (panel.dates[in_span][0], panel.dates[in_span][-1])
        return self.fit(panel.features, obs, train_span=span)

    def fit_transform(self, *args: object, **kwargs: object) -> NoReturn:
        """Deliberately absent. Defined only so the sklearn reflex fails loudly."""
        raise AlreadyFittedError(
            "RobustZScoreNorm has no fit_transform(). The sklearn shorthand is how "
            "validation and test data get fitted on by accident. Fit once on the "
            "training span — norm.fit(x, train_mask) or norm.fit_on_panel(panel, "
            "train_end) — then call transform() on each split separately."
        )

    # ------------------------------------------------------------------ #
    # Transforming                                                        #
    # ------------------------------------------------------------------ #

    def transform(self, x: FloatArray) -> FeatureArray:
        """Apply stored training statistics, clip to [-clip, clip], map NaN to 0.

        Returns an array of the same shape as `x`. Missingness information is
        destroyed by the NaN-to-0 step; use `transform_with_masks()` on any path
        that feeds a model, so the indicator columns survive.
        """
        stats = self.stats
        x = self._validate_feature_array(x)
        if x.shape[2] != stats.n_features:
            raise ValueError(
                f"x has {x.shape[2]} features, normalizer was fitted on {stats.n_features}"
            )

        with np.errstate(invalid="ignore"):
            z = (x.astype(np.float64) - stats.median) / stats.scale
        np.clip(z, -self.clip, self.clip, out=z)
        z[~np.isfinite(z)] = 0.0
        return z.astype(np.float32)

    def transform_with_masks(self, x: FloatArray) -> NormalizedFeatures:
        """`transform()` plus one binary missingness indicator per feature group.

        The indicator is True (1.0) where ANY feature in the group was NaN in the
        input, evaluated before the NaN-to-0 substitution.
        """
        stats = self.stats
        x = self._validate_feature_array(x)
        if x.shape[2] != stats.n_features:
            raise ValueError(
                f"x has {x.shape[2]} features, normalizer was fitted on {stats.n_features}"
            )

        values = self.transform(x)
        groups = self._resolve_groups(stats.n_features)
        was_nan = np.isnan(x)

        indicators = np.empty((x.shape[0], x.shape[1], len(groups)), dtype=np.float32)
        for j, (_, idx) in enumerate(groups.items()):
            indicators[:, :, j] = was_nan[:, :, idx].any(axis=2)

        base_names = (
            stats.feature_names
            if stats.feature_names is not None
            else tuple(f"f{i:03d}" for i in range(stats.n_features))
        )
        resolved_groups = tuple(groups)
        return NormalizedFeatures(
            values=np.concatenate([values, indicators], axis=2),
            names=base_names + tuple(f"{g}__isnan" for g in resolved_groups),
            n_base_features=stats.n_features,
            group_names=resolved_groups,
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Persist fitted statistics so inference borrows train's numbers verbatim."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @classmethod
    def load(cls, path: str | Path) -> RobustZScoreNorm:
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)  # our own artifact, never untrusted input
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a RobustZScoreNorm")
        return obj

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_feature_array(x: FloatArray) -> FloatArray:
        arr = np.asarray(x)
        if arr.ndim != 3:
            raise ValueError(f"features must be (T, N, F), got shape {arr.shape}")
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(f"features must be a floating dtype, got {arr.dtype}")
        return arr

    @staticmethod
    def _resolve_train_mask(train_mask: MaskArray, shape: tuple[int, ...]) -> MaskArray:
        t, n = shape[0], shape[1]
        m = np.asarray(train_mask)
        if m.dtype != np.bool_:
            raise TypeError(f"train_mask must be bool, got {m.dtype}")
        if m.shape == (t,):
            return np.broadcast_to(m[:, None], (t, n))
        if m.shape == (t, n):
            return m
        raise ValueError(f"train_mask must have shape ({t},) or ({t}, {n}), got {m.shape}")

    def _resolve_groups(self, n_features: int) -> dict[str, CountArray]:
        spec = self._feature_groups_spec

        if spec == "auto" and self.feature_names is None:
            # Nothing to classify — degrade to per-feature rather than guess.
            spec = "per_feature"

        if spec == "per_feature":
            names = self.feature_names or tuple(f"f{i:03d}" for i in range(n_features))
            return {name: np.array([i]) for i, name in enumerate(names)}

        if spec == "auto":
            assert self.feature_names is not None  # narrowed above
            spec = group_names(self.feature_names, strict=self.strict_groups)

        assert not isinstance(spec, str)  # only the two literals above are strings
        name_to_idx = {name: i for i, name in enumerate(self.feature_names or ())}
        groups: dict[str, CountArray] = {}
        seen: set[int] = set()
        for group, members in spec.items():
            idx = []
            for member in members:
                if isinstance(member, str):
                    if member not in name_to_idx:
                        raise ValueError(f"group {group!r} references unknown feature {member!r}")
                    idx.append(name_to_idx[member])
                else:
                    if not 0 <= member < n_features:
                        raise ValueError(f"group {group!r} references index {member} out of range")
                    idx.append(int(member))
            if not idx:
                raise ValueError(f"group {group!r} is empty")
            seen.update(idx)
            groups[group] = np.array(sorted(set(idx)))

        missing = sorted(set(range(n_features)) - seen)
        if missing:
            raise ValueError(
                f"{len(missing)} feature(s) belong to no group: "
                f"{self._describe_features(np.array(missing))}. Every feature must be "
                "assigned, otherwise its missingness is silently unrecorded."
            )
        return groups

    def _describe_features(self, idx: npt.NDArray[np.integer[Any]]) -> str:
        shown = idx[:8]
        if self.feature_names is not None:
            label = ", ".join(self.feature_names[i] for i in shown)
        else:
            label = ", ".join(str(int(i)) for i in shown)
        return label + (f", ... (+{len(idx) - len(shown)} more)" if len(idx) > len(shown) else "")

    def __repr__(self) -> str:
        if self._stats is None:
            return f"RobustZScoreNorm(clip={self.clip}, unfitted)"
        span = ""
        if self._stats.train_span is not None:
            span = f", train={self._stats.train_span[0].date()}..{self._stats.train_span[1].date()}"
        return (
            f"RobustZScoreNorm(clip={self.clip}, features={self._stats.n_features}, "
            f"n_obs={self._stats.n_observations}{span})"
        )


def infer_feature_groups(feature_names: Sequence[str], sep: str = "_") -> dict[str, list[str]]:
    """Group feature names by their prefix up to the first `sep`.

    A convenience for the Phase-0 feature bank, whose names follow a
    `group_detail` convention (`ret_5d`, `vol_20d`, ...). Names without `sep`
    form their own single-member group.
    """
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        prefix = name.split(sep, 1)[0] if sep in name else name
        groups.setdefault(prefix, []).append(name)
    return groups


# --------------------------------------------------------------------- #
# Label processing                                                       #
# --------------------------------------------------------------------- #


def process_labels(
    labels: FloatArray,
    mask: MaskArray | None = None,
    drop_pct: float = 5.0,
    per_tail: bool = False,
    min_names: int = 2,
) -> ProcessedLabels:
    """Spec section 4.5: drop NaN, drop the most extreme 5%, cross-sectional z-score.

    Every step is applied WITHIN each date. Cross-sectional processing is the
    whole point of the label — trimming pooled across dates would delete the
    tails of high-volatility periods wholesale rather than the outliers within
    each of them.

    Parameters
    ----------
    labels:
        (T, N) forward returns, NaN where absent.
    mask:
        Optional (T, N) universe mask. Names outside the universe are excluded
        from the trim counts and from the z-score statistics.
    drop_pct:
        Total percentage of each date's cross-section discarded as extreme,
        split evenly between the tails: 5.0 removes 2.5% from each side. Pass
        `per_tail=True` to read it as 5% from each side instead. The spec's
        "the most extreme 5% (both tails)" is ambiguous; see NOTES.md.
    per_tail:
        Interpret `drop_pct` as per-tail rather than total.
    min_names:
        A date with fewer surviving names than this is voided entirely — a
        cross-sectional z-score over a handful of names is noise wearing the
        costume of a label.

    Returns
    -------
    ProcessedLabels
        `.values` has the same (T, N) shape as the input, NaN where the
        observation was absent, outside the universe, or trimmed.
    """
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"labels must be (T, N), got shape {arr.shape}")
    if not 0.0 <= drop_pct < 100.0:
        raise ValueError(f"drop_pct must be in [0, 100), got {drop_pct}")
    if min_names < 2:
        raise ValueError(f"min_names must be >= 2, got {min_names}")

    per_tail_fraction = (drop_pct / 100.0) if per_tail else (drop_pct / 200.0)
    if per_tail_fraction >= 0.5:
        raise ValueError(
            f"drop_pct={drop_pct} with per_tail={per_tail} would discard the entire cross-section"
        )

    if mask is None:
        eligible = np.ones(arr.shape, dtype=bool)
    else:
        eligible = np.asarray(mask)
        if eligible.shape != arr.shape:
            raise ValueError(f"mask shape {eligible.shape} does not match labels {arr.shape}")
        if eligible.dtype != np.bool_:
            raise TypeError(f"mask must be bool, got {eligible.dtype}")

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    n_input = int(eligible.sum())
    n_missing = int((eligible & np.isnan(arr)).sum())
    n_trimmed = 0
    n_output = 0
    n_dates_dropped = 0

    for t in range(arr.shape[0]):
        cols = np.flatnonzero(eligible[t] & ~np.isnan(arr[t]))
        k = cols.size
        if k < min_names:
            n_dates_dropped += 1 if k > 0 else 0
            continue

        n_tail = int(np.floor(k * per_tail_fraction))
        if k - 2 * n_tail < min_names:
            # Trimming would leave too thin a cross-section to standardize.
            n_dates_dropped += 1
            continue

        vals = arr[t, cols].astype(np.float64)
        if n_tail > 0:
            order = np.argsort(vals, kind="stable")
            keep_pos = order[n_tail : k - n_tail]
            n_trimmed += 2 * n_tail
        else:
            keep_pos = np.arange(k)

        keep_cols = cols[keep_pos]
        kept = vals[keep_pos]

        sd = kept.std(ddof=0)
        if not np.isfinite(sd) or sd == 0.0:
            # Degenerate cross-section: no dispersion to standardize against.
            n_dates_dropped += 1
            n_trimmed -= 2 * n_tail
            continue

        out[t, keep_cols] = ((kept - kept.mean()) / sd).astype(np.float32)
        n_output += keep_cols.size

    return ProcessedLabels(
        values=out,
        n_input=n_input,
        n_missing=n_missing,
        n_trimmed=n_trimmed,
        n_output=n_output,
        n_dates_dropped=n_dates_dropped,
        per_tail_fraction=per_tail_fraction,
    )


def _frozen(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return a read-only copy, so fitted statistics cannot be edited in place."""
    out = np.array(arr, copy=True)
    out.flags.writeable = False
    return out
