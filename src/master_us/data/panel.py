"""The canonical data contract for this project.

Every downstream module — features, models, backtest, risk — consumes a `Panel`.
Its assertions are the point, not boilerplate: they catch shape and alignment
bugs at construction time rather than three phases later, when a silently
misaligned label has already produced a plausible-looking Sharpe ratio.

Alignment convention, stated once and relied on everywhere:

    features[t, n, :]   information known AS OF the close of dates[t]
    labels[t, n]        FORWARD return, realized AFTER dates[t]
    mask[t, n]          True iff ticker n is in the tradeable universe on dates[t]

If that convention is ever violated, every result in this repository is void.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REQUIRED_METADATA_COLUMNS = frozenset({"mcap", "sector", "adv", "price"})

# The dtypes are part of the contract, not incidental — `_validate_dtypes`
# enforces at runtime exactly what these annotations claim statically.
TickerArray = npt.NDArray[np.str_]
FeatureArray = npt.NDArray[np.float32]
MaskArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Panel:
    """Aligned (date, ticker) panel of features, labels, and universe membership.

    Attributes
    ----------
    dates:
        Sorted, unique trading dates. Shape (T,).
    tickers:
        Stable, sorted ticker ordering. Shape (N,). The ordering is fixed for the
        life of the Panel; downstream code may index positionally.
    features:
        Shape (T, N, F), float32. NaN permitted where a name is absent or a
        feature is genuinely unavailable; consumers must respect `mask`.
    feature_names:
        Length F, unique.
    market:
        Shape (T, M), float32. The market state vector that drives the gate.
        NaN is NOT permitted here — a NaN market vector would gate every
        prediction on that date to NaN.
    market_names:
        Length M, unique.
    labels:
        Shape (T, N), float32. Forward return relative to `dates[t]`. NaN where
        no label exists (typically the final `horizon` rows).
    mask:
        Shape (T, N), bool. True iff tradeable. Single source of truth for
        universe membership — no module may fit on or trade a masked-out name.
    label_horizon:
        Number of trading days forward the label looks. Recorded so that
        embargo logic and decay analysis cannot silently disagree about it.
    metadata:
        MultiIndex (date, ticker) frame carrying at least mcap, sector, adv,
        price. Used by the risk model and the cost model.
    """

    dates: pd.DatetimeIndex
    tickers: TickerArray
    features: FeatureArray
    feature_names: list[str]
    market: FeatureArray
    market_names: list[str]
    labels: FeatureArray
    mask: MaskArray
    label_horizon: int = 1
    metadata: pd.DataFrame | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_dates()
        self._validate_tickers()
        self._validate_shapes()
        self._validate_market()
        self._validate_dtypes()
        self._validate_mask()
        self._validate_metadata()

    def _validate_dates(self) -> None:
        if not isinstance(self.dates, pd.DatetimeIndex):
            raise TypeError(f"dates must be a DatetimeIndex, got {type(self.dates).__name__}")
        if len(self.dates) == 0:
            raise ValueError("dates is empty")
        if not self.dates.is_monotonic_increasing:
            raise ValueError("dates must be sorted ascending")
        if not self.dates.is_unique:
            dupes = self.dates[self.dates.duplicated()].unique()
            raise ValueError(f"dates contains {len(dupes)} duplicate entries, first: {dupes[0]}")

    def _validate_tickers(self) -> None:
        if not isinstance(self.tickers, np.ndarray):
            raise TypeError(f"tickers must be an ndarray, got {type(self.tickers).__name__}")
        if self.tickers.ndim != 1:
            raise ValueError(f"tickers must be 1-D, got shape {self.tickers.shape}")
        if len(self.tickers) == 0:
            raise ValueError("tickers is empty")
        if len(set(self.tickers.tolist())) != len(self.tickers):
            raise ValueError("tickers contains duplicates")

    def _validate_shapes(self) -> None:
        t, n, f, m = self.n_dates, self.n_tickers, self.n_features, self.n_market

        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names contains duplicates")
        if len(set(self.market_names)) != len(self.market_names):
            raise ValueError("market_names contains duplicates")

        expected = {
            "features": ((t, n, f), self.features.shape),
            "market": ((t, m), self.market.shape),
            "labels": ((t, n), self.labels.shape),
            "mask": ((t, n), self.mask.shape),
        }
        for name, (want, got) in expected.items():
            if got != want:
                raise ValueError(f"{name} has shape {got}, expected {want}")

    def _validate_market(self) -> None:
        if np.isnan(self.market).any():
            bad = np.argwhere(np.isnan(self.market))
            date = self.dates[bad[0, 0]]
            col = self.market_names[bad[0, 1]]
            raise ValueError(
                f"market contains {len(bad)} NaN values — the gate cannot consume NaN. "
                f"First at {date.date()} in '{col}'."
            )
        if np.isinf(self.market).any():
            raise ValueError("market contains infinite values")

    def _validate_dtypes(self) -> None:
        if self.features.dtype != np.float32:
            raise TypeError(f"features must be float32, got {self.features.dtype}")
        if self.market.dtype != np.float32:
            raise TypeError(f"market must be float32, got {self.market.dtype}")
        if self.labels.dtype != np.float32:
            raise TypeError(f"labels must be float32, got {self.labels.dtype}")
        if self.mask.dtype != np.bool_:
            raise TypeError(f"mask must be bool, got {self.mask.dtype}")
        if self.label_horizon < 1:
            raise ValueError(f"label_horizon must be >= 1, got {self.label_horizon}")

    def _validate_mask(self) -> None:
        if not self.mask.any():
            raise ValueError("mask is entirely False — no tradeable names anywhere")
        empty_dates = int((~self.mask.any(axis=1)).sum())
        if empty_dates > 0:
            first = self.dates[int(np.argmax(~self.mask.any(axis=1)))]
            raise ValueError(
                f"{empty_dates} dates have no tradeable names, first at {first.date()}. "
                "Drop these dates rather than carrying them."
            )

    def _validate_metadata(self) -> None:
        if self.metadata is None:
            return
        if not isinstance(self.metadata.index, pd.MultiIndex):
            raise TypeError("metadata must have a MultiIndex of (date, ticker)")
        if self.metadata.index.nlevels != 2:
            raise ValueError(
                f"metadata index must have 2 levels (date, ticker), "
                f"got {self.metadata.index.nlevels}"
            )
        missing = REQUIRED_METADATA_COLUMNS - set(self.metadata.columns)
        if missing:
            raise ValueError(f"metadata missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------------ #
    # Shape accessors                                                     #
    # ------------------------------------------------------------------ #

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_market(self) -> int:
        return len(self.market_names)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_dates, self.n_tickers, self.n_features)

    # ------------------------------------------------------------------ #
    # Slicing                                                             #
    # ------------------------------------------------------------------ #

    def date_slice(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> Panel:
        """Return a Panel restricted to [start, end] inclusive.

        Used for chronological splits. Ticker ordering is preserved so that a
        model trained on one slice can score another positionally.
        """
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end)
        sel = (self.dates >= lo) & (self.dates <= hi)
        if not sel.any():
            raise ValueError(f"no dates in range [{lo.date()}, {hi.date()}]")

        meta = None
        if self.metadata is not None:
            idx = self.metadata.index.get_level_values(0)
            meta = self.metadata[(idx >= lo) & (idx <= hi)]

        return Panel(
            dates=self.dates[sel],
            tickers=self.tickers.copy(),
            features=self.features[sel],
            feature_names=list(self.feature_names),
            market=self.market[sel],
            market_names=list(self.market_names),
            labels=self.labels[sel],
            mask=self.mask[sel],
            label_horizon=self.label_horizon,
            metadata=meta,
            attrs=dict(self.attrs),
        )

    def split(
        self,
        train: tuple[str, str],
        valid: tuple[str, str],
        test: tuple[str, str],
        embargo_days: int = 21,
    ) -> dict[str, Panel]:
        """Chronological three-way split with an embargo gap between segments.

        The embargo exists because a label at date t looks `label_horizon` days
        forward; without a gap, the tail of train overlaps the head of valid and
        leaks. The gap is applied by trimming the END of each earlier segment.
        """
        for name, (lo, hi) in {"train": train, "valid": valid, "test": test}.items():
            if pd.Timestamp(lo) >= pd.Timestamp(hi):
                raise ValueError(f"{name} range is not increasing: {lo} -> {hi}")
        if pd.Timestamp(train[1]) >= pd.Timestamp(valid[0]):
            raise ValueError("train must end before valid begins")
        if pd.Timestamp(valid[1]) >= pd.Timestamp(test[0]):
            raise ValueError("valid must end before test begins")

        gap = pd.Timedelta(days=embargo_days)
        return {
            "train": self.date_slice(train[0], pd.Timestamp(train[1]) - gap),
            "valid": self.date_slice(valid[0], pd.Timestamp(valid[1]) - gap),
            "test": self.date_slice(test[0], test[1]),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @classmethod
    def load(cls, path: str | Path) -> Panel:
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)  # our own artifact, never untrusted input
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a Panel")
        return obj

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def coverage(self) -> pd.DataFrame:
        """Per-year coverage table. Goes straight into NOTES.md at Phase 0."""
        years = self.dates.year
        rows = []
        for year in sorted(set(years)):
            sel = years == year
            m = self.mask[sel]
            feats = self.features[sel]
            rows.append(
                {
                    "year": int(year),
                    "dates": int(sel.sum()),
                    "mean_names": float(m.sum(axis=1).mean()),
                    "min_names": int(m.sum(axis=1).min()),
                    "feature_nan_rate": float(np.isnan(feats[m]).mean()) if m.any() else np.nan,
                    "label_nan_rate": float(np.isnan(self.labels[sel][m]).mean())
                    if m.any()
                    else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("year")

    def __repr__(self) -> str:
        return (
            f"Panel(dates={self.n_dates} "
            f"[{self.dates[0].date()}..{self.dates[-1].date()}], "
            f"tickers={self.n_tickers}, features={self.n_features}, "
            f"market={self.n_market}, horizon={self.label_horizon}, "
            f"mean_universe={self.mask.sum(axis=1).mean():.0f})"
        )
