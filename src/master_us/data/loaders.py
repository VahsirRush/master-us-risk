"""Price acquisition via yfinance, hardened — data-sources-contract section 3.

yfinance is an unofficial scraper of Yahoo endpoints. It throttles, it returns
an empty frame instead of raising, and it only serves currently-listed tickers.
Each of those is handled explicitly here, because each of them fails silently by
default and a silent failure in the price layer is a survivorship bias that
nothing downstream can detect.

On the adjustment-ratio check — a DEVIATION from the contract, measured rather
than assumed. Sections 3.1, 3.2 and 7 all say `adj_close / close` must be
"monotone non-increasing over time". Under Yahoo's convention it is monotone
non-DECREASING: `adj_close` equals `close` on the most recent bar and is
progressively smaller going back, since each past dividend scales earlier prices
down. Measured over AAPL, KO and JNJ 2014-2024, every downward step in the ratio
is at most 8.3e-07 relative (float rounding in Yahoo's ~8 significant digits)
while genuine dividend steps are positive and reach 6e-03. The check below is
therefore non-decreasing within a relative tolerance of 1e-06. The contract's
intent — a non-monotone jump means a corrupted split adjustment — is preserved;
only the stated direction was wrong. See NOTES.md, Session 3.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from master_us.data.sources import RAW_ROOT, FetchError, batched, with_retry

PRICE_CACHE_DIR = RAW_ROOT / "prices"
FAILED_TICKERS_PATH = RAW_ROOT / "failed_tickers.json"

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# Calibrated against the full pull, not a sample. Across all 613 retrievable
# S&P 500 constituents over 2010-2025, the most negative relative step in
# adj_close/close is -1.451e-06 and the 99th percentile is -1.220e-06 — a hard
# ceiling, because it is float rounding in Yahoo's ~8 significant digits, not a
# data property. Genuine dividend adjustments are positive with a median
# magnitude of 8.9e-03. The two populations are separated by nearly four orders
# of magnitude, so anything in between works; 1e-05 sits ~7x above the observed
# noise ceiling and ~900x below the median true adjustment.
#
# An earlier value of 1e-06 was calibrated on three tickers over ten years and
# sat INSIDE the rounding distribution, flagging 73 of 613 tickers (11.9%) as
# corrupted when none were. See NOTES.md, Session 3.
ADJUSTMENT_MONOTONE_RTOL = 1e-5


@dataclass(frozen=True)
class TickerDiagnostics:
    """Per-ticker validation outcome. Recorded for every ticker, pass or fail."""

    ticker: str
    n_rows: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    requested_coverage: float  # observed days / business days in the request window
    internal_coverage: float  # observed days / business days in the OBSERVED span
    n_nonpositive_prices: int
    n_zero_volume: int
    n_negative_volume: int
    worst_adjustment_step: float  # most negative relative step in adj_close/close
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class YFinancePrices:
    """Cache-first OHLCV from Yahoo. Satisfies `PriceSource`.

    Batching, retry and validation parameters come from `config/data.yaml`
    (`prices:`) and default to the values in contract section 3.1.
    """

    cache_dir: Path = PRICE_CACHE_DIR
    batch_size: int = 30
    sleep_between_batches_sec: float = 1.5
    backoff_sec: tuple[float, ...] = (2.0, 8.0, 32.0)
    min_coverage_ratio: float = 0.8
    failed_tickers: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, TickerDiagnostics] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Cache                                                               #
    # ------------------------------------------------------------------ #

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def cached_tickers(self) -> set[str]:
        if not self.cache_dir.exists():
            return set()
        return {p.stem for p in self.cache_dir.glob("*.parquet")}

    def _read_cache(self, ticker: str) -> pl.DataFrame | None:
        path = self._path(ticker)
        return pl.read_parquet(path) if path.exists() else None

    def _write_cache(self, ticker: str, frame: pl.DataFrame) -> None:
        """Merge with anything already cached and rewrite. Append-only in effect.

        Re-runs fetch only new dates, so an existing row is never replaced by a
        later pull. That matters: Yahoo occasionally re-adjusts history, and
        silently overwriting would change past returns between runs.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_cache(ticker)
        merged = frame if existing is None else pl.concat([existing, frame])
        merged = merged.unique(subset=["date"], keep="first").sort("date")
        merged.write_parquet(self._path(ticker))

    # ------------------------------------------------------------------ #
    # Fetching                                                            #
    # ------------------------------------------------------------------ #

    def _download_batch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        def fetch() -> pd.DataFrame:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=False,  # keep BOTH close and adj_close
                    group_by="ticker",
                    progress=False,
                    threads=False,
                    actions=False,
                )
            if frame is None:
                raise FetchError("yfinance batch", ["download returned None"])
            return frame

        return with_retry(fetch, f"yfinance batch of {len(tickers)}", backoff=self.backoff_sec)

    @staticmethod
    def _extract(frame: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        """Pull one ticker out of a grouped multi-ticker frame."""
        if frame.empty:
            return None
        if isinstance(frame.columns, pd.MultiIndex):
            if ticker not in frame.columns.get_level_values(0):
                return None
            sub = frame[ticker]
        else:
            sub = frame
        sub = sub.dropna(how="all")
        return None if sub.empty else sub

    def _to_long(self, ticker: str, sub: pd.DataFrame) -> pl.DataFrame:
        wanted = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        missing = set(wanted) - set(sub.columns)
        if missing:
            raise FetchError(ticker, [f"missing columns {sorted(missing)}"])
        adj_col = "Adj Close" if "Adj Close" in sub.columns else None
        if adj_col is None:
            raise FetchError(
                ticker,
                [
                    "no 'Adj Close' column — yfinance was called with auto_adjust=True "
                    "somewhere, which destroys the split/dividend factor"
                ],
            )

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(sub.index).tz_localize(None),
                "ticker": ticker,
                **{dst: pd.to_numeric(sub[src], errors="coerce") for src, dst in wanted.items()},
                "adj_close": pd.to_numeric(sub[adj_col], errors="coerce"),
            }
        )
        out = out.dropna(subset=["close", "adj_close"])
        return pl.from_pandas(out[OHLCV_COLUMNS]).with_columns(
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("volume").cast(pl.Float64),
        )

    # ------------------------------------------------------------------ #
    # Validation — contract section 3.1                                   #
    # ------------------------------------------------------------------ #

    def validate(self, frame: pl.DataFrame, ticker: str, start: str, end: str) -> TickerDiagnostics:
        """Every assertion from section 3.1, each recorded rather than raised.

        Recorded, because a hard raise on the first odd ticker would abort a
        two-hour pull over one name. The caller decides what to do with the
        problems; `problems` being non-empty is never silently ignored.
        """
        pdf = frame.to_pandas()
        n = len(pdf)
        first = pdf["date"].min() if n else None
        last = pdf["date"].max() if n else None

        requested_days = max(len(pd.bdate_range(start, end)), 1)
        observed_days = max(len(pd.bdate_range(first, last)), 1) if n else 1

        problems: list[str] = []

        nonpositive = int(((pdf[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1)).sum())
        if nonpositive:
            problems.append(f"{nonpositive} rows with a non-positive price")

        neg_volume = int((pdf["volume"] < 0).sum())
        if neg_volume:
            problems.append(f"{neg_volume} rows with negative volume")

        zero_volume = int((pdf["volume"] == 0).sum())
        # Zero-volume days happen legitimately (halts, very thin names). A large
        # share of them means the series is padded, which is a different thing.
        if n and zero_volume / n > 0.05:
            problems.append(f"{zero_volume}/{n} rows ({zero_volume / n:.1%}) have zero volume")

        internal = n / observed_days
        if n and internal < self.min_coverage_ratio:
            problems.append(
                f"internal coverage {internal:.2f} < {self.min_coverage_ratio} "
                f"({n} rows over {observed_days} business days from "
                f"{first.date() if first is not None else '?'} to "
                f"{last.date() if last is not None else '?'}) — the series has holes"
            )

        worst_step = self._worst_adjustment_step(pdf)
        if worst_step < -ADJUSTMENT_MONOTONE_RTOL:
            problems.append(
                f"adj_close/close fell by {abs(worst_step):.2e} relative "
                f"(tolerance {ADJUSTMENT_MONOTONE_RTOL:.0e}) — corrupted split adjustment"
            )

        return TickerDiagnostics(
            ticker=ticker,
            n_rows=n,
            first_date=first,
            last_date=last,
            requested_coverage=n / requested_days,
            internal_coverage=internal,
            n_nonpositive_prices=nonpositive,
            n_zero_volume=zero_volume,
            n_negative_volume=neg_volume,
            worst_adjustment_step=worst_step,
            problems=tuple(problems),
        )

    @staticmethod
    def _worst_adjustment_step(pdf: pd.DataFrame) -> float:
        """Most negative relative step in adj_close/close. 0.0 if never negative."""
        if len(pdf) < 2:
            return 0.0
        ratio = (pdf["adj_close"] / pdf["close"]).to_numpy(dtype=float)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 2:
            return 0.0
        steps = np.diff(ratio) / ratio[:-1]
        return float(min(steps.min(), 0.0))

    # ------------------------------------------------------------------ #
    # The Protocol method                                                 #
    # ------------------------------------------------------------------ #

    def get_ohlcv(self, tickers: list[str], start: str, end: str) -> pl.DataFrame:
        """Cache-first OHLCV in long format. Only uncached tickers are fetched."""
        unique = sorted(set(tickers))
        cached = self.cached_tickers()
        to_fetch = [t for t in unique if t not in cached]

        for i, batch in enumerate(batched(to_fetch, self.batch_size)):
            self._fetch_batch(batch, start, end)
            if i + 1 < (len(to_fetch) + self.batch_size - 1) // self.batch_size:
                time.sleep(self.sleep_between_batches_sec)

        frames: list[pl.DataFrame] = []
        for ticker in unique:
            frame = self._read_cache(ticker)
            if frame is None:
                self.failed_tickers.setdefault(ticker, "no data retrieved")
                continue
            frames.append(frame)

        if not frames:
            raise FetchError(
                "price pull",
                [f"none of {len(unique)} tickers returned data — check connectivity"],
            )

        lo = pd.Timestamp(start).to_pydatetime()
        hi = pd.Timestamp(end).to_pydatetime()
        return (
            pl.concat(frames)
            .filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
            .sort(["date", "ticker"])
        )

    def _fetch_batch(self, batch: list[str], start: str, end: str) -> None:
        try:
            raw = self._download_batch(batch, start, end)
        except FetchError as exc:
            for ticker in batch:
                self.failed_tickers[ticker] = f"batch fetch failed: {exc.attempts[-1][:160]}"
            return

        returned = (
            set(raw.columns.get_level_values(0))
            if isinstance(raw.columns, pd.MultiIndex)
            else set(batch)
        )
        unexpected = returned - set(batch)
        if unexpected:
            raise FetchError(
                "yfinance batch",
                [f"returned tickers not requested: {sorted(unexpected)[:8]}"],
            )

        for ticker in batch:
            sub = self._extract(raw, ticker)
            if sub is None:
                # The section 3.1 case: delisted names come back empty, not as an error.
                self.failed_tickers[ticker] = "empty frame (likely delisted or unknown symbol)"
                continue
            try:
                long = self._to_long(ticker, sub)
            except FetchError as exc:
                self.failed_tickers[ticker] = exc.attempts[-1][:160]
                continue

            if long.is_empty():
                self.failed_tickers[ticker] = "all rows dropped as unparseable"
                continue

            diag = self.validate(long, ticker, start, end)
            self.diagnostics[ticker] = diag
            if diag.problems:
                self.failed_tickers[ticker] = "; ".join(diag.problems)[:300]
            self._write_cache(ticker, long)

    def revalidate_cache(self, start: str, end: str, tickers: list[str] | None = None) -> None:
        """Re-run validation over every cached ticker, without refetching.

        Validation otherwise runs only at fetch time, which leaves two holes: a
        resumed run produces no diagnostics for the tickers cached by the
        previous run, and a tolerance that is later recalibrated never gets
        applied to data already on disk. It is also the check that catches Yahoo
        silently re-adjusting history between pulls, which is the failure mode
        contract section 3.2 asks to be able to detect.

        Repopulates `diagnostics` and `failed_tickers` for validation problems.
        Retrieval failures recorded by `_fetch_batch` are left alone — a ticker
        with no cache entry has nothing to revalidate.
        """
        wanted = sorted(set(tickers)) if tickers is not None else sorted(self.cached_tickers())
        for ticker in wanted:
            frame = self._read_cache(ticker)
            if frame is None or frame.is_empty():
                continue
            diag = self.validate(frame, ticker, start, end)
            self.diagnostics[ticker] = diag
            if diag.problems:
                self.failed_tickers[ticker] = "; ".join(diag.problems)[:300]
            else:
                # A previously-flagged ticker that now passes must stop being flagged.
                self.failed_tickers.pop(ticker, None)

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def save_failures(self, path: Path = FAILED_TICKERS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "failed": self.failed_tickers,
            "diagnostics": {
                t: {
                    "n_rows": d.n_rows,
                    "first_date": str(d.first_date.date()) if d.first_date is not None else None,
                    "last_date": str(d.last_date.date()) if d.last_date is not None else None,
                    "requested_coverage": round(d.requested_coverage, 4),
                    "internal_coverage": round(d.internal_coverage, 4),
                    "n_zero_volume": d.n_zero_volume,
                    "worst_adjustment_step": d.worst_adjustment_step,
                    "problems": list(d.problems),
                }
                for t, d in self.diagnostics.items()
            },
        }
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path
