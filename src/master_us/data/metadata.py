"""Panel metadata: mcap, sector, adv, price — the minimum the cost and risk
paths need, built from data already on disk.

PROVISIONAL STATUS, stated loudly because downstream must not treat this as
final (Session 6; to be replaced/promoted in Phase 5 with the Barra work):

* **sector** — today's GICS sector from the Wikipedia constituents table,
  applied to the name's ENTIRE history. Sector changes and departed names'
  sectors are wrong by construction; departed names fall back to "Unknown".
  Phase 5 needs a proper historical mapping.
* **mcap** — shares outstanding from the CACHED SEC XBRL facts, joined on
  the `filed` date (as-of backward join, so it is point-in-time on the same
  rule as everything else), times the RAW close (as-traded price x as-filed
  count — deliberately not the adjusted close, which would mix today's
  split basis with a historical share count). Two known coarsenesses to fix
  in Phase 5: multi-class issuers (GOOG/GOOGL) each carry the entity-level
  share count, double-counting the company across classes; and the count
  staleness is capped at 400 days, beyond which mcap is NaN rather than a
  stale guess.
* **adv** — 21-day mean dollar volume, same definition the tradeable filter
  uses. Not provisional.
* **price** — raw close. Not provisional.

`attrs["metadata_provenance"]` on the Panel records all of this in machine-
readable form so no consumer can quietly assume finality.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from master_us.data.sec import TAG_FALLBACKS
from master_us.data.sources import DATA_ROOT, RAW_ROOT, with_retry
from master_us.data.universe import CONSTITUENTS_URL, _clean_ticker

SECTORS_CACHE = RAW_ROOT / "sp500_sectors.parquet"
FUNDAMENTALS_INTERIM = DATA_ROOT / "interim" / "fundamentals.parquet"

SHARES_STALENESS_DAYS = 400

METADATA_PROVENANCE = {
    "sector": "PROVISIONAL: current GICS sector applied to full history; 'Unknown' for departed names",
    "mcap": "PROVISIONAL: SEC as-filed shares (filed-date as-of join, entity-level, "
    f"staleness cap {SHARES_STALENESS_DAYS}d) x raw close; multi-class issuers double-counted",
    "adv": "21d mean dollar volume (same definition as the tradeable filter)",
    "price": "raw close",
    "replace_in": "Phase 5 (Barra descriptors)",
}


def fetch_sectors(user_agent: str, cache_path: Path = SECTORS_CACHE) -> pl.DataFrame:
    """[ticker, sector] from the constituents table. Cached; scrape on miss."""
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    import requests

    def get() -> str:
        resp = requests.get(CONSTITUENTS_URL, headers={"User-Agent": user_agent}, timeout=60)
        resp.raise_for_status()
        return resp.text

    tables = pd.read_html(io.StringIO(with_retry(get, f"GET {CONSTITUENTS_URL}")))
    table = next(t for t in tables if "Symbol" in t.columns and "GICS Sector" in t.columns)
    frame = pl.DataFrame(
        {
            "ticker": [_clean_ticker(s) for s in table["Symbol"].astype(str)],
            "sector": table["GICS Sector"].astype(str).to_list(),
        }
    ).unique(subset=["ticker"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache_path)
    return frame


def shares_outstanding_asof(tickers: list[str]) -> pl.DataFrame:
    """[ticker, filed, shares] from the cached XBRL facts, tag-priority resolved.

    One row per (ticker, filed date): the share count knowable from that day
    on. The as-of join onto trading dates happens in `build_metadata`.
    """
    if not FUNDAMENTALS_INTERIM.exists():
        raise FileNotFoundError(
            f"{FUNDAMENTALS_INTERIM} missing — run scripts/03_build_panel.py first"
        )
    priority = {tag: i for i, tag in enumerate(TAG_FALLBACKS["shares_outstanding"])}
    facts = (
        pl.read_parquet(FUNDAMENTALS_INTERIM)
        .filter(
            pl.col("tag").is_in(list(priority))
            & pl.col("ticker").is_in(tickers)
            & (pl.col("value") > 0)
        )
        .with_columns(pl.col("tag").replace_strict(priority, return_dtype=pl.Int32).alias("_rank"))
        .sort(["_rank"])
        .group_by(["ticker", "filed"])
        .first()  # best tag per filing date
        .select(
            "ticker",
            pl.col("filed").cast(pl.Datetime("ns")),  # match the price frames' unit
            pl.col("value").alias("shares"),
        )
        .sort(["ticker", "filed"])
    )
    return _harmonize_split_basis(facts)


def _harmonize_split_basis(facts: pl.DataFrame) -> pl.DataFrame:
    """Express every historical share count on the LATEST filing's split basis.

    Necessary because Yahoo's `close` column is SPLIT-adjusted (measured:
    AAPL's cached June-2015 close is ~$31.75, the post-2014/2020-splits basis,
    not the ~$127 it traded at). As-filed share counts are on the FILING
    date's basis, so raw shares x cached close understates pre-split mcap by
    the split factor — AAPL 2015 came out $188B instead of ~$730B.

    The cache has no split events, but the filings themselves reveal them: a
    split shows as a ~2x/4x/7x jump between consecutive filed counts, far
    outside buyback/issuance drift (a few percent per quarter). Any jump with
    ratio outside [0.75, 1.33] is treated as a basis change, and earlier
    counts are scaled by the cumulative product of the jumps after them.
    Validated: AAPL June-2015 mcap comes out $731B on the harmonized basis.
    The residual error (true basis changes inside +/-33%, e.g. huge one-shot
    issuance) is part of this column's PROVISIONAL status.
    """
    parts: list[pl.DataFrame] = []
    for _, group in facts.group_by("ticker", maintain_order=True):
        shares = group["shares"].to_numpy()
        if len(shares) > 1:
            ratio = shares[1:] / shares[:-1]
            jump = np.where((ratio > 4.0 / 3.0) | (ratio < 0.75), ratio, 1.0)
            # factor[i] = product of jumps after filing i -> today's basis
            factor = np.concatenate([np.cumprod(jump[::-1])[::-1], [1.0]])
            shares = shares * factor
        parts.append(group.with_columns(pl.Series("shares", shares)))
    return pl.concat(parts)


def build_metadata(
    ohlcv: pl.DataFrame,
    panel_start: pd.Timestamp,
    tickers: list[str],
    user_agent: str,
) -> pd.DataFrame:
    """The MultiIndex (date, ticker) frame the Panel contract requires.

    `ohlcv` is the assembly's long frame (date, ticker, close, and the _adv21
    column it already computes). Rows outside `tickers` or before
    `panel_start` are dropped.
    """
    required = {"date", "ticker", "close", "_adv21"}
    if not required <= set(ohlcv.columns):
        raise ValueError(f"ohlcv is missing {sorted(required - set(ohlcv.columns))}")

    base = (
        ohlcv.filter(
            (pl.col("date") >= panel_start.to_pydatetime()) & pl.col("ticker").is_in(tickers)
        )
        .select(
            "date",
            "ticker",
            pl.col("close").alias("price"),
            pl.col("_adv21").alias("adv"),
        )
        .sort(["ticker", "date"])
    )

    shares = shares_outstanding_asof(tickers)
    joined = base.join_asof(
        shares,
        left_on="date",
        right_on="filed",
        by="ticker",
        strategy="backward",
        tolerance=f"{SHARES_STALENESS_DAYS}d",
    ).with_columns((pl.col("shares") * pl.col("price")).alias("mcap"))

    sectors = fetch_sectors(user_agent)
    joined = joined.join(sectors, on="ticker", how="left").with_columns(
        pl.col("sector").fill_null("Unknown")
    )

    out = (
        joined.select(["date", "ticker", "mcap", "sector", "adv", "price"])
        .to_pandas()
        .set_index(["date", "ticker"])
        .sort_index()
    )
    return out
