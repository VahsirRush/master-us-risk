"""Industry classification for the risk model — spec §9.1's industry dummies.

WHY THIS MODULE EXISTS AT ALL. `Panel.metadata["sector"]` is today's GICS
sector scraped from the Wikipedia constituents table and applied backwards
over each name's whole history. It has two defects, and the second is fatal
for §9.2:

1. It is not point-in-time. A name that moved sectors carries its current
   sector back through history.
2. **17.0% of the universe (99 of 584 names) resolves to "Unknown"** — every
   name that has since left the index, because it is not in today's table.

Defect 2 is disqualifying on its own. §9.2 requires z-scoring *within
industry*, so "Unknown" would become a 12th pseudo-industry whose membership
is precisely "names that were later removed from the S&P 500". That group is
correlated with distress and delisting, so standardizing within it would bake
a survivorship artifact into every descriptor. An industry scheme has to
cover the whole cross-section or it is not an industry scheme.

THE FIX. SEC assigns every filer an SIC code, and it keeps the record after
a company leaves an index — so SIC covers the departed names that Wikipedia
structurally cannot. This module fetches SIC per CIK from the submissions
API and maps SIC divisions onto GICS-L1-shaped buckets.

WHAT THIS IS NOT. SIC is a different taxonomy from GICS, coarser and
famously dated in its treatment of technology. The mapping below is a
documented approximation, not a GICS licence. Two honesty measures come with
it: `agreement_with_gics()` measures how often the SIC-derived sector matches
the real GICS sector on the 83% of names where GICS is known, and the
resulting column is named `industry`, never `gics_sector`, so no downstream
consumer can mistake it for the licensed classification.

THE HYBRID, AND WHY IT WON. The first design used the SIC-derived scheme for
every name, on the argument that one consistent taxonomy beats a mixture.
The measurement killed that: SIC-derived buckets agree with real GICS only
**77.5%** of the time (Utilities 100%, Real Estate 96.7%, but Consumer
Discretionary 60.9% and Materials 62.5%). Using them everywhere would
misclassify roughly a fifth of the 485 names whose true sector is known, to
buy consistency that does not pay for itself — both paths emit the same 11
GICS-L1 labels, so this is one label set with two assignment mechanisms of
different accuracy, not two taxonomies.

`resolve_industry` therefore takes GICS where it exists and SIC only for the
99 names it cannot cover, and records which was used per name in
`industry_source`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from master_us.data.sources import RAW_ROOT, sec_user_agent, with_retry

SIC_CACHE = RAW_ROOT / "sec_sic_by_cik.parquet"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC's own SIC division boundaries, collapsed onto GICS-L1-shaped buckets.
# Ranges are (low, high, bucket) inclusive, tested in order.
SIC_RANGES: tuple[tuple[int, int, str], ...] = (
    (100, 999, "Materials"),          # agriculture, forestry, fishing
    (1000, 1099, "Materials"),        # metal mining
    (1200, 1299, "Energy"),           # coal
    (1300, 1399, "Energy"),           # oil and gas extraction
    (1400, 1499, "Materials"),        # nonmetallic minerals
    (1500, 1799, "Industrials"),      # construction
    (2000, 2199, "Consumer Staples"), # food, tobacco
    (2200, 2399, "Consumer Discretionary"),  # textiles, apparel
    (2400, 2599, "Industrials"),      # lumber, furniture
    (2600, 2699, "Materials"),        # paper
    (2700, 2799, "Communication Services"),  # printing and publishing
    (2800, 2829, "Materials"),        # industrial chemicals
    (2830, 2836, "Health Care"),      # drugs and biologicals
    (2840, 2899, "Consumer Staples"), # soap, cosmetics
    (2900, 2999, "Energy"),           # petroleum refining
    (3000, 3299, "Materials"),        # rubber, stone, clay, glass
    (3300, 3399, "Materials"),        # primary metals
    (3400, 3569, "Industrials"),      # fabricated metal, machinery
    (3570, 3579, "Information Technology"),  # computers and office equipment
    (3580, 3599, "Industrials"),
    (3600, 3639, "Industrials"),      # electrical equipment
    (3640, 3669, "Information Technology"),
    (3670, 3679, "Information Technology"),  # semiconductors
    (3680, 3699, "Information Technology"),
    (3700, 3799, "Consumer Discretionary"),  # motor vehicles, aerospace splits below
    (3800, 3829, "Information Technology"),  # instruments
    (3830, 3859, "Health Care"),      # medical instruments
    (3860, 3999, "Consumer Discretionary"),
    (4000, 4299, "Industrials"),      # railroads, trucking
    (4400, 4599, "Industrials"),      # water, air transport
    (4600, 4699, "Energy"),           # pipelines
    (4700, 4799, "Industrials"),      # transportation services
    (4800, 4899, "Communication Services"),
    (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"),      # wholesale
    (5200, 5399, "Consumer Discretionary"),
    (5400, 5499, "Consumer Staples"), # food stores
    (5500, 5799, "Consumer Discretionary"),
    (5800, 5899, "Consumer Discretionary"),  # eating and drinking
    (5900, 5912, "Consumer Discretionary"),
    (5912, 5912, "Consumer Staples"), # drug stores
    (5913, 5999, "Consumer Discretionary"),
    (6000, 6499, "Financials"),
    (6500, 6599, "Real Estate"),
    (6798, 6798, "Real Estate"),      # REITs
    (6600, 6797, "Financials"),
    (6799, 6999, "Financials"),
    (7000, 7099, "Consumer Discretionary"),  # hotels
    (7200, 7299, "Consumer Discretionary"),
    (7300, 7369, "Industrials"),      # business services
    (7370, 7379, "Information Technology"),  # computer services / software
    (7380, 7399, "Industrials"),
    (7500, 7699, "Consumer Discretionary"),
    (7700, 7999, "Communication Services"),  # entertainment
    (8000, 8099, "Health Care"),
    (8100, 8199, "Industrials"),
    (8200, 8299, "Consumer Discretionary"),  # educational services
    (8300, 8399, "Health Care"),
    (8400, 8699, "Consumer Discretionary"),
    (8700, 8799, "Industrials"),      # engineering, accounting, research
    (8800, 8999, "Industrials"),
)

# Aerospace/defence sits inside the 3700s motor-vehicle block but is
# Industrials in GICS; SIC 3720-3729 and 3760-3769 are the aircraft and
# missile codes.
SIC_OVERRIDES: tuple[tuple[int, int, str], ...] = (
    (3720, 3729, "Industrials"),
    (3760, 3769, "Industrials"),
    (3812, 3812, "Industrials"),  # search/navigation/defence electronics
    (2833, 2836, "Health Care"),
    (5122, 5122, "Health Care"),  # drug wholesalers
)

UNKNOWN = "Unclassified"


def sic_to_industry(sic: int | None) -> str:
    """Map one SIC code onto a GICS-L1-shaped bucket.

    Overrides are tested before the main ranges so the narrow corrections
    (aerospace inside the motor-vehicle block) win over the broad bucket.
    """
    if sic is None or sic <= 0:
        return UNKNOWN
    for lo, hi, bucket in SIC_OVERRIDES:
        if lo <= sic <= hi:
            return bucket
    for lo, hi, bucket in SIC_RANGES:
        if lo <= sic <= hi:
            return bucket
    return UNKNOWN


def fetch_sic(
    ciks: list[int],
    cache_path: Path = SIC_CACHE,
    user_agent: str | None = None,
    pause: float = 0.12,
    refresh: bool = False,
) -> pl.DataFrame:
    """[cik, sic, sic_description, entity_name] from SEC submissions.

    Cache-first per the project convention: the whole frame is cached, and
    only CIKs missing from it are fetched. `pause` keeps the request rate
    under SEC's 10/second limit.

    A CIK that 404s or carries no SIC is recorded with `sic = 0` rather than
    dropped, so a later run does not retry it forever and the coverage table
    can count it honestly.
    """
    import requests

    cached = pl.read_parquet(cache_path) if cache_path.exists() and not refresh else None
    have = set(cached["cik"].to_list()) if cached is not None else set()
    todo = [c for c in dict.fromkeys(ciks) if c not in have]
    if not todo:
        assert cached is not None
        return cached

    agent = user_agent or sec_user_agent()
    rows: list[dict[str, object]] = []
    for i, cik in enumerate(todo):
        url = SUBMISSIONS_URL.format(cik=cik)

        def get(u: str = url) -> str:
            resp = requests.get(u, headers={"User-Agent": agent}, timeout=30)
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            return resp.text

        try:
            body = with_retry(get, f"GET submissions CIK{cik:010d}")
        except Exception as exc:
            rows.append({"cik": cik, "sic": 0, "sic_description": f"FETCH FAILED: {exc}",
                         "entity_name": ""})
            continue

        if not body:
            rows.append({"cik": cik, "sic": 0, "sic_description": "404", "entity_name": ""})
        else:
            doc = json.loads(body)
            raw = str(doc.get("sic") or "").strip()
            rows.append(
                {
                    "cik": cik,
                    "sic": int(raw) if raw.isdigit() else 0,
                    "sic_description": str(doc.get("sicDescription") or ""),
                    "entity_name": str(doc.get("name") or ""),
                }
            )
        if i % 50 == 0:
            print(f"  SIC {i + 1}/{len(todo)}")
        time.sleep(pause)

    fresh = pl.DataFrame(
        rows,
        schema={"cik": pl.Int64, "sic": pl.Int64, "sic_description": pl.Utf8,
                "entity_name": pl.Utf8},
    )
    out = pl.concat([cached, fresh]) if cached is not None else fresh
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache_path)
    return out


def industry_map(
    tickers: list[str],
    cik_map: pl.DataFrame,
    sic: pl.DataFrame,
) -> pl.DataFrame:
    """[ticker, cik, sic, industry] for every requested ticker.

    Tickers with no CIK or no SIC land in `UNKNOWN`; they are kept in the
    frame so the caller can count them rather than discover a silent drop.
    """
    base = pl.DataFrame({"ticker": list(dict.fromkeys(tickers))})
    joined = (
        base.join(cik_map.select("ticker", "cik").unique(subset=["ticker"]), on="ticker", how="left")
        .join(sic.select("cik", "sic"), on="cik", how="left")
        .with_columns(
            pl.col("sic")
            .map_elements(lambda s: sic_to_industry(s if s is not None else None),
                          return_dtype=pl.Utf8)
            .alias("industry")
        )
    )
    return joined


def resolve_industry(
    tickers: list[str],
    cik_map: pl.DataFrame,
    sic: pl.DataFrame,
    gics: pl.DataFrame,
) -> pl.DataFrame:
    """[ticker, industry, industry_source] — GICS where known, SIC elsewhere.

    The module docstring argues for one scheme over a hybrid; the measurement
    overturned that. SIC-derived buckets agree with real GICS only 77.5% of
    the time, so using them everywhere would misclassify roughly a fifth of
    the 485 names whose true sector is known. Taking GICS where it exists and
    SIC only for the 99 names it cannot cover gives ~96% expected accuracy
    against ~78%.

    The consistency objection does not survive contact either: both paths
    emit the same 11 GICS-L1 labels, so this is one label set with two
    assignment mechanisms of different accuracy, not two taxonomies. The
    mechanism is recorded per name in `industry_source` so any later analysis
    can condition on it.
    """
    sic_based = industry_map(tickers, cik_map, sic).select("ticker", "cik", "sic",
                                                           pl.col("industry").alias("_sic_ind"))
    out = (
        sic_based.join(gics.select("ticker", "sector"), on="ticker", how="left")
        .with_columns(
            pl.when(pl.col("sector").is_not_null() & (pl.col("sector") != "Unknown"))
            .then(pl.col("sector"))
            .otherwise(pl.col("_sic_ind"))
            .alias("industry"),
            pl.when(pl.col("sector").is_not_null() & (pl.col("sector") != "Unknown"))
            .then(pl.lit("gics"))
            .otherwise(
                pl.when(pl.col("_sic_ind") != UNKNOWN)
                .then(pl.lit("sic"))
                .otherwise(pl.lit("none"))
            )
            .alias("industry_source"),
        )
        .select("ticker", "cik", "sic", "industry", "industry_source")
    )
    return out


def primary_class(tickers: list[str], cik_map: pl.DataFrame, adv: pl.DataFrame) -> pl.DataFrame:
    """[ticker, cik, is_primary_class] — one primary listing per issuer.

    Multi-class issuers (GOOG/GOOGL, FOX/FOXA, NWS/NWSA, UA/UAA here) each
    carry the ENTITY-level share count, so cap-weighting over both classes
    counts the company twice. The share count cannot be split per class from
    what SEC's financial statement datasets provide, so the fix is not to
    divide it: the class with the higher mean dollar volume is marked
    primary, and cap-weighted aggregates use only primary classes.

    `mcap` itself stays entity-level on every class, which is the right
    quantity for the size descriptor — Barra's size is company size, not
    share-class size. Only the *weighting* needs de-duplication.
    """
    base = (
        pl.DataFrame({"ticker": list(dict.fromkeys(tickers))})
        .join(cik_map.select("ticker", "cik").unique(subset=["ticker"]), on="ticker", how="left")
        .join(adv.select("ticker", "mean_adv"), on="ticker", how="left")
        .with_columns(pl.col("mean_adv").fill_null(0.0))
    )
    ranked = base.with_columns(
        pl.col("mean_adv").rank("ordinal", descending=True).over("cik").alias("_rank")
    )
    return ranked.with_columns(
        (pl.col("_rank") == 1).alias("is_primary_class")
    ).select("ticker", "cik", "is_primary_class")


def agreement_with_gics(industry: pl.DataFrame, gics: pl.DataFrame) -> dict[str, object]:
    """How often the SIC-derived bucket matches the real GICS sector.

    Measured only on names where GICS is actually known — the point is to
    validate the mapping, and 'Unknown' carries no information to agree with.
    A low rate here is a reason to distrust the mapping, so it is reported
    rather than assumed.
    """
    merged = (
        industry.join(gics, on="ticker", how="inner")
        .filter((pl.col("sector") != "Unknown") & (pl.col("industry") != UNKNOWN))
    )
    if merged.height == 0:
        return {"n": 0, "agreement": None, "by_sector": pl.DataFrame()}

    merged = merged.with_columns(
        (pl.col("industry") == pl.col("sector")).cast(pl.Float64).alias("match")
    )
    by_sector = (
        merged.group_by("sector")
        .agg(pl.len().alias("n"), pl.col("match").mean().alias("rate"))
        .sort("n", descending=True)
    )
    return {
        "n": merged.height,
        "agreement": float(merged["match"].sum()) / merged.height,
        "by_sector": by_sector,
        "confusion": (
            merged.filter(~pl.col("match"))
            .group_by(["sector", "industry"])
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        ),
    }
