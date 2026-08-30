"""SEC Financial Statement Data Sets — data-sources-contract section 4.

The single reason this source exists rather than an EDGAR scrape: `sub.txt`
carries a `filed` field, the SEC acceptance date, and that field is what makes
the data genuinely point-in-time.

THE PIT RULE, which the rest of this module exists to enforce:

    A fact is usable on date t if and only if filed <= t.

`period` (in `sub.txt`) and `ddate` (in `num.txt`) are fiscal period ends. They
precede public availability by 30 to 90 days — the 2015Q1 archive has filings
with period 2014-12-31 and filed 2015-02-11, a 42-day gap. Joining on either of
those hands the model a balance sheet weeks before anyone could have traded on
it, and manufactures returns that were never available. That is the single most
common source of look-ahead in retail quant work, and it is invisible in the
output: the numbers simply come out better.

Both dates are carried through to the panel, `period_end` for diagnostics only.
`as_of()` is the only sanctioned way to ask what was knowable on a date, and it
filters on `filed`.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import requests

from master_us.data.sources import (
    RAW_ROOT,
    FetchError,
    RateLimiter,
    require_valid_user_agent,
    with_retry,
)

SEC_CACHE_DIR = RAW_ROOT / "sec"
ARCHIVE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CIK_MAP_CACHE = RAW_ROOT / "sec_cik_ticker_map.parquet"

KEEP_FORMS = ("10-K", "10-Q")

# Ordered fallbacks per concept — contract section 4.3. XBRL tags are not stable
# across filers or years; `Revenues` was largely displaced by the
# RevenueFromContractWithCustomer* tags after ASC 606 adoption around 2018. Order
# is preference order: the first tag that resolves for a (cik, period) wins.
TAG_FALLBACKS: Mapping[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "assets": ("Assets",),
    "liabilities": ("Liabilities", "LiabilitiesAndStockholdersEquity"),
    "long_term_debt": (
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ),
    "shares_outstanding": (
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ),
}

ALL_TAGS: frozenset[str] = frozenset(t for tags in TAG_FALLBACKS.values() for t in tags)

FUNDAMENTAL_COLUMNS = ["ticker", "cik", "filed", "period_end", "tag", "value"]


@dataclass(frozen=True)
class TagCoverage:
    """Which tag resolved a concept, and for what share of observations."""

    concept: str
    year: int
    tag: str
    n_ciks: int
    share: float


def quarters_between(start_quarter: str, end: str) -> list[str]:
    """Inclusive list of `YYYYqN` labels from `start_quarter` to the quarter of `end`."""
    year, q = int(start_quarter[:4]), int(start_quarter[-1])
    last = pd.Timestamp(end)
    out: list[str] = []
    while (year, q) <= (last.year, (last.month - 1) // 3 + 1):
        out.append(f"{year}q{q}")
        q += 1
        if q > 4:
            year, q = year + 1, 1
    return out


class SECFundamentals:
    """Bulk XBRL archives, cached per quarter. Satisfies `FundamentalSource`."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path = SEC_CACHE_DIR,
        rate_limit_per_sec: float = 8.0,
        timeout: int = 300,
    ) -> None:
        self.user_agent = require_valid_user_agent(user_agent)
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(per_second=rate_limit_per_sec)
        self.timeout = timeout
        self.failed_quarters: dict[str, str] = {}
        # Per quarter, how many facts were discarded for having filed < period_end.
        self.dropped_impossible_dates: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # CIK -> ticker                                                       #
    # ------------------------------------------------------------------ #

    def cik_ticker_map(self, refresh: bool = False) -> pl.DataFrame:
        """[cik, ticker] from the SEC's own mapping file.

        The SEC keys on CIK; everything else in this project keys on ticker. The
        mapping is many-to-one for multi-class issuers (GOOG/GOOGL share a CIK),
        so this is deliberately not deduplicated to one row per CIK — the join
        downstream fans out and that is correct.
        """
        if CIK_MAP_CACHE.exists() and not refresh:
            return pl.read_parquet(CIK_MAP_CACHE)

        self.limiter.wait()

        def fetch() -> dict[str, dict[str, object]]:
            resp = requests.get(
                COMPANY_TICKERS_URL, headers={"User-Agent": self.user_agent}, timeout=self.timeout
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise FetchError("company_tickers.json", ["payload was not a mapping"])
            return payload

        payload = with_retry(fetch, "company_tickers.json")
        frame = pl.DataFrame(
            {
                "cik": [int(str(v["cik_str"])) for v in payload.values()],
                "ticker": [str(v["ticker"]).upper().replace(".", "-") for v in payload.values()],
            }
        ).unique()
        CIK_MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(CIK_MAP_CACHE)
        return frame

    # ------------------------------------------------------------------ #
    # Quarterly archives                                                  #
    # ------------------------------------------------------------------ #

    def _quarter_path(self, quarter: str) -> Path:
        return self.cache_dir / f"{quarter}.parquet"

    def fetch_quarter(self, quarter: str, force: bool = False) -> pl.DataFrame:
        """Download, parse and cache one quarterly archive.

        The ZIPs are 60-100MB each and contain millions of numeric facts. Only
        the tags in `ALL_TAGS` survive to the parquet, which is what keeps the
        cache in the tens of megabytes rather than the tens of gigabytes.
        """
        path = self._quarter_path(quarter)
        if path.exists() and not force:
            return pl.read_parquet(path)

        self.limiter.wait()
        url = ARCHIVE_URL.format(quarter=quarter)

        def fetch() -> bytes:
            resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.content

        payload = with_retry(fetch, f"SEC archive {quarter}")
        frame = self._parse_archive(payload, quarter)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        return frame

    def _parse_archive(self, payload: bytes, quarter: str) -> pl.DataFrame:
        """Join sub.txt to num.txt, keeping only 10-K/10-Q and the tags we need."""
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = set(archive.namelist())
            missing = {"sub.txt", "num.txt"} - members
            if missing:
                raise FetchError(
                    f"SEC archive {quarter}", [f"archive is missing {sorted(missing)}"]
                )
            sub = pd.read_csv(
                archive.open("sub.txt"),
                sep="\t",
                dtype=str,
                usecols=["adsh", "cik", "name", "form", "period", "filed", "fy", "fp"],
            )
            num = pd.read_csv(
                archive.open("num.txt"),
                sep="\t",
                dtype=str,
                usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom", "value", "coreg"],
            )

        sub = sub[sub["form"].isin(KEEP_FORMS)].copy()
        if sub.empty:
            return pl.DataFrame(schema=_fundamental_schema())

        sub = self._prefer_originals(sub)

        num = num[num["tag"].isin(ALL_TAGS)]
        # coreg marks a co-registrant's figures (a subsidiary filing alongside the
        # parent). Those are not the consolidated entity, so they are dropped.
        num = num[num["coreg"].isna() | (num["coreg"] == "")]
        if num.empty:
            return pl.DataFrame(schema=_fundamental_schema())

        merged = num.merge(sub[["adsh", "cik", "period", "filed"]], on="adsh", how="inner")
        if merged.empty:
            return pl.DataFrame(schema=_fundamental_schema())

        out = pd.DataFrame(
            {
                "cik": pd.to_numeric(merged["cik"], errors="coerce"),
                "filed": pd.to_datetime(merged["filed"], format="%Y%m%d", errors="coerce"),
                "period_end": pd.to_datetime(merged["period"], format="%Y%m%d", errors="coerce"),
                "ddate": pd.to_datetime(merged["ddate"], format="%Y%m%d", errors="coerce"),
                "tag": merged["tag"],
                "value": pd.to_numeric(merged["value"], errors="coerce"),
                "qtrs": pd.to_numeric(merged["qtrs"], errors="coerce"),
                "quarter": quarter,
            }
        ).dropna(subset=["cik", "filed", "value"])

        # A filing accepted before its own period ended is impossible. Measured
        # against the real archives these are upstream data-entry errors, not a
        # parsing fault on our side, and they are vanishingly rare: 2 of 6,439
        # 10-K/10-Q submissions in 2024q2 (0.03%), both a Q1 10-Q whose `period`
        # was mis-tagged as the following December.
        #
        # They are dropped rather than kept or raised on. Kept is wrong because a
        # filing whose period is nonsense may well have a nonsense value attached;
        # raising is wrong because it discards a whole quarter of good data over a
        # couple of bad filings. The count is recorded so it stays visible.
        impossible = out["filed"] < out["period_end"]
        n_impossible = int(impossible.sum())
        if n_impossible:
            share = n_impossible / len(out)
            if share > 0.01:
                raise FetchError(
                    f"SEC archive {quarter}",
                    [
                        f"{n_impossible} of {len(out)} facts ({share:.2%}) have "
                        "filed < period_end. Isolated cases are upstream filer errors, "
                        "but this many means the date parsing is wrong."
                    ],
                )
            out = out[~impossible]
        self.dropped_impossible_dates[quarter] = n_impossible

        out["cik"] = out["cik"].astype("int64")
        return pl.from_pandas(out)

    @staticmethod
    def _prefer_originals(sub: pd.DataFrame) -> pd.DataFrame:
        """Drop amendments where an original exists for the same (cik, period, form).

        Contract section 8.3: the as-filed value is the correct PIT value. A
        restatement is information the model would not have had, so preferring
        originals is deliberate, not a shortcut.
        """
        sub = sub.copy()
        sub["is_amendment"] = sub["form"].str.contains("/A", regex=False)
        sub = sub.sort_values(["cik", "period", "is_amendment", "filed"])
        return sub.drop_duplicates(subset=["cik", "period"], keep="first").drop(
            columns=["is_amendment"]
        )

    # ------------------------------------------------------------------ #
    # The Protocol method                                                 #
    # ------------------------------------------------------------------ #

    def get_fundamentals(self, start: str, end: str) -> pl.DataFrame:
        """[ticker, cik, filed, period_end, tag, value] over the requested window.

        Filtered on `filed`, never on `period_end` — a fact filed inside the
        window but covering an earlier period belongs here; one covering a
        period inside the window but filed after it does not.
        """
        quarters = quarters_between("2009q2", end)
        frames: list[pl.DataFrame] = []
        for quarter in quarters:
            path = self._quarter_path(quarter)
            if path.exists():
                frames.append(pl.read_parquet(path))

        if not frames:
            raise FetchError(
                "SEC fundamentals",
                [f"no cached quarters under {self.cache_dir} — run scripts/02_fetch_fundamentals.py"],
            )

        facts = pl.concat(frames, how="diagonal")
        lo, hi = pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()
        facts = facts.filter((pl.col("filed") >= lo) & (pl.col("filed") <= hi))

        mapping = self.cik_ticker_map()
        joined = facts.join(mapping, on="cik", how="inner")
        return joined.select(FUNDAMENTAL_COLUMNS).sort(["filed", "ticker", "tag"])


# --------------------------------------------------------------------- #
# The PIT accessor                                                       #
# --------------------------------------------------------------------- #


def as_of(facts: pl.DataFrame, date: str | pd.Timestamp) -> pl.DataFrame:
    """Facts knowable on `date` — the most recent filing per (ticker, tag) with filed <= date.

    The ONLY sanctioned way to ask what was known on a date. It filters on
    `filed`; `period_end` never enters the predicate. Everything about the PIT
    guarantee reduces to this function being the only door.
    """
    cutoff = pd.Timestamp(date).to_pydatetime()
    return (
        facts.filter(pl.col("filed") <= cutoff)
        .sort("filed")
        .group_by(["ticker", "tag"])
        .last()
        .sort(["ticker", "tag"])
    )


def resolve_tag(facts: pl.DataFrame, concept: str) -> pl.DataFrame:
    """Resolve one concept to a single value per (cik, period_end) via the fallback list.

    Tries `TAG_FALLBACKS[concept]` in order and takes the first tag that has a
    value for that filing. Returns [cik, ticker, filed, period_end, concept,
    value, resolved_tag] — `resolved_tag` is kept because which tag answered is
    a genuine diagnostic, not bookkeeping: a concept that switches tags mid-sample
    is exactly where a descriptor quietly changes meaning.
    """
    if concept not in TAG_FALLBACKS:
        raise KeyError(f"unknown concept {concept!r}; known: {sorted(TAG_FALLBACKS)}")

    priority = {tag: i for i, tag in enumerate(TAG_FALLBACKS[concept])}
    subset = facts.filter(pl.col("tag").is_in(list(priority)))
    if subset.is_empty():
        return pl.DataFrame(
            schema={
                "cik": pl.Int64,
                "ticker": pl.String,
                "filed": pl.Datetime("ns"),
                "period_end": pl.Datetime("ns"),
                "concept": pl.String,
                "value": pl.Float64,
                "resolved_tag": pl.String,
            }
        )

    return (
        subset.with_columns(
            pl.col("tag").replace_strict(priority, return_dtype=pl.Int32).alias("_rank")
        )
        .sort(["_rank", "filed"])
        .group_by(["cik", "period_end"])
        .first()
        .with_columns(pl.lit(concept).alias("concept"), pl.col("tag").alias("resolved_tag"))
        .select(["cik", "ticker", "filed", "period_end", "concept", "value", "resolved_tag"])
        .sort(["ticker", "period_end"])
    )


def tag_coverage(facts: pl.DataFrame, concepts: Iterable[str] | None = None) -> pl.DataFrame:
    """Per concept, per year: which tag resolved and for what share of filers.

    Contract section 4.3 asks for this table in NOTES.md, and calls it a genuine
    diagnostic rather than boilerplate. It is: a concept resolving for under 70%
    of the universe in some year means the Barra descriptor built on it is
    unreliable in that year and must be flagged in the risk model output.
    """
    wanted = list(concepts) if concepts is not None else list(TAG_FALLBACKS)
    rows: list[dict[str, object]] = []

    universe_by_year = (
        facts.with_columns(pl.col("filed").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("cik").n_unique().alias("n_universe"))
    )
    universe = dict(zip(universe_by_year["year"], universe_by_year["n_universe"], strict=True))

    for concept in wanted:
        resolved = resolve_tag(facts, concept)
        if resolved.is_empty():
            continue
        grouped = (
            resolved.with_columns(pl.col("filed").dt.year().alias("year"))
            .group_by(["year", "resolved_tag"])
            .agg(pl.col("cik").n_unique().alias("n_ciks"))
        )
        for row in grouped.iter_rows(named=True):
            total = universe.get(row["year"], 0)
            rows.append(
                {
                    "concept": concept,
                    "year": row["year"],
                    "resolved_tag": row["resolved_tag"],
                    "n_ciks": row["n_ciks"],
                    "share_of_filers": row["n_ciks"] / total if total else 0.0,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "concept": pl.String,
                "year": pl.Int32,
                "resolved_tag": pl.String,
                "n_ciks": pl.UInt32,
                "share_of_filers": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort(["concept", "year", "n_ciks"], descending=[False, False, True])


def concept_coverage_by_year(facts: pl.DataFrame) -> pl.DataFrame:
    """Per concept, per year: share of filers for whom the concept resolved at all.

    The <70% flag from contract section 4.3 reads off this table.

    Counts DISTINCT CIKs per (concept, year), not the sum of per-tag counts. One
    filer can resolve a concept under two different tags in the same year — which
    is precisely what happens across the ASC 606 revenue transition around 2018 —
    and summing the per-tag counts double-counts it. An earlier version of this
    function did exactly that and reported revenue coverage of 161.8% in 2018.
    """
    wanted = list(TAG_FALLBACKS)
    universe_by_year = (
        facts.with_columns(pl.col("filed").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("cik").n_unique().alias("n_universe"))
    )

    rows: list[pl.DataFrame] = []
    for concept in wanted:
        resolved = resolve_tag(facts, concept)
        if resolved.is_empty():
            continue
        rows.append(
            resolved.with_columns(pl.col("filed").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("cik").n_unique().alias("n_ciks"),
                pl.col("resolved_tag").n_unique().alias("n_tags_used"),
            )
            .with_columns(pl.lit(concept).alias("concept"))
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "concept": pl.String,
                "year": pl.Int32,
                "n_ciks": pl.UInt32,
                "n_tags_used": pl.UInt32,
                "share_of_filers": pl.Float64,
                "unreliable": pl.Boolean,
            }
        )

    return (
        pl.concat(rows)
        .join(universe_by_year, on="year", how="left")
        .with_columns((pl.col("n_ciks") / pl.col("n_universe")).alias("share_of_filers"))
        .with_columns((pl.col("share_of_filers") < 0.70).alias("unreliable"))
        .select(["concept", "year", "n_ciks", "n_tags_used", "share_of_filers", "unreliable"])
        .sort(["concept", "year"])
    )


def _fundamental_schema() -> pl.Schema:
    """The schema an empty quarter must still carry, so concat stays type-stable."""
    fields: list[tuple[str, pl.DataType]] = [
        ("cik", pl.Int64()),
        ("filed", pl.Datetime("ns")),
        ("period_end", pl.Datetime("ns")),
        ("ddate", pl.Datetime("ns")),
        ("tag", pl.String()),
        ("value", pl.Float64()),
        ("qtrs", pl.Float64()),
        ("quarter", pl.String()),
    ]
    return pl.Schema(fields)


def save_failed_quarters(failures: Mapping[str, str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(dict(failures), fh, indent=2, sort_keys=True)
    return path


def summarize_pit_lag(facts: pl.DataFrame) -> pl.DataFrame:
    """Distribution of `filed - period_end` in days.

    The headline diagnostic for the PIT rule: if this is near zero the dates are
    being parsed wrong, and if it is large the fundamentals are staler than the
    risk model assumes. Either way it needs to be looked at, not assumed.
    """
    return (
        facts.with_columns(
            (pl.col("filed") - pl.col("period_end")).dt.total_days().alias("lag_days")
        )
        .select(
            pl.col("lag_days").min().alias("min"),
            pl.col("lag_days").quantile(0.05).alias("p05"),
            pl.col("lag_days").median().alias("median"),
            pl.col("lag_days").quantile(0.95).alias("p95"),
            pl.col("lag_days").max().alias("max"),
            pl.col("lag_days").mean().alias("mean"),
            pl.len().alias("n_facts"),
        )
    )


def known_tickers(facts: pl.DataFrame) -> Sequence[str]:
    return sorted(facts["ticker"].unique().to_list())
