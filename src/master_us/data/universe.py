"""S&P 500 point-in-time membership — data-sources-contract section 2.

Reconstructed by taking today's constituent list and walking the index-changes
table backward: a name added on date d was absent before d; a name removed on
date d was present before d. Repeat to the start of the sample.

The reconstruction is only as good as the changes table, and the changes table
is a Wikipedia page. Section 2.2 says to document the defects rather than hide
them, so:

* Coverage is poor before ~2000 and good from 2010, which is why the sample
  starts in 2010.
* Wikipedia is mutable. Every scrape is cached to parquet with its timestamp
  and source URL, and that parquet is committed, so the build reproduces even
  after the page changes underneath it.
* Ticker symbols are reused and renamed. A name removed under one symbol and
  re-added under another looks like two companies here.

DEVIATION from section 2.1: the contract says both tables live on
`List_of_S&P_500_companies` as tables 0 and 1. Wikipedia has since split the
changes table onto its own article, `Historical_components_of_the_S&P_500`, and
what is now table 1 on the original page is a navigation box. Both URLs are
below; see NOTES.md, Session 3.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import requests
import yaml

from master_us.data.sources import CONFIG_ROOT, RAW_ROOT, FetchError, with_retry

CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"

ALIASES_PATH = CONFIG_ROOT / "ticker_aliases.yaml"
MEMBERSHIP_CACHE = RAW_ROOT / "sp500_membership.parquet"
CHANGES_CACHE = RAW_ROOT / "sp500_changes.parquet"

# Free-text reasons, bucketed for the survivorship report. Ordered: first match
# wins, so the specific patterns must precede the generic ones.
REMOVAL_REASON_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bankruptcy", r"bankrupt|chapter 11|chapter 7|liquidat"),
    ("taken_private", r"taken private|going private|private equity|investor consortium"),
    ("acquired_or_merged", r"acquir|merg|purchas|bought|takeover|combin"),
    ("spun_off", r"spin[- ]?off|spun off|split into|separat"),
    ("index_rebalance", r"market cap|market capitalization|index rebalanc|represent"),
    ("delisted_for_cause", r"delist|no longer meet|failed to|non-compliance|deficien"),
)


@dataclass(frozen=True)
class MembershipScrape:
    """A reconstruction run, with the provenance needed to reproduce it."""

    membership: pl.DataFrame  # [date, ticker, in_universe]
    changes: pl.DataFrame  # [date, added, removed, reason, reason_class]
    current_constituents: tuple[str, ...]
    scraped_at: datetime
    n_changes_applied: int

    @property
    def tickers(self) -> tuple[str, ...]:
        col = self.membership.filter(pl.col("in_universe"))["ticker"].unique().sort()
        return tuple(col.to_list())


def load_ticker_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    """Historical -> current symbol map. See config/ticker_aliases.yaml.

    Returns an empty map if the file is absent, because the reconstruction is
    still meaningful without it — just less correct for renamed names, which is
    a documented limitation rather than a crash.
    """
    if not path.exists():
        return {}
    with path.open() as fh:
        loaded = yaml.safe_load(fh) or {}
    aliases = loaded.get("aliases") or {}
    return {str(k).upper(): str(v).upper() for k, v in aliases.items()}


def classify_removal_reason(reason: str | None) -> str:
    """Bucket a free-text Wikipedia reason into a survivorship category.

    The distinction that matters for section 3.3: acquisitions and index
    rebalances usually leave a retrievable price history behind, bankruptcies
    and delistings usually do not. Getting a name into the wrong bucket
    understates or overstates the bias, so `unclassified` is a real outcome and
    is reported rather than folded into a neighbour.
    """
    if reason is None or not str(reason).strip() or str(reason).lower() == "nan":
        return "unstated"
    text = str(reason).lower()
    for label, pattern in REMOVAL_REASON_PATTERNS:
        if re.search(pattern, text):
            return label
    return "unclassified"


class SP500Universe:
    """PIT S&P 500 membership reconstructed from Wikipedia. Satisfies `UniverseSource`."""

    def __init__(
        self,
        user_agent: str,
        cache_path: Path = MEMBERSHIP_CACHE,
        changes_cache_path: Path = CHANGES_CACHE,
        timeout: int = 60,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.cache_path = cache_path
        self.changes_cache_path = changes_cache_path
        self.timeout = timeout
        self.aliases = load_ticker_aliases() if aliases is None else dict(aliases)

    # ------------------------------------------------------------------ #
    # Scraping                                                            #
    # ------------------------------------------------------------------ #

    def _get(self, url: str) -> str:
        def fetch() -> str:
            resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text

        return with_retry(fetch, f"GET {url}")

    def fetch_current_constituents(self) -> list[str]:
        """Today's members, from the `constituents` table on the main article."""
        tables = pd.read_html(io.StringIO(self._get(CONSTITUENTS_URL)))
        for table in tables:
            if "Symbol" in table.columns and len(table) > 400:
                return [_clean_ticker(s) for s in table["Symbol"].astype(str)]
        raise FetchError(
            "S&P 500 constituents table",
            [
                f"no table with a 'Symbol' column and >400 rows among {len(tables)} tables "
                f"at {CONSTITUENTS_URL} — the page layout changed"
            ],
        )

    def fetch_changes(self) -> pl.DataFrame:
        """The index-changes table: [date, added, removed, reason, reason_class]."""
        tables = pd.read_html(io.StringIO(self._get(CHANGES_URL)))
        table = next(
            (
                t
                for t in tables
                if isinstance(t.columns, pd.MultiIndex)
                and {"Added", "Removed"} <= {c[0] for c in t.columns}
            ),
            None,
        )
        if table is None:
            raise FetchError(
                "S&P 500 changes table",
                [
                    f"no table with 'Added'/'Removed' column groups among {len(tables)} tables "
                    f"at {CHANGES_URL} — the page layout changed"
                ],
            )

        table.columns = [a if a == b else f"{a}_{b}" for a, b in table.columns]
        renamed = table.rename(
            columns={
                "Effective Date": "date",
                "Date": "date",
                "Added_Ticker": "added",
                "Removed_Ticker": "removed",
                "Reason": "reason",
            }
        )
        missing = {"date", "added", "removed"} - set(renamed.columns)
        if missing:
            raise FetchError(
                "S&P 500 changes table",
                [f"table is missing columns {sorted(missing)}; got {list(renamed.columns)}"],
            )

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(renamed["date"], errors="coerce", format="mixed"),
                "added": renamed["added"].map(_clean_ticker),
                "removed": renamed["removed"].map(_clean_ticker),
                "reason": renamed.get("reason", pd.Series([None] * len(renamed))).astype(str),
            }
        )

        unparsed = int(frame["date"].isna().sum())
        if unparsed:
            raise FetchError(
                "S&P 500 changes table",
                [f"{unparsed} of {len(frame)} change dates did not parse"],
            )

        frame["reason_class"] = frame["reason"].map(classify_removal_reason)
        return pl.from_pandas(frame.sort_values("date", ascending=False).reset_index(drop=True))

    # ------------------------------------------------------------------ #
    # Reconstruction                                                      #
    # ------------------------------------------------------------------ #

    def reconstruct(self, start: str, end: str) -> MembershipScrape:
        """Walk the changes backward from today's list to build PIT membership.

        Snapshots on the last trading day of each month, then forward-fills to
        daily with membership effective the NEXT trading day — a change
        announced effective date d is only tradeable from d+1, and treating it
        as tradeable on d is a one-day look-ahead on every index addition.
        """
        scraped_at = datetime.now(UTC)
        current = [self.aliases.get(t, t) for t in self.fetch_current_constituents()]
        changes = self.fetch_changes().with_columns(
            # Both sides of the walk must live in the SAME symbol space. The
            # changes table records an addition under the symbol in use at the
            # time; the current list carries today's. Without this mapping a
            # renamed name never reconnects to its own addition event and looks
            # like a member from the start of the sample.
            pl.col("added").replace(self.aliases),
            pl.col("removed").replace(self.aliases),
        )

        lo, hi = pd.Timestamp(start), pd.Timestamp(end)

        # Month-end snapshot dates, from `end` back to `start`.
        month_ends = pd.date_range(lo, hi, freq="BME")
        if len(month_ends) == 0:
            raise ValueError(f"no month ends in range [{lo.date()}, {hi.date()}]")

        change_rows = changes.sort("date", descending=True).to_dicts()
        members = set(current)
        snapshots: dict[pd.Timestamp, frozenset[str]] = {}
        applied = 0

        cursor = 0
        for snapshot_date in reversed(month_ends):
            # Reverse every change strictly AFTER this snapshot date.
            while cursor < len(change_rows) and change_rows[cursor]["date"] > snapshot_date:
                row = change_rows[cursor]
                if row["added"]:
                    members.discard(row["added"])  # not yet a member before its add date
                if row["removed"]:
                    members.add(row["removed"])  # still a member before its removal date
                applied += 1
                cursor += 1
            snapshots[snapshot_date] = frozenset(members)

        return MembershipScrape(
            membership=_snapshots_to_daily(snapshots, lo, hi),
            changes=changes,
            current_constituents=tuple(sorted(current)),
            scraped_at=scraped_at,
            n_changes_applied=applied,
        )

    # ------------------------------------------------------------------ #
    # The Protocol method, cache-first                                    #
    # ------------------------------------------------------------------ #

    def get_membership(self, start: str, end: str) -> pl.DataFrame:
        """Cached daily membership. Scrapes only on a cache miss."""
        if self.cache_path.exists():
            cached = pl.read_parquet(self.cache_path)
            lo, hi = pd.Timestamp(start), pd.Timestamp(end)
            covered = cached["date"].min(), cached["date"].max()
            if covered[0] is not None and covered[0] <= lo and covered[1] >= hi:
                return cached.filter(
                    (pl.col("date") >= lo.to_pydatetime()) & (pl.col("date") <= hi.to_pydatetime())
                )

        scrape = self.reconstruct(start, end)
        self.save(scrape)
        return scrape.membership

    def save(self, scrape: MembershipScrape) -> Path:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        stamped = scrape.membership.with_columns(
            pl.lit(scrape.scraped_at.isoformat()).alias("scraped_at"),
            pl.lit(CHANGES_URL).alias("source_url"),
        )
        stamped.write_parquet(self.cache_path)
        scrape.changes.with_columns(
            pl.lit(scrape.scraped_at.isoformat()).alias("scraped_at")
        ).write_parquet(self.changes_cache_path)
        return self.cache_path


def _clean_ticker(raw: object) -> str:
    """Normalize a Wikipedia ticker cell to a Yahoo-compatible symbol.

    Wikipedia writes class shares with a dot (`BRK.B`); Yahoo wants a dash
    (`BRK-B`). Footnote markers and non-breaking spaces both appear.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text or text.lower() in ("nan", "none", "—", "-"):
        return ""
    text = text.replace("\xa0", " ").split("[")[0].strip()
    text = re.sub(r"\s+", " ", text).split(" ")[0]
    return text.replace(".", "-").upper()


def _snapshots_to_daily(
    snapshots: dict[pd.Timestamp, frozenset[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pl.DataFrame:
    """Expand month-end snapshots to a daily [date, ticker, in_universe] frame.

    A snapshot taken on month-end d governs the days AFTER d, up to and
    including the next month-end. That offset is the "effective the next trading
    day" rule, and it is the difference between a clean backtest and a one-day
    look-ahead on every index change.
    """
    days = pd.bdate_range(start, end)
    ordered = sorted(snapshots)

    rows: list[dict[str, object]] = []
    for i, snapshot_date in enumerate(ordered):
        first = snapshot_date + pd.Timedelta(days=1)
        last = ordered[i + 1] if i + 1 < len(ordered) else end
        window = days[(days >= first) & (days <= last)]
        for day in window:
            for ticker in snapshots[snapshot_date]:
                rows.append({"date": day, "ticker": ticker, "in_universe": True})

    if not rows:
        raise ValueError("membership reconstruction produced no rows")

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .sort(["date", "ticker"])
    )
