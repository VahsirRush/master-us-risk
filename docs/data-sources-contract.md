# Data Sources Contract — Free Path (yfinance + SEC XBRL)

**Spec amendment.** This replaces §4.1 and §4.2 of the implementation spec. Everything downstream of the `Panel` object is unchanged.

---

## 0. What Changes and Why

| Spec section | Original | Amended |
|---|---|---|
| §4.1 Universe | Russell 1000, PIT membership | **S&P 500**, PIT membership from Wikipedia constituent history |
| §4.2 Prices | CRSP | yfinance, hardened with caching + retry |
| §4.2 Fundamentals | Compustat / Sharadar `datekey` | **SEC Financial Statement Data Sets** (`filed` field = true PIT) |
| §4.1 Delisting | Terminal return applied | **Measured and reported as a named limitation** |

S&P 500 over Russell 1000 because free constituent history exists for it and doesn't for the Russell. Side benefit: ~500 names/day makes inter-stock attention cheaper to train and is closer to the paper's CSI300 setup.

---

## 1. Module Contract

```python
# src/master_us/data/sources.py
"""
Free-path data acquisition. Every function is cache-first: check parquet,
fetch only on miss, write immediately. Raw pulls are immutable — never
mutate anything under data/raw/.
"""

from typing import Protocol
import polars as pl
import pandas as pd


class PriceSource(Protocol):
    def get_ohlcv(self, tickers: list[str], start: str, end: str) -> pl.DataFrame:
        """Returns long format: [date, ticker, open, high, low, close, adj_close,
        volume]. Split- and dividend-adjusted. Missing days absent, not NaN-filled."""

class FundamentalSource(Protocol):
    def get_fundamentals(self, start: str, end: str) -> pl.DataFrame:
        """Returns long format: [ticker, cik, filed, period_end, tag, value].
        `filed` is the SEC acceptance date and is the ONLY date usable for PIT joins."""

class UniverseSource(Protocol):
    def get_membership(self, start: str, end: str) -> pl.DataFrame:
        """Returns [date, ticker, in_universe: bool], daily, forward-filled
        from monthly reconstitution."""
```

Implementations must satisfy these Protocols so a later WRDS upgrade is a drop-in swap.

---

## 2. Universe — S&P 500 PIT Membership

### 2.1 Source

Wikipedia's *List of S&P 500 companies* page has two tables: current constituents, and a **selected changes** table with `Date · Added ticker · Removed ticker · Reason`. Reconstructing backwards from the current list through the changes gives point-in-time membership.

```python
class SP500Universe:
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def get_membership(self, start: str, end: str) -> pl.DataFrame:
        """
        1. Scrape both tables (pandas.read_html, tables 0 and 1)
        2. Start from the current constituent set at scrape date
        3. Walk the changes table BACKWARD in time: for each change,
           reverse it (a name 'added' on date d was absent before d;
           a name 'removed' on date d was present before d)
        4. Snapshot membership on the last trading day of each month
        5. Forward-fill daily; membership effective the NEXT trading day
        6. Cache to data/raw/sp500_membership.parquet with scrape timestamp
        """
```

### 2.2 Known defects — document, don't hide

- The changes table is incomplete before ~2000. **Start the sample at 2010**; coverage is good from there.
- Ticker symbols change (mergers, rebrands). Maintain `config/ticker_aliases.yaml` mapping historical → current, seeded from the changes table's reason column.
- Wikipedia is mutable. Cache the scrape with a timestamp and commit the parquet so the build is reproducible even if the page changes.

### 2.3 Filters at reconstitution

Unchanged from spec: price ≥ $5, 21d ADV ≥ $2M, ≥252 days history.

---

## 3. Prices — yfinance, Hardened

### 3.1 The problem

yfinance is an unofficial scraper of Yahoo endpoints. It throttles, it returns empty frames instead of raising, and it **only serves currently-listed tickers**. Every one of these needs explicit handling.

```python
class YFinancePrices:
    def get_ohlcv(self, tickers, start, end) -> pl.DataFrame:
        """
        Batching:  30 tickers per request, 1.5s sleep between batches
        Retry:     3 attempts, exponential backoff (2s, 8s, 32s)
        Validation: after each batch, assert
                    - returned tickers ⊆ requested tickers
                    - each ticker has >= 0.8 * expected trading days
                    - no negative prices, no zero volume on non-holidays
                    - adj_close / close ratio is monotone non-increasing over time
                      (catches corrupted split adjustments)
        Failure:   record in self.failed_tickers with reason. NEVER silently drop.
        Cache:     one parquet per ticker under data/raw/prices/{ticker}.parquet,
                   appended incrementally so re-runs fetch only new dates
        """
```

### 3.2 Adjustment handling

Use `auto_adjust=False` and keep **both** `close` and `adj_close`:
- `adj_close` for return computation
- raw `close` for the ≥$5 price filter and for spread estimation
- Their ratio is your split/dividend factor — the monotonicity assertion above catches Yahoo's occasional retroactive re-adjustments, which silently change historical returns between runs

**Pin your data.** Once a full pull succeeds, commit a hash of the price panel. If a later re-pull changes historical values, you need to know.

### 3.3 Survivorship measurement — required, not optional

```python
def survivorship_report(membership: pl.DataFrame,
                        retrieved: set[str]) -> pd.DataFrame:
    """
    For every ticker that appears in historical membership, record whether
    price data was retrievable. Group by year and by removal reason.

    Emit to reports/survivorship.md:
      - % of historical constituents retrievable, overall and by year
      - Breakdown by removal reason (acquired / merged / index rebalance /
        delisted for cause)
      - Estimated direction and rough magnitude of the resulting bias

    Acquisitions and mergers usually remain retrievable. Failures usually
    do not. The bias is therefore OPTIMISTIC, and the report must say so
    in those words.
    """
```

This report goes in the README. A named, measured limitation reads as rigor; an unnamed one reads as ignorance.

---

## 4. Fundamentals — SEC Financial Statement Data Sets

### 4.1 Why this source, not EDGAR scraping

The SEC publishes quarterly ZIP archives of parsed XBRL from every filing. Each contains `sub.txt` (submissions) and `num.txt` (numeric facts). The `sub.txt` file carries a **`filed`** field — the acceptance date. That field is what makes this genuinely point-in-time. Scraping individual filings gets you the same data with far more work and more failure modes.

Archives live under the SEC's *Financial Statement Data Sets* page, one ZIP per quarter.

```python
class SECFundamentals:
    """
    Acquisition:
      - One ZIP per quarter from 2009Q2 forward
      - Extract sub.txt (adsh, cik, name, form, period, filed, fy, fp)
                 num.txt (adsh, tag, version, ddate, qtrs, value)
      - Keep form in {10-K, 10-Q}; discard amendments unless no original exists
      - Cache extracted parquet per quarter; the ZIPs are large, extract once

    PIT join rule:
      A fact is usable on date t iff filed <= t.
      Use `filed`. NEVER use `period` or `ddate` — those are fiscal period ends
      and precede availability by 30-90 days. This is the single most common
      source of look-ahead in retail quant projects.

    Ticker mapping:
      SEC keys on CIK, not ticker. Use the SEC's company_tickers.json for
      CIK -> ticker. Handle the many-to-one cases (multiple share classes).
    """
```

### 4.2 Tags needed for the Barra descriptors

| Barra factor | XBRL tags (US-GAAP) |
|---|---|
| Value — B/P | `StockholdersEquity` |
| Value — E/P | `NetIncomeLoss` |
| Value — CF/P | `NetCashProvidedByUsedInOperatingActivities` |
| Leverage | `Liabilities`, `Assets`, `LongTermDebtNoncurrent` |
| Growth | `Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax`, `NetIncomeLoss` (3yr) |
| Quality — ROE | `NetIncomeLoss` / `StockholdersEquity` |
| Quality — accruals | `NetIncomeLoss` − operating cash flow, scaled by `Assets` |
| Market cap | `dei:EntityCommonStockSharesOutstanding` × price |

### 4.3 The tag-drift problem

XBRL tags are not stable across filers or years. `Revenues` was widely replaced by `RevenueFromContractWithCustomerExcludingAssessedTax` after ASC 606 adoption (~2018).

```python
TAG_FALLBACKS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    # ... one ordered list per concept
}

def resolve_tag(facts: pl.DataFrame, concept: str) -> pl.DataFrame:
    """Try fallbacks in order per (cik, period). Log which tag resolved for what
    fraction of observations — this coverage table goes in NOTES.md and is a
    genuine diagnostic, not boilerplate."""
```

Report per-concept coverage by year. If a concept resolves for <70% of the universe in some year, that Barra descriptor is unreliable there and should be flagged in the risk model output.

### 4.4 Rate limits

SEC requires a declared `User-Agent` containing a real name and email, and caps requests at roughly 10/second. Exceeding it gets your IP blocked.

```python
HEADERS = {"User-Agent": "Rush <your-email@domain.com>"}
RATE_LIMIT_PER_SEC = 8   # deliberately under the cap
```

The bulk ZIPs are only ~60 files total, so this is a one-time cost of a few minutes, not a scraping campaign.

---

## 5. Spread Estimation — for the Cost Model

Free data has no bid-ask spreads. §5.2 of the spec calls for the Corwin-Schultz estimator, which derives spread from daily high-low ranges alone.

```python
def corwin_schultz_spread(high, low, window=21) -> np.ndarray:
    """Two-day high-low range estimator. Produces negative values on some
    days — standard practice is to floor at zero, then take a rolling median
    over `window` for stability.

    Cross-check: mean estimated spread for mega-caps should land in the
    single-digit bps range; small-caps meaningfully higher. If mega-caps
    come out at 50bps, the implementation is wrong."""
```

Use this for the "realistic" cost tier. Keep the flat 10bps tier as the headline comparison, since it's the convention readers expect.

---

## 6. Assembly Order

```
scripts/00_fetch_universe.py     -> data/raw/sp500_membership.parquet
scripts/01_fetch_prices.py       -> data/raw/prices/*.parquet  (+ failed_tickers.json)
scripts/02_fetch_fundamentals.py -> data/raw/sec/{year}q{q}.parquet
scripts/03_build_panel.py        -> data/processed/panel.pkl
scripts/04_survivorship_report.py-> reports/survivorship.md
```

Each script is idempotent and cache-first. Re-running `01` after a month fetches only the new dates.

---

## 7. Additional Phase-0 Tests

Add these to the four in spec §4.7:

```python
def test_sec_pit_uses_filed_date():
    """Construct a fact with period_end well before filed. Assert it is NOT
    available in the panel on any date between period_end and filed."""

def test_membership_reconstruction():
    """Spot-check known index changes against the reconstructed membership —
    a name added in a known year must be absent the year prior."""

def test_price_adjustment_monotone():
    """adj_close/close ratio must be monotone non-increasing per ticker.
    A violation means a corrupted split adjustment."""

def test_survivorship_report_exists():
    """The panel cannot be built without a survivorship report. This is a
    hard dependency, deliberately."""
```

---

## 8. What This Path Costs You, Stated Plainly

For the README's limitations section:

1. **Survivorship bias**, direction optimistic, magnitude measured in `reports/survivorship.md`. Names delisted for cause are disproportionately unretrievable.
2. **S&P 500 only** — large-cap universe, so results don't speak to small-cap behavior, where much published anomaly return concentrates.
3. **Restatements invisible.** The SEC datasets carry the as-filed values, which is correct for PIT, but amended filings are handled by preferring originals. A model can't see a restatement it wouldn't have known about — this is the right behavior, but worth naming.
4. **Estimated spreads, not observed.** Corwin-Schultz is a proxy; true execution costs may differ, particularly in stressed periods.
5. **Yahoo data quality** is unwarranted-by-contract. Data is pinned by hash; any change between runs is detected but the underlying values are not independently verifiable.

Each of these is a sentence a hiring manager will respect. The alternative — a project that doesn't mention them — invites the assumption that you didn't know.
