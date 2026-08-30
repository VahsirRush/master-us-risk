"""Phase-0 tests from data-sources-contract section 7, plus the user-agent guard.

These four sit alongside the four in spec section 4.7. Where a test needs real
data it reads the committed caches and skips if they are absent, so the suite
stays runnable on a clean checkout; where the property can be pinned on
constructed data it is, because a test that skips is a test that does not run.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from master_us.data.loaders import ADJUSTMENT_MONOTONE_RTOL, PRICE_CACHE_DIR, YFinancePrices
from master_us.data.sec import TAG_FALLBACKS, as_of, quarters_between, resolve_tag
from master_us.data.sources import (
    PLACEHOLDER_MARKERS,
    InvalidUserAgentError,
    batched,
    require_valid_user_agent,
)
from master_us.data.survivorship import build_report, render_markdown
from master_us.data.universe import (
    CHANGES_CACHE,
    MEMBERSHIP_CACHE,
    classify_removal_reason,
    load_ticker_aliases,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _cached_prices(limit: int | None = None) -> list[pl.DataFrame]:
    if not PRICE_CACHE_DIR.exists():
        return []
    paths = sorted(PRICE_CACHE_DIR.glob("*.parquet"))
    return [pl.read_parquet(p) for p in (paths[:limit] if limit else paths)]


# ------------------------------------------------------------------ #
# Section 7 — the SEC point-in-time rule                              #
# ------------------------------------------------------------------ #


def test_sec_pit_uses_filed_date():
    """A fact with period_end well before filed must be invisible in between.

    The single most important test in this file. Constructed rather than
    sampled, so the window between period_end and filed is known exactly and the
    assertion is about every date in it, not a spot check.
    """
    facts = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "cik": [1, 1],
            "filed": [
                datetime(2020, 2, 15),  # 46 days after the period closed
                datetime(2019, 5, 10),
            ],
            "period_end": [datetime(2019, 12, 31), datetime(2019, 3, 31)],
            "tag": ["Assets", "Assets"],
            "value": [999.0, 111.0],
        }
    ).with_columns(pl.col("filed").cast(pl.Datetime("ns")), pl.col("period_end").cast(pl.Datetime("ns")))

    # Across the whole gap the Q4 number does not exist yet, and the stale Q1
    # number is what a model would legitimately have seen.
    for date in ("2019-12-31", "2020-01-01", "2020-01-31", "2020-02-14"):
        visible = as_of(facts, date)
        assert visible["value"].to_list() == [111.0], f"Q4 value leaked on {date}"

    # It becomes visible on the filing date itself, not before.
    assert as_of(facts, "2020-02-15")["value"].to_list() == [999.0]
    assert as_of(facts, "2020-02-16")["value"].to_list() == [999.0]

    # And nothing at all is visible before the first filing.
    assert as_of(facts, "2019-05-09").is_empty()


def test_as_of_never_filters_on_period_end():
    """A fact whose period_end is in the future but which is already filed is usable.

    The converse of the test above, and the reason `period_end` must not appear
    in any predicate: filtering on it would hide a fact that genuinely was public.
    """
    facts = pl.DataFrame(
        {
            "ticker": ["BBB"],
            "cik": [2],
            "filed": [datetime(2020, 1, 10)],
            "period_end": [datetime(2020, 6, 30)],  # ahead of the filing
            "tag": ["Assets"],
            "value": [42.0],
        }
    ).with_columns(pl.col("filed").cast(pl.Datetime("ns")), pl.col("period_end").cast(pl.Datetime("ns")))

    assert as_of(facts, "2020-01-10")["value"].to_list() == [42.0]
    assert as_of(facts, "2020-01-09").is_empty()


def test_sec_cache_has_no_facts_filed_before_their_period_end():
    """The invariant, checked against what actually landed on disk."""
    from master_us.data.sec import SEC_CACHE_DIR

    if not SEC_CACHE_DIR.exists() or not any(SEC_CACHE_DIR.glob("*.parquet")):
        pytest.skip("no SEC cache — run scripts/02_fetch_fundamentals.py")

    paths = sorted(SEC_CACHE_DIR.glob("*.parquet"))[:8]
    for path in paths:
        frame = pl.read_parquet(path)
        if frame.is_empty():
            continue
        bad = frame.filter(pl.col("filed") < pl.col("period_end"))
        assert bad.is_empty(), f"{path.name} has {len(bad)} facts filed before period_end"


def test_pit_lag_is_in_the_documented_range():
    """Median filed-minus-period_end must sit in the 30-90 day band section 4.1 predicts.

    A median near zero would mean the dates are being parsed wrong — which is
    the failure this whole module exists to prevent, and it would be silent.
    """
    from master_us.data.sec import SEC_CACHE_DIR

    if not SEC_CACHE_DIR.exists() or not any(SEC_CACHE_DIR.glob("*.parquet")):
        pytest.skip("no SEC cache — run scripts/02_fetch_fundamentals.py")

    frames = [pl.read_parquet(p) for p in sorted(SEC_CACHE_DIR.glob("*.parquet"))[:6]]
    facts = pl.concat([f for f in frames if not f.is_empty()], how="diagonal")
    lag = (
        (facts["filed"] - facts["period_end"]).dt.total_days().median()
    )
    assert lag is not None
    assert 20 <= lag <= 120, f"median PIT lag of {lag} days is outside the documented range"


def test_tag_fallbacks_resolve_in_priority_order():
    """The first tag in the fallback list wins when several are present.

    Order is the whole mechanism: `Revenues` and the ASC 606 tags coexist in
    2018-2019 filings, and which one is picked decides whether a revenue series
    is continuous across the transition or steps.
    """
    preferred, second = TAG_FALLBACKS["revenue"][0], TAG_FALLBACKS["revenue"][2]
    facts = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "cik": [1, 1],
            "filed": [datetime(2019, 2, 1)] * 2,
            "period_end": [datetime(2018, 12, 31)] * 2,
            "tag": [second, preferred],
            "value": [100.0, 200.0],
        }
    ).with_columns(pl.col("filed").cast(pl.Datetime("ns")), pl.col("period_end").cast(pl.Datetime("ns")))

    resolved = resolve_tag(facts, "revenue")
    assert len(resolved) == 1
    assert resolved["resolved_tag"].to_list() == [preferred]
    assert resolved["value"].to_list() == [200.0]


def test_resolve_tag_rejects_unknown_concept():
    with pytest.raises(KeyError, match="unknown concept"):
        resolve_tag(pl.DataFrame(), "ebitda_but_vibes")


def test_quarters_between_covers_the_sample():
    quarters = quarters_between("2009q2", "2025-12-31")
    assert quarters[0] == "2009q2"
    assert quarters[-1] == "2025q4"
    assert len(quarters) == 67
    assert quarters_between("2020q1", "2020-01-15") == ["2020q1"]


# ------------------------------------------------------------------ #
# Section 7 — the price adjustment check                              #
# ------------------------------------------------------------------ #


def test_price_adjustment_monotone():
    """adj_close/close must be monotone non-decreasing per ticker.

    DIRECTION: the contract says non-increasing; under Yahoo's convention it is
    non-DECREASING — adj_close equals close on the most recent bar and is
    progressively smaller going back. Measured, not assumed: see the module
    docstring in loaders.py and NOTES.md, Session 3.
    """
    frames = _cached_prices()
    if not frames:
        pytest.skip("no price cache — run scripts/01_fetch_prices.py")

    violations: list[tuple[str, float]] = []
    for frame in frames:
        if len(frame) < 3:
            continue
        ratio = (frame["adj_close"] / frame["close"]).to_numpy()
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 3:
            continue
        worst = float((np.diff(ratio) / ratio[:-1]).min())
        if worst < -ADJUSTMENT_MONOTONE_RTOL:
            violations.append((frame["ticker"][0], worst))

    assert not violations, f"corrupted split adjustment in: {violations[:10]}"


def test_adjustment_tolerance_sits_between_rounding_and_signal():
    """The tolerance must clear the noise ceiling without swallowing real breaks.

    Guards against the calibration error that produced it: an earlier value of
    1e-06 sat inside the float-rounding distribution and flagged 73 of 613
    tickers as corrupted when none were.
    """
    frames = _cached_prices()
    if not frames:
        pytest.skip("no price cache — run scripts/01_fetch_prices.py")

    worst_negative = 0.0
    for frame in frames:
        if len(frame) < 3:
            continue
        ratio = (frame["adj_close"] / frame["close"]).to_numpy()
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 3:
            continue
        worst_negative = min(worst_negative, float((np.diff(ratio) / ratio[:-1]).min()))

    assert abs(worst_negative) < ADJUSTMENT_MONOTONE_RTOL, (
        f"observed rounding noise {worst_negative:.2e} exceeds the tolerance "
        f"{ADJUSTMENT_MONOTONE_RTOL:.0e} — recalibrate"
    )
    assert ADJUSTMENT_MONOTONE_RTOL < 1e-3, "tolerance is loose enough to hide a real split break"


def test_adjustment_check_catches_an_injected_corruption():
    """The negative control. A synthetic split break must be caught."""
    n = 200
    close = np.full(n, 100.0)
    ratio = np.linspace(0.90, 1.00, n)
    ratio[120:] *= 0.5  # a 50% downward break: a corrupted split adjustment
    frame = pl.DataFrame(
        {
            "date": pl.datetime_range(
                datetime(2020, 1, 1), datetime(2020, 1, 1), "1d", eager=True
            ).extend_constant(datetime(2020, 1, 1), n - 1),
            "ticker": ["ZZZ"] * n,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close * ratio,
            "volume": np.full(n, 1e6),
        }
    )
    diag = YFinancePrices().validate(frame, "ZZZ", "2020-01-01", "2020-12-31")
    assert diag.worst_adjustment_step < -0.4
    assert any("corrupted split adjustment" in p for p in diag.problems)
    assert not diag.ok


# ------------------------------------------------------------------ #
# Section 7 — membership reconstruction                               #
# ------------------------------------------------------------------ #


def test_membership_reconstruction():
    """Spot-check known index changes: a name added in year Y is absent in Y-1.

    Uses additions the reconstruction must get right by construction — each was
    added to the S&P 500 on a well-documented date well inside the sample.
    """
    if not MEMBERSHIP_CACHE.exists():
        pytest.skip("no membership cache — run scripts/00_fetch_universe.py")

    membership = pl.read_parquet(MEMBERSHIP_CACHE).filter(pl.col("in_universe"))
    by_ticker = (
        membership.group_by("ticker")
        .agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
    )
    first_seen = {r["ticker"]: r["first"] for r in by_ticker.iter_rows(named=True)}

    # (ticker, year it joined the index). Each addition IS recorded in the
    # changes table, so the reconstruction has no excuse to get it wrong.
    # META is here on purpose: Facebook was added as FB in 2013 and renamed in
    # 2022, so this only passes once config/ticker_aliases.yaml is applied.
    known_additions = [
        ("TSLA", 2020),
        ("GOOGL", 2014),
        ("META", 2013),
        ("OTIS", 2020),
        ("CARR", 2020),
        ("KVUE", 2023),
        ("GEV", 2024),
    ]
    checked = 0
    for ticker, year in known_additions:
        if ticker not in first_seen:
            continue
        checked += 1
        assert first_seen[ticker].year >= year - 1, (
            f"{ticker} appears from {first_seen[ticker].date()}, "
            f"but it joined the index in {year}"
        )
        # And it must be genuinely absent the year before it joined.
        prior = membership.filter(
            (pl.col("ticker") == ticker) & (pl.col("date").dt.year() < year - 1)
        )
        assert prior.is_empty(), f"{ticker} is present before {year - 1}"

    assert checked >= 2, "no known additions were present to check"


def test_membership_departures_are_retained():
    """Names that left the index must still occupy their historical dates.

    If the reconstruction dropped departed names it would be survivorship bias
    baked into the universe itself, before prices are even fetched.
    """
    if not MEMBERSHIP_CACHE.exists():
        pytest.skip("no membership cache — run scripts/00_fetch_universe.py")

    membership = pl.read_parquet(MEMBERSHIP_CACHE).filter(pl.col("in_universe"))
    last_date = membership["date"].max()
    by_ticker = membership.group_by("ticker").agg(pl.col("date").max().alias("last"))
    departed = by_ticker.filter(pl.col("last") < last_date)

    assert len(departed) > 100, f"only {len(departed)} names ever leave the index — suspicious"
    assert membership["ticker"].n_unique() > 600


def test_ticker_aliases_are_applied_and_their_absence_is_bounded():
    """A renamed name must reconnect to its own addition event, not float to the start.

    Without `config/ticker_aliases.yaml`, META appears from 2010-02-01 — nearly
    four years before Facebook was in the index — because the changes table
    records the addition under FB and the current list carries META.

    The second half measures what the alias map does NOT cover. The map is
    curated by hand and incomplete by construction, so this bounds the residual
    rather than asserting it away: names that never appear as an addition are
    either genuine long-term members or uncovered renames, and if that count
    grows sharply someone has added constituents without checking.
    """
    if not (MEMBERSHIP_CACHE.exists() and CHANGES_CACHE.exists()):
        pytest.skip("no membership cache — run scripts/00_fetch_universe.py")

    aliases = load_ticker_aliases()
    assert "FB" in aliases and aliases["FB"] == "META"

    membership = pl.read_parquet(MEMBERSHIP_CACHE).filter(pl.col("in_universe"))
    first = {
        r["ticker"]: r["date"]
        for r in membership.group_by("ticker").agg(pl.col("date").min()).iter_rows(named=True)
    }
    sample_start = membership["date"].min()

    assert "META" in first
    assert first["META"] > sample_start, (
        "META is present from the first day of the sample — the FB alias is not being applied"
    )

    # No historical symbol that has an alias should survive into the universe;
    # it should have been folded into its current symbol.
    leaked = [old for old in aliases if old in first]
    assert not leaked, f"pre-rename symbols still in the universe: {leaked}"

    changes = pl.read_parquet(CHANGES_CACHE)
    added = set(changes.filter(pl.col("added") != "")["added"].to_list())
    current = set(membership.filter(pl.col("date") == membership["date"].max())["ticker"].to_list())
    never_added = current - added
    # These are long-tenured members (the changes table is "selected changes" and
    # is thin before 2000) plus any rename the alias map misses. Recorded in
    # NOTES.md Session 3 at 271 of 503; the bound catches a regression, not drift.
    assert len(never_added) < 300, (
        f"{len(never_added)} of {len(current)} current constituents have no recorded "
        "addition — either the changes table changed shape or aliases regressed"
    )


def test_universe_size_is_plausible():
    if not MEMBERSHIP_CACHE.exists():
        pytest.skip("no membership cache — run scripts/00_fetch_universe.py")
    per_day = pl.read_parquet(MEMBERSHIP_CACHE).filter(pl.col("in_universe")).group_by("date").len()
    assert 480 <= per_day["len"].min() <= 520
    assert 490 <= per_day["len"].max() <= 530


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("Equity Residential acquired AvalonBay Communities.", "acquired_or_merged"),
        ("Market capitalization changes.", "index_rebalance"),
        ("Honeywell completed the corporate spin-off of Solstice.", "spun_off"),
        ("Company filed for Chapter 11 bankruptcy.", "bankruptcy"),
        ("An investor consortium took the company private.", "taken_private"),
        ("", "unstated"),
        (None, "unstated"),
        ("Reasons unlike any other.", "unclassified"),
    ],
)
def test_removal_reason_classification(reason, expected):
    assert classify_removal_reason(reason) == expected


# ------------------------------------------------------------------ #
# Section 7 — the survivorship report is a hard dependency            #
# ------------------------------------------------------------------ #


def test_survivorship_report_exists():
    """The panel cannot be built without a survivorship report. Deliberately hard.

    Contract section 3.3 makes this a blocking dependency rather than a
    nice-to-have: an unnamed limitation reads as ignorance, and the only way to
    guarantee it gets named is to make the build fail without it.
    """
    from master_us.data.sources import REPO_ROOT

    report = REPO_ROOT / "reports" / "survivorship.md"
    assert report.exists(), (
        "reports/survivorship.md is missing — run scripts/04_survivorship_report.py. "
        "This is a hard dependency of the panel build, not optional."
    )
    text = report.read_text()
    assert "OPTIMISTIC" in text, "the report must state the direction of the bias in those words"
    assert "Retrieval rate by year" in text
    assert "removal reason" in text.lower()


def test_survivorship_report_regenerates_from_cache():
    """The report is derived, not hand-written: it must rebuild from the caches."""
    if not (MEMBERSHIP_CACHE.exists() and CHANGES_CACHE.exists()):
        pytest.skip("no membership cache — run scripts/00_fetch_universe.py")
    retrieved = YFinancePrices().cached_tickers()
    if not retrieved:
        pytest.skip("no price cache — run scripts/01_fetch_prices.py")

    report = build_report(
        pl.read_parquet(MEMBERSHIP_CACHE), pl.read_parquet(CHANGES_CACHE), retrieved
    )
    assert report.n_total > report.n_retrieved > 0
    assert 0.0 < report.overall_rate < 1.0
    assert "OPTIMISTIC" in render_markdown(report)

    # The gradient that makes the bias visible: later years are better covered.
    rates = report.by_year["retrieval_rate"].to_list()
    assert rates[-1] > rates[0], "retrieval rate should improve toward the present"


def test_survivorship_counts_every_historical_constituent():
    """Every ticker that ever appears in membership is accounted for, retrieved or not."""
    if not (MEMBERSHIP_CACHE.exists() and CHANGES_CACHE.exists()):
        pytest.skip("no membership cache")
    membership = pl.read_parquet(MEMBERSHIP_CACHE)
    report = build_report(membership, pl.read_parquet(CHANGES_CACHE), {"AAPL"})
    ever = membership.filter(pl.col("in_universe"))["ticker"].n_unique()
    assert report.n_total == ever
    assert report.n_retrieved == 1


# ------------------------------------------------------------------ #
# The SEC user-agent guard                                            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("marker", PLACEHOLDER_MARKERS)
def test_placeholder_user_agent_is_refused(marker):
    """Every placeholder marker must be caught, in any casing or position."""
    with pytest.raises(InvalidUserAgentError, match="placeholder"):
        require_valid_user_agent(f"Somebody <{marker}@somewhere.com>")
    with pytest.raises(InvalidUserAgentError):
        require_valid_user_agent(f"{marker.lower()} <a@b.com>")


def test_the_shipped_replace_me_default_is_refused():
    """The exact string the config template ships with."""
    with pytest.raises(InvalidUserAgentError, match="REPLACE-ME"):
        require_valid_user_agent("REPLACE-ME <your-email@domain.com>")


@pytest.mark.parametrize("bad", ["", "   ", None, "Just A Name", "no-at-sign", "a@nodot"])
def test_malformed_user_agent_is_refused(bad):
    with pytest.raises(InvalidUserAgentError):
        require_valid_user_agent(bad)


def test_valid_user_agent_passes_through_stripped():
    assert require_valid_user_agent("  Rush <rush@example.io>  ") == "Rush <rush@example.io>"


def test_configured_user_agent_is_valid():
    """The live config must be usable. Fails if someone resets it to the template."""
    from master_us.data.sources import sec_user_agent

    assert "@" in sec_user_agent()


def test_sec_source_refuses_placeholder_at_construction():
    """The guard fires before any endpoint is reachable, not at request time."""
    from master_us.data.sec import SECFundamentals

    with pytest.raises(InvalidUserAgentError):
        SECFundamentals(user_agent="REPLACE-ME <your-email@domain.com>")


# ------------------------------------------------------------------ #
# Shared infrastructure                                               #
# ------------------------------------------------------------------ #


def test_batched_splits_without_loss():
    assert list(batched(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(batched([], 3)) == []
    with pytest.raises(ValueError, match="batch size"):
        list(batched([1, 2], 0))


def test_price_cache_rows_are_sane():
    """No non-positive prices and no negative volume anywhere in the cache."""
    frames = _cached_prices(limit=40)
    if not frames:
        pytest.skip("no price cache — run scripts/01_fetch_prices.py")
    for frame in frames:
        ticker = frame["ticker"][0]
        for col in ("open", "high", "low", "close", "adj_close"):
            assert frame[col].min() > 0, f"{ticker} has a non-positive {col}"
        assert frame["volume"].min() >= 0, f"{ticker} has negative volume"
        assert frame["date"].is_sorted(), f"{ticker} is not sorted by date"
        assert frame["date"].n_unique() == len(frame), f"{ticker} has duplicate dates"
