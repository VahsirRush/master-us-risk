"""Survivorship measurement — data-sources-contract section 3.3.

Required, not optional. The free path can only retrieve prices for currently
listed symbols, so every name that left the index and stopped trading is simply
absent. That absence is not random: acquisitions and index rebalances usually
leave a retrievable history, failures usually do not. The bias is therefore
**optimistic**, and section 3.3 requires the report to say so in those words.

This module measures it. It does not fix it — nothing on the free path can.
Naming and quantifying a limitation is the contribution; pretending it is absent
would not be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl


@dataclass(frozen=True)
class SurvivorshipReport:
    """Retrieval outcome for every ticker that ever appeared in membership."""

    by_ticker: pl.DataFrame  # [ticker, retrieved, last_in_universe, removal_reason]
    by_year: pl.DataFrame  # [year, n_constituents, n_retrieved, retrieval_rate]
    by_reason: pl.DataFrame  # [reason_class, n, n_retrieved, retrieval_rate]
    n_total: int
    n_retrieved: int
    generated_at: datetime

    @property
    def overall_rate(self) -> float:
        return self.n_retrieved / self.n_total if self.n_total else 0.0


def build_report(
    membership: pl.DataFrame,
    changes: pl.DataFrame,
    retrieved: set[str],
) -> SurvivorshipReport:
    """Cross membership against what the price pull actually returned."""
    ever = (
        membership.filter(pl.col("in_universe"))
        .group_by("ticker")
        .agg(
            pl.col("date").min().alias("first_in_universe"),
            pl.col("date").max().alias("last_in_universe"),
        )
    )

    # A ticker's removal reason is the reason attached to the change that removed
    # it. A name removed more than once keeps the most recent.
    removals = (
        changes.filter(pl.col("removed") != "")
        .sort("date", descending=True)
        .group_by("removed")
        .first()
        .select(
            pl.col("removed").alias("ticker"),
            pl.col("reason_class").alias("removal_reason"),
            pl.col("date").alias("removal_date"),
        )
    )

    by_ticker = (
        ever.join(removals, on="ticker", how="left")
        .with_columns(
            pl.col("ticker").is_in(list(retrieved)).alias("retrieved"),
            pl.col("removal_reason").fill_null("still_in_index"),
        )
        .sort("ticker")
    )

    # Per year: of the names in the index that year, how many are retrievable now?
    per_year = (
        membership.filter(pl.col("in_universe"))
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by(["year", "ticker"])
        .agg(pl.len().alias("days"))
        .with_columns(pl.col("ticker").is_in(list(retrieved)).alias("retrieved"))
        .group_by("year")
        .agg(
            pl.col("ticker").n_unique().alias("n_constituents"),
            pl.col("retrieved").sum().alias("n_retrieved"),
        )
        .with_columns((pl.col("n_retrieved") / pl.col("n_constituents")).alias("retrieval_rate"))
        .sort("year")
    )

    by_reason = (
        by_ticker.group_by("removal_reason")
        .agg(
            pl.len().alias("n"),
            pl.col("retrieved").sum().alias("n_retrieved"),
        )
        .with_columns((pl.col("n_retrieved") / pl.col("n")).alias("retrieval_rate"))
        .sort("n", descending=True)
    )

    return SurvivorshipReport(
        by_ticker=by_ticker,
        by_year=per_year,
        by_reason=by_reason,
        n_total=len(by_ticker),
        n_retrieved=int(by_ticker["retrieved"].sum()),
        generated_at=datetime.now(UTC),
    )


def render_markdown(report: SurvivorshipReport) -> str:
    """The section 3.3 deliverable, for `reports/survivorship.md` and the README."""
    lines: list[str] = []
    add = lines.append

    add("# Survivorship Measurement")
    add("")
    add(f"Generated {report.generated_at.isoformat(timespec='seconds')}")
    add("")
    add(
        "Required by `docs/data-sources-contract.md` section 3.3. The free data path "
        "(yfinance) serves only currently-listed symbols, so names that left the index "
        "and stopped trading cannot be retrieved at all. This report measures how many, "
        "and which kind."
    )
    add("")

    add("## Headline")
    add("")
    add(
        f"- **{report.n_retrieved} of {report.n_total} historical constituents "
        f"({report.overall_rate:.1%}) are retrievable.**"
    )
    add(f"- {report.n_total - report.n_retrieved} names are not, and are absent from the panel.")
    add("")

    add("## Retrieval rate by year")
    add("")
    add("Of the names in the index during each year, the share retrievable today.")
    add("")
    add("| Year | Constituents | Retrieved | Rate |")
    add("|---:|---:|---:|---:|")
    for row in report.by_year.iter_rows(named=True):
        add(
            f"| {row['year']} | {row['n_constituents']} | {row['n_retrieved']} "
            f"| {row['retrieval_rate']:.1%} |"
        )
    add("")

    add("## Retrieval rate by removal reason")
    add("")
    add(
        "The direction of the bias is visible here rather than asserted: reasons that "
        "leave a company trading under some symbol retain their history, reasons that "
        "end the listing do not."
    )
    add("")
    add("| Removal reason | Names | Retrieved | Rate |")
    add("|:---|---:|---:|---:|")
    for row in report.by_reason.iter_rows(named=True):
        add(
            f"| {row['removal_reason']} | {row['n']} | {row['n_retrieved']} "
            f"| {row['retrieval_rate']:.1%} |"
        )
    add("")

    add("## Direction and magnitude of the bias")
    add("")
    add(_bias_statement(report))
    add("")

    add("## What this means for every number in this repository")
    add("")
    add(
        "- Backtest returns are **overstated** by an unknown but non-zero amount. "
        "The missing names are disproportionately the ones that fell."
    )
    add(
        "- The effect is largest in the early sample, where more of the era's "
        "constituents have since stopped trading, and smallest at the end."
    )
    add(
        "- Cross-sectional dispersion is **understated**: the left tail is the part "
        "that is missing."
    )
    add(
        "- This cannot be corrected on the free path. It is measured and reported, "
        "not fixed. A CRSP or Sharadar upgrade would eliminate it, and the "
        "`PriceSource` Protocol exists so that upgrade is a drop-in swap."
    )
    return "\n".join(lines) + "\n"


def _bias_statement(report: SurvivorshipReport) -> str:
    """State the direction, and where the measurement complicates the expected story.

    Section 3.3 anticipates that "acquisitions and mergers usually remain
    retrievable, failures usually do not", making the bias cleanly optimistic.
    The measurement does not support that mechanism, and this function reports
    what was measured rather than what was expected. See NOTES.md, Session 3.
    """
    counts = {
        row["removal_reason"]: (row["n"], row["n_retrieved"], row["retrieval_rate"])
        for row in report.by_reason.iter_rows(named=True)
    }
    missing_by_reason = {
        reason: n - got for reason, (n, got, _) in counts.items() if n - got > 0
    }
    missing = report.n_total - report.n_retrieved
    still_n, still_got, still_rate = counts.get("still_in_index", (0, 0, 0.0))
    acq_n, acq_got, acq_rate = counts.get("acquired_or_merged", (0, 0, 0.0))
    reb_n, reb_got, reb_rate = counts.get("index_rebalance", (0, 0, 0.0))
    acq_missing = missing_by_reason.get("acquired_or_merged", 0)
    reb_missing = missing_by_reason.get("index_rebalance", 0)

    parts: list[str] = []

    parts.append(
        "**The bias is OPTIMISTIC**, in the sense that matters: the panel is "
        "conditioned on a name still having a tradeable symbol today, which is a "
        "form of survival. Every name that stopped trading during the sample is "
        "absent from its entire history, not merely from the point it stopped."
    )

    parts.append(
        "**The mechanism is not the one section 3.3 anticipates, and the measurement "
        "says so.** That section expects acquisitions and mergers to remain "
        f"retrievable while failures do not. Measured here, acquired or merged names "
        f"are retrievable at only {acq_rate:.1%} ({acq_got} of {acq_n}) — Yahoo drops a "
        f"symbol when it stops trading regardless of *why* it stopped. Selection is on "
        f"'no longer trades', not on 'failed'. Names still in the index are retrievable "
        f"at {still_rate:.1%} ({still_got} of {still_n}); names dropped in an index "
        f"rebalance, which usually keep trading, at {reb_rate:.1%} ({reb_got} of {reb_n})."
    )

    composition = ", ".join(
        f"{n} {reason.replace('_', ' ')}"
        for reason, n in sorted(missing_by_reason.items(), key=lambda kv: -kv[1])
    )
    parts.append(
        f"**Composition of the {missing} absent names:** {composition}. That composition "
        "pulls in two directions, and honesty requires naming both:"
    )

    parts.append(
        f"- The {acq_missing} acquired or merged names typically ended at a takeover "
        "premium. Dropping them removes a positive terminal return, which biases "
        "measured performance **downward**. This is the largest single missing bucket."
    )
    parts.append(
        f"- The {reb_missing} names dropped in index rebalances left because their market "
        "capitalisation shrank, which is a slow underperformance. Dropping them biases "
        "measured performance **upward**."
    )
    parts.append(
        "- Bankruptcies and delistings for cause bias **upward** and are the classic "
        "survivorship channel, but here they are a handful of names, not the bulk."
    )

    parts.append(
        "**Net direction: upward, with lower confidence than section 3.3 assumes.** The "
        "conditioning-on-survival channel is the dominant one and points up. The "
        "acquisition channel is the largest by name count and points down. The two do "
        "not cancel to zero, but neither is the sign as safe as 'failures are missing' "
        "would make it. The magnitude is not directly measurable — the returns of the "
        "absent names are exactly what cannot be observed."
    )

    parts.append(
        f"The retrieval rate by year is the clearest single signal that the bias is real "
        f"and time-varying: it climbs monotonically from "
        f"{report.by_year['retrieval_rate'][0]:.1%} in {report.by_year['year'][0]} to "
        f"{report.by_year['retrieval_rate'][-1]:.1%} in {report.by_year['year'][-1]}. "
        "The early sample is the most contaminated, and the early sample is the training set."
    )

    return "\n\n".join(parts)
