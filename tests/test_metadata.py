"""Tests for the provisional metadata layer — Session 6.

The split-basis harmonization gets the closest look: it is the piece that was
measurably wrong on the first build (AAPL 2015 mcap out by 4x) and the piece
most likely to silently drift.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from master_us.data.assemble import PANEL_PATH
from master_us.data.metadata import METADATA_PROVENANCE, _harmonize_split_basis
from master_us.data.panel import Panel


def _facts(ticker: str, counts: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [ticker] * len(counts),
            "filed": [datetime(2015 + i // 4, 1 + 3 * (i % 4), 1) for i in range(len(counts))],
            "shares": counts,
        }
    ).with_columns(pl.col("filed").cast(pl.Datetime("ns")))


def test_harmonization_scales_pre_split_counts():
    """A 4:1 split mid-series: earlier counts scaled x4, later untouched."""
    facts = _facts("AAA", [5.0e9, 5.1e9, 20.4e9, 20.5e9])  # split between filings 2 and 3
    out = _harmonize_split_basis(facts)["shares"].to_numpy()
    np.testing.assert_allclose(out, [20.0e9, 20.4e9, 20.4e9, 20.5e9])


def test_harmonization_leaves_buyback_drift_alone():
    """A steady 2%/quarter buyback never crosses the jump threshold."""
    counts = [1e9 * (0.98**i) for i in range(8)]
    out = _harmonize_split_basis(_facts("BBB", counts))["shares"].to_numpy()
    np.testing.assert_allclose(out, counts)


def test_harmonization_handles_reverse_split():
    facts = _facts("CCC", [10.0e9, 1.0e9, 1.02e9])  # 10:1 reverse split
    out = _harmonize_split_basis(facts)["shares"].to_numpy()
    np.testing.assert_allclose(out, [1.0e9, 1.0e9, 1.02e9])


def test_harmonization_is_per_ticker():
    both = pl.concat([_facts("AAA", [5.0e9, 20.0e9]), _facts("BBB", [1.0e9, 1.01e9])])
    out = _harmonize_split_basis(both)
    aaa = out.filter(pl.col("ticker") == "AAA")["shares"].to_numpy()
    bbb = out.filter(pl.col("ticker") == "BBB")["shares"].to_numpy()
    np.testing.assert_allclose(aaa, [20.0e9, 20.0e9])
    np.testing.assert_allclose(bbb, [1.0e9, 1.01e9])


# ------------------------------------------------------------------ #
# The live panel's metadata                                           #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def panel() -> Panel:
    if not PANEL_PATH.exists():
        pytest.skip("no real panel — run scripts/07_assemble_panel.py")
    return Panel.load(PANEL_PATH)


def test_panel_metadata_satisfies_the_contract(panel):
    md = panel.metadata
    assert md is not None
    assert {"mcap", "sector", "adv", "price"} <= set(md.columns)
    assert md.index.nlevels == 2
    assert len(md) > 2_000_000


def test_panel_metadata_is_flagged_provisional(panel):
    """No consumer may quietly treat this as final."""
    prov = panel.attrs["metadata_provenance"]
    assert "PROVISIONAL" in prov["sector"]
    assert "PROVISIONAL" in prov["mcap"]
    assert prov["replace_in"].startswith("Phase 5")
    assert prov == METADATA_PROVENANCE


def test_panel_mcap_spot_checks(panel):
    """Known market caps within provisional tolerance (30%)."""
    md = panel.metadata
    checks = [
        ("MSFT", "2024-06-03", 3.2e12),
        ("NVDA", "2024-06-03", 2.9e12),
        ("KO", "2020-06-01", 1.95e11),
    ]
    for ticker, when, expected in checks:
        got = float(md.xs(ticker, level=1).loc[when:, "mcap"].iloc[0])
        assert expected / 1.3 < got < expected * 1.3, (
            f"{ticker} {when}: got ${got / 1e9:.0f}B, expected ~${expected / 1e9:.0f}B"
        )


def test_panel_mcap_coverage_is_reported_not_perfect(panel):
    """Coverage well above half, below 100% — matching the XBRL shares table."""
    md = panel.metadata
    coverage = float(md["mcap"].notna().mean())
    assert 0.7 < coverage < 1.0
