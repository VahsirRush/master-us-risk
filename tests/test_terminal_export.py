"""Terminal export — RISK / ATTR panels follow phase_ladder, not a hard-code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_us.terminal.export import (
    PHASE7_TABLES,
    _load_phase7_tables,
    _phase_passed,
    _risk_attr_payload,
)


def _phases(*, five=True, six=True, seven=True):
    def row(n: int, ok: bool) -> dict:
        return {
            "phase": n,
            "label": f"P{n}",
            "status": "pass" if ok else "pending",
            "gate_passed": True if ok else None,
            "gate": "ok" if ok else None,
            "notes": ["headline"] if n == 7 and ok else [],
            "metrics": {
                "in_gate_fraction": {"gross": 0.946, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "bias_random": {"gross": 1.0, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "bias_factor_mimicking": {"gross": 0.94, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "bias_market": {"gross": 1.07, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "market_ann": {"gross": 0.14, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "momentum_ann": {"gross": 0.009, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "value_ann": {"gross": 0.008, "net": None, "std": None, "net_std": None, "n_seeds": None},
                "mean_r2": {"gross": 0.26, "net": None, "std": None, "net_std": None, "n_seeds": None},
            }
            if ok
            else {},
        }

    return [row(5, five), row(6, six), row(7, seven)]


@pytest.fixture
def tables() -> dict:
    raw = _load_phase7_tables()
    assert raw is not None, f"missing {PHASE7_TABLES} — run scripts/71_join_report.py"
    return raw


def test_phase7_tables_are_structured(tables):
    assert len(tables["books"]) == 4
    assert {b["key"] for b in tables["books"]} == {
        "decile",
        "plain_eigen",
        "neutral_eigen",
        "neutral_no_eigen",
    }
    assert len(tables["timing"]) == 8
    assert "specific_gross_sharpe_caveat" in tables["attribution"]
    assert "method_note" in tables["timing_summary"]
    assert tables["eigen"]["prediction_confirmed"] is False


def test_risk_attr_ready_when_phases_pass(tables):
    risk, attr, pending = _risk_attr_payload(_phases(), tables)
    assert pending == []
    assert risk is not None and attr is not None
    assert len(risk["books"]) == 4
    assert risk["attribution"]["specific_gross_sharpe_caveat"]
    assert len(attr["timing"]) == 8
    assert "SUBSTITUTE" in attr["timing_summary"]["method_note"].upper() or "substitute" in attr[
        "timing_summary"
    ]["method_note"].lower() or "CONSEQUENCE" in attr["timing_summary"]["method_note"].upper()


def test_risk_pending_when_phase_six_missing(tables):
    risk, attr, pending = _risk_attr_payload(_phases(six=False), tables)
    assert risk is None
    assert attr is not None  # ATTR only needs phase 7
    assert [p["id"] for p in pending] == ["risk"]


def test_both_pending_without_tables():
    risk, attr, pending = _risk_attr_payload(_phases(), None)
    assert risk is None and attr is None
    assert [p["id"] for p in pending] == ["risk", "attr"]


def test_phase_passed_requires_gate():
    assert _phase_passed(
        [{"phase": 7, "status": "pass", "gate_passed": True}], 7
    )
    assert not _phase_passed(
        [{"phase": 7, "status": "pass", "gate_passed": False}], 7
    )
    assert not _phase_passed(
        [{"phase": 7, "status": "pending", "gate_passed": None}], 7
    )


def test_committed_tables_match_phase7_status():
    """The status artifact must point at the structured twin we export from."""
    status = json.loads(
        Path("reports/status/phase7.json").read_text()
    )
    assert any(str(a).endswith("phase7_tables.json") for a in status["artifacts"])
    assert PHASE7_TABLES.exists()
