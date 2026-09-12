"""PhaseResult / ablation tables -> the terminal's JSON blob — spec §1, §5.

Every number in the dashboard comes from here, and every number here is
computed from committed artifacts: the cached score arrays, the metrics
bundle, `reports/status/phase*.json`, and `reports/survivorship.md`. Nothing
is typed in by hand and nothing is a placeholder — a dashboard that invents a
figure to fill a panel is worse than one with an empty panel, because the
empty panel is honest about what was measured.

The ticker frame is the spec's organizing idea (§0): each variant is an
instrument with a price (net Sharpe), a volatility (seed dispersion), and a
carrying cost (turnover).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_us.data.sources import REPO_ROOT
from master_us.reporting.results import MetricValue, PhaseResult

TERMINAL_DIR = REPO_ROOT / "reports" / "terminal"
SURVIVORSHIP = REPO_ROOT / "reports" / "survivorship.md"

# spec §0 — the ticker table. `isolates` is what the cell tests, which is what
# turns a row of numbers into a claim.
TICKERS: dict[str, dict[str, str]] = {
    "master": {"ticker": "MSTR.US", "name": "Full MASTER", "isolates": "—  reference"},
    "ungated": {"ticker": "MSTR.NG", "name": "No gating", "isolates": "the market gate"},
    "no_inter_stock": {
        "ticker": "MSTR.NX",
        "name": "No inter-stock attention",
        "isolates": "cross-sectional modelling",
    },
    "time_aligned": {
        "ticker": "MSTR.TA",
        "name": "Time-aligned",
        "isolates": "the cross-time claim",
    },
    "market_shuffled": {
        "ticker": "MSTR.SH",
        "name": "Market vector shuffled",
        "isolates": "whether the gate reads real market structure",
    },
    "lgbm": {"ticker": "LGBM.BL", "name": "LightGBM (tuned)", "isolates": "the strongest baseline"},
    "lgbmspec": {
        "ticker": "LGBM.SP",
        "name": "LightGBM (shipped cfg)",
        "isolates": "the config the spec ships",
    },
    "lstm": {"ticker": "LSTM.BL", "name": "Per-stock LSTM", "isolates": "temporal modelling alone"},
    "ridge": {"ticker": "RIDG.BL", "name": "Cross-sectional ridge", "isolates": "the linear floor"},
    "master_full": {
        "ticker": "MSTR.UF",
        "name": "Full MASTER · 100/10",
        "isolates": "— reference, full budget",
    },
    "ungated_full": {
        "ticker": "MSTR.NF",
        "name": "No gating · 100/10",
        "isolates": "the market gate, full budget",
    },
}


@dataclass(frozen=True)
class Serialized:
    payload: dict[str, Any]

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=2, sort_keys=True))
        return path


def _mv(m: MetricValue | None) -> dict[str, Any] | None:
    if m is None:
        return None
    return {
        "gross": m.gross,
        "net": m.net,
        "std": m.std,
        "net_std": m.net_std,
        "n_seeds": m.n_seeds,
    }


def survivorship_by_year(path: Path = SURVIVORSHIP) -> list[dict[str, Any]]:
    """Parse the committed survivorship report's per-year table."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        m = re.match(r"^\|\s*(\d{4})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|$", line.strip())
        if m:
            rows.append(
                {
                    "year": int(m.group(1)),
                    "constituents": int(m.group(2)),
                    "retrieved": int(m.group(3)),
                    "rate": float(m.group(4)) / 100.0,
                }
            )
    return rows


def survivorship_headline(path: Path = SURVIVORSHIP) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    m = re.search(r"\*\*(\d+) of (\d+) historical constituents \(([\d.]+)%\)", text)
    return {
        "retrieved": int(m.group(1)) if m else None,
        "total": int(m.group(2)) if m else None,
        "rate": float(m.group(3)) / 100.0 if m else None,
        "bias_direction": "OPTIMISTIC" if "OPTIMISTIC" in text else None,
    }


def phase_ladder() -> list[dict[str, Any]]:
    """The BUILD panel (§3.1): real cached PhaseResults, nothing assumed."""
    results = PhaseResult.load_all()
    out: list[dict[str, Any]] = []
    names = {
        0: "DATA", 1: "ENGINE", 2: "BASELINE", 3: "MASTER",
        4: "ABLATION", 5: "FACTORS", 6: "RISK", 7: "JOIN", 8: "REPORTS",
    }
    for phase, label in names.items():
        r = results.get(phase)
        out.append(
            {
                "phase": phase,
                "label": label,
                "status": r.status if r else "pending",
                "gate_passed": r.gate_passed if r else None,
                "gate": r.gate if r else None,
                "notes": r.notes if r else [],
                "metrics": {k: _mv(v) for k, v in (r.metrics.items() if r else [])},
            }
        )
    return out


def build_payload(tables: dict[str, dict[str, MetricValue]]) -> Serialized:
    """The full blob the frontend consumes."""
    variants = []
    for key, meta in TICKERS.items():
        t = tables.get(key)
        if t is None:
            continue
        variants.append(
            {
                "key": key,
                **meta,
                "rank_ic": _mv(t.get("rank_ic")),
                "icir": _mv(t.get("icir")),
                "ls_sharpe": _mv(t.get("ls_sharpe")),
                "lo_sharpe": _mv(t.get("lo_sharpe") or t.get("sharpe")),
                "turnover": _mv(t.get("turnover")),
                "ls_turnover": _mv(t.get("ls_turnover")),
                "breakeven_bps": _mv(t.get("breakeven_bps")),
            }
        )

    return Serialized(
        {
            "generated_at": __import__("datetime").datetime.now(
                __import__("datetime").UTC
            ).isoformat(timespec="seconds"),
            "variants": variants,
            "phases": phase_ladder(),
            "survivorship": {
                "headline": survivorship_headline(),
                "by_year": survivorship_by_year(),
            },
        }
    )
