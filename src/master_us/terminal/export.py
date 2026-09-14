"""Static export of the terminal payload — spec §1 (static mode).

The terminal has two modes off one codebase: a live FastAPI server and a
static JSON blob committed for GitHub Pages. This builds the blob. The
frontend never computes a statistic; it only formats what lands here, so a
number on the screen can always be traced to an array on disk.

Everything is computed from committed artifacts:

    data/processed/phase4/metrics_bundle.npz   labels, mask, returns, sectors
    data/processed/phase2/*_seed*_scores.npy   per-seed model scores
    reports/status/phase*.json                 cached PhaseResults
    reports/phase7_tables.json                 join books / timing / attribution
    reports/survivorship.md                    Phase 0 coverage

There are no literals in this file that stand in for a measurement. RISK and
ATTR are pending only when their phases have not passed (or the join tables
are absent); once PhaseResults say pass, the panels carry real numbers.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from scipy import stats as scipy_stats

from master_us.backtest.construct import decile_long_short
from master_us.backtest.costs import CostConfig
from master_us.backtest.engine import run_backtest
from master_us.data.sources import REPO_ROOT
from master_us.experiments.ablations import BETA_VALUES, cost_breakeven, net_sharpe_at_bps
from master_us.experiments.phase2 import PHASE2_DIR
from master_us.experiments.stress import (
    Bundle,
    cap_tiers,
    decay_curve,
    load_bundle,
    rank_ic,
    regime_slices,
    sector_neutralized,
)
from master_us.reporting.results import MetricValue
from master_us.terminal.serialize import (
    TICKERS,
    phase_ladder,
    survivorship_by_year,
    survivorship_headline,
)

REQUIRED_SEEDS = 5
BPS_GRID = (0.0, 5.0, 10.0, 20.0, 50.0)

# The blotter order: MASTER first as the reference, then its ablations, then
# the baselines. The two full-budget cells are held out of the blotter and
# shown only in the gate-null table, where their budget is labelled — a
# 100/10 row sitting unlabelled beside 12/4 rows would be a budget
# comparison masquerading as an architecture comparison.
BLOTTER = (
    "master",
    "ungated",
    "no_inter_stock",
    "time_aligned",
    "market_shuffled",
    "lgbm",
    "lgbmspec",
    "lstm",
    "ridge",
)
EQUITY = ("master", "ungated", "lgbm", "lstm", "ridge")

PHASE7_TABLES = REPO_ROOT / "reports" / "phase7_tables.json"

# Spec §3.5 / §3.6 empty-state copy. Used only when phase_ladder says the
# underlying phases have not passed — never hardcoded as permanently pending.
_RISK_PENDING = {
    "id": "risk",
    "label": "RISK",
    "spec": "§3.5",
    "phases": "5-7",
    "needs": "Barra-style factor model and the join: style factors, covariance "
    "calibration, and the four book variants under style-neutralization.",
}
_ATTR_PENDING = {
    "id": "attr",
    "label": "ATTR",
    "spec": "§3.6",
    "phases": "7",
    "needs": "Factor attribution of the signal — the join of the model's "
    "scores onto the risk model's factor exposures, plus the gate factor-timing check.",
}


def seeds_for(tag: str) -> list[int]:
    return sorted(
        int(p.stem.split("_seed")[1].split("_")[0])
        for p in PHASE2_DIR.glob(f"{tag}_seed*_scores.npy")
    )


def _daily_returns(bundle: Bundle) -> npt.NDArray[np.float64]:
    """Same-day realized return, lagged out of the forward label."""
    out = np.full_like(bundle.raw_forward, np.nan, dtype=np.float64)
    out[1:] = bundle.raw_forward[:-1]
    return out


def _ic_series(bundle: Bundle, scores: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    vals = []
    for t in bundle.test_idx:
        ok = bundle.label_valid[t] & np.isfinite(scores[t]) & np.isfinite(bundle.labels[t])
        if ok.sum() >= 10:
            vals.append(float(scipy_stats.spearmanr(scores[t, ok], bundle.labels[t, ok]).statistic))
    return np.asarray(vals, dtype=np.float64)


def evaluate(
    bundle: Bundle, tag: str, cost_cfg: CostConfig
) -> dict[str, Any] | None:
    """Per-seed metrics plus the series the terminal charts.

    Returns None below `REQUIRED_SEEDS`. An incomplete cell is not a weak
    result, it is not a result — Session 10 killed a sweep after 1 of 30 runs
    and the single score file would otherwise have rendered as a row
    indistinguishable from a finished one.
    """
    seeds = seeds_for(tag)
    if len(seeds) < REQUIRED_SEEDS:
        return None

    per_seed: dict[str, list[float]] = {
        k: [] for k in ("rank_ic", "icir", "ls_gross", "ls_net", "breakeven")
    }
    test = bundle.test_idx
    daily = _daily_returns(bundle)
    equity_curves: list[npt.NDArray[np.float64]] = []
    turnover_pool: list[npt.NDArray[np.float64]] = []
    cost_curves: list[list[float]] = []

    for seed in seeds:
        scores = np.load(PHASE2_DIR / f"{tag}_seed{seed}_scores.npy")
        ic = _ic_series(bundle, scores)
        per_seed["rank_ic"].append(float(ic.mean()))
        sd = float(ic.std(ddof=1))
        per_seed["icir"].append(float(ic.mean() / sd) if sd > 0 else 0.0)

        res = run_backtest(
            dates=bundle.dates[test],
            returns=daily[test],
            scores=scores[test].astype(np.float64),
            mask=bundle.mask[test],
            constructor=lambda s, m, p: decile_long_short(s, m, n_deciles=10),
            rebalance_idx=np.arange(len(test), dtype=np.int64),
            cost_cfg=cost_cfg,
            tier="baseline",
        )
        g, n, to = res.series.gross, res.series.net, res.series.turnover
        per_seed["ls_gross"].append(_sharpe(g))
        per_seed["ls_net"].append(_sharpe(n))
        per_seed["breakeven"].append(cost_breakeven(g, to))
        equity_curves.append(
            (np.cumprod(1.0 + np.nan_to_num(n)) - 1.0).astype(np.float64)
        )
        turnover_pool.append(np.asarray(to, dtype=np.float64))
        cost_curves.append([net_sharpe_at_bps(g, to, b) for b in BPS_GRID])

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "equity": np.mean(equity_curves, axis=0),
        "turnover": np.concatenate(turnover_pool),
        "cost_curve": np.mean(cost_curves, axis=0).tolist(),
    }


def _sharpe(x: npt.NDArray[np.floating], trading_days: int = 252) -> float:
    a = np.asarray(x, dtype=np.float64)
    sd = a.std(ddof=1)
    return float(a.mean() / sd * math.sqrt(trading_days)) if sd > 0 else 0.0


def _mv(values: list[float]) -> MetricValue:
    a = np.asarray(values, dtype=np.float64)
    return MetricValue(
        gross=float(a.mean()),
        net=float(a.mean()),
        std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        n_seeds=int(a.size),
        net_std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
    )


def _pair(values_g: list[float], values_n: list[float]) -> MetricValue:
    g, n = np.asarray(values_g), np.asarray(values_n)
    return MetricValue(
        gross=float(g.mean()),
        net=float(n.mean()),
        std=float(g.std(ddof=1)) if g.size > 1 else 0.0,
        n_seeds=int(g.size),
        net_std=float(n.std(ddof=1)) if n.size > 1 else 0.0,
    )


def metrics_of(ev: dict[str, Any]) -> dict[str, MetricValue]:
    p = ev["per_seed"]
    return {
        "rank_ic": _mv(p["rank_ic"]),
        "icir": _mv(p["icir"]),
        "ls_sharpe": _pair(p["ls_gross"], p["ls_net"]),
        "breakeven_bps": _mv(p["breakeven"]),
    }


def _net(m: MetricValue) -> float:
    """The net figure, asserted present.

    `MetricValue.net` is optional in the contract because a gross-only
    metric is a legitimate object. Nothing in this export is one — every
    variant here is built by `_pair` from a real net return array — so a
    None means the payload was assembled wrong. Raise rather than coerce to
    0.0: a silent zero would publish as a real net Sharpe.
    """
    if m.net is None:
        raise ValueError("MetricValue.net is None; expected a net series for this metric")
    return m.net


def _m(mv: MetricValue | None) -> dict[str, Any] | None:
    if mv is None:
        return None
    return {
        "gross": mv.gross,
        "net": mv.net,
        "std": mv.std,
        "net_std": mv.net_std,
        "n_seeds": mv.n_seeds,
    }


def _sample_idx(size: int, n: int = 260) -> list[int]:
    """Evenly spaced indices with the endpoint always kept.

    The equity curve is ~1,750 test days; the chart is ~700 px wide. Sampling
    to 260 points is a plotting decision, not a statistical one — every
    reported number is computed on the full series before this runs.
    """
    if size <= n:
        return list(range(size))
    return sorted(set(np.linspace(0, size - 1, n).astype(int).tolist()) | {size - 1})


def _downsample(a: npt.NDArray[np.floating], n: int = 260) -> list[float]:
    arr = np.asarray(a, dtype=np.float64)
    return [round(float(arr[i]), 6) for i in _sample_idx(arr.size, n)]


def _histogram(a: npt.NDArray[np.floating], bins: int = 26) -> dict[str, Any]:
    arr = np.asarray(a, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"edges": [], "counts": [], "mean": None, "p50": None}
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "edges": [round(float(e), 5) for e in edges],
        "counts": [int(c) for c in counts],
        "mean": round(float(arr.mean()), 5),
        "p50": round(float(np.median(arr)), 5),
    }


def build(cost_cfg: CostConfig | None = None) -> dict[str, Any]:
    """The whole payload. Every key here is backed by an array on disk."""
    if cost_cfg is None:
        with (REPO_ROOT / "config" / "costs.yaml").open() as fh:
            cost_cfg = CostConfig.from_yaml(yaml.safe_load(fh))
    bundle = load_bundle()

    tags = list(TICKERS) + [f"beta_{b}" for b in BETA_VALUES if b != 1.0]
    evs = {t: ev for t in tags if (ev := evaluate(bundle, t, cost_cfg)) is not None}
    mets = {t: metrics_of(ev) for t, ev in evs.items()}

    test_dates = [str(d.date()) for d in bundle.dates[bundle.test_idx]]

    variants = []
    for key in BLOTTER:
        if key not in evs:
            continue
        m, p = mets[key], evs[key]["per_seed"]
        variants.append(
            {
                "key": key,
                **TICKERS[key],
                "rank_ic": _m(m["rank_ic"]),
                "icir": _m(m["icir"]),
                "ls_sharpe": _m(m["ls_sharpe"]),
                "breakeven_bps": _m(m["breakeven_bps"]),
                "turnover_mean": round(float(np.mean(evs[key]["turnover"])), 5),
                "seeds": evs[key]["seeds"],
                "per_seed": {k: [round(v, 6) for v in vals] for k, vals in p.items()},
                "turnover_hist": _histogram(evs[key]["turnover"]),
                "cost_curve": [round(v, 5) for v in evs[key]["cost_curve"]],
            }
        )

    ref = mets["master"]
    for v in variants:
        m = mets[v["key"]]
        v["d_net_ls"] = round(_net(m["ls_sharpe"]) - _net(ref["ls_sharpe"]), 6)
        v["d_rank_ic"] = round(m["rank_ic"].gross - ref["rank_ic"].gross, 6)
        v["net_distinguishable"] = (
            False if v["key"] == "master" else m["ls_sharpe"].net_distinguishable_from(ref["ls_sharpe"])
        )
        v["gross_distinguishable"] = (
            False if v["key"] == "master" else m["rank_ic"].distinguishable_from(ref["rank_ic"])
        )

    phases = phase_ladder()
    tables = _load_phase7_tables()
    risk, attr, pending = _risk_attr_payload(phases, tables)

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provenance": {
            "bundle": "data/processed/phase4/metrics_bundle.npz",
            "scores": "data/processed/phase2/<variant>_seed<n>_scores.npy",
            "phases": "reports/status/phase*.json",
            "join_tables": "reports/phase7_tables.json",
            "survivorship": "reports/survivorship.md",
            "cost_model": {
                "baseline_bps": 10.0,
                "spread": "Corwin-Schultz, overnight-gap adjusted",
                "impact": "sqrt market impact",
            },
            "splits": {"train": "<=2016", "valid": "2017-2018", "test": "2019-2025", "embargo_days": 21},
            "required_seeds": REQUIRED_SEEDS,
        },
        "variants": variants,
        "gate_null": _gate_null(mets),
        "ablations": _ablations(mets),
        "beta_sweep": _beta(mets),
        "cost": {
            "bps_grid": list(BPS_GRID),
            "rows": [
                {
                    "key": v["key"],
                    "ticker": v["ticker"],
                    "name": v["name"],
                    "curve": v["cost_curve"],
                    "breakeven": v["breakeven_bps"]["gross"],
                    "turnover_hist": v["turnover_hist"],
                    "turnover_mean": v["turnover_mean"],
                }
                for v in variants
            ],
        },
        "equity": {
            "dates": [test_dates[i] for i in _sample_idx(len(test_dates))],
            "series": [
                {"key": k, "ticker": TICKERS[k]["ticker"], "curve": _downsample(evs[k]["equity"])}
                for k in EQUITY
                if k in evs
            ],
            "basis": "net of costs at the 10 bps baseline, dollar-neutral decile book, "
            "seed-averaged",
        },
        "stress": _stress(bundle, evs),
        "phases": phases,
        "pending_panels": pending,
        "risk": risk,
        "attr": attr,
        "survivorship": {
            "headline": survivorship_headline(),
            "by_year": survivorship_by_year(),
        },
        "deferred": [
            {
                "row": "§8.1 row 6",
                "name": "Lookback sweep — L = 40 / 60 / 120, both arms",
                "status": "DEFERRED — NOT RUN",
                "compute": "70-95 h projected",
                "detail": "Deferred rather than run at reduced power. No results exist.",
            },
            {
                "row": "§8.1 row 7",
                "name": "Head sweep — (4,2) / (8,8) / (16,4), both arms",
                "status": "DEFERRED — NOT RUN",
                "compute": "~12.2 h projected",
                "detail": "Launched and stopped after 1 of 30 runs; that output was "
                "deleted. No results exist.",
            },
        ],
    }


def _load_phase7_tables(path: Path = PHASE7_TABLES) -> dict[str, Any] | None:
    """Committed join tables — machine-readable twin of `reports/phase7.md`."""
    if not path.exists():
        return None
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} did not parse to a mapping")
    return loaded


def _phase_passed(phases: list[dict[str, Any]], n: int) -> bool:
    hit = next((p for p in phases if p["phase"] == n), None)
    return hit is not None and hit.get("status") == "pass" and hit.get("gate_passed") is True


def _risk_attr_payload(
    phases: list[dict[str, Any]], tables: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """RISK / ATTR content, or pending records, derived from phase_ladder.

    RISK needs Phases 5-7 (Barra factors + calibrated covariance + the join
    books). ATTR needs Phase 7 (timing table + eigenfactor check). Neither
    panel is hard-coded pending: once the PhaseResults say pass and the
    structured tables exist, the panels carry numbers.
    """
    pending: list[dict[str, Any]] = []
    risk_ready = (
        _phase_passed(phases, 5)
        and _phase_passed(phases, 6)
        and _phase_passed(phases, 7)
        and tables is not None
    )
    attr_ready = _phase_passed(phases, 7) and tables is not None

    risk: dict[str, Any] | None = None
    attr: dict[str, Any] | None = None

    if risk_ready:
        assert tables is not None
        risk = {
            "books": tables["books"],
            "control_note": tables["control_note"],
            "attribution": tables["attribution"],
            "neutral_vs_unconstrained": tables["neutral_vs_unconstrained"],
            "n_seeds": tables["n_seeds"],
            "headline": next(
                (p["notes"][0] for p in phases if p["phase"] == 7 and p.get("notes")),
                None,
            ),
            "bias": {
                "in_gate_fraction": _phase_metric(phases, 6, "in_gate_fraction"),
                "random": _phase_metric(phases, 6, "bias_random"),
                "factor_mimicking": _phase_metric(phases, 6, "bias_factor_mimicking"),
                "market": _phase_metric(phases, 6, "bias_market"),
                "note": "validates CALIBRATION only — says nothing about factor-return strength",
            },
            "factors": {
                "market_ann": _phase_metric(phases, 5, "market_ann"),
                "momentum_ann": _phase_metric(phases, 5, "momentum_ann"),
                "value_ann": _phase_metric(phases, 5, "value_ann"),
                "mean_r2": _phase_metric(phases, 5, "mean_r2"),
            },
        }
    else:
        pending.append(dict(_RISK_PENDING))

    if attr_ready:
        assert tables is not None
        attr = {
            "timing": tables["timing"],
            "timing_summary": tables["timing_summary"],
            "eigen": tables["eigen"],
            "n_seeds": tables["n_seeds"],
        }
    else:
        pending.append(dict(_ATTR_PENDING))

    return risk, attr, pending


def _phase_metric(phases: list[dict[str, Any]], n: int, key: str) -> float | None:
    hit = next((p for p in phases if p["phase"] == n), None)
    if hit is None:
        return None
    m = (hit.get("metrics") or {}).get(key)
    if not isinstance(m, dict):
        return None
    g = m.get("gross")
    return float(g) if g is not None else None


def _gate_null(mets: dict[str, dict[str, MetricValue]]) -> list[dict[str, Any]]:
    out = []
    for label, gk, uk in (("short 12/4", "master", "ungated"), ("full 100/10", "master_full", "ungated_full")):
        if gk not in mets or uk not in mets:
            continue
        measures = []
        for metric, basis, pretty in (
            ("rank_ic", "gross", "gross RankIC"),
            ("ls_sharpe", "net", "net L/S Sharpe"),
        ):
            a, b = mets[gk][metric], mets[uk][metric]
            if basis == "gross":
                av, bv, asd, bsd = a.gross, b.gross, a.std, b.std
                real = a.distinguishable_from(b)
            else:
                av, bv, asd, bsd = _net(a), _net(b), a.net_std, b.net_std
                real = a.net_distinguishable_from(b)
            pooled = math.hypot(asd or 0.0, bsd or 0.0)
            measures.append(
                {
                    "metric": pretty,
                    "basis": basis,
                    "gated": av,
                    "gated_sd": asd,
                    "ungated": bv,
                    "ungated_sd": bsd,
                    "gap": av - bv,
                    "pooled": pooled,
                    "ratio": abs(av - bv) / pooled if pooled else None,
                    "distinguishable": real,
                }
            )
        out.append({"budget": label, "gated": TICKERS[gk]["ticker"], "ungated": TICKERS[uk]["ticker"], "measures": measures})
    return out


def _ablations(mets: dict[str, dict[str, MetricValue]]) -> list[dict[str, Any]]:
    ref = mets["master"]
    out = []
    for k in ("ungated", "no_inter_stock", "time_aligned", "market_shuffled"):
        if k not in mets:
            continue
        m = mets[k]
        out.append(
            {
                "key": k,
                **TICKERS[k],
                "rank_ic": _m(m["rank_ic"]),
                "ls_sharpe": _m(m["ls_sharpe"]),
                "d_rank_ic": m["rank_ic"].gross - ref["rank_ic"].gross,
                "gross_distinguishable": m["rank_ic"].distinguishable_from(ref["rank_ic"]),
                "d_net_ls": _net(m["ls_sharpe"]) - _net(ref["ls_sharpe"]),
                "net_distinguishable": m["ls_sharpe"].net_distinguishable_from(ref["ls_sharpe"]),
            }
        )
    return out


def _beta(mets: dict[str, dict[str, MetricValue]]) -> dict[str, Any]:
    u = mets["ungated"]
    pts = []
    for b in BETA_VALUES:
        key = "master" if b == 1.0 else f"beta_{b}"
        if key not in mets:
            continue
        m = mets[key]
        pts.append(
            {
                "beta": b,
                "is_default": b == 1.0,
                "rank_ic": m["rank_ic"].gross,
                "sd": m["rank_ic"].std,
                "net_ls": _net(m["ls_sharpe"]),
                "net_sd": m["ls_sharpe"].net_std,
                "breakeven": m["breakeven_bps"].gross,
                "distinguishable": m["rank_ic"].distinguishable_from(u["rank_ic"]),
            }
        )
    return {
        "reference": {
            "ticker": TICKERS["ungated"]["ticker"],
            "rank_ic": u["rank_ic"].gross,
            "sd": u["rank_ic"].std,
            "net_ls": _net(u["ls_sharpe"]),
            "net_sd": u["ls_sharpe"].net_std,
        },
        "points": pts,
    }


def _stress(bundle: Bundle, evs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("master", "ungated"):
        if key not in evs:
            continue
        stacked = np.nanmean(
            [np.load(PHASE2_DIR / f"{key}_seed{s}_scores.npy") for s in evs[key]["seeds"]], axis=0
        )
        out[key] = {
            "regimes": [
                {"name": n, "ic": None if not np.isfinite(v) else float(v), "dates": int(d)}
                for n, (v, d) in regime_slices(bundle, stacked).items()
            ],
            "cap_tiers": [
                {"tier": n, "ic": None if not np.isfinite(v) else float(v)}
                for n, v in cap_tiers(bundle, stacked).items()
            ],
            "decay": [{"h": int(h), "ic": float(v)} for h, v in decay_curve(bundle, stacked).items()],
            "sector_neutral": float(
                rank_ic(sector_neutralized(bundle, stacked), bundle.labels, bundle.label_valid, bundle.test_idx)
            ),
        }
    return out


def write(path: Path, payload: dict[str, Any] | None = None) -> Path:
    payload = payload if payload is not None else build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), default=float))
    return path
