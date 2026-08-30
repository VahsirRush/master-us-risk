# CLAUDE.md

Project context for Claude Code. Read this before any work in this repo.

---

## What this is

A port of the MASTER architecture (AAAI-2024, Market-Guided Stock Transformer) to US equities, paired with a Barra-style multi-factor risk model, with a research-terminal UI for tracking results.

**This is a portfolio artifact for quantitative research and risk roles.** Correctness and honest reporting matter more than good-looking numbers. A negative result, properly measured, is a success. A good number that can't be defended is a failure.

## What this is NOT

Not a fidelity replication. The original authors disclosed that their published val/test splits were dumped with training-set processors and contain ~95% of stocks per day; their data access has expired and correct splits can't be regenerated. Do not treat their published table as a target.

The contribution is: faithful architecture port + turnover-aware net-of-cost evaluation + factor attribution of the signal. The original published none of the latter two.

---

## Specifications

Four documents govern this build. Read the relevant one before starting a phase; follow it literally where it specifies a signature or a gate.

| Doc | Governs |
|---|---|
| `docs/implementation-spec.md` | Architecture, phases, module contracts, gates |
| `docs/data-sources-contract.md` | Free-path data (yfinance + SEC XBRL) — amends spec §4.1–4.2 |
| `docs/output-layer-spec.md` | Terminal rendering, `MetricValue` / `PhaseResult` contract |
| `docs/research-terminal-spec.md` | Web dashboard — replaces output-layer §5 |

When a spec and your instinct disagree, follow the spec and raise the disagreement in `NOTES.md`.

---

## Non-negotiable rules

1. **No look-ahead.** Features at `t` use only data known at `t`. Fundamentals join on the SEC `filed` date, never `period_end`. This is the single most common failure mode in this domain.
2. **Normalization statistics come from training only.** `RobustZScoreNorm` fits on train, transforms val/test with borrowed statistics. Refitting on val/test is leakage.
3. **Chronological splits, 21-day embargo.** Train ≤2016 · Valid 2017–2018 · Test 2019–2025. No shuffling, no k-fold. Test is touched once per model, at the end.
4. **Five seeds minimum.** Every headline number is mean ± std over ≥5 initializations. A gap smaller than pooled seed dispersion must be reported as "not distinguishable" in those words.
5. **Every metric is a (gross, net) pair.** Turnover is a permanent column, never optional.
6. **Fail loudly.** No bare `except`. No forward-fill beyond 5 days. Assert alignment on every join. Silent degradation is worse than a crash.
7. **Gates are blocking.** Do not start Phase N+1 until Phase N's gate passes. Specifically: no transformer work before the backtest engine reproduces 12-1 momentum including the 2009 crash.
8. **Logic lives in `src/`.** Scripts and notebooks are thin callers. Nothing that reaches a report is computed in a notebook.

---

## Conventions

- Python 3.11, `ruff` + `mypy` clean before any commit
- `polars` for data pipeline, `pandas` only at boundaries where a library demands it
- Type hints on every public function; `@dataclass(frozen=True)` for all result objects
- Raw pulls under `data/raw/` are immutable — never mutate, only append
- Cache-first everywhere: check parquet, fetch on miss, write immediately
- One commit per phase minimum, with that phase's numbers logged in `NOTES.md`

## NOTES.md protocol

Append-only. After every phase, log: what was built, the gate result, the actual numbers, what broke, and any deviation from spec with reasoning. This file is how the README gets written at the end — treat it as the lab notebook, not a changelog.

---

## Current state

**Phase: 0 fully discharged and Phase 1 passed (Session 5).** The real Panel exists: `data/processed/panel.pkl` (h=1) and `panel_h5.pkl` (h=5) — 4,004 dates (2010-02 → 2025-12) × 584 tickers × 130 features, market vector 67 dims, mean universe 420/day, cache load 0.1s. All four §4.7 leakage gates now pass against REAL data (`tests/test_real_panel_gates.py`), not just synthetic. 220 tests, ruff + mypy clean.

Feature bank (`data/features.py`): 65 base + 65 cross-sectional ranks, every name strictly classified in the feature-group registry. Market vector (`data/market_vector.py`): zero NaN by construction — 2008 warm-up, ≤5-day ffill for calendar gaps, hard raise beyond; ^SP400 dollar volume proxied by MDY (Yahoo reports zero volume for the index itself, measured). Labels are §4.5-processed (trim + per-date z-score); raw forward returns ride in `attrs["raw_forward_returns"]`. Real IC sanity: max |IC| 0.0154 (`xs_ret_1d`, negative = short-term reversal); no feature improves when delayed.

Engine (Session 4) gate numbers: momentum Sharpe −0.13, 2009 crash −106.6% (short-leg-driven), rebalance turnover 57.9%. `make status`: Phases 0 and 1 green.

**Next: Phase 2 baselines** (ridge / lgbm / lstm / ungated, spec §6) — the panel and engine both exist now. Remaining debts: Panel.metadata (mcap needs the SEC shares join — attach at Phase 5), price-data hash pinning (do at Phase-2 start).

## Environment

- venv: `~/.venvs/master-us` (Python 3.11.14). **Not** in the project directory — the
  directory name contains a colon, which Python refuses as a venv path.
- Run everything through it: `~/.venvs/master-us/bin/python -m pytest`
- `lightgbm` needs `brew install libomp`; the pip wheel alone fails to load.
- `config/data.yaml` `user_agent` must be a real name and email. The SEC blocks
  by IP for placeholders; `require_valid_user_agent` refuses before any request.

Update this section at the end of every session.

---

## Session start checklist

1. Read `NOTES.md` to see where the last session ended
2. Read the spec section for the current phase
3. Confirm the previous phase's gate still passes (`make status`)
4. State what you intend to build this session before building it

## Session end checklist

1. Tests pass, `ruff` and `mypy` clean
2. `NOTES.md` updated with numbers and issues
3. "Current state" above updated
4. Committed
