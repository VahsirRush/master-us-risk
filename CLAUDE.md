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

**Phase: 3 COMPLETE (Session 8). Full MASTER measured at 5 seeds. Next: Phase 4 — the β sweep.**

### THE PROJECT HAS BEEN REFRAMED — read `reports/framing.md` before writing anything that explains this project

The original question ("does MASTER beat the baselines?", spec §0/§6) is **answered, and not by MASTER**: the ungated transformer is not distinguishable from tuned LightGBM on gross RankIC, net L/S Sharpe, or net long-only Sharpe. The architecture reaches parity with the strongest baseline **without** the paper's mechanism.

The open question is now: **does the market-guided gate add anything over an ungated architecture already at parity?** The control is the **ungated row**, not the LightGBM row. `reports/framing.md` is a living document and the source the Phase-8 README draws from — keep it current.

### Phase 3 result: the gate is not distinguishable from no gate

| measure | ungated | MASTER | gap | verdict |
|---|---:|---:|---:|---|
| gross RankIC | +0.0201 ±0.0006 | +0.0212 ±0.0009 | +0.0011 | NOT distinguishable (ratio **0.949** — a 5% near-miss) |
| net L/S Sharpe | −0.7383 ±0.0967 | −0.6806 ±0.1221 | +0.0577 | NOT distinguishable |
| net long-only Sharpe | +0.4177 ±0.0448 | +0.4374 ±0.0539 | +0.0198 | NOT distinguishable |

**"Not established", not "no effect".** All three point estimates favour MASTER (same sign, 1-in-8 under a null), and the RankIC gap misses its threshold by 5%. The gate is **mechanically active** — same-seed score agreement with ungated is 0.74–0.94 (mean ≈0.88) against ungated's own seed-to-seed agreement of 0.8166 — so it changes predictions about as much as reseeding does. The answer is *power*, not a verdict.

Six-model table and the full distinguishability grid: NOTES.md Session 8.

**Phase 4 should NOT re-derive the gate ablation headline** (measured above). It should add seeds to resolve the 0.949 near-miss, and run the β sweep (0.1–10.0) against the ungated horizontal reference — spec §7.1's "single most informative plot in this project", and still genuinely open.

**Budget deviation that must be preserved:** MASTER ran 12 epochs / patience 4 / lookback 20 — *identical to the ungated row*, not spec §7.4's 100/10. A gated model trained 8× longer than its control measures budget, not gating. If either row is re-run at 100/10, **both** must be.

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

## Long-run discipline (learned in Sessions 6-7)

- Launch training **detached under `caffeinate`** (`nohup caffeinate -dimsu ... & disown`). The machine sleeps; harness-tracked background tasks get reaped; detached jobs survive both. A 2.4h grid survived a 5-day sleep this way.
- **Never let a watcher be the only evidence.** `scripts/09_heartbeat.py` writes a self-dating artifact; a stale file reads `UNKNOWN`, never `RUNNING`. Check it with `--read` (free) or `make status`.
- **`pgrep -f X` matches every tool watching for X**, including itself — this cost five days of a spinning waiter. Prefer `--pidfile`. Note BSD/macOS `pgrep` has no `-a`.
- Don't run two memory-heavy processes at once on this 8 GB box; a 1.2 GB panel load beside a training run gets the trainer OOM-killed with no traceback.
- Keep at most one waiter and one Monitor alive; stop them when done.
