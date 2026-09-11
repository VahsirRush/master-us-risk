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

**Phase: 4 COMPLETE (Session 10). The project's headline question is SETTLED. Next: Phase 5 (Barra risk model).**

## THE GATE-NULL RESULT IS CLOSED — do not re-open it casually

**The market-guided gate does not produce a distinguishable improvement over the ungated architecture.** Settled at Phase 4 with the spec's full 100/10 budget and 10 confirmatory seeds per condition, backed by a 25-run β sweep and a market-shuffle control.

| Budget | Measure | MASTER | ungated | gap | ratio | verdict |
|---|---|---:|---:|---:|---:|---|
| short 12/4 | gross RankIC | +0.0212 ±0.0009 | +0.0201 ±0.0006 | +0.0011 | 0.949 | NOT distinguishable |
| **full 100/10** | gross RankIC | +0.0224 ±0.0022 | +0.0210 ±0.0020 | +0.0014 | **0.464** | NOT distinguishable |
| **full 100/10** | net L/S Sharpe | −0.4556 ±0.2611 | −0.5590 ±0.2731 | +0.1034 | **0.274** | NOT distinguishable |

**It held up — and strengthened — under more budget and more power.** Phase 3's 0.949 was a 5% near-miss that plausibly meant "under-trained or under-powered". It was neither: the full budget doubles the gap (+0.0011 → +0.0014) and more than doubles the dispersion (±0.0009 → ±0.0022), so the ratio *falls* to 0.464. A real effect behaves the opposite way. The extra budget was genuine work, not a formality — best epochs moved 1,1,2,3,1 → 1,2,2,6,**11** and wall time roughly doubled.

Independent corroboration: **no β from 0.1 to 10.0 separates from no-gating**, and `market_shuffled` (gate fed a date-permuted market vector) costs +0.0212 → +0.0207 ±0.0002 — the gate is not reading market structure. Wider still: **none of MASTER's three structural mechanisms is distinguishable from its ablation** — gating −0.0011, inter-stock attention −0.0000, cross-time attention −0.0007.

**This is CLOSED, not live.** Do not re-run it for more seeds or more epochs — both were tried and both moved the answer *away* from significance. Re-open only on (a) an architectural change to the gating mechanism itself, or (b) a materially different experimental setup — different universe, label horizon, or feature bank. Anything else is re-litigating a settled result.

Secondary finding worth carrying: **longer training is less reproducible.** Seed dispersion roughly doubles at 100/10 (RankIC ±0.0009 → ±0.0022; net L/S ±0.12 → ±0.26) with turnover unchanged. More budget buys a slightly higher mean at materially worse run-to-run stability.

Full detail: `reports/phase4.md`, `reports/framing.md`, NOTES Session 10.

## Still open

- **§8.1 rows 6-7** (lookback {20,40,60,120}, heads {(4,2),(8,4),(8,8),(16,4)}), both arms per point. Head sweep running (~12.5h); lookback extrapolates to ~70-95h and was not completable in-session. These cannot overturn the headline — only show whether the gate helps at some *other* hyperparameter setting, which would be a new claim needing its own evidence.
- **Phases 5-7**: Barra risk model, attribution, the join. Untouched.
- `Panel.metadata` mcap/sector remain PROVISIONAL — due for replacement at Phase 5.

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
