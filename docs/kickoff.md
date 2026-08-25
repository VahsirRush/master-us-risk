# Claude Code Kickoff

## Setup (do this yourself, before opening Claude Code)

```bash
mkdir master-us && cd master-us
git init
mkdir -p docs
# copy the four specs into docs/
#   implementation-spec.md
#   data-sources-contract.md
#   output-layer-spec.md
#   research-terminal-spec.md
# copy CLAUDE.md to the repo root
git add -A && git commit -m "specs"
```

Claude Code reads `CLAUDE.md` automatically at session start. The specs in `docs/` get read on demand when you point at them.

---

## Session 1 — Scaffold and the Panel contract

**Goal:** a repo that builds, tests, and can exercise every downstream module against synthetic data before any real data exists. No network calls this session.

**Prompt:**

> Read CLAUDE.md and docs/implementation-spec.md sections 1–3.
>
> Build the scaffold: pyproject.toml with the dependency set from §1, the full directory tree from §2 with empty `__init__.py` files, a Makefile with all eight phase targets stubbed to echo "not implemented", and NOTES.md.
>
> Then implement `src/master_us/data/panel.py` — the `Panel` dataclass from §3 with all its `__post_init__` assertions — plus a `make_synthetic_panel(n_dates, n_tickers, n_features, seed)` factory in `tests/fixtures.py` that produces a valid Panel with realistic shapes and a known injected signal.
>
> Write tests that confirm Panel rejects misaligned inputs: wrong feature shape, NaN in market vector, unsorted dates, duplicate dates.
>
> Do not implement any data loading, models, or backtest logic yet.

**Done when:** `pytest` passes, `ruff` and `mypy` clean, `make status` runs.

---

## Session 2 — Phase 0 tests against synthetic data

**Goal:** the leakage tests exist and are proven to catch leakage, before real data can hide behind them.

**Prompt:**

> Read docs/implementation-spec.md §4.5–4.7.
>
> Implement `RobustZScoreNorm` per §4.5 — training-only statistics, borrowed on transform, clip at ±3.
>
> Then write the four Phase-0 tests from §4.7 against synthetic panels. Each test must be demonstrated to work in both directions: pass on a clean synthetic panel, and fail on a deliberately corrupted one. Write the corruption fixtures explicitly — a panel with a forward-shifted feature, one with val-fitted normalization, one with a dropped delisted ticker, one with misaligned labels.
>
> A test that only ever passes is not a test.

**Done when:** eight tests total — four clean-pass, four corrupt-fail.

---

## Session 3 — Data acquisition

**Goal:** real data on disk, cached, with the survivorship report.

**Prompt:**

> Read docs/data-sources-contract.md in full.
>
> Implement the three Protocol classes from §1, then `SP500Universe` (§2), `YFinancePrices` (§3), and `SECFundamentals` (§4). Then the four scripts in §6.
>
> Priorities in order: cache-first behavior, the validation assertions in §3.1, and the PIT `filed`-date rule in §4.1. The tag fallback resolution in §4.3 needs a coverage report — I want to see which concepts resolve for what fraction of the universe by year.
>
> Run the full pull. Produce reports/survivorship.md per §3.3 with actual measured numbers.

**Done when:** `panel.pkl` exists, all Phase-0 tests pass against it, survivorship report has real numbers.

**Expect this session to take the longest and break the most.** yfinance will throttle. Some tickers won't resolve. That's the work.

---

## Session 4 — Backtest engine

**Prompt:**

> Read docs/implementation-spec.md §5 and docs/output-layer-spec.md §2–3.
>
> Implement the `MetricValue` and `PhaseResult` contracts first — everything downstream renders through them.
>
> Then `backtest/construct.py`, `costs.py`, and `metrics.py` per §5.1–5.3. Every metric function returns a (gross, net) pair.
>
> Then the Phase 1 gate: build 12-1 momentum, run it through the engine, and show me the equity curve, Sharpe, and drawdown. I'm looking for a visible 2009 momentum crash. If it's not there, the engine is wrong.
>
> Build the Rich phase-panel renderer and `make status` from output-layer-spec §3.2 and §3.5.

**Done when:** momentum reproduces, `make status` shows phases 0–1 green.

---

## After Session 4

The remaining sequence follows the phase ladder — baselines, MASTER, ablations, factors, risk, join, terminal. By then the patterns are established and sessions get faster.

---

## Working notes

**Keep sessions to one phase.** Context degrades on long sessions and the gate structure exists to give you clean stopping points.

**Push back on the spec when it's wrong.** These docs were written before any code existed. If §4.3's feature list is impractical against actual yfinance output, say so and adjust — but log the deviation in `NOTES.md` with reasoning.

**Watch for the enthusiastic failure mode.** If a model produces a Sharpe above 2 or an IC above 0.08, do not celebrate — go find the bug. In this domain, good numbers are evidence of error until proven otherwise.

**The gates are the spec's actual value.** Everything else is detail that can flex. The gates are what keep you from spending three weeks on a transformer sitting on a broken backtest.
