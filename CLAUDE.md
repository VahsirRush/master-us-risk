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

**Phase: 4 COMPLETE (Session 10). Phase-8 output layer built (Session 11). Phases 5-7 COMPLETE (Sessions 12-15) — risk model built, bias gate PASS at 94.6%, and the join done. **All measurement phases are finished.** Next: Phase 8 (reports/README, §13).**

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

## OPEN (not settled)

- **Lookback sweep (§8.1 row 6, L=40/60/120)**: not run. Projected 70-95h compute. See NOTES for detail.
- **Head sweep (§8.1 row 7, 3 configs × 2 arms × 5 seeds = 30 runs)**: launched, stopped after 1/30 runs. Projected ~12.2h compute. **No results exist — do not treat the single completed run as indicative of anything.** Its score file was deleted and `41_phase4_report.py` now refuses any variant with fewer than 5 seeds, so a partial cell cannot be silently averaged into a table. See NOTES for detail.

**Neither open item affects the gate-null conclusion (see above), which is independently closed and does not depend on either sweep completing.**

(Numbering: §8.1 row 6 is lookback, row 7 is the head grid.)

## Research terminal (Session 11)

The Phase-8 output layer is built: `terminal/`, a Vite + React static export
following `docs/research-terminal-spec.md` §3+ with an Apple/iOS dark visual
language replacing §2. Panels MONITOR / QUOTE / ABLA / COST / DATA are live;
RISK and ATTR render a "PHASE 5-7 · NOT YET RUN" empty state driven by the
payload's `pending_panels`, and must stay that way until those phases produce
numbers.

```
~/.venvs/master-us/bin/python scripts/50_export_terminal.py   # -> terminal/public/results.json
cd terminal && npm run build && npm run preview               # http://localhost:4173
npm run smoke                                                 # headless render check, 20 assertions
~/.venvs/master-us/bin/python scripts/51_bundle_single.py     # one self-contained .html
```

**Re-run the export after any change to a score array or a PhaseResult** —
`results.json` is a cache, and the frontend computes nothing, so a stale blob
is the only way a wrong number can reach the screen.

**`npm` binaries must be invoked as `node node_modules/<pkg>/...`.** The
project directory name contains a colon, `PATH` is colon-separated, and npm's
`node_modules/.bin` entry therefore splits into two broken paths — every
binary resolves as "command not found". Same hazard as the venv path note
below.

**The single-file target must stay IIFE** (`vite.config.single.ts`). Chrome
fetches `<script type="module">` with CORS and a `file://` page has a null
origin, so an ESM bundle opens to a blank page.

## Phase 5 — risk model, descriptors and factor returns (Session 12)

`src/master_us/risk/` implements spec §9.1-9.3. **Gate PASS**: the factor
returns reproduce known published patterns.

```
~/.venvs/master-us/bin/python scripts/60_build_risk_model.py   # --rebuild to ignore the cache
```

Monthly (177 estimated periods, mean 371 names, weighted R2 0.257): market
+14.21%/yr, momentum +0.92% (t=1.04), value +0.83%. Weekly variant agrees.
Worst momentum month is **2020-11 at -3.68% (-3.8 SD)** — the vaccine
rotation, found independently, with 2019-09 second.

These are the post-`exposure_lag=1` figures and they match `reports/phase5.md`
and `reports/status/phase5.json` exactly. The pre-fix values are superseded;
NOTES Session 14 records the before/after if the size of the correction is
ever needed.

**Two things the gate does NOT establish.** The **2009 momentum crash is
outside the sample** (panel starts 2010-02, first estimable period 2011-04),
so unlike Phase 1's backtest engine this cannot reproduce it. And momentum's
premium is **positive but NOT significant** (t=1.04) — the sign matches the
literature, the sample cannot reject zero.

**`long_term_debt` is the weak concept**, 72.9-81.9% tag resolution on the
universe, so `debt_equity` (64.6%) and `earnings_var` (57.2%) are the least
reliable descriptors. Leverage is built but should not be leaned on.
Measurement trap: run over ALL SEC filers rather than the universe it reads
40-67% and looks like a failure — **always restrict to the universe before
comparing to Phase 0's table.**

**Missing exposures are imputed to zero, not dropped** (`MIN_FACTOR_FRACTION
= 0.75`). Requiring all eight factors produced no regression at all before
2013. Not free: momentum falls +1.69% -> +1.01% on a matched window as the
fraction relaxes. No conclusion turns on it.

## The yfinance-shares placeholder does not exist — do not go looking for it

A future session may be told `Panel.metadata` mcap is a yfinance placeholder
awaiting an SEC join. **It is not, and never was.** `metadata.py` has one
commit ever (cb3f8ce, Session 6) and it reads cached XBRL facts joined as-of
on `filed`, with split-basis harmonization. Verified in Session 12 by grep,
by git log, and by `git diff --quiet`. Nothing was swapped; nothing
previously reported changed.

What WAS provisional and is now fixed: sector (17.0% "Unknown" — every
departed name) and multi-class double-counting (4 pairs). What remains:
**mcap is NaN on 11.6% of tradeable cells**, structural from early XBRL
adoption, not the 400d staleness cap (relaxing to 1100d moves it to 11.0%).

## Industry classification is a hybrid, by measurement

`risk/industry.py` takes **GICS where known (485 names) and SEC SIC for the
99 it cannot cover** (departed names absent from today's Wikipedia table);
100% coverage, mechanism recorded per name in `industry_source`. A
SIC-for-everything scheme was tried first and rejected: it agrees with real
GICS only **77.5%** of the time. Do not "simplify" this back to one scheme.

## Phase 6 — covariance and bias tests (Session 14)

`risk/{covariance,bias_tests,forecast}.py` implement §9.4-9.5. **Gate PASS:
bias statistic in [0.9, 1.1] for 94.6% of portfolios** (random 1.004,
factor-mimicking 0.939, cap-weighted market 1.072), against sample covariance
94.1% and Ledoit-Wolf 91.8%.

```
~/.venvs/master-us/bin/python scripts/61_risk_covariance.py   # --rebuild to re-estimate daily
```

**THE GATE VALIDATES CALIBRATION, NOT FACTOR STRENGTH.** It shows predicted
vol matches realized dispersion. It says nothing about whether any factor
earns a return. Momentum is +0.92%/yr at t=1.04 — right sign, not
distinguishable from zero — and 94.6% does not change that. Never quote the
bias gate as evidence for a factor, in NOTES, framing, or dashboard copy.

**None of §9.4's three refinements is distinguishable on this data.**
Separate half-lives 95.2% vs single-HL 95.2% (the spec's claimed degradation
does NOT reproduce); Newey-West no effect on the headline; the eigenfactor
adjustment measurably HURTS (95.2% -> 94.6%). The eigen implementation is
correct — it detects the textbook dispersion pattern — but it targets
minimum-variance directions and §9.5's portfolios are random and
factor-mimicking, neither optimized. **Expect it to matter in Phase 7 if the
join optimizes.** Shipped config follows the spec regardless; see NOTES.

**A bias statistic can fail for HARNESS reasons that look like model
reasons.** The first run read 49.9% (fail). Cause was `run_bias_tests`
charging predicted variance for names with no realized return that period —
only ~64% of names are in the regression daily, inflating predicted variance
by ~1.56x. The tell was two internally inconsistent measurements (specific
risk under-forecast per name at 1.066, over-forecast per portfolio at 0.820),
not the failing number. Regression test:
`test_missing_names_do_not_inflate_predicted_vol`.

## Phase 7 — the join (Session 15)

`experiments/join.py` implements §10. The headline, filled in:

> MASTER-US long-short net Sharpe is **-0.68**. After Barra
> style-neutralization it is **-0.91** — the book carries **77%** of its
> *risk* in style and industry factors while only **15%** of its *return*
> comes from them, so **85%** is specific alpha — of which **none survives
> realistic costs**, neutralized or not. Cost breakeven occurs at
> **10.6 bps** against a 10 bps assumption.

```
nohup caffeinate -dimsu ~/.venvs/master-us/bin/python scripts/70_join.py \
    --eigen-check --pidfile /tmp/join.pid > /tmp/join.log 2>&1 & disown
~/.venvs/master-us/bin/python scripts/71_join_report.py
```

**The asymmetry is the finding.** 77% of risk in factors, 15% of return from
them — factor exposure costs 14.61%/yr of vol to earn 2.31%/yr (Sharpe
0.158). Largest exposure is volatility at **-0.88**, a short-high-vol tilt.

**Neutralization makes net WORSE** (-0.68 -> -0.91, distinguishable). It
removes volatility faster than return, so the same cost drag becomes a
larger fraction of a smaller denominator. The unconstrained optimized arm
(-0.624) is the control that keeps this from confounding neutralization with
the switch to an optimizer.

**Gross specific-alpha Sharpe is +1.287 and that is NOT achievable** — it
assumes free factor hedging. The neutralized book is the achievable version
at -0.907 net. Do not quote +1.287 as an alpha.

**The Session 14 eigenfactor prediction is NOT CONFIRMED.** It predicted the
adjustment would matter once something optimized against the covariance;
the net Sharpe gap on vs off is **+0.0002, not distinguishable**. Recorded
as a failed prediction, not explained away.

**§10.5 gate interpretation: a clean null.** 0 of 8 factors show significant
timing; largest gate delta 0.0247 (leverage) against seed dispersion 0.0211.
No Phase-3 checkpoint exists, so this measures the gate's CONSEQUENCE
(gated vs ungated book timing), not the activations themselves — stated as a
substitute, not an implementation, of the spec's activation regression.
**This completes the gate null**: the gate neither predicts returns better
nor times factors.

## Exposure/return alignment — `exposure_lag=1`, do not revert

`pipeline.estimate()` reads exposures at the last trading day BEFORE the
return window opens. Session 12 read them at the window's first day, putting
day-t information on both sides (`vol_60d`, `beta`, `log_mcap` all use data
through t). Harmless-ish monthly, fatal daily. Every Phase 5 figure was
restated as a result — the Phase 5 section above carries the corrected set,
and NOTES Session 14 has the before/after table. The gate still passes on all
six checks.

## Test suite concurrency guard (narrowed, Session 13)

`tests/conftest.py` no longer aborts the whole session when a training job is
live. It skips only tests marked `heavy` — the one module that loads the
1.2 GB panel (`test_real_panel_gates.py`, 9 tests). Everything else runs,
including the two tests that would catch a Phase-6 regression in the
factor-return estimation path (`test_recovers_known_factor_returns`,
`test_industry_constraint_holds`). Verified against a simulated RUNNING
heartbeat: 336 passed / 9 skipped, both critical tests passing.

Mark any NEW test that loads the real panel with `pytestmark =
pytest.mark.heavy`, or it will run beside a training job and can OOM-kill it.

## Also open

- **Phases 6-7**: covariance + bias tests (§9.4-9.5), attribution, the join.
  The terminal's RISK and ATTR panels are wired and waiting on these.
- **`Panel.metadata` sector is still the old current-GICS column.** The
  Phase-5 risk layer does NOT read it — it builds its own industry from
  `risk/industry.py`. A future panel rebuild should adopt the same resolver.

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
