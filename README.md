# MASTER-US

A port of the MASTER architecture (AAAI-2024, *Market-Guided Stock
Transformer*) to US equities, evaluated net of costs and attributed against a
Barra-style multi-factor risk model built from scratch in the same repository.

What it reports is a null result, measured carefully enough to be worth
reporting. The paper's distinctive contribution — market-guided feature
gating — does not improve on the same architecture with the gate removed, at
either training budget, at any gate temperature, and with or without the
market vector's information intact. Along the way the evaluation establishes
something about the baselines themselves: at daily rebalancing, every
market-neutral book among them is net-negative after realistic costs.

---

## The two findings

**1. The market gate is not distinguishable from no gate.**

The ungated transformer — MASTER minus the market gate — is already at parity
with tuned LightGBM, the strongest baseline. So the question was never whether
the architecture is competitive; it was whether the gate adds anything on top.
It does not, and the answer got *more* decisive with more statistical power.

| Budget | Measure | MASTER | ungated | gap | pooled σ | ratio | verdict |
|:---|:---|---:|---:|---:|---:|---:|:---|
| short 12/4 | gross RankIC | +0.0212 ±0.0009 | +0.0201 ±0.0006 | +0.0011 | 0.0011 | 0.949 | not distinguishable |
| **full 100/10** | gross RankIC | +0.0224 ±0.0022 | +0.0210 ±0.0020 | +0.0014 | 0.0030 | **0.464** | not distinguishable |
| **full 100/10** | net L/S Sharpe | −0.4556 ±0.2611 | −0.5590 ±0.2731 | +0.1034 | 0.3778 | **0.274** | not distinguishable |

A ratio below 1.0 means the gap sits inside pooled seed dispersion. The short
budget produced a 0.949 — a 5% near-miss, exactly the shape of a real-but-small
effect masked by too little power, and it would have been dishonest to call
that a null. Running both arms at the spec's full schedule roughly doubled the
gap and more than doubled the dispersion, so the ratio *fell* to 0.464. A real
effect behaves the opposite way.

Three independent controls agree:

- **No gate temperature separates from no gating.** A 25-run β sweep from 0.1
  to 10.0 is flat above β=0.5; hard selection at β=0.1 actively hurts, at 4–7x
  the seed dispersion.
- **Destroying the gate's input costs almost nothing.** Feeding it a
  date-permuted market vector — every marginal preserved, all timing
  information gone — moves RankIC +0.0212 → +0.0207 ±0.0002. The gate is not
  reading market structure.
- **The gate does no factor timing either.** 0 of 8 style factors show
  significant regime-conditional timing; the largest gated-vs-ungated
  difference (0.0247, leverage) sits inside seed dispersion of 0.0211.

The result generalizes past the gate: **none of MASTER's three structural
mechanisms is distinguishable from its own ablation** — gating −0.0011,
inter-stock attention −0.0000, cross-time attention −0.0007.

**2. The signal's risk lives in factors; its return does not. Neither survives costs.**

> MASTER-US long-short net Sharpe is −0.68. After Barra
> style-neutralization it is −0.91 — the book carries 77% of its
> *risk* in style and industry factors while only 15% of its *return*
> comes from them, so 85% is specific alpha — of which **none survives
> realistic costs, neutralized or not. Cost breakeven occurs at **10.6 bps**
> against a 10 bps assumption.

The asymmetry in the middle is the most useful thing the risk model produced.
The book spends 14.61%/yr of volatility on factor exposure to earn 2.31%/yr
from it — a Sharpe of 0.158 on the factor component, close to unrewarded risk,
and invisible without a factor model. The largest single exposure is
volatility at −0.88, a persistent short-high-vol tilt.

Neutralizing that exposure makes the net result *worse*, not better, because
it removes volatility faster than it removes return and the same cost drag then
falls on a smaller denominator.

---

## Results

### The model table

Test 2019–2025, 5 seeds each, net = flat 10 bps, daily rebalancing. All deep
rows at a matched 12-epoch / patience-4 budget, so every comparison is
controlled.

| Model | RankIC | ICIR | L/S Sharpe (gross → net) | L/S turn | Long-only (gross → net) | LO turn |
|:---|---:|---:|---:|---:|---:|---:|
| Ridge | +0.0162 ±0.0000 | +0.110 | +0.99 → **−0.89** | 129% | +1.01 → +0.45 | 56% |
| LightGBM (spec config) | +0.0185 ±0.0005 | +0.147 | +0.95 → **−1.17** | 140% | +0.94 → +0.31 | 68% |
| LightGBM (tuned) | +0.0207 ±0.0007 | +0.142 | +0.83 → **−0.89** | 134% | +0.98 → +0.40 | 64% |
| LSTM | +0.0139 ±0.0034 | +0.074 | +0.38 → **−0.28** | 55% | +0.87 → +0.65 | 21% |
| Ungated transformer | +0.0201 ±0.0006 | +0.112 | +0.71 → **−0.74** | 117% | +0.94 → +0.42 | 51% |
| **MASTER (gated)** | **+0.0212 ±0.0009** | +0.122 | +0.77 → **−0.68** | 116% | +1.01 → +0.44 | 54% |

Three things in this table needed the seed-dispersion machinery to state, and
would have been invisible without it:

1. **The gate metric and the thesis metric rank the models differently, and
   both orderings are statistically real.** On gross RankIC: tuned LGBM >
   ungated > spec LGBM > Ridge > LSTM. On net L/S Sharpe: LSTM > Ridge ≈ tuned
   LGBM > ungated > spec LGBM. The model ranking *last* on the signal metric
   ranks *first* net of costs — and its advantage comes from turnover (55% vs
   117–140%), not from a better signal.
2. **Passing the RankIC gate bought nothing net.** Tuned LightGBM beats Ridge
   distinguishably on gross RankIC and is not distinguishable from it on either
   net measure.
3. **The LSTM is underdetermined**, at ±0.0034 RankIC dispersion — roughly 5x
   LightGBM's. It cannot be told apart from linear ridge on the signal metric.

### Cost is the binding constraint, not signal

Every market-neutral book above is net-negative: gross L/S Sharpe 0.38–0.99,
all underwater after 10 bps at 55–140% one-way daily turnover. Breakeven lands
between 8 and 11 bps against a 10 bps assumption. The long-only column
stays positive only because it carries market beta.

This is the project's stated contribution landing on its own baselines before
the model under study is introduced. A gross-only table — the way these
comparisons are usually published — would have shown five plausible alpha
models.

### The risk model

Eight style factors plus industry, estimated by cross-sectional WLS.
**Gate: PASS** — the factor returns reproduce known published patterns without
being told to look for them.

Monthly, 177 estimated periods, mean 371 names, weighted R² 0.257:

| | annualized |
|:---|---:|
| market | +14.21% |
| momentum | +0.92% (t=1.04) |
| value | +0.83% |

The worst momentum month the model finds is November 2020 at −3.68%
(−3.8 SD) — the vaccine-announcement rotation — with September 2019 second.
Value is weak through the 2017–2020 growth regime (−1.56%/yr) and strong in the
2022–2023 rotation (+5.65%/yr).

The covariance built on those returns is well calibrated: the bias
statistic falls in [0.9, 1.1] for 94.6% of test portfolios, against 94.1%
for a rolling sample covariance and 91.8% for Ledoit-Wolf shrinkage. The model
wins, but narrowly enough that "matches the naive benchmark" is the honest
description.

### The join

5 seeds, test 2019–2025.

| book | gross Sharpe | net Sharpe | turnover | breakeven |
|:---|---:|---:|---:|---:|
| MASTER decile L/S | +0.768 ±0.130 | −0.681 ±0.122 | 116% | 10.6 bps |
| optimized, unconstrained | +0.189 ±0.103 | −0.624 ±0.120 | 30% | 2.3 bps |
| optimized, style-neutral | +0.218 ±0.120 | −0.907 ±0.198 | 31% | 2.0 bps |

The unconstrained row is a control, not a candidate strategy. Without it,
comparing the decile book against the neutralized one would confound
style-neutralization with the separate change from a decile rule to an
optimizer. Both optimized arms run through the same function with identical
penalties, so the neutral-vs-unconstrained comparison isolates neutralization.

Return attribution, gross annualized: total 15.66%, factor +2.31%
(14.8%), specific +13.35% (85.2%). Risk attribution: factor share of
predicted variance 76.8%, specific 23.2%.

> The specific component's gross Sharpe of +1.287 is not an alpha number and
> is not quoted as one here. It is what remains after factor exposure is
> removed arithmetically, which assumes factor hedging is free. Constructing
> the hedge and paying for it is the style-neutral book, at −0.907 net. The gap
> between +1.287 and −0.907 is the difference between an attribution and a
> portfolio.

---

## Method

The protocol is fixed before any model runs and is identical across every row
of every table.

| | |
|:---|:---|
| Universe | S&P 500 historical constituents, point-in-time membership |
| Panel | 4,004 dates (2010-02-01 → 2025-12-30) × 584 tickers × 130 features |
| Market vector | 67 dimensions, per spec §4.4 |
| Splits | train ≤2016 · validation 2017–2018 · test 2019–2025 |
| Embargo | 21 days between splits |
| Seeds | ≥5 per condition; 10 for the confirmatory gate pair |
| Costs | flat 10 bps one-way, applied to every book |
| Data | yfinance prices + SEC XBRL fundamentals (free path) |

Five rules the code enforces mechanically rather than by convention:

1. **No look-ahead.** Fundamentals join on the SEC `filed` date, never
   `period_end`. Features at `t` use only data known at `t`.
2. **Normalization statistics come from training only.** Validation and test
   are transformed with borrowed statistics; refitting on them is leakage.
3. **Test is touched once per model**, at the end. No shuffling, no k-fold.
4. **A gap smaller than pooled seed dispersion is reported as "not
   distinguishable"** — in those words, computed, never eyeballed.
5. **Every metric is a (gross, net) pair**, and turnover is a permanent
   column.

Phase gates are blocking: no transformer work happened before the backtest
engine reproduced 12-1 momentum including the 2009 crash, which it does — the
March–September 2009 window returns −106.6% gross with a −123.6% within-window
drawdown, and the engine independently surfaced the 2020 spike-down as a second
crash.

---

## Limitations

Stated here rather than buried, because several of them bound what the numbers
above can claim.

Survivorship bias is present, measured, and not correctable on the free
path. yfinance serves only currently-listed symbols, so names that stopped
trading are absent from their *entire* history. 613 of 795 historical
constituents (77.1%) are retrievable; 182 are not. The retrieval rate climbs
monotonically from 73.0% in 2010 to 97.7% in 2025, so the early sample is the
most contaminated — and the early sample is the training set.

The direction is upward but less safely so than usual. The largest missing
bucket is 120 acquired-or-merged names, which typically ended at a takeover
premium and whose absence biases performance *downward*; the
conditioning-on-survival channel dominates and points up. Both are named in
`reports/survivorship.md` rather than one being asserted. A CRSP or Sharadar
upgrade would eliminate this, and the `PriceSource` Protocol exists so that
swap is a drop-in.

**The 2009 momentum crash is outside the risk model's sample.** The panel
starts 2010-02 and the first estimable factor period is 2011-04. The backtest
engine reproduced 2009 independently; the factor model cannot, and does not
claim to.

**Momentum's premium is positive but not statistically significant**
(+0.92%/yr, t=1.04). The sign matches the literature; the sample cannot reject
zero, and it is reported that way.

**Leverage is the weakest of the eight factors.** `long_term_debt` resolves for
only 72.9–81.9% of the universe, making `debt_equity` (64.6%) and
`earnings_var` (57.2%) the least reliable descriptors. Leverage is built but
should not be leaned on. Market cap is NaN on 11.6% of tradable cells, which
is structural to early XBRL adoption rather than a staleness-cap artifact.

**Two ablation sweeps are scoped but deliberately not run.** The lookback
dimension (L=40/60/120) needs 70–95 hours of continuous compute; the head-count
grid needs ~12 hours and was stopped after 1 of 30 runs. Rather than publish a
weakly-powered result beside the fully-powered ones, both are left as explicit
future work — and the report generator refuses any variant with fewer than 5
seeds, so a partial cell cannot be silently averaged into a table. Neither
affects the gate-null conclusion, which was completed independently at full
power.

**One prediction this project made and failed to confirm**, recorded because a
failed prediction is worth as much as a confirmed one: the eigenfactor
covariance adjustment slightly hurt the bias statistic, and the argument was
that it should help once a real optimizer ran against the covariance. The
optimizer ran. The net Sharpe difference with the adjustment on versus off is
+0.0002 — not distinguishable.

---

## Layout

```
src/master_us/          all logic — scripts and notebooks are thin callers
├── data/               sources, features, market vector, panel assembly
├── models/             MASTER, ungated transformer, LSTM, tabular baselines
├── backtest/           portfolio construction, costs, turnover
├── risk/               Barra-style descriptors, factor returns, covariance
├── experiments/        phase drivers, ablations, stress tests, the join
├── reporting/          MetricValue / PhaseResult, distinguishability
└── terminal/           results payload construction
scripts/                numbered thin callers, 00 → 71
docs/                   the four specifications this build follows
reports/                per-phase results, figures, status JSON
terminal/               Vite + React results dashboard
tests/                  387 tests
```

`docs/project-conventions.md` is the engineering contract — the rules above,
plus the standing decisions that were settled by measurement and should not be
silently reverted. `NOTES.md` is the lab notebook: what was built each phase,
the gate result, what broke, and every deviation from spec with its reasoning.
`reports/framing.md` records how the headline question changed in response to a
measurement, which is the single most important piece of context for reading
the results.

## Reproducing

```bash
python3.11 -m venv ~/.venvs/master-us
~/.venvs/master-us/bin/pip install -e ".[dev]"

# The SEC requires a declared agent with a real name and email.
export MASTER_US_SEC_USER_AGENT="Your Name <you@domain.com>"

make status        # confirm every phase gate still passes
make test          # 387 tests
```

Data acquisition and training are driven by the numbered scripts in order;
Every stage is cache-first, so re-running is cheap and only fetches on a miss.
Raw pulls under `data/raw/` are immutable. Training is long enough that it
should be launched detached — see `docs/project-conventions.md`.

`lightgbm` needs `brew install libomp` on macOS; the pip wheel alone fails to
load.

## Research terminal

`terminal/` is a static results dashboard — Vite + React, no backend, and it
computes no statistics. Every panel is a pure function of an exported payload,
so any number on screen traces to an array under `data/processed/`. It also
builds to one self-contained HTML file that opens with no network.

```bash
~/.venvs/master-us/bin/python scripts/50_export_terminal.py
cd terminal && npm run build && npm run preview
```

Three honesty constraints are enforced structurally rather than by convention:
A delta inside pooled seed dispersion renders struck-through and grey, so noise
cannot be scanned as a result; the deferred sweeps appear only as deferred
entries carrying their projected compute, with no chart implying data exists;
and the export refuses to emit any cell with fewer than 5 seeds.

## License

MIT — see [LICENSE](LICENSE).
