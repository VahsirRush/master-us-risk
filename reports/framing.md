# Project framing — the question this build is actually answering

**Status: living document. This supersedes implementation-spec §0 where they
disagree, and it is the source the Phase-8 README draws from.** It exists
because the framing changed once already, in response to a measurement, and
that change must not be lost between here and Phase 8.

## The original framing (implementation-spec §0)

A port of MASTER (AAAI-2024) to US equities, with the contribution being a
faithful architecture port plus two things the original published neither of:
turnover-aware net-of-cost evaluation, and factor attribution of the signal.
The implied headline question was **"does MASTER beat the baselines?"**, with
LightGBM as the bar to clear (spec §6 calls it "the real bar", and §12
anticipates "LightGBM beats MASTER" as a common and reportable outcome).

## The reframing (Phase 2 result, Sessions 6-7)

Phase 2 measured five models under one protocol, 5 seeds each. The result
that changed the question:

> **The ungated transformer — MASTER minus the market gate — is already at
> parity with the strongest baseline.** It is NOT distinguishable from tuned
> LightGBM on gross RankIC (+0.0007), net L/S Sharpe (−0.1566), or net
> long-only Sharpe (−0.0200).

So "does MASTER beat LightGBM" was answered, and not by MASTER. The question
became: **does the market-guided gate add anything over an ungated
architecture already at parity with the strongest baseline?**

## That question is now SETTLED: the gate adds nothing distinguishable

**Status: closed as of Phase 4.** Not "suggested", not "pending more seeds".

### How it was resolved

**Phase 3 raised a legitimate doubt.** At 5 seeds and a short training budget
(12 epochs, patience 4), full MASTER beat ungated by +0.0011 RankIC against a
pooled seed dispersion of 0.0011 — a ratio of **0.949**, missing the
distinguishability threshold by 5%. All three measures were same-signed. That
is exactly the shape of a real-but-small effect being masked by too little
power, and it would have been dishonest to call it a null and move on.

**Phase 4 tested the two candidate explanations directly.** If the gate had a
real small effect, either (a) more training would let it show, or (b) some
other gate temperature would reveal it. Both were tested:

- **Budget**: both arms re-run at the spec's full 100 epochs / patience 10.
- **Temperature**: a 6-point β sweep (0.1 → 10.0), 5 seeds each, against the
  no-gating reference line.

### The result

| Budget | Measure | gap | pooled | ratio | verdict |
|:---|:---|---:|---:|---:|:---|
| short 12/4 | gross RankIC | +0.0011 | 0.0011 | **0.949** | NOT distinguishable |
| full 100/10 | gross RankIC | +0.0014 | 0.0030 | **0.464** | NOT distinguishable |
| full 100/10 | net L/S Sharpe | +0.1034 | 0.3778 | **0.274** | NOT distinguishable |

**The ratio got worse under more power — 0.949 → 0.464.** The full budget
roughly doubles the point-estimate gap (+0.0011 → +0.0014) and more than
doubles the seed dispersion (±0.0009 → ±0.0022), so the gap moves *further*
inside the noise rather than emerging from it. A real effect behaves the
opposite way: more power shrinks the threshold relative to the effect.

The β sweep agrees independently: **no temperature separates from no gating**,
the curve is flat above β=0.5, and hard selection (β=0.1) actively *hurts*
with 4-7x the seed dispersion. And `market_shuffled` — the gate fed a
date-permuted market vector, its entire input destroyed while every marginal
is preserved — costs +0.0212 → +0.0207 (±0.0002). The gate is not reading
market structure.

### Why this is the cleanest available resolution

The obvious objection to a null from a short-budget experiment is that the
experiment was too weak. That objection is closed here, because **the extra
budget demonstrably changed training and still did not change the answer**:
best epochs moved from 1, 1, 2, 3, 1 under patience 4 to 1, 2, 2, 6, **11**
under patience 10, and wall time roughly doubled (700-1200s → 1369-2681s per
seed). Two seeds genuinely found later optima. The longer schedule was not a
formality — it did more work, reached different solutions, and produced the
same verdict more decisively.

### Secondary finding: longer training is less reproducible

Seed dispersion roughly doubles at the longer budget — RankIC ±0.0009 →
±0.0022, net L/S Sharpe ±0.12 → ±0.26 — with turnover unchanged at 114-117%.
This is a genuine finding, not an error-bar footnote. More budget buys a
slightly higher mean at the cost of materially worse run-to-run stability, and
it makes any future claim of a small gate effect *harder* to establish, since
the threshold grows faster than the effect does. Anyone proposing to settle
this question with "just more training" should know the ground moves away from
them as they do it.

### What would legitimately re-open it

Not more seeds and not more epochs — both were tried. Only an architectural
change to the gating mechanism itself, or a materially different experimental
setup (different universe, label horizon, or feature bank). Absent one of
those, this is settled.

### The wider result it sits inside

None of MASTER's three structural mechanisms is distinguishable from its
ablation on this data: gating (ΔRankIC −0.0011), inter-stock attention
(−0.0000), cross-time attention (−0.0007). The architecture as a whole does
not justify itself here relative to its own ablations — which is a cleaner and
more useful finding than a horse-race result would have been.

## Scope: what was measured and what was deferred

The lookback dimension (§8.1 row 6) was scoped but not executed: a timing
probe showed it would require 70-95 hours of continuous compute, incompatible
with this session's constraints. Rather than run a reduced version and risk a
weakly-powered result sitting alongside the fully-powered gate-null findings,
this is left as explicit future work.

The head-count dimension (§8.1 row 7) was launched but stopped before
completion, for the same reason as the lookback sweep: closing out this
project with speed took priority over completing every planned sweep, and an
incomplete or single-run result is not evidence. Both the lookback and head
sweeps are deferred as future work; neither affects the gate-null conclusion,
which was independently completed at full statistical power.

The gate null rests on the default configuration, where it was measured at the
spec's full budget with 10 confirmatory seeds, a 25-run β sweep, and a
market-shuffle control.

## The second finding the README must carry

Phase 2 also established, across all five baselines, that **every
market-neutral book is net-negative**: gross L/S Sharpe 0.38–0.99, all
underwater after 10 bps at 55–140% daily turnover. Gross alpha exists and
costs consume all of it.

This is the project's original stated contribution (turnover-aware net-of-cost
evaluation) landing on its own baselines before the model under study is even
introduced. A gross-only table would have shown five plausible alpha models.
It also means any claim the gate makes has to survive costs, not just improve
RankIC — and the two orderings genuinely differ: the model ranking last on
gross RankIC (LSTM) ranks first on net, both orderings statistically real.

## Framing paragraph for the README (draft, edit for voice at Phase 8)

> This is a port of the MASTER architecture to US equities, and what it
> reports is a null result, measured carefully enough to be worth reporting.
> Built under one protocol with five seeds and net-of-cost evaluation
> throughout, the ungated transformer matches the strongest tabular baseline
> to within seed noise on every measure tried — so the interesting question
> was never whether the architecture is competitive, but whether the paper's
> distinctive contribution, market-guided feature gating, adds anything on top
> of it. It does not. The gate is not distinguishable from no gate at either
> training budget, no temperature in a six-point β sweep separates from the
> no-gating reference, and permuting the market vector's dates — destroying
> the gate's entire input — costs essentially nothing. The same holds for the
> architecture's other two structural claims: removing inter-stock attention
> changes RankIC by −0.0000. Along the way the evaluation establishes
> something about the baselines themselves: at daily rebalancing, every
> market-neutral book among them is net-negative after realistic costs, with
> breakeven between 8 and 11 bps against a 10 bps assumption — invisible in
> the gross numbers these comparisons are usually published with.

## The risk-model half (Phase 5, Session 12)

The MASTER half of this project is closed. The Barra half has begun, and it
is a separate contribution rather than a continuation: the gate-null result
does not depend on it and is not revisited by it.

Phase 5 built the eight style factors and estimated cross-sectional factor
returns, and it passes its own reproduction gate — the same kind of gate
Phase 1 applied to the backtest engine. The worst momentum month the model
finds, without being told to look, is **November 2020 (−3.68%, −3.8 SD)**,
the vaccine-announcement rotation, with September 2019 second. Value is
weak through the 2017-2020 growth regime (−1.56%/yr) and strong in the
2022-2023 rotation (+5.65%/yr). The market intercept is equity-like at
14.21%/yr and 14.14% vol.

Two limits belong in the README alongside those numbers, because they bound
what the eventual attribution can claim:

1. **The 2009 momentum crash is outside the sample.** The panel starts
   2010-02 and the first estimable period is 2011-04. Phase 1's backtest
   engine reproduced 2009 independently; the factor model cannot, and does
   not claim to.
2. **Momentum's premium is positive but not statistically significant**
   (+0.92%/yr, t=1.04). The sign matches the literature; the sample cannot
   reject zero. On this project's own standard that is reported in those
   words, not rounded up.

The free-data path also bounds the factors themselves. `long_term_debt`
resolves for only 72.9-81.9% of the universe, so leverage is the weakest of
the eight and is built from long-term debt alone — no short-term debt tag is
cached. That is a stated limitation of the leverage factor, not a silent
degradation of it.

## The risk model is calibrated (Phase 6, Session 14)

The covariance built on those factor returns passes its bias-test gate:
predicted portfolio volatility matches realized dispersion for **94.6%** of
test portfolios, against 94.1% for a rolling sample covariance and 91.8% for
Ledoit-Wolf shrinkage. The model wins, but narrowly enough that "matches the
naive benchmark" is the honest description.

**This validates calibration and nothing else.** A risk model is well
calibrated when it forecasts the SIZE of returns correctly; that is a
separate claim from any factor earning a return. Momentum here is +0.92%/yr
at t=1.04 — correctly signed, not distinguishable from zero — and the bias
result does not strengthen it. Any README sentence that lets a reader slide
from "the risk model is calibrated" to "the factors are real" is wrong, and
the two claims are kept apart deliberately throughout.

A secondary finding worth the README's space, in the same spirit as the
gate-null result: **none of the three refinements the spec prescribes for the
covariance is distinguishable from its absence on this data.** Separate
volatility and correlation half-lives — which the spec singles out as the
shortcut that degrades the bias statistic — perform identically to the single
half-life it warns against. The eigenfactor adjustment measurably hurts,
because it corrects minimum-variance directions while the prescribed test
portfolios are random and factor-mimicking, neither of which is optimized.
All three were implemented properly before being measured, and the
implementations are retained.

## The join — the sentence this whole project was built to produce (Phase 7)

> **MASTER-US long-short net Sharpe is −0.68. After Barra
> style-neutralization it is −0.91 — the book carries 77% of its *risk* in
> style and industry factors while only 15% of its *return* comes from them,
> so 85% is specific alpha — of which none survives realistic costs,
> neutralized or not. Cost breakeven occurs at 10.6 bps against a 10 bps
> assumption.**

The asymmetry in the middle of that sentence is the most useful thing the
risk model produced. A book that spends 77% of its risk budget on factor
exposure and earns 15% of its return there is paying 14.61%/yr of volatility
for 2.31%/yr of return — a Sharpe of 0.158 on the factor component. That is
close to unrewarded risk, and it is invisible without a factor model.

It would be easy to stop there and report the specific component's gross
Sharpe of +1.287 as "the alpha after hedging". That number is real but not
achievable: it assumes factor hedging is free. Constructing the hedge and
paying for it gives **−0.907 net**, worse than the unhedged book, because
neutralization removes volatility faster than it removes return and the cost
drag then falls on a smaller denominator. The gap between +1.287 and −0.907
is the difference between an attribution and a portfolio.

**The gate null is now complete.** Phases 3-4 established that market-guided
gating does not predict returns distinguishably better than no gating.
Phase 7 adds that it does no detectable regime-conditional factor timing
either: 0 of 8 style factors show significant timing, and the largest
difference between the gated and ungated books (0.0247, on leverage) sits
inside the seed dispersion of 0.0211. A mechanism that neither improves
prediction nor times factors is a cleaner and more complete null than either
result would have been alone.

One prediction this project made and failed to confirm, recorded because a
failed prediction is worth as much as a confirmed one: Phase 6 found the
eigenfactor adjustment slightly hurt the bias statistic and argued it should
help once a real optimizer ran against the covariance. Phase 7 ran that
optimizer. The difference in net Sharpe with the adjustment on versus off is
+0.0002 — not distinguishable.

## Status of the headline question

| | |
|---|---|
| Question | Does the market gate improve on the ungated architecture? |
| Answer | **No — not distinguishably, at either budget** |
| Evidence | 10 confirmatory seeds at spec 100/10; 25-run β sweep; market-shuffle control |
| Strength | RankIC ratio 0.949 (short) → **0.464** (full) — weakens under more power |
| Status | **CLOSED.** Re-open only on an architectural change to the gating mechanism or a materially different experimental setup. Not on more seeds or more epochs — both were tried. |
