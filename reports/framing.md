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
that changes the question:

> **The ungated transformer — MASTER minus the market gate — is already at
> parity with the strongest baseline.** It is NOT distinguishable from tuned
> LightGBM on gross RankIC (+0.0007), net L/S Sharpe (−0.1566), or net
> long-only Sharpe (−0.0200). Three measures, three verdicts of "not
> distinguishable from seed noise."

So "does MASTER beat LightGBM" is no longer the interesting question. The
architecture-minus-the-gate answers it already, to a draw. What remains open,
and what this project is now built to answer:

> **Does the market-guided gate add anything over an ungated architecture that
> is already at parity with the strongest baseline?**

## What follows from the reframing

1. **The gate ablation and the β sweep are the load-bearing result of the
   whole project**, not one cell in a Phase-4 grid. Everything else —
   the data layer, the engine, the baselines — is scaffolding that exists to
   make that one comparison trustworthy.
2. **The control is the `ungated` row, not the LightGBM row.** MASTER has to
   beat the architecture it is a superset of. Beating LightGBM would prove
   nothing that the ungated model has not already proven.
3. **The comparison must be exact.** Full MASTER and ungated differ by
   `use_gate` and nothing else: same layers, widths, optimizer, schedule,
   seeds, and training budget. A gate result contaminated by a longer
   schedule or a wider model would be worthless.
4. **A null result here is the finding.** If the gate is not distinguishable
   from ungated, that is a publishable, honest answer about whether the
   paper's central mechanism transfers to US equities — and per CLAUDE.md it
   must be reported in those words, not buried.

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

> This is a port of the MASTER architecture to US equities, and the question
> it answers is narrower and more useful than "does the model work." Built
> under one protocol with five seeds and net-of-cost evaluation throughout,
> the ungated transformer turns out to match the strongest tabular baseline
> to within seed noise on every measure tried. The open question is therefore
> not whether the architecture is competitive, but whether the paper's
> distinctive contribution — market-guided feature gating — adds anything on
> top of it. That is what the gate ablation and β sweep here measure, and the
> answer is reported whichever way it falls. Along the way the evaluation
> establishes something about the baselines themselves: at daily rebalancing,
> every market-neutral book among them is net-negative after realistic costs,
> which is invisible in the gross numbers these comparisons are usually
> published with.
