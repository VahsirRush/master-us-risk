# The gate-null result — SETTLED at full budget

The question this project narrowed to (see `reports/framing.md`): does the
market-guided gate add anything over an ungated architecture already at parity
with the strongest baseline? The answer is no, and it is no at BOTH training
budgets — the longer one more decisively than the short one.

| Budget | Measure | MASTER | ungated | gap | pooled | ratio | verdict |
|:---|:---|---:|---:|---:|---:|---:|:---|
| short 12/4 | gross RankIC | +0.0212 ±0.0009 | +0.0201 ±0.0006 | +0.0011 | 0.0011 | **0.949** | NOT distinguishable |
| short 12/4 | net L/S Sharpe | -0.6806 ±0.1221 | -0.7383 ±0.0967 | +0.0577 | 0.1557 | **0.371** | NOT distinguishable |
| full 100/10 | gross RankIC | +0.0224 ±0.0022 | +0.0210 ±0.0020 | +0.0014 | 0.0030 | **0.464** | NOT distinguishable |
| full 100/10 | net L/S Sharpe | -0.4556 ±0.2611 | -0.5590 ±0.2731 | +0.1034 | 0.3778 | **0.274** | NOT distinguishable |

**The RankIC ratio falls from 0.949 to 0.464 when both models train to the spec's schedule.**
A ratio below 1.0 means the gap is inside pooled seed dispersion. Phase 3's
0.949 was an uncomfortable near-miss — 5% short of the
threshold — which left open the possibility that a real small effect was being
masked by too little training or too few seeds. It was not. Giving both arms
the full budget roughly doubles the point-estimate gap (+0.0011 → +0.0014) and
more than doubles the seed dispersion (±0.0009 → ±0.0022), so the gap moves
*further* inside the noise. This is a resolved null, not a near-miss.

## The extra budget was real — it just did not change the answer

The short-budget schedule was not quietly truncating training in a way that
hid the effect. Under patience 4 the best epochs were 1, 1, 2, 3, 1; under
patience 10 they were 1, 2, 2, 6, **11**, and wall time roughly doubled
(700-1200s → 1369-2681s per seed). Two seeds genuinely found later optima.
The longer schedule changed training and left the conclusion intact, which is
the cleanest available resolution: the Phase 3 decision to run short was not
the cause of the near-miss.

## Secondary finding: longer training is LESS reproducible

Seed dispersion roughly doubles at the longer budget — RankIC ±0.0009 →
±0.0022, net L/S Sharpe ±0.12 → ±0.26 — while turnover is unchanged at
114-117%. This is a finding in its own right, not a footnote about error bars:
more training buys a slightly higher mean at the cost of materially worse
run-to-run reproducibility. It also means any future claim of a small gate
effect gets HARDER to establish with more budget, not easier, because the
threshold it must clear grows faster than the effect does.

---

# Phase 4 — ablation grid, β sweep, cost analysis, stress tests

Read `reports/framing.md` first. The question this grid resolves is whether
the market gate does anything over an ungated architecture already at parity
with the strongest baseline.

All cells: 5 seeds, 12 epochs / patience 4 / lookback 20 — the same budget as
every deep row in the six-model table. See NOTES Session 9 for the decision.

## Ablation table

| Variant | n | RankIC | L/S Sharpe (gross → net) | L/S turn | Breakeven bps |
|:---|---:|---:|---:|---:|---:|
| Full MASTER | 5 | +0.0212 ±0.0009 | +0.77 → -0.68 | 116% | 10.6 |
| minus market gating | 5 | +0.0201 ±0.0006 | +0.71 → -0.74 | 117% | 9.8 |
| Full MASTER (100/10) | 5 | +0.0224 ±0.0022 | +0.91 → -0.46 | 114% | 13.4 |
| minus gating (100/10) | 5 | +0.0210 ±0.0020 | +0.87 → -0.56 | 117% | 12.2 |
| no_inter_stock | 5 | +0.0211 ±0.0014 | +0.76 → -0.62 | 113% | 11.0 |
| time_aligned | 5 | +0.0205 ±0.0007 | +0.82 → -0.58 | 117% | 11.8 |
| market_shuffled | 5 | +0.0207 ±0.0002 | +0.60 → -0.64 | 107% | 9.5 |
| beta_0.1 | 5 | +0.0183 ±0.0040 | +0.52 → -0.62 | 98% | 8.2 |
| beta_0.5 | 5 | +0.0201 ±0.0033 | +0.67 → -0.63 | 108% | 10.0 |
| beta_2.0 | 5 | +0.0209 ±0.0010 | +0.73 → -0.66 | 113% | 10.5 |
| beta_5.0 | 5 | +0.0207 ±0.0009 | +0.74 → -0.64 | 112% | 10.7 |
| beta_10.0 | 5 | +0.0207 ±0.0008 | +0.68 → -0.65 | 110% | 10.3 |

## Every variant vs full MASTER

| Variant | ΔRankIC | gross verdict | Δnet L/S Sharpe | net verdict |
|:---|---:|:---|---:|:---|
| minus market gating | -0.0011 | NOT distinguishable | -0.058 | NOT distinguishable |
| Full MASTER (100/10) | +0.0013 | NOT distinguishable | +0.225 | NOT distinguishable |
| minus gating (100/10) | -0.0001 | NOT distinguishable | +0.122 | NOT distinguishable |
| no_inter_stock | -0.0000 | NOT distinguishable | +0.064 | NOT distinguishable |
| time_aligned | -0.0007 | NOT distinguishable | +0.099 | NOT distinguishable |
| market_shuffled | -0.0004 | NOT distinguishable | +0.038 | NOT distinguishable |
| beta_0.1 | -0.0029 | NOT distinguishable | +0.062 | NOT distinguishable |
| beta_0.5 | -0.0010 | NOT distinguishable | +0.050 | NOT distinguishable |
| beta_2.0 | -0.0003 | NOT distinguishable | +0.025 | NOT distinguishable |
| beta_5.0 | -0.0005 | NOT distinguishable | +0.042 | NOT distinguishable |
| beta_10.0 | -0.0005 | NOT distinguishable | +0.035 | NOT distinguishable |

## β sweep (§8.2)

Horizontal reference = the no-gating variant.

| β | RankIC | vs ungated | verdict |
|---:|---:|---:|:---|
| 0.1 | +0.0183 ±0.0040 | -0.0018 | NOT distinguishable |
| 0.5 | +0.0201 ±0.0033 | +0.0000 | NOT distinguishable |
| 1.0 | +0.0212 ±0.0009 | +0.0011 | NOT distinguishable |
| 2.0 | +0.0209 ±0.0010 | +0.0008 | NOT distinguishable |
| 5.0 | +0.0207 ±0.0009 | +0.0006 | NOT distinguishable |
| 10.0 | +0.0207 ±0.0008 | +0.0006 | NOT distinguishable |
| — (no gating) | +0.0201 ±0.0006 | — | reference |

## §8.1 rows 6-7 — lookback and head sweeps

Both arms (gated / ungated) at every point, so each row answers not just
"does this hyperparameter matter" but "does the gate null survive here".
All at the full 100/10 budget, matching the confirmatory pair.

| Sweep point | gated RankIC | ungated RankIC | gap | ratio | verdict |
|:---|---:|---:|---:|---:|:---|
| lookback 20 / heads 8-4 (default) | +0.0224 ±0.0022 | +0.0210 ±0.0020 | +0.0014 | 0.464 | NOT distinguishable |
| lookback 40 | — | — | — | — | not yet run |
| lookback 60 | — | — | — | — | not yet run |
| lookback 120 | — | — | — | — | not yet run |
| heads 4/2 | — | — | — | — | not yet run |
| heads 8/8 | — | — | — | — | not yet run |
| heads 16/4 | — | — | — | — | not yet run |

*1 of 7 sweep points complete.*


## Stress tests (§8.5) — full MASTER vs ungated

### Full MASTER (seed-averaged scores)

| slice | RankIC |
|:---|---:|
| 2018Q4 selloff | outside test split (0 dates) |
| COVID crash | +0.0028 (62 dates) |
| 2022 bear | +0.0264 (251 dates) |
| 2023-2025 recovery | +0.0158 (751 dates) |
| cap tier: large | +0.0211 |
| cap tier: mid | +0.0201 |
| cap tier: small | +0.0257 |
| sector-neutralized | +0.0177 |
| decay h=1d | +0.0200 |
| decay h=3d | +0.0160 |
| decay h=5d | +0.0118 |
| decay h=10d | +0.0083 |
| decay h=21d | +0.0058 |

### minus market gating (seed-averaged scores)

| slice | RankIC |
|:---|---:|
| 2018Q4 selloff | outside test split (0 dates) |
| COVID crash | +0.0018 (62 dates) |
| 2022 bear | +0.0239 (251 dates) |
| 2023-2025 recovery | +0.0159 (751 dates) |
| cap tier: large | +0.0219 |
| cap tier: mid | +0.0204 |
| cap tier: small | +0.0249 |
| sector-neutralized | +0.0175 |
| decay h=1d | +0.0204 |
| decay h=3d | +0.0169 |
| decay h=5d | +0.0132 |
| decay h=10d | +0.0099 |
| decay h=21d | +0.0082 |


## Cost breakeven (§8.4)

bps at which net alpha reaches zero, dollar-neutral decile book.

| Variant | breakeven bps | net Sharpe @0 | @10 | @20 | @50 |
|:---|---:|---:|---:|---:|---:|
| Full MASTER | 10.6 | +0.77 | +0.05 | -0.68 | -2.88 |
| minus market gating | 9.8 | +0.71 | -0.01 | -0.74 | -2.95 |
| Full MASTER (100/10) | 13.4 | +0.91 | +0.23 | -0.46 | -2.53 |
| minus gating (100/10) | 12.2 | +0.87 | +0.16 | -0.56 | -2.73 |
| no_inter_stock | 11.0 | +0.76 | +0.08 | -0.62 | -2.71 |
| time_aligned | 11.8 | +0.82 | +0.12 | -0.58 | -2.72 |
| market_shuffled | 9.5 | +0.60 | -0.02 | -0.64 | -2.52 |
| beta_0.1 | 8.2 | +0.52 | -0.05 | -0.62 | -2.35 |
| beta_0.5 | 10.0 | +0.67 | +0.02 | -0.63 | -2.60 |
| beta_2.0 | 10.5 | +0.73 | +0.04 | -0.66 | -2.76 |
| beta_5.0 | 10.7 | +0.74 | +0.05 | -0.64 | -2.73 |
| beta_10.0 | 10.3 | +0.68 | +0.02 | -0.65 | -2.66 |
