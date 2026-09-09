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

## Every variant vs full MASTER

| Variant | ΔRankIC | gross verdict | Δnet L/S Sharpe | net verdict |
|:---|---:|:---|---:|:---|
| minus market gating | -0.0011 | NOT distinguishable | -0.058 | NOT distinguishable |

## β sweep (§8.2)

Horizontal reference = the no-gating variant.

| β | RankIC | vs ungated | verdict |
|---:|---:|---:|:---|
| 1.0 | +0.0212 ±0.0009 | +0.0011 | NOT distinguishable |
| — (no gating) | +0.0201 ±0.0006 | — | reference |

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
