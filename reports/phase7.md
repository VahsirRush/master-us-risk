# Phase 7 — the join

## Headline

> MASTER-US long-short net Sharpe is **-0.68**. After Barra style-neutralization it is **-0.91** — the book carries **77%** of its *risk* in style and industry factors while only **15%** of its *return* comes from them, so **85%** is specific alpha — of which **none survives realistic costs**, neutralized or not. Cost breakeven occurs at **10.6 bps** against a 10 bps assumption.

## Books (5 seeds, test 2019-2025)

| book | gross Sharpe | net Sharpe | turnover | breakeven |
|---|---:|---:|---:|---:|
| MASTER decile L/S | +0.768 ±0.130 | -0.681 ±0.122 | 116% | 10.6 bps |
| optimized, unconstrained | +0.189 ±0.103 | -0.624 ±0.120 | 30% | 2.3 bps |
| optimized, STYLE-NEUTRAL | +0.218 ±0.120 | -0.907 ±0.198 | 31% | 2.0 bps |
|   same, eigenfactor OFF | +0.218 ±0.120 | -0.907 ±0.198 | 31% | 2.0 bps |

The **unconstrained** row is a control, not a candidate strategy. Without it, comparing the decile book against the neutralized one would confound style-neutralization with the separate change from a decile rule to an optimizer. Both optimized arms run through the same function with identical lambdas, so the neutral-vs-unconstrained comparison isolates neutralization.

### Distinguishability (net Sharpe)

- MASTER decile L/S vs zero: **distinguishable**
- optimized, unconstrained vs zero: **distinguishable**
- optimized, STYLE-NEUTRAL vs zero: **distinguishable**
- same, eigenfactor OFF vs zero: **distinguishable**
- neutral vs unconstrained: **distinguishable**

## §10.2 Return attribution (gross, annualized)

- total realized **+15.66%/yr**
- factor contribution **+2.31%/yr** (14.8% of return)
- specific (alpha) **+13.35%/yr** (85.2% of return), gross Sharpe +1.28

> **The +1.28 gross specific Sharpe is not an alpha number and must not be quoted as one.** This is NOT an alpha number and must not be quoted as one. It is the return left after factor exposure is removed arithmetically, which assumes factor hedging is free. It is not achievable. Constructing the hedge and paying for it is the style-neutral book, which is WORSE than the unhedged book. The gap between the two is the difference between an attribution and a portfolio. Here that hedged book is the style-neutral arm above, at **-0.907 net**.

## §10.3 Risk attribution

- factor share of predicted variance **76.8%**, specific **23.2%**

**The asymmetry is the finding**: the book spends most of its risk budget on factor exposure and earns almost none of its return there — **14.61%/yr of volatility to earn +2.31%/yr**, a Sharpe of +0.163 on the factor component. That is close to unrewarded risk, and it is invisible without a factor model.

## Eigenfactor prediction check

Net Sharpe gap **+0.0002**, **NOT distinguishable**. Phase 6 found the eigenfactor adjustment slightly hurt the bias statistic and argued it should help once a real optimizer ran against the covariance, because the adjustment corrects minimum-variance directions while Phase 6's test portfolios were random and factor-mimicking, neither optimized. Phase 7 ran that optimizer. The prediction is NOT confirmed, and is recorded as a failed prediction rather than explained away.

## §10.5 Gate interpretation — factor timing

**Method note.** §10.5 specifies regressing the learned gate activations on the market state vector and on contemporaneous factor returns. No Phase-3 checkpoint was saved and no activations were cached, so the activations are not available. What is measured here instead is the gate's CONSEQUENCE: whether the gated book times factors better than the ungated one. The two models share seeds, data and protocol and differ only in the gate, so the difference is attributable to it. This is a SUBSTITUTE for the specified activation regression, not an implementation of it.

| factor | MASTER | ungated | gate delta | t (gated) |
|---|---:|---:|---:|---:|
| size | +0.0240 | +0.0064 | +0.0176 | +1.01 |
| value | +0.0190 | +0.0195 | -0.0005 | +0.80 |
| momentum | +0.0163 | +0.0206 | -0.0043 | +0.69 |
| volatility | +0.0168 | +0.0132 | +0.0036 | +0.71 |
| liquidity | +0.0014 | +0.0042 | -0.0028 | +0.06 |
| leverage | +0.0269 | +0.0022 | +0.0247 | +1.13 |
| growth | -0.0348 | -0.0222 | -0.0126 | -1.46 |
| quality | -0.0207 | -0.0176 | -0.0031 | -0.87 |

**0/8 factors show significant timing.** Largest gate delta 0.0247 against seed dispersion 0.0211 — inside the noise. The gate does no detectable regime-conditional factor timing.
