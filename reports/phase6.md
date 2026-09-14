# Phase 6 — factor covariance and bias tests

## What this gate does and does not establish

It validates **calibration**: predicted portfolio volatility matches
realized dispersion. It says nothing about whether any factor earns a
return. Phase 5's momentum premium is +0.92%/yr at t=1.04 — right sign,
not distinguishable from zero — and a good bias statistic here does not
change that. The two claims are independent.

## Bias statistics (median by portfolio family)

```
estimator                           random    factor_mimicking cap_weighted_market  in [0.9,1.1]
------------------------------------------------------------------------------------------------
MASTER-US (full)                     1.004               0.939               1.072         94.6%
MASTER-US (no eigen)                 1.010               0.960               1.056         95.2%
single-HL shortcut                   1.012               0.964               1.060         95.2%
no Newey-West                        1.004               0.951               1.011         95.2%
sample cov (252d)                    1.011               0.963               1.093         94.1%
Ledoit-Wolf (252d)                   0.989               0.935               1.119         91.8%
```

**Gate: 94.6% of portfolios inside [0.9, 1.1] — PASS**

Out-of-sample by construction: the forecast from data through period t
is paired with the realized return of period t+1.
