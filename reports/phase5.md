# Phase 5 — Barra descriptors and factor returns

## Tag resolution coverage (S&P-500 universe)

| year | equity | net_income | operating_cash_flow | assets | long_term_debt | revenue |
|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 99.3% | 100.0% | 99.1% | 99.1% | 73.6% | 90.5% |
| 2013 | 99.8% | 99.6% | 99.6% | 99.6% | 74.2% | 91.3% |
| 2016 | 99.6% | 100.0% | 99.6% | 99.8% | 79.5% | 90.9% |
| 2018 | 99.8% | 100.0% | 100.0% | 100.0% | 80.9% | 96.3% |
| 2021 | 100.0% | 100.0% | 100.0% | 100.0% | 82.3% | 95.6% |
| 2025 | 100.0% | 100.0% | 100.0% | 100.0% | 80.6% | 95.5% |

## Descriptor coverage (conditional on the name trading)

| descriptor | overall | worst year | rate |
|---|---:|---:|---:|
| earnings_var | 57.2% | 2011 | 0.7% |
| debt_equity | 64.6% | 2017 | 60.4% |
| sales_growth_3y | 75.0% | 2011 | 0.0% |
| debt_assets | 77.9% | 2012 | 73.5% |
| roe | 78.9% | 2017 | 73.7% |
| turn_252d | 80.3% | 2011 | 16.7% |
| earnings_growth_3y | 82.0% | 2011 | 0.0% |
| turn_60d | 86.4% | 2011 | 67.8% |
| turn_21d | 87.7% | 2011 | 79.1% |
| ep | 88.4% | 2011 | 84.2% |
| cfp | 88.4% | 2011 | 84.2% |
| bp | 88.4% | 2011 | 84.2% |
| log_mcap | 88.4% | 2011 | 84.4% |
| mom_12_1 | 91.5% | 2011 | 0.4% |
| resid_vol | 91.5% | 2011 | 0.8% |
| beta | 91.5% | 2011 | 0.8% |
| accruals | 94.8% | 2016 | 90.9% |
| vol_60d | 97.9% | 2011 | 76.5% |

## Factor returns (monthly, annualized)

| factor | ann mean | ann vol | t |
|---|---:|---:|---:|
| market | +14.21% | 14.14% | 3.86 |
| size | +0.66% | 2.73% | 0.93 |
| value | +0.83% | 3.06% | 1.04 |
| momentum | +0.92% | 3.40% | 1.04 |
| volatility | +0.43% | 5.11% | 0.32 |
| liquidity | +0.36% | 3.16% | 0.44 |
| leverage | +0.08% | 1.37% | 0.23 |
| growth | +0.12% | 1.25% | 0.37 |
| quality | -0.36% | 1.18% | -1.16 |

## Gate

```
  [PASS] momentum sign: +0.92%/yr (t=1.04) — sign matches the literature; NOT significant at 5%
  [PASS] momentum crash: worst month 2020-11 at -3.68% (-3.8 SD); known in-sample crashes ('2020-11', '2021-01'); ('2009-03', '2009-04') predate the panel and are NOT testable
  [PASS] value cyclicality: 2017-2020 mean -1.56%/yr, 2022-2023 mean +5.65%/yr — weak in the growth regime and strong in the rotation, as documented
  [PASS] market intercept: 14.21%/yr at 14.14% vol — equity-like
  [PASS] cross-sectional fit: mean weighted R^2 0.257 over 177 estimated periods (a monthly cross-sectional factor model typically lands 0.15-0.40)

  GATE: PASS
  [PASS] specific ⟂ exposures: largest |corr| = 0.0078 on 'liquidity' (tolerance 0.05)
```
