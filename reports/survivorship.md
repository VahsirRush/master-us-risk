# Survivorship Measurement

Generated 2026-08-26T02:37:47+00:00

Required by `docs/data-sources-contract.md` section 3.3. The free data path (yfinance) serves only currently-listed symbols, so names that left the index and stopped trading cannot be retrieved at all. This report measures how many, and which kind.

## Headline

- **613 of 795 historical constituents (77.1%) are retrievable.**
- 182 names are not, and are absent from the panel.

## Retrieval rate by year

Of the names in the index during each year, the share retrievable today.

| Year | Constituents | Retrieved | Rate |
|---:|---:|---:|---:|
| 2010 | 512 | 374 | 73.0% |
| 2011 | 517 | 382 | 73.9% |
| 2012 | 519 | 392 | 75.5% |
| 2013 | 516 | 397 | 76.9% |
| 2014 | 516 | 402 | 77.9% |
| 2015 | 529 | 416 | 78.6% |
| 2016 | 534 | 436 | 81.6% |
| 2017 | 534 | 447 | 83.7% |
| 2018 | 524 | 450 | 85.9% |
| 2019 | 521 | 463 | 88.9% |
| 2020 | 521 | 472 | 90.6% |
| 2021 | 520 | 479 | 92.1% |
| 2022 | 519 | 486 | 93.6% |
| 2023 | 515 | 494 | 95.9% |
| 2024 | 516 | 500 | 96.9% |
| 2025 | 518 | 506 | 97.7% |

## Retrieval rate by removal reason

The direction of the bias is visible here rather than asserted: reasons that leave a company trading under some symbol retain their history, reasons that end the listing do not.

| Removal reason | Names | Retrieved | Rate |
|:---|---:|---:|---:|
| still_in_index | 465 | 465 | 100.0% |
| acquired_or_merged | 154 | 34 | 22.1% |
| index_rebalance | 122 | 86 | 70.5% |
| spun_off | 28 | 14 | 50.0% |
| unclassified | 17 | 11 | 64.7% |
| unstated | 4 | 2 | 50.0% |
| taken_private | 4 | 0 | 0.0% |
| bankruptcy | 1 | 1 | 100.0% |

## Direction and magnitude of the bias

**The bias is OPTIMISTIC**, in the sense that matters: the panel is conditioned on a name still having a tradeable symbol today, which is a form of survival. Every name that stopped trading during the sample is absent from its entire history, not merely from the point it stopped.

**The mechanism is not the one section 3.3 anticipates, and the measurement says so.** That section expects acquisitions and mergers to remain retrievable while failures do not. Measured here, acquired or merged names are retrievable at only 22.1% (34 of 154) — Yahoo drops a symbol when it stops trading regardless of *why* it stopped. Selection is on 'no longer trades', not on 'failed'. Names still in the index are retrievable at 100.0% (465 of 465); names dropped in an index rebalance, which usually keep trading, at 70.5% (86 of 122).

**Composition of the 182 absent names:** 120 acquired or merged, 36 index rebalance, 14 spun off, 6 unclassified, 4 taken private, 2 unstated. That composition pulls in two directions, and honesty requires naming both:

- The 120 acquired or merged names typically ended at a takeover premium. Dropping them removes a positive terminal return, which biases measured performance **downward**. This is the largest single missing bucket.

- The 36 names dropped in index rebalances left because their market capitalisation shrank, which is a slow underperformance. Dropping them biases measured performance **upward**.

- Bankruptcies and delistings for cause bias **upward** and are the classic survivorship channel, but here they are a handful of names, not the bulk.

**Net direction: upward, with lower confidence than section 3.3 assumes.** The conditioning-on-survival channel is the dominant one and points up. The acquisition channel is the largest by name count and points down. The two do not cancel to zero, but neither is the sign as safe as 'failures are missing' would make it. The magnitude is not directly measurable — the returns of the absent names are exactly what cannot be observed.

The retrieval rate by year is the clearest single signal that the bias is real and time-varying: it climbs monotonically from 73.0% in 2010 to 97.7% in 2025. The early sample is the most contaminated, and the early sample is the training set.

## What this means for every number in this repository

- Backtest returns are **overstated** by an unknown but non-zero amount. The missing names are disproportionately the ones that fell.
- The effect is largest in the early sample, where more of the era's constituents have since stopped trading, and smallest at the end.
- Cross-sectional dispersion is **understated**: the left tail is the part that is missing.
- This cannot be corrected on the free path. It is measured and reported, not fixed. A CRSP or Sharadar upgrade would eliminate it, and the `PriceSource` Protocol exists so that upgrade is a drop-in swap.
