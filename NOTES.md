# NOTES

Append-only lab notebook. One entry per session. Log the numbers, not just the
narrative — this file is where the README gets written from at Phase 8.

Format per entry: what was built, gate result, actual numbers, what broke, any
deviation from spec with reasoning.

---

## Session 1 — 2026-08-18 — Scaffold + Panel contract

**Built**
- Repository scaffold per implementation-spec section 2
- `pyproject.toml`, strict ruff + mypy config
- `Makefile` with eight phase targets, all stubbed
- `src/master_us/data/panel.py` — the Panel contract from spec section 3
- `src/master_us/utils/validation.py` — `set_determinism`, `assert_aligned`
- `tests/fixtures.py` — synthetic panel factory + four corruption fixtures
- `tests/test_panel.py` — 19 tests

**Gate:** n/a (pre-Phase 0)

**Numbers**
- 19 tests pass
- Synthetic signal recovery: injected 0.15, measured within [0.10, 0.22] tolerance

**Deviations from spec**
- Added `label_horizon` field to Panel (not in spec section 3). Reason: embargo
  logic and signal-decay analysis both need it, and having each infer it
  independently invites silent disagreement.
- Added `attrs` dict for provenance (synthetic flag, seed, signal feature index).

**Open**
- Corruption fixtures exist but the Phase-0 leakage tests that consume them are
  not yet written. That is Session 2.

---

## Session 2 — 2026-08-25 — Normalization layer + Phase-0 leakage gates

**Built**
- `src/master_us/data/normalize.py` — `RobustZScoreNorm` (spec §4.5),
  `process_labels`, `infer_feature_groups`, and three frozen result objects
  (`NormalizationStats`, `NormalizedFeatures`, `ProcessedLabels`)
- `tests/test_normalization.py` — 40 tests
- `tests/test_no_lookahead.py` — 14 tests, the four §4.7 gates plus three extras
- `tests/fixtures.py` — extended: `n_delistings` on the factory,
  `corrupt_survivorship_drop_delisted`, `corrupt_regime_shift`, a `lead`
  parameter on `corrupt_lookahead`, negative `shift` on
  `corrupt_misaligned_labels`

**Gate:** Phase 0 §4.7 — all four pass, each demonstrated to fail on a panel
carrying the corresponding defect. Still synthetic-only; the gate is not
discharged until it runs against a real panel.

**Numbers** — clean vs. corrupted, 750 dates × 120 names × 20 features, seed 7.
Mean per-date cross-sectional Pearson IC throughout.

*Gate 1 — shifted features lose signal.* "Delayed" = feature from t−1 used
against the return following t.

| panel | IC lag 0 | IC delayed | flagged features |
|---|---|---|---|
| clean | +0.1505 | −0.0007 | none |
| `corrupt_lookahead(feature=5, lead=1)` | +0.0020 | **+1.0000** | [5] |
| `corrupt_misaligned_labels(shift=-1)` | −0.0048 | **+0.1500** | [0] |
| `corrupt_lookahead(feature=5, lead=0)` | +1.0000 | +0.0020 | none — see below |

*Gate 1b — implausible IC screen (added, not in spec).* Clean max \|IC\| 0.1505
at feature 0 (the injected signal); with `corrupt_lookahead(lead=0)`, 1.0000 at
feature 5.

*Gate 2 — normalization uses training statistics.* Panel given a regime break at
the split (features ×4, +2 from t=450) so that a leak has something to move.

| procedure | median[0] | scale[0] | n obs |
|---|---|---|---|
| `fit(full array, train-only mask)` | +0.0012 | 1.0013 | 51 254 |
| `fit(train slice only)` | +0.0012 | 1.0013 | 51 254 |
| `fit(full array, all-True mask)` — the leak | **+0.2769** | **1.6110** | 85 435 |

Honest vs. train-slice output on the val/test span: max \|diff\| = 0.0, bitwise
identical. Honest vs. leaked: mean \|diff\| 0.4242, max 1.3218.

*Gate 3 — survivorship.* Clean panel built with 18 delistings.

| panel | never tradeable | delistings missing a terminal return | names exiting before sample end | mean universe |
|---|---|---|---|---|
| clean, 18 delistings | 0 | 0 | 19 | 106.5 |
| `corrupt_survivorship_drop_delisted` | **18** | **18** | 1 | 96.8 |
| `corrupt_survivorship(0.2)` | **23** | 3 | 16 | 85.7 |

Mean terminal return across the 18 delisted names: −0.4065.

*Gate 4 — panel alignment.* Lead–lag IC profile of the signal feature; a correct
panel peaks at lag 0.

| panel | −3 | −2 | −1 | 0 | +1 | +2 | +3 | peak |
|---|---|---|---|---|---|---|---|---|
| clean | −.0017 | +.0003 | −.0048 | **+.1505** | −.0007 | −.0018 | −.0019 | 0 |
| `corrupt_misaligned_labels(+1)` | −.0002 | −.0035 | **+.1513** | −.0007 | −.0017 | −.0024 | +.0001 | −1 |
| `corrupt_misaligned_labels(−1)` | −.0055 | −.0015 | −.0005 | −.0048 | **+.1500** | −.0012 | −.0021 | +1 |

Ticker axis permuted within each date: IC 0.1505 → −0.0007, with every lead/lag
below the noise floor. Shapes and dtypes are unchanged by that corruption, which
is the point — no structural assertion sees it.

*Arithmetic.* Transform output on the clean panel: min −3.0000, max 3.0000,
input NaN rate 0.0507, output NaN count 0. Labels (120 dates × 1000 names,
`drop_pct=5.0`): 120 000 eligible, 1 000 missing, 5 950 trimmed, 113 050
surviving; trimmed fraction 0.050000; per-date max \|mean\| 1.03e−08; per-date
std ∈ [1.000000, 1.000000].

**Tests:** 73 pass in 1.0s (19 panel + 40 normalization + 14 leakage).
`ruff check src tests` and `mypy src` both clean.

**Spec ambiguities and how they were resolved**

1. *§4.5 "the most extreme 5% (both tails)"* — 5% total or 5% per tail? Read as
   **5% total, 2.5% from each tail**, which is the ordinary meaning of "the most
   extreme 5%" and matches `label_extreme_drop_pct: 5.0` in `config/data.yaml`.
   `per_tail=True` gives the other reading. If the intent was 10% total, that is
   a one-argument change.
2. *§4.5 "a companion binary mask feature per feature group"* — `Panel` carries
   `feature_names` but no group metadata, so "feature group" is not expressible
   against the contract as it stands. Resolved by taking `feature_groups` as a
   constructor argument, with `infer_feature_groups()` deriving them from the
   `group_detail` naming convention §4.3 implies. **Default when no grouping is
   supplied is one indicator per feature**, which doubles F. Phase 0's feature
   bank should pass an explicit grouping; flagging it because the default is
   safe but wasteful at F≈150.
3. *§4.5 `transform()` signature vs. emitting extra columns* — the spec's
   signature returns one array, but "emit a companion mask feature" changes the
   column count. Split rather than guessed: `transform()` keeps the spec
   signature and is shape-preserving; `transform_with_masks()` returns the
   augmented array. **Model paths must call the latter**, or missingness is
   silently discarded by the NaN→0 step.
4. *§4.5 `fit(x, train_mask)` — what is `train_mask`?* Accepts `(T,)` date-level
   or `(T, N)` observation-level. Observation level is preferred so the training
   span can be intersected with `Panel.mask`; `fit_on_panel(panel, train_end)`
   does that intersection for you.
5. *Zero-MAD features* — not addressed by the spec. Raises by default per
   CLAUDE.md rule 6, with `zero_mad="unit_scale"` as an explicit opt-out. Real
   data will hit this and the decision should be logged, not silent.
6. *"Drop" in the label rule* — implemented as set-to-NaN, not remove. The
   `(T, N)` grid is load-bearing everywhere downstream.

**Deviations from spec**
- Added a fifth Phase-0 test, `test_no_feature_has_an_implausible_information_coefficient`.
  §4.7 names four; the fourth does not cover `corrupt_lookahead(lead=0)` (see
  "What broke" below).
- Added `test_alignment_gate_fires_when_the_ticker_axis_is_permuted` — a
  cross-sectional misalignment that every shape assertion passes.
- `pandas.*` added to the mypy `ignore_missing_imports` overrides. pandas-stubs
  was tried first and rejected: it disagrees with pandas 2.x/3.x at enough call
  sites that adopting it means editing `panel.py` for pandas typing rather than
  for anything real. Eight other untyped dependencies already take this route.

**What broke**

- **`corrupt_lookahead(lead=0)` does not move the metric §4.7's shift test
  measures.** The fixture sets `feature[t] = label[t]`. Delaying the feature
  gives `label[t−1]` against `label[t]`, which is uncorrelated in the synthetic
  panel, so IC collapses 1.0000 → 0.0020 and the shift test sees a textbook
  clean pass. Asserting the shift test "catches lookahead" on that fixture would
  have been asserting nothing. Two changes: `corrupt_lookahead` gained a `lead`
  parameter, where `lead=1` sets `feature[t] = label[t+1]` — the realistic
  `.shift(-1)`-for-`.shift(1)` bug, invisible at lag 0 (IC +0.0020) and glaring
  under a one-day delay (IC +1.0000) — and the contemporaneous case is now
  caught by the separate implausible-IC screen. The blind spot is asserted
  explicitly in `test_contemporaneous_lookahead_is_invisible_to_the_shift_gate`
  so it cannot quietly close or widen.
- **`corrupt_misaligned_labels` code and docstring disagreed.** The docstring
  claimed "labels[t] describes a return already realized by dates[t]" — leakage.
  The code did `labels[:-shift] = labels[shift:]`, giving `labels[t] =
  old_labels[t+1]`, a return *further* forward: an off-by-one that destroys
  signal rather than leaking. Both directions now exist (`shift=-1` is the
  leaky one the docstring described) and the docstring says which is which.
- **Session 1's "ruff + mypy clean" did not hold.** Neither tool was installed
  in this environment, so nothing had actually been run. Under ruff 0.16.4 and
  mypy 1.19.1, `panel.py`, `validation.py`, and `test_panel.py` produced 6 ruff
  and 15 mypy errors — mostly bare `np.ndarray` under `disallow_any_generics`.
  Fixed rather than configured away: `panel.py` now declares `TickerArray`,
  `FeatureArray`, `MaskArray` aliases, which also makes the dtype contract that
  `_validate_dtypes` enforces at runtime visible to the type checker.
- Environment note: the default `python3` here is 3.9.6; the project targets
  3.11 and `/opt/homebrew/bin/python3.11` is what the suite was run under. Worth
  a venv before Phase 1, since `torch` and `lightgbm` are not installed anywhere
  yet.

**Open**
- The Phase-0 gates run against synthetic panels only. They are necessary, not
  sufficient — the real test is the first panel built from yfinance + SEC, where
  the fundamentals `filed`-date join is the thing most likely to leak.
- `min_names` in `process_labels` defaults to 2, which is far too permissive for
  a real cross-section. It should be driven from `config/data.yaml` once the
  universe builder exists and the realistic floor is known.
- Feature groups for the ~150-feature bank are undefined. Until §4.3 lands,
  `transform_with_masks()` emits one indicator per feature.

---

## Session 3 — 2026-08-26 — Infrastructure, feature groups, and the real data pull

**Built**
- `git init` + initial commit of Sessions 1–2; `.gitignore`; venv at
  `~/.venvs/master-us` (3.11.14) with `pip install -e ".[dev]"`
- `src/master_us/data/feature_groups.py` — the §4.3 group registry, wired into
  `RobustZScoreNorm` as the new default (`feature_groups="auto"`)
- `src/master_us/data/sources.py` — the three §1 Protocols, the SEC user-agent
  guard, `RateLimiter`, `with_retry`
- `src/master_us/data/universe.py` — `SP500Universe`, PIT reconstruction
- `src/master_us/data/loaders.py` — `YFinancePrices`, hardened per §3.1–3.2
- `src/master_us/data/sec.py` — `SECFundamentals`, `TAG_FALLBACKS`, `as_of`,
  `resolve_tag`, `concept_coverage_by_year`
- `src/master_us/data/survivorship.py` + `reports/survivorship.md`
- `config/ticker_aliases.yaml` — §2.2's rename map
- `scripts/00`–`04`, each idempotent and cache-first
- `tests/test_feature_groups.py` (14), `tests/test_data_sources.py` (45)

**Gate:** Phase 0 §4.7 still passes (synthetic). The four §7 data-contract tests
now pass against real data. Phase 0 is NOT complete — §4.3 features and §4.4
market vector remain, so no `Panel` is written yet.

**Numbers**

*Environment.* Full dependency set installs cleanly on macOS arm64 with no extra
index URLs — torch 2.13.0 (MPS available), lightgbm 4.7.0, polars 1.44.0,
pandas 3.0.5, numpy 2.4.6. One non-pip dependency: lightgbm's wheel needs
`libomp.dylib`, fixed with `brew install libomp`. Without it `import lightgbm`
raises `OSError: Library not loaded: @rpath/libomp.dylib`.

*Universe.* 795 unique tickers ever in membership, 503 current, 354 changes
applied, 504.4 names/day (min 502, max 507), 2010-02-01 → 2025-12-31.

*Prices.* 613 of 795 retrieved (77.1%), 2,270,255 rows, 5.7 min for the full
pull, 59 MB cached. 191 failures: 182 empty frames (unretrievable delisted
symbols) and 9 excessive-zero-volume names (worst: SW 59.9%, CPWR 57.5%).

*Fundamentals.* 67 quarters, 2009q2 → 2025q4, 171 MB cached, 21,453,464 facts
over 6,389 tickers / 5,048 CIKs. **PIT lag (`filed − period_end`): median 39
days, p05 26, p95 75, min 2, max 4018.** That median is the concrete size of the
look-ahead a `period_end` join would inject.

*Survivorship* (`reports/survivorship.md`):

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

Retrieval rate by year climbs monotonically 72.7% (2010) → 97.7% (2025).

*Tag resolution coverage* (share of S&P-500 filing CIKs resolving each concept):

| Year | assets | equity | liabilities | long_term_debt | net_income | op_cash_flow | revenue | shares_out |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 99.1% | 99.3% | 99.1% | 73.5% | 100.0% | 99.1% | 90.6% | 88.1% |
| 2013 | 99.6% | 99.6% | 99.6% | 73.5% | 99.6% | 99.6% | 91.2% | 90.8% |
| 2016 | 99.8% | 99.4% | 99.6% | 78.8% | 100.0% | 99.6% | 90.8% | 90.6% |
| 2018 | 100.0% | 99.6% | 100.0% | 80.4% | 100.0% | 100.0% | 96.0% | 90.5% |
| 2021 | 100.0% | 99.8% | 100.0% | 81.7% | 100.0% | 100.0% | 95.5% | 90.8% |
| 2025 | 100.0% | 99.8% | 100.0% | 79.4% | 100.0% | 100.0% | 95.7% | 92.5% |

**Zero concept-years below the 70% reliability threshold.** `long_term_debt` is
the weakest at 72.5–87.5% and is the one to watch for the leverage descriptor.

The ASC 606 transition is visible exactly where §4.3 predicts — `TAG_FALLBACKS`
earning its place:

| Year | `SalesRevenueNet` | `Revenues` | `RevenueFromContractWithCustomerExcludingAssessedTax` |
|---:|---:|---:|---:|
| 2017 | 222 | 223 | 0 |
| 2018 | 222 | 316 | 228 |
| 2019 | 15 | 225 | 304 |
| 2020 | 0 | 207 | 323 |

*Panel inputs.* 2,270,255 rows over 4,023 dates; 1,603,854 tradeable after the
§2.3 price/ADV/history filters; 425 names/day mean (min 336, max 496).

**Tests:** 147 pass in 2.7s (19 panel + 44 normalization + 14 leakage + 14
feature groups + 45 data sources + 11 other). ruff and mypy clean over
`src tests scripts` / 19 source files.

**Deviations from spec, all measured rather than assumed**

1. **§2.1 — the changes table moved.** The contract says both tables are on
   `List_of_S&P_500_companies` as tables 0 and 1. `read_html` now finds only two
   tables there, and table 1 is a navigation box; the string "Selected changes"
   has zero occurrences on the page. The changes table is now its own article,
   `Historical_components_of_the_S&P_500` (407 rows, 1976→2026, 354 in-sample).
   `SP500Universe` carries both URLs and fails loudly with the table count if
   either layout changes again.
2. **§3.1/§3.2/§7 — the adjustment monotonicity direction is backwards.** The
   contract says `adj_close/close` is "monotone non-increasing over time". Under
   Yahoo's convention it is non-DECREASING: `adj_close == close` on the most
   recent bar and is progressively smaller going back, because each past dividend
   scales earlier prices down. Measured across all 613 retrievable tickers, the
   most negative relative step is −1.451e-06 (99th pct −1.220e-06) — a hard
   ceiling, since it is float rounding in Yahoo's ~8 significant digits — while
   genuine dividend steps are positive with median magnitude 8.9e-03. The
   contract's *intent* (a non-monotone jump means a corrupted split adjustment)
   is preserved; only the stated direction was wrong.
3. **§3.3 — the stated bias mechanism is not what the data shows.** The contract
   says "acquisitions and mergers usually remain retrievable. Failures usually do
   not." Measured, acquired/merged names are retrievable at **22.1%**. Yahoo drops
   a symbol when it stops trading regardless of why. Selection is on "no longer
   trades", not on "failed". This matters for the sign: of the 182 absent names,
   121 are acquisitions — which typically ended at a takeover premium, biasing
   measured performance *downward* — against 37 index-rebalance removals, which
   bias *upward*. The report still states the direction is OPTIMISTIC (the
   conditioning-on-survival channel dominates and the by-year gradient is
   unambiguous) but now says so with the composition attached rather than
   asserting a mechanism the data contradicts.
4. **§4.5 label drop stays at 5% total / 2.5% per tail** — confirmed already the
   default in `normalize.py`; `per_tail=True` remains the documented alternative,
   not the default. No change needed.
5. `pandas.*` was already in the mypy ignore list (Session 2); `types-PyYAML`
   added to dev deps — a real maintained stub package, unlike pandas-stubs.

**What broke**

- **The venv could not be created inside the project.** The directory name
  `Resume Project 2:3` contains a colon, which Python refuses as a venv path
  ("Refusing to create a venv ... because it contains the PATH separator :").
  The venv lives at `~/.venvs/master-us` instead. Everything else tolerates the
  colon. Worth renaming the directory at some point; not done unilaterally.
- **My first adjustment tolerance was calibrated on three tickers and was wrong.**
  1e-06 sat *inside* the rounding distribution and flagged 73 of 613 tickers
  (11.9%) as having corrupted split adjustments when none did. Recalibrated to
  1e-05 against the full pull — ~7× above the observed noise ceiling and ~900×
  below the median true adjustment, with zero tickers flagged at any tolerance
  from 2e-06 upward. Failures fell 270 → 197 on the re-run. Two tests now pin the
  calibration from both sides so this cannot silently drift back.
- **Validation only ran at fetch time, never on cached reads.** A resumed run
  produced no diagnostics for tickers cached by the previous run, and a
  recalibrated tolerance was never applied to data already on disk. Added
  `YFinancePrices.revalidate_cache()`, called from script 01. This is also the
  check that would catch Yahoo silently re-adjusting history between pulls, which
  §3.2 asks to be able to detect.
- **I aborted 10 whole SEC quarters over 2 bad filings.** My `filed < period_end`
  invariant raised on the entire archive. Investigated rather than loosened: in
  2024q2 exactly 2 of 6,439 10-K/10-Q submissions (0.031%) trip it — POWER REIT
  and POWERSCHOOL each filed a Q1 10-Q with `period` mis-tagged as the following
  December. Upstream filer errors, not a parsing fault. Now dropped and counted
  per quarter, with the raise retained above a 1% share, where it would genuinely
  indicate broken date parsing.
- **`concept_coverage_by_year` double-counted and reported 161.8% revenue
  coverage in 2018.** It summed per-tag CIK counts, and a filer using both a
  legacy and an ASC 606 revenue tag in the same year was counted twice — the
  error was largest exactly at the transition it was meant to measure. Rewritten
  to count distinct CIKs per concept-year. Max share is now 100.0%.
- **META appeared in the universe from 2010-02-01, four years before Facebook
  joined the index.** Not a bad test — a real defect. Facebook was added
  2013-12-23 as FB, which the changes table records correctly; the 2022 FB→META
  rename is absent from the table, so the backward walk never connected today's
  META to FB's addition event and left it in the universe for all history. Fixed
  with `config/ticker_aliases.yaml` per §2.2, applied to both the current list and
  the changes table so the walk operates in one symbol space. META now first
  appears 2014-01-01. Unique tickers 801 → 795.
  - The alias map **cannot be auto-derived** and is curated. A single changes-table
    row pairs an addition with an unrelated removal on the same date and the
    reason text usually describes only one side, so a mechanical pass produces
    false pairs — `AGN→AAL`, `FOSL→WLTW`, `XEC→IR` all come out of one and are all
    wrong. Deliberately excluded: DOW/DWDP/DD and UTX/RTN/RTX, which are
    reorganisations rather than renames.
- Every other addition the changes table records reconstructs correctly, verified
  against GOOGL (2014-05), OTIS/CARR (2020-05), KVUE (2023-09), VLTO (2023-11),
  GEV (2024-05), WBD (2022-05), TSLA (2020).

**Open**

- **271 of 503 current constituents have no recorded addition event.** Most are
  genuine long-tenured members — the changes table is "selected changes" and is
  thin before 2000 — but an unknown subset are renames the alias map misses. This
  is the largest remaining uncertainty in the universe reconstruction and it is
  bounded, not eliminated, by `test_ticker_aliases_are_applied_and_their_absence_is_bounded`.
- The 9 zero-volume names are flagged but still in the cache. Decide before the
  feature bank whether to exclude them: 40–60% zero-volume days will distort
  Amihud illiquidity and the volume features badly.
- `data/processed/panel.pkl` is deliberately NOT written. A `Panel` needs
  `features` (§4.3) and `market` (§4.4); emitting one with empty arrays would pass
  `Panel.__post_init__` and fail every downstream gate for non-obvious reasons.
- The §4.7 leakage gates have still only ever run on synthetic panels. They must
  be re-run against the real panel once §4.3/§4.4 land — that is where the
  fundamentals `filed` join gets its first genuine test.
- Price data is not yet pinned by hash (§3.2). Should happen before any result is
  quoted, so a Yahoo re-adjustment between runs is detectable.

---

## Session 4 — 2026-08-30 — Backtest engine + the Phase-1 momentum gate

**Built**
- `src/master_us/reporting/results.py` — `MetricValue`, `PhaseResult`,
  `AblationResult` (output-layer §2), with JSON persistence under
  `reports/status/` for the ladder
- `src/master_us/reporting/theme.py` + `status.py` — the §3.1 theme, §3.2
  phase panels, §3.5 `make status` ladder, rendered from cached PhaseResults
- `src/master_us/backtest/construct.py` — `topk_dropout` (buffer rule),
  `decile_long_short`, `cost_aware_optimize` (cvxpy/CLARABEL; solves clean
  with budget, cap, and factor-neutrality rows)
- `src/master_us/backtest/costs.py` — flat tier + Corwin-Schultz spread with
  overnight-gap adjustment + sqrt impact
- `src/master_us/backtest/metrics.py` — every §5.3 metric as a (gross, net)
  `MetricValue` pair; `BacktestSeries` refuses NaN and negative costs
- `src/master_us/backtest/engine.py` — the loop; weights formed at close of t
  earn from t+1, costs charged at the rebalance, no trades off-schedule
- `src/master_us/backtest/momentum.py` — the gate: signal, wide-panel
  assembly, thresholds with reasoning, equity-curve artifact
- `scripts/05_phase0_result.py`, `10_fetch_gate_data.py`, `11_momentum_gate.py`
- `tests/test_results.py` (14), `test_backtest.py` (27),
  `test_engine_reproduces_momentum.py` (5, the §5.4 gate test)
- Makefile now runs through the venv; `make status` and `make engine` are live

**Gate: PASSED.** 12-1 momentum (skip most recent month), decile long-short,
equal weight, monthly rebalance, on the extended real panel (5,030 dates ×
632 tickers, 384 tradeable/day mean), eval window 2008-01-02 → 2025-12-30,
216 rebalances.

| metric | gross | net (10bps flat) | net (CS+impact) |
|---|---|---|---|
| Sharpe | −0.13 | −0.16 | −0.17 |
| Ann. return | −3.5% | −4.2% | — |
| Max drawdown (cumsum) | −159.0% | −159.9% | — |
| Hit rate | 51.9% | 51.8% | — |
| Mean one-way turnover per rebalance | 57.9% | | |
| Mean daily turnover / holding period | 2.76% / 36.2 days | | |

*The 2009 slice.* Mar–Sep 2009 gross return **−106.6%** (arithmetic sum,
constant-notional), within-window drawdown −123.6%. Decomposition: long leg
+17.2%, **short leg −123.8%** — the documented anatomy (shorted losers
rallying), not winners falling. Worst days are the documented junk-rally
days: 2009-03-10 (−10.7%), 03-23 (−10.2%), 04-09 (−14.4%), 05-08 (−8.7%).
Max single-day move among shorted names +66%; 26 crash-window days where a
shorted name moved >+20%. Unprompted second validation: the sample's worst
single day is **2020-11-09 at −24.2% — Pfizer vaccine day**, momentum's
documented worst day in decades. The engine reproduced a second crash nobody
aimed it at.

*Plausibility.* Sharpe −0.13 gross sits inside the gate band (−0.5, 1.0):
large-cap-only 12-1 momentum through BOTH 2009 and 2020 crashes, on a
survivorship-thinned pre-2010 universe, is expected to be weak-to-negative.
Nothing here trips the §12 "too good" alarms; the equity curve's shape
(gains into mid-2008, crash, decade of drift, 2020 spike-down) matches the
published record feature for feature.

**Deviations from spec, each measured**
1. *Output-layer §2 writes `distinguishable_from` as a `@property` taking an
   argument* — not valid Python (a property receives only self). Implemented
   as a method; compares |Δgross| against pooled std √(σa²+σb²), strict
   inequality, and returns False when neither side has any dispersion
   estimate.
2. *Corwin-Schultz aggregation:* the contract says "floor at zero, then
   rolling median." That ordering fails the contract's own sanity check —
   15–22 bps mean for mega-caps (flooring first turns symmetric noise into
   one-sided bias), and overnight gaps landing in γ add more. With the
   paper's own overnight-gap adjustment, signed dailies, rolling mean, floor
   the aggregate: AAPL 1.9 / XOM 3.9 / MSFT 5.1 / KO 6.5 / JNJ 6.5 bps,
   smaller names higher (ALK 9.4, NWSA 14.2). Single-digit mega-caps, as
   demanded.
3. *Gate data window:* canonical sample starts 2010; a 2010 start contains no
   2009 crash. Gate runs on separate caches (`sp500_membership_2007.parquet`,
   prices backfilled to 2006, SPY full-window) — canonical 2010+ files and
   splits untouched. Gate membership: 821 tickers ever, 632 with prices.
4. *Sub-checks added to the gate:* short-leg-driven crash shape,
   zero off-rebalance trades/costs, realistic ≥ flat costs, and
   backward-looking-ness of the momentum signal itself.

**What broke**
- **The monotonicity check caught a real Yahoo re-adjustment.** After the
  2006–2010 backfill, LEG failed `test_price_adjustment_monotone` with a
  −5.2e−03 step exactly at the merge seam (2009-12-31 → 2010-01-04): Yahoo
  re-based LEG's adjustments between the Session-3 pull (Aug 26) and the
  backfill (Aug 30), so the merged file mixed two adjustment bases. Exactly
  the failure mode §3.2 says to detect. Fix: single-basis full-history
  refetch of LEG (worst step now −9.6e−07, rounding noise). One ticker in
  633; the seam check is now standing guard for every future backfill.
- **`scripts/05` initially reported 0 tests passed** — pyproject `addopts`
  already carries `-q`, and the script's own `-q` made it `-qq`, which
  suppresses the "N passed" summary line it parsed. Phase 0 briefly rendered
  as FAILED with 188 tests green. Parse now reads the full stdout without
  adding another -q.
- pandas `.to_numpy()` on a reindexed Series returned a read-only array;
  SPY benchmark returns are now copied before mutation.
- First CS synthetic-recovery test was under-sampled (24 trades/day) and
  understated the estimator's recoverable spread; at 200 trades/day CS
  recovers 20/50/200 bps as 19/44/189. Test rewritten with the measured
  noise floor (~10 bps) documented.

**Numbers**: 193 tests pass in ~7s (188 prior + backtest/results/gate suites,
minus consolidation), ruff + mypy clean over 27 source files. `make status`
renders Phase 0 ✓ (188 tests, 4,023 dates, 425 names/day) and Phase 1 ✓
(Sharpe −0.13, crash −106.6%, rebalance turnover 57.9%).

**Open**
- The −159% full-sample cumsum drawdown is dominated by 2009 plus a decade of
  drift; fine for a gate factor, but worth remembering that the arithmetic
  (constant-notional) convention makes long-horizon drawdowns look deeper
  than compounded ones.
- Turnover 57.9% per rebalance is at the high end of the published 25–60%
  band for monthly deciles — daily-recomputed scores churn decile edges.
  Fine for the gate; a production momentum book would form scores at
  rebalance dates only.
- Gate thresholds (`GATE_*` in momentum.py) are constants with reasoning
  attached; revisit if the sample window ever changes.
- Price data still not pinned by hash (§3.2) — the LEG incident makes the
  case; do it at Phase-2 start.

---

## Session 5 — 2026-08-30 — Feature bank, market vector, and the real Panel

**Built**
- `src/master_us/data/features.py` — the §4.3 bank: 65 base features + 65
  cross-sectional ranks = **130** (target "~150"; every listed content is
  covered, breadth comes from window expansion within the table's own rows).
  Groups: returns 8 · volatility 21 · volume 10 · price_structure 19 ·
  technical 7 · cross_sectional_rank 65. Every name classifies strictly in
  the Session-3 registry — `build_panel` refuses an unclassifiable name.
- `src/master_us/data/market_vector.py` — the §4.4 vector, exactly M=67.
- `src/master_us/data/assemble.py` + `scripts/06_fetch_market.py`,
  `scripts/07_assemble_panel.py` — the real Panel, both horizons:
  `data/processed/panel.pkl` (h=1) and `panel_h5.pkl` (h=5), 1.2 GB each.
- `tests/test_features.py` (12), `test_market_vector.py` (7),
  `test_real_panel_gates.py` (9) — the §4.7 gates' first run on real data.

**Gate: Phase 0 fully discharged.** All four §4.7 gates now pass against the
REAL assembled panel, end to end. 220 tests, ruff + mypy clean.

**The real panel**: 4,004 dates (2010-02-01 → 2025-12-30) × 584 tickers ×
130 features, market 67, mean universe 420/day. Load from cache 0.1s (gate:
<30s). Build 40s per horizon.

*Coverage (Panel.coverage()):* names/day climbs 334.6 (2010) → 492.6 (2025),
min 329 — the survivorship gradient from Session 3, as expected. In-mask
feature NaN: **4 cells out of 218,593,440 (1.8e-08)** — one bar with a null
high/low nulled by the finiteness guard, hitting vol_parkinson_2d/vol_gk_2d
and their ranks. The zero rate is structural, not luck: the mask requires
≥252 observed days and polars rolling windows are observation-based (they
span halts), so every ≤60d window is full for any masked name. No feature
group has elevated NaN in any year; the Session-3 XBRL concept-resolution
issue cannot resurface here because no §4.3 feature touches fundamentals.
Label NaN ≈4.7%/yr = the §4.5 trim (2×floor(N×0.025)/N at N≈420) plus the
final horizon rows. Market vector: 0 NaN, 0 inf, all 4,004×67 cells.

*Real-feature IC profile* (per-date Pearson vs h=1 labels): max |IC| =
0.0154 at `xs_ret_1d`, NEGATIVE — short-term reversal, the best-documented
daily cross-sectional effect, with the correct sign. Top ten are all
reversal-family (1-3d returns, VWAP deviation, close position). Every
feature weakens under a one-day delay; max delayed gain +0.0017, inside the
0.01 noise tolerance. Nothing implausible, nothing leaking.

**Decisions the user asked to be told about**
1. *Market-vector gaps:* three mechanisms, in order — (a) warm-up not fill:
   index data fetched from 2008, so every rolling head is filled before the
   panel starts; (b) calendar mismatches forward-fill AT MOST 5 days
   (CLAUDE.md rule 6 ceiling) — an index level is a state and yesterday's
   state is the honest belief about an unobserved today, whereas
   interpolation would manufacture information; (c) anything longer raises
   with column and date named. In practice the real data needed zero fills.
2. *S&P 400 dollar volume:* Yahoo reports **zero volume for ^SP400 (and
   ^MID) across their entire history** (measured). Level/returns use the
   true index; the dollar-volume columns use MDY (SPDR MidCap ETF, own
   close × volume). Segments' dv columns are in different units — fine,
   each column is only compared with its own history and normalization is
   per-column. ^GSPC and ^RUT carry real volume (0-0.1% zero days).
3. *Labels:* Panel.labels = the §4.5-processed labels (trim 5% total →
   per-date z-score; measured per-date |mean| ≤ 1.7e-08, std = 1.000). The
   RAW forward-return matrix rides in `attrs["raw_forward_returns"]` — the
   engine and IC-in-return-units need it, and recomputing downstream would
   invite a second, subtly different definition. Trimmed fraction measured
   0.0478 (nominal 5%; floor rounding at N≈420).

**Real-data assertion findings and how they were resolved**
- **Panel validation itself never tripped** — but only because assembly was
  designed around the two trips it WOULD have hit: (a) features computed on
  the 2010+ canonical window would leave the first 252 days with zero
  tradeable names (`no tradeable names` assertion) — features are computed
  over the extended 2006+ history and sliced, so windows are warm on day
  one; (b) the panel calendar starts at the universe's first daily snapshot
  (2010-02-01), not the price calendar's start.
- **42 of 626 ticker columns were never tradeable** — almost all REUSED
  SYMBOLS: DV (DeVry→DoubleVerify), BEAM, MMI (Motorola
  Mobility→Marcus & Millichap), BTU (Peabody relisted post-bankruptcy),
  AET/EMC/ESRX-class ghosts. Yahoo's series belongs to the newer company;
  its dates never overlap the old company's membership window, so the
  membership∧filters conjunction correctly never fires — the §2.2
  ticker-reuse defect caught by construction. Assembly now drops these
  columns and records them in `attrs["dropped_never_tradeable"]`. Panel:
  626 → 584 tickers.
- **89/89 exiting names carry a terminal forward return.** Panel exits are
  index removals of still-trading companies; true delistings never enter
  (Yahoo doesn't serve them) and are counted in reports/survivorship.md.
  The real-data survivorship gate asserts: no never-tradeable columns,
  >50 exits (no scrubbing), terminal label on every exit.
- **The synthetic shift gate does not transfer as written.** On synthetic
  panels, delaying features collapses IC to zero; real features are
  autocorrelated (a 60d vol barely moves overnight), so their delayed IC
  legitimately persists. The property that survives is one-sided: delayed
  |IC| must not EXCEED |IC| now beyond noise (0.01). That is Session 2's
  `shift_gate_violations` logic, now the real-data form of gate 1.
- Two of my own test-construction errors (breadth hand-check with 4 names
  against a deliberate 10-name floor; `_to_wide` after dropping tickers) —
  both fixed in the tests/assembler, not routed around.

**Deviations from spec**
- §4.3 signature `build_features(ohlcv, cfg)` → `build_features(ohlcv, spy)`:
  beta/corr to S&P need the benchmark series, which no cfg dict carries.
- Bank is 130, not ~150: every content row of the §4.3 table is implemented;
  I did not pad with concepts outside the table to hit a round number.
- RSI is Cutler's (windowed) not Wilder's (recursive): Wilder's value depends
  on the series start point, so cached vs fresh computations would differ.
  MACD terms are divided by price for cross-sectional comparability.
  GK per-day variance floored at 0 before averaging (standard).
- Volume ratio / dollar-volume z-score compare today against a reference
  window ending at t-1 (self-exclusion — the literal closed='left' reading);
  all other rolling stats end at and include t (info as of close of t).
- `assemble.py` is not in the spec §2 layout (scripts must be thin, the
  assembly logic had to live somewhere in src/).
- Panel.metadata = None: mcap needs the SEC shares join, which belongs with
  the Barra descriptors (Phase 5). Open item, not smuggled in half-built.

**Open**
- panel.pkl is 1.2 GB/horizon (float32 130-feature tensor). Fine locally;
  never committing it. Reproducible via scripts 00→07.
- metadata (mcap/sector/adv/price) still to be attached at Phase 5.
- The 4 stray NaN cells trace to one bar with a null high/low that price
  validation doesn't currently flag (it checks non-positive, not null).
  Harmless (NaN→0+mask machinery), but the loaders could tighten.
- Price data still not pinned by hash — carried from Session 4; do at
  Phase-2 start.

---
