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
