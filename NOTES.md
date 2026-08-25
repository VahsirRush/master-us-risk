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
