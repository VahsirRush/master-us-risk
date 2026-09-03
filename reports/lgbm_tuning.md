# LightGBM tuning — validation only

The Phase-2 gate is LightGBM out-of-sample RankIC > 0.02. The stock
`config/baselines.yaml` configuration misses it. Every row below is scored on
the VALIDATION split; the test split is not touched by this search.

| Config | Valid RankIC | Best iter | Features | Hypothesis |
|:---|---:|---:|:---|:---|
| `all_tuned` | +0.0224 | 248 | all | same combination, full feature bank |
| `spec` | +0.0214 | 26 | all | the configuration config/baselines.yaml ships with |
| `leaves_15` | +0.0210 | 44 | all | quarter capacity |
| `leaves_15_slow` | +0.0210 | 124 | all | shallow + slow: more trees, smaller steps |
| `leaves_15_child1000` | +0.0206 | 40 | all | shallow + large leaves: forbid fitting date-specific noise |
| `leaves_7` | +0.0200 | 64 | all | shallow trees; near-additive model over weak features |
| `leaves_31` | +0.0197 | 35 | all | halve capacity: 64 leaves overfits a max-|IC|-0.0154 signal |
| `ranks_only_tuned` | +0.0194 | 153 | ranks | best-guess combination of the above, ranks only |
| `leaves_15_colsample04` | +0.0186 | 43 | all | decorrelate trees across the raw/rank near-duplicate pairs |
| `ranks_only_leaves_15` | +0.0159 | 26 | ranks | rank features match the per-date z-scored target's geometry |

**Winner: `all_tuned`**

```yaml
n_estimators: 2000
learning_rate: 0.02
num_leaves: 15
min_child_samples: 500
subsample: 0.8
colsample_bytree: 0.5
early_stopping_rounds: 100
feature_subset: all
```

---

## What happened on test (applied once, after the search closed)

`all_tuned` was run on the test split with 5 seeds, one time, after the table
above was final. No configuration was reconsidered afterwards.

| Config | Valid RankIC | Test RankIC (5 seeds) | Gate (> 0.02) |
|:---|---:|---:|:---|
| `spec` (shipped) | +0.0214 | +0.0185 ±0.0005 | **miss** |
| `all_tuned` | +0.0224 | +0.0207 ±0.0007 | **pass** |

Three things worth stating plainly:

1. **Validation flatters test by ~0.002-0.003 for both configs.** The ranking
   between them survived the transfer, but the level did not. Anyone reading
   the validation column as a forecast of out-of-sample performance would be
   over-optimistic by more than the entire gate margin.
2. **The gate margin is one seed-standard-deviation.** 0.0207 against a bar of
   0.02, with seed dispersion 0.0007. One of the five seeds (0.0199) misses
   the bar on its own. This passes, but it is not a comfortable pass and
   should not be quoted as one.
3. **The tuning gain IS distinguishable from noise.** +0.0022 against a pooled
   seed std of 0.0009. The improvement is real, unlike the differences among
   the middle rows of the search table, which are not.
