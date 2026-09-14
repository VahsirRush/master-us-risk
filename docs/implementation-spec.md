# MASTER-US + Barra Risk Model — Implementation Specification

This document is the build contract. Follow it literally. Where it specifies a signature, implement that signature. Where it specifies a gate, do not proceed past it until the gate passes.

---

## 0. Project Framing (goes in README verbatim, edited for voice)

This is a **port with cost accounting**, not a fidelity replication.

The original MASTER authors (AAAI-2024, SJTU-DMTai) later disclosed that their published validation and test splits were dumped using training-set processors and contain roughly 95% of stocks per day rather than the full universe used in the paper's experiments. Their data access has since expired and the correct splits cannot be regenerated. Reproducing their exact table is therefore not a well-posed target.

What is well-posed, and what this project does:

1. Faithfully implement the architecture — cross-time temporal attention, inter-stock attention, market-guided gating — and port it to US equities.
2. Measure the two things the original evaluation left open: **turnover-aware net-of-cost performance**, and **factor attribution of the resulting signal**.
3. Inherit the original's one methodological strength: **5-seed averaging with dispersion reported**, and treat seed variance as the threshold for calling any difference real.

Headline deliverable: a net Sharpe, a cost breakeven point, and a factor-neutralized alpha number.

---

## 1. Environment

```toml
# pyproject.toml — target Python 3.11
[project]
dependencies = [
  "numpy>=1.26", "pandas>=2.1", "polars>=0.20",
  "torch>=2.2", "lightgbm>=4.3", "scikit-learn>=1.4",
  "statsmodels>=0.14", "scipy>=1.12", "cvxpy>=1.4",
  "pyarrow>=15", "pyyaml", "tqdm",
  "matplotlib", "seaborn",
]
[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "hypothesis"]
```

Determinism block, called at the top of every training entry point:

```python
def set_determinism(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

---

## 2. Repository Layout

```
master-us/
├── README.md
├── NOTES.md                       # append-only log: phase, decision, numbers, what broke
├── pyproject.toml
├── Makefile                       # one target per phase
├── config/
│   ├── data.yaml
│   ├── costs.yaml
│   ├── master.yaml
│   ├── baselines.yaml
│   └── barra.yaml
├── data/                          # gitignored
│   ├── raw/                       # immutable pulls
│   ├── interim/                   # aligned panels
│   └── processed/                 # model-ready tensors
├── src/master_us/
│   ├── data/
│   │   ├── universe.py
│   │   ├── loaders.py
│   │   ├── features.py
│   │   ├── market_vector.py
│   │   ├── normalize.py
│   │   └── panel.py               # the canonical Panel object
│   ├── models/
│   │   ├── layers.py
│   │   ├── master.py
│   │   ├── loss.py
│   │   ├── train.py
│   │   └── baselines/{ridge,lgbm,lstm,ungated}.py
│   ├── backtest/
│   │   ├── construct.py
│   │   ├── costs.py
│   │   ├── engine.py
│   │   └── metrics.py
│   ├── risk/
│   │   ├── descriptors.py
│   │   ├── standardize.py
│   │   ├── factor_returns.py
│   │   ├── covariance.py
│   │   ├── specific_risk.py
│   │   ├── attribution.py
│   │   └── bias_tests.py
│   ├── experiments/
│   │   ├── ablations.py
│   │   ├── beta_sweep.py
│   │   └── join.py
│   └── utils/{validation,io,plotting}.py
├── scripts/                       # thin CLI, argparse → src call, zero logic
├── tests/
└── reports/
```

**Invariant:** nothing in `scripts/` or notebooks computes a number that reaches a report. All logic in `src/`.

---

## 3. Core Data Contract

Every downstream module consumes this object. Define it once, in `data/panel.py`.

```python
@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex        # (T,) sorted, unique, trading days
    tickers: np.ndarray            # (N,) sorted, stable ordering
    features: np.ndarray           # (T, N, F) float32, NaN where absent
    feature_names: list[str]       # len F
    market: np.ndarray             # (T, M) float32, no NaN
    market_names: list[str]        # len M
    labels: np.ndarray             # (T, N) float32, forward return, NaN where absent
    mask: np.ndarray               # (T, N) bool — True = tradeable in universe on that date
    metadata: pd.DataFrame         # MultiIndex (date, ticker): mcap, sector, adv, price, spread

    def __post_init__(self):
        assert self.features.shape == (len(self.dates), len(self.tickers), len(self.feature_names))
        assert self.labels.shape == self.mask.shape == (len(self.dates), len(self.tickers))
        assert self.market.shape == (len(self.dates), len(self.market_names))
        assert not np.isnan(self.market).any()
        assert self.dates.is_monotonic_increasing and self.dates.is_unique
```

`mask` is the single source of truth for universe membership. No module may trade or fit on a masked-out name.

---

## 4. Phase 0 — Data Layer

### 4.1 Universe (`data/universe.py`)

```python
def build_universe(start: str, end: str, cfg: dict) -> pd.DataFrame:
    """Returns MultiIndex (date, ticker) -> bool membership, monthly reconstituted,
    forward-filled to daily. Delisted names remain until their delisting date."""
```

Rules:
- Target: Russell 1000. Fallback if no PIT membership source: top 1000 by trailing 60-day median dollar volume, reconstituted on the last trading day of each month, membership effective the following trading day.
- Filters applied at reconstitution: price ≥ $5, 21d ADV ≥ $2M, ≥ 252 days of history.
- **Delisting returns must be applied.** A name that leaves the universe by delisting gets its terminal return, not a silent drop. Dropping them is survivorship bias and it inflates every downstream number.

### 4.2 Sources (`data/loaders.py`)

Preference order, with a uniform interface so the rest of the code is source-agnostic:
1. CRSP/Compustat via WRDS — correct if available
2. Sharadar SEP + SF1 (Nasdaq Data Link) — has `datekey` for true PIT fundamentals
3. yfinance + SEC EDGAR — free; requires manual handling of splits/dividends and a filing-lag rule

**PIT rule for fundamentals:** a fundamental value is usable on date `t` only if `datekey <= t`. If using EDGAR, use the actual filing date, never the fiscal period end. If a source cannot supply a filing date, do not use its fundamentals — build the model on price/volume features only and state this in the README.

All raw pulls cached to `data/raw/*.parquet`, keyed by source and date range, never re-fetched.

### 4.3 Feature bank (`data/features.py`)

Target ~150 features. Every function has signature `(ohlcv: pl.DataFrame) -> pl.DataFrame` and must be expressible as a backward-looking rolling operation.

| Group | Contents |
|---|---|
| Returns | close-to-close over 1,2,3,5,10,20,30,60d |
| Volatility | realized std over same windows; Parkinson; Garman-Klass |
| Volume | volume ratio vs. 20/60d mean; dollar-volume z-score; Amihud illiquidity |
| Price structure | (high−low)/close; close position within day range; VWAP deviation; overnight gap |
| Technical | RSI(14), MACD(12,26,9), Bollinger %B(20,2), rolling beta(60) to S&P, rolling corr(60) to S&P |
| Cross-sectional | per-date rank transform of every feature above |

```python
def build_features(ohlcv: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """All features strictly backward-looking. Every rolling window uses closed='left'
    semantics where the current bar's close is the last included observation."""
```

### 4.4 Market state vector (`data/market_vector.py`)

Translating the original's construction (which used CSI300/500/800) to US indices. For each of **S&P 500, S&P 400 MidCap, Russell 2000**:
- Latest index return
- For d ∈ {5, 10, 20, 30, 60}: rolling mean and std of index return; rolling mean and std of index dollar volume

That is 3 × (1 + 20) = 63 dimensions. Append:
- VIX level, VIX 20d change
- Cross-sectional return dispersion of the universe
- Fraction of universe above its 200d moving average

Total M ≈ 67. **The market vector must never contain NaN** — it gates every prediction.

### 4.5 Normalization (`data/normalize.py`)

Implement the original authors' method exactly. This is the leakage-critical component.

```python
class RobustZScoreNorm:
    """Median/MAD normalization with training-only statistics.

    fit()       computes per-feature median and MAD over the TRAINING span only.
    transform() applies stored training statistics to any split, then clips to [-3, 3].

    Validation and test data are transformed with BORROWED training statistics.
    Refitting on val/test is leakage and is the single most common replication error.
    """
    def fit(self, x: np.ndarray, train_mask: np.ndarray) -> "RobustZScoreNorm": ...
    def transform(self, x: np.ndarray) -> np.ndarray: ...
```

After transform: NaN → 0, and emit a companion binary mask feature per feature group.

**Label processing:** forward return → drop NaN → drop the most extreme 5% (both tails) → cross-sectional z-score within each date.

### 4.6 Splits

```yaml
# config/data.yaml
splits:
  train: ["2010-01-01", "2016-12-31"]
  valid: ["2017-01-01", "2018-12-31"]
  test:  ["2019-01-01", "2025-12-31"]
embargo_days: 21   # gap between splits, prevents label overlap bleed
```

Chronological only. No shuffling, no k-fold, no re-tuning on test. Test is touched exactly once per model, at the end.

### 4.7 Phase 0 gates

```python
# tests/test_no_lookahead.py
def test_shifted_features_lose_signal():
    """Shift every feature forward one day. IC must fall toward zero.
    If it doesn't, a feature contains future information."""

def test_normalization_uses_training_stats():
    """Assert transform() on val/test produces the same output whether or not
    val/test data was present at fit() time."""

def test_universe_no_survivorship():
    """Assert delisted tickers appear in the panel up to their delisting date
    with a terminal return applied."""

def test_panel_alignment():
    """Assert features[t, n] and labels[t, n] refer to the same (date, ticker),
    and that label[t] is a FORWARD return relative to features[t]."""
```

All four must pass. Load time from cache < 30s. Coverage table (rows, dates, mean names/day, NaN rate per feature group) logged to `NOTES.md`.

---

## 5. Phase 1 — Backtest Engine (before any model)

### 5.1 Portfolio construction (`backtest/construct.py`)

```python
def topk_dropout(scores, mask, k=50, dropout_buffer=25) -> np.ndarray:
    """Original paper's convention. Hold top-k by score; an existing holding is only
    sold when it falls below rank k + dropout_buffer. The buffer is the cheapest
    turnover control available and materially changes net performance."""

def decile_long_short(scores, mask, n_deciles=10, weighting="equal") -> np.ndarray:
    """Academic standard. Long top decile, short bottom decile."""

def cost_aware_optimize(scores, prev_w, cov, mask, lam_turnover, lam_risk,
                        constraints: dict) -> np.ndarray:
    """max  s'w - lam_risk * w'Σw - lam_turnover * ||w - prev_w||_1
       s.t. sum(w)=0 (or 1), |w_i| <= cap, optional factor-neutrality rows.
    cvxpy. Used in Phase 7 for the neutralized variant."""
```

### 5.2 Cost model (`backtest/costs.py`)

```yaml
# config/costs.yaml
baseline_bps: 10          # round-trip proportional
sweep_bps: [0, 5, 10, 20, 50]
realistic:
  spread_estimator: "corwin_schultz"   # or "chen_velikov"
  impact:
    model: "sqrt"
    coefficient: 0.1                    # bps per sqrt(participation)
    adv_lookback: 21
```

```python
def apply_costs(weights_t, weights_tm1, metadata_t, cfg) -> float:
    """Returns cost in return units for the rebalance. Proportional component from
    half-spread; impact component as coef * sqrt(trade_value / ADV)."""
```

### 5.3 Metrics (`backtest/metrics.py`)

Every metric returned as a `(gross, net)` pair. No function returns a single number.

- Annualized return, excess return vs. SPY, Sharpe, Information Ratio
- Max drawdown, Calmar
- **Daily turnover** — `0.5 * sum(|w_t - w_{t-1}|)`
- Hit rate, average holding period
- IC, RankIC, ICIR, RankICIR (Pearson/Spearman per date, then mean/std)

### 5.4 Phase 1 gate — the momentum reproduction

```python
# tests/test_engine_reproduces_momentum.py
def test_momentum_matches_literature():
    """Build 12-1 momentum (skip most recent month), run through the engine.
    Assert: long-short Sharpe in a documented plausible range for the sample,
    a visible 2009 momentum crash drawdown, and turnover consistent with
    monthly rebalancing. If this fails, the engine cannot evaluate anything."""
```

**Do not proceed to Phase 2 until this passes.**

---

## 6. Phase 2 — Baselines

All four run under identical protocol: same panel, same splits, same 5 seeds, same engine.

| Model | Config | Role |
|---|---|---|
| Ridge | cross-sectional, α tuned on valid | Floor. Below this, debug rather than continue. |
| **LightGBM** | 500 trees, lr 0.05, num_leaves 64, early stop on valid RankIC | The real bar. Gradient boosting routinely beats deep models on tabular alpha. |
| LSTM | 2 layers, hidden 128, per-stock | Isolates temporal modeling value |
| **Ungated transformer** | MASTER minus the gate, identical elsewhere | The critical control |

Gate: LightGBM out-of-sample RankIC > 0.02. If not, the feature bank is broken — fix it before building the transformer.

---

## 7. Phase 3 — MASTER Architecture

### 7.1 Layers (`models/layers.py`)

```python
class CrossTimeAttention(nn.Module):
    """Intra-stock temporal attention. The query is the token at the forecast date;
    keys/values span all lookback positions. This is CROSS-TIME, not time-aligned —
    it is the paper's first structural claim. Sinusoidal positional encoding.
    Input  (B*N, L, d) -> Output (B*N, d)"""
    def __init__(self, d_model=128, n_heads=8, n_layers=2, dropout=0.2): ...

class InterStockAttention(nn.Module):
    """Attention ACROSS stocks on a single date. Models momentary peer correlation.
    This is what distinguishes the architecture from a stack of per-stock RNNs.
    Input  (B, N, d) -> Output (B, N, d)"""
    def __init__(self, d_model=128, n_heads=8, n_layers=1, dropout=0.2): ...

class MarketGate(nn.Module):
    """Market-guided feature gating. m_t -> MLP -> sigmoid(logits / beta) -> gate in [0,1]^F.
    Gate multiplies the raw feature embedding elementwise before attention.

    beta is the temperature and the paper's key hyperparameter: beta -> 0 approaches
    hard selection, beta -> inf approaches uniform (i.e. no gating). The beta sweep
    against a no-gating horizontal reference is the single most informative plot
    in this project."""
    def __init__(self, m_dim: int, f_dim: int, hidden=64, beta=1.0): ...
    def forward(self, m: Tensor) -> Tensor:  # (B, M) -> (B, F)
```

### 7.2 Model (`models/master.py`)

```python
class MASTER(nn.Module):
    """Forward path:
       x (B, N, L, F), m (B, M)
       -> gate = MarketGate(m)                      (B, F)
       -> x = x * gate[:, None, None, :]            feature selection
       -> h = Linear(F -> d)(x)
       -> h = CrossTimeAttention(h)                 (B, N, d)
       -> h = InterStockAttention(h)                (B, N, d)
       -> y = Linear(d -> 1)(h).squeeze(-1)         (B, N)
    Masked positions receive -inf attention bias and are excluded from loss."""
```

### 7.3 Loss (`models/loss.py`)

```python
def ic_loss(pred, target, mask) -> Tensor:
    """Negative Pearson correlation computed WITHIN each date, averaged over dates.
    Plain MSE materially underperforms here — the task is cross-sectional ranking,
    not level prediction. Masked names excluded from both moments."""
```

### 7.4 Training (`models/train.py`)

```yaml
# config/master.yaml
lookback: 60
d_model: 128
n_heads_temporal: 8      # N1 in the paper
n_heads_cross: 4         # N2 in the paper
n_layers_temporal: 2
n_layers_cross: 1
dropout: 0.2
beta: 1.0
optimizer: adamw
lr: 1.0e-4
weight_decay: 1.0e-4
scheduler: cosine
grad_clip: 1.0
max_epochs: 100
patience: 10             # on validation RankIC
seeds: [0, 1, 2, 3, 4]
label_horizon: 1         # also run 5 as robustness
```

**Batching:** one batch = one trading date, all names present that date. The inter-stock attention requires the full cross-section; do not batch by stock.

---

## 8. Phase 4 — Ablations, β Sweep, Cost Analysis

### 8.1 Ablation grid (`experiments/ablations.py`) — 5 seeds each

| # | Variant | Isolates |
|---|---|---|
| 1 | Full MASTER | — |
| 2 | − market gating | The gate |
| 3 | − inter-stock attention | Cross-sectional modeling |
| 4 | Time-aligned instead of cross-time | The cross-time claim |
| 5 | **Market vector shuffled** | Whether the gate learns real signal (same params, destroyed input) |
| 6 | Lookback ∈ {20, 40, 60, 120} | Memory horizon |
| 7 | (N1, N2) head grid | Matches the paper's sweep |

### 8.2 β sweep (`experiments/beta_sweep.py`)

β ∈ {0.1, 0.5, 1, 2, 5, 10}, 5 seeds each, plotted with a **horizontal reference line for the no-gating variant** — reproducing the paper's figure on US data.

### 8.3 Result table format

Every cell is `mean ± std` over 5 seeds. Any variant whose gap to the baseline is smaller than the pooled seed std must be reported as **not distinguishable**, in those words.

Columns: IC · RankIC · ICIR · RankICIR · Decile spread (gross) · Decile spread (net) · **Daily turnover** · Annualized net Sharpe

### 8.4 Cost analysis — the project's original contribution

```python
def cost_breakeven(returns_gross, turnover) -> float:
    """bps level at which net alpha reaches zero. Report for MASTER,
    ungated transformer, and LightGBM."""
```

Produce a **net Sharpe vs. assumed cost** curve for all three models on one axis. This is the analysis the original evaluation omitted — it reported Excess Annualized Return and Information Ratio from a top-30 daily-rebalance strategy without publishing turnover or isolating the gross/net gap.

### 8.5 Stress tests

- Regime slices: 2018Q4 · Feb–Apr 2020 · full-year 2022 · 2023–2025
- Sector-neutralized predictions (residualize on GICS L1)
- Cap tiers: large / mid / small
- Signal decay: IC at horizons 1, 3, 5, 10, 21 days

---

## 9. Phases 5–6 — Barra-Style Risk Model

### 9.1 Descriptors (`risk/descriptors.py`)

| Factor | Descriptors |
|---|---|
| Size | log(market cap) |
| Value | B/P, E/P, CF/P |
| Momentum | 12-1 return (skip most recent month) |
| Volatility | 60d realized vol, CAPM residual vol, beta |
| Liquidity | 21/60/252d turnover ratios |
| Leverage | debt/assets, debt/equity |
| Growth | 3yr sales growth, 3yr earnings growth |
| Quality | ROE, accruals, earnings variability |

Plus GICS L1 industry dummies.

### 9.2 Standardization (`risk/standardize.py`)

Order matters:
1. Winsorize each descriptor at ±3σ
2. **Z-score within industry** (Barra convention — not global)
3. Multi-descriptor factors: equal-weight component z-scores, then re-z-score
4. Demean so cap-weighted mean exposure is zero

### 9.3 Factor returns (`risk/factor_returns.py`)

```python
def estimate_factor_returns(exposures, returns, mcap, industry) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-sectional WLS per period. Weights = sqrt(mcap).
    Constraint: cap-weighted industry factor returns sum to zero, so the intercept
    is interpretable as the market factor.
    Returns (factor_returns [T x K], specific_returns [T x N])."""
```

Monthly frequency. Also produce a weekly variant for robustness.

### 9.4 Covariance (`risk/covariance.py`)

```python
def factor_covariance(factor_returns, hl_vol=40, hl_corr=90, nw_lags=5,
                      eigen_adjust=True) -> np.ndarray:
    """EWMA with SEPARATE half-lives for volatility and correlation — Barra uses
    different decay for each and combining them into one is the common shortcut
    that degrades the bias statistic.
    Newey-West adjustment with nw_lags for serial correlation.
    Optional eigenfactor adjustment: simulate returns from the estimated cov,
    re-estimate, measure eigenvalue bias by eigenvector, apply scaling correction."""
```

```python
def specific_risk(specific_returns, metadata, hl=60, shrink=True) -> np.ndarray:
    """EWMA of squared residuals, shrunk toward a structural model regressing
    log specific vol on size, leverage, and industry."""
```

### 9.5 Phase 6 gate — bias tests (`risk/bias_tests.py`)

```python
def bias_statistic(portfolio_returns, predicted_vols) -> float:
    """b = std(realized_return / predicted_vol). Target ≈ 1.0.
    b > 1 => underforecasting risk.  b < 1 => overforecasting."""
```

Run on: 1000 random portfolios, each factor-mimicking portfolio, and the cap-weighted market. Compare against sample covariance and Ledoit-Wolf shrinkage benchmarks.

**Gate: bias statistic ∈ [0.9, 1.1] for the majority of test portfolios.** Do not proceed to Phase 7 otherwise — an uncalibrated covariance makes the attribution meaningless.

---

## 10. Phase 7 — The Join

`experiments/join.py`. This is where the two projects become one result.

1. **Exposure time series** — run the MASTER long-short portfolio through the risk model; plot style exposures across the test period.
2. **Return attribution** — decompose realized P&L into factor return contribution vs. specific (alpha) return.
3. **Risk attribution** — factor vs. specific share of predicted variance; marginal contribution to risk by factor.
4. **Neutralized portfolio** — `cost_aware_optimize` with factor-neutrality constraints on all style factors. Report its **net** Sharpe.
5. **Gate interpretation** — regress learned gate activations on (a) the market state vector and (b) contemporaneous Barra factor returns. If the gate performs regime-conditional factor timing, this shows it in a vocabulary a portfolio manager already uses. This is the highest-value single analysis in the project.

### The headline sentence, filled in

> MASTER-US long-short net Sharpe is **X**. After Barra style-neutralization it is **Y** — so **Z%** of the raw signal is factor exposure and **(1−Z)%** is specific alpha, of which the portion surviving realistic costs is **W%**. Cost breakeven occurs at **B** bps.

**If Y ≈ 0, that is the result, and it is a good one.** Write it as a finding. A candidate who reports honest neutralized alpha is more credible than one who reports a suspiciously strong raw number.

---

## 11. Build Order & Gates

| Phase | Make target | Gate to pass |
|---|---|---|
| 0 | `make data` | All four Phase-0 tests pass; coverage logged |
| 1 | `make engine` | 12-1 momentum reproduces incl. 2009 crash |
| 2 | `make baselines` | LightGBM out-of-sample RankIC > 0.02 |
| 3 | `make master` | Beats ridge; compared to LGBM with seed error bars |
| 4 | `make ablations` | Full grid + β sweep + breakeven table |
| 5 | `make factors` | Momentum/value factor return series match known patterns |
| 6 | `make risk` | Bias statistic ∈ [0.9, 1.1] |
| 7 | `make join` | Neutralized net Sharpe computed |
| 8 | `make reports` | Clean clone reproduces the headline table |

**Phase 3 does not begin until Phase 1 passes.** A transformer on an unvalidated engine produces confident nonsense with high precision.

---

## 12. Failure Modes

| Symptom | Diagnosis |
|---|---|
| Strong in-sample IC, ~0 out-of-sample | Leakage. Audit normalization statistics first, then feature timestamps, then label alignment. |
| Sharpe > 3 | Bug. Check costs applied, universe filters, delisting handling, date alignment. |
| Wide seed dispersion | Model underdetermined. Reduce capacity or regularize before believing any comparison. |
| Daily turnover > 50% | You have a cost model, not an alpha model. Raise `dropout_buffer` or `lam_turnover`. |
| Bias statistic far from 1.0 | Covariance issue — usually the single-half-life shortcut or missing Newey-West. |
| Gate ablation shows no effect | Legitimate and reportable. The mechanism may not transfer to US markets. Say so plainly. |
| LightGBM beats MASTER | Also legitimate and common. Report it. It's the most useful thing in the writeup. |

---

## 13. Reports

`reports/master_us.md` · `reports/risk_model.md` · `reports/join.md`, plus a README containing:

- The framing paragraph from §0, including the original repo's data disclosure
- Headline table: gross and net, with seed error bars
- The cost breakeven table and net-Sharpe-vs-cost curve
- The β sweep figure with no-gating reference line
- Bias test table vs. sample covariance and Ledoit-Wolf
- The neutralized alpha result
- "Limitations and what I'd do next"
- Exact reproduction: clone → install → `make` per phase
- Data access notes: what's free, what needs a subscription, what the fallback path costs in rigor
