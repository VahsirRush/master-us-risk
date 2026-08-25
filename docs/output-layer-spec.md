# Output & Presentation Layer — Specification

**Spec addendum.** This governs everything the project *shows* — terminal output during runs, and the report artifacts at the end. It does not change any computation.

---

## 0. Principle

Two audiences, two surfaces, both from one source of truth:

| Surface | Audience | Purpose |
|---|---|---|
| **Terminal** | You, mid-build | See a phase's state at a glance; catch a broken run in 5 seconds, not 40 minutes |
| **HTML report** | Hiring managers | One scrollable artifact that answers "does this person know what they're doing" |

**Hard rule:** both surfaces render from the same `Result` objects. No number is ever formatted by hand or copied between them. If a metric appears in the terminal and the report with different values, the architecture is wrong.

---

## 1. Libraries

```toml
# add to pyproject.toml
"rich>=13.7",          # tables, panels, progress, live displays
"textual>=0.60",       # the interactive dashboard (Phase 4+)
"plotext>=5.2",        # terminal-native plots, no window needed
"typer>=0.12",         # CLI with automatic Rich formatting
"jinja2>=3.1",         # HTML report templating
"matplotlib>=3.8",     # report figures
```

**Division of labor:** Rich for anything linear and scrolling. Textual only for the ablation dashboard, where you need to navigate a grid of results. plotext for during-run sanity checks — a loss curve you can see over SSH without opening a window. Don't build a Textual app where a Rich table would do.

---

## 2. The Result Contract

Every phase emits one of these. Renderers consume them. Nothing renders raw dicts.

```python
# src/master_us/reporting/results.py

@dataclass(frozen=True)
class MetricValue:
    """A number that always knows its own uncertainty and cost basis."""
    gross: float
    net: float | None = None
    std: float | None = None          # across seeds
    n_seeds: int | None = None

    def render(self, fmt: str = ".4f") -> str:
        """'0.0342 ± 0.0061' or '0.0342 (net 0.0198) ± 0.0061'"""

    @property
    def distinguishable_from(self, other: "MetricValue") -> bool:
        """True iff |self.gross - other.gross| > pooled std.
        Renderers use this to gray out non-results."""


@dataclass(frozen=True)
class PhaseResult:
    phase: int
    name: str
    status: Literal["pass", "fail", "running", "skipped"]
    gate: str                          # human-readable gate condition
    gate_passed: bool
    metrics: dict[str, MetricValue]
    duration_sec: float
    artifacts: list[Path]              # figures, tables written
    notes: list[str]                   # anything the phase wants surfaced


@dataclass(frozen=True)
class AblationResult:
    variant: str
    isolates: str                      # what this cell tests
    metrics: dict[str, MetricValue]
    baseline_delta: dict[str, float]   # signed gap vs. full model
```

---

## 3. Terminal Surface

### 3.1 Visual language

One consistent scheme across every renderer. Define once in `reporting/theme.py`.

```python
THEME = Theme({
    "pass":        "bold green",
    "fail":        "bold red",
    "running":     "bold yellow",
    "skipped":     "dim white",
    "gross":       "cyan",
    "net":         "bold cyan",        # net is emphasized — it's what matters
    "degraded":    "yellow",           # net materially below gross
    "noise":       "dim italic",       # difference within seed dispersion
    "metric":      "white",
    "label":       "dim cyan",
    "warn":        "bold yellow",
})
```

**The one opinionated choice:** net metrics are rendered brighter than gross, and any comparison smaller than seed dispersion is dimmed to `noise` style. The terminal's visual hierarchy encodes the project's thesis — costs and dispersion are the point, so they get the emphasis. A result you can't distinguish from noise should *look* like noise on screen.

### 3.2 Phase runner display

Every `make` target runs through this.

```python
def run_phase(phase: Phase) -> PhaseResult:
    """
    ┌─ Phase 4 · Ablations ────────────────────────────────────┐
    │ Gate: full grid + β sweep + breakeven table               │
    └──────────────────────────────────────────────────────────┘

      ⠹ variant 3/7 · seed 2/5 · epoch 34/100   ETA 12m 40s
      ▸ RankIC (valid)  0.0341   best 0.0356 @ epoch 29

    [on completion]
      ✓ Phase 4 complete · 47m 12s · 7 artifacts → reports/
    """
```

Rules:
- One `Live` display per phase, updated in place — no scrolling wall of epoch logs
- Always show: current unit of work, position in the grid, ETA
- Always show the *metric being optimized*, live, with its best-so-far
- On failure, the panel turns red and prints the assertion that fired, with the file and line. Never a bare traceback dump.

### 3.3 Results tables

```python
def render_ablation_table(results: list[AblationResult]) -> Table:
    """
    Ablation · US S&P 500 · test 2019-2025 · 5 seeds

    Variant                  RankIC          Net Sharpe      Turnover   vs. full
    ─────────────────────────────────────────────────────────────────────────────
    Full MASTER              0.0341 ±0.0058  0.94 ±0.21      38.2%      —
    − market gating          0.0298 ±0.0061  0.81 ±0.19      41.0%      -0.0043
    − inter-stock attn       0.0244 ±0.0049  0.62 ±0.17      36.8%      -0.0097
    − cross-time attn        0.0331 ±0.0071  0.89 ±0.24      39.1%      -0.0010  ← noise
    market vector shuffled   0.0295 ±0.0055  0.79 ±0.18      40.4%      -0.0046
    ─────────────────────────────────────────────────────────────────────────────
    LightGBM baseline        0.0312 ±0.0022  0.88 ±0.09      22.1%
    """
```

- The `← noise` marker and dim styling fire automatically from `distinguishable_from`. You should never have to decide by eye whether a gap is real.
- Baselines sit below a rule, not interleaved — they're a reference line, not a competitor row.
- Turnover is a permanent column, never optional.

### 3.4 In-terminal plots

```python
def plot_training_curve(history) -> None:
    """plotext line chart: train loss + valid RankIC, dual axis.
    Rendered inline after every seed completes. Catches divergence
    without opening a window or leaving the SSH session."""

def plot_cost_curve(models: dict[str, CostCurve]) -> None:
    """Net Sharpe vs. assumed bps, all models on one axis, with a
    marked zero-crossing per model. This is the project's headline
    finding — it should be visible from the terminal."""
```

### 3.5 Run summary

`make status` renders the full phase ladder from cached `PhaseResult` objects:

```
MASTER-US · build status

  ✓ 0  Data layer              4 tests pass · 3,142 dates · 498 names/day avg
  ✓ 1  Backtest engine         momentum Sharpe 0.61 · 2009 crash -34.2% ✓
  ✓ 2  Baselines               LGBM RankIC 0.0312 > 0.02 ✓
  ✓ 3  MASTER                  RankIC 0.0341 ±0.0058
  ⠹ 4  Ablations               variant 3/7 · running 22m
  ○ 5  Barra factors
  ○ 6  Risk model
  ○ 7  Join
  ○ 8  Reports
```

This is the single most useful thing in the whole layer. It answers "where am I" instantly and it's cheap to build.

---

## 4. The Ablation Dashboard (Textual)

Build this only after Phase 4 produces real results. It's the one place a TUI earns itself.

```
┌ MASTER-US · Ablation Explorer ───────────────────────────────────────┐
│ [g]rid  [b]eta sweep  [c]ost curves  [r]egimes  [q]uit               │
├──────────────────────────────────────────────────────────────────────┤
│  Variant             │ RankIC   │ Net Sh  │ Turn  │ Breakeven        │
│ ▸ Full MASTER        │ 0.0341   │ 0.94    │ 38.2% │ 24 bps           │
│   − gating           │ 0.0298   │ 0.81    │ 41.0% │ 19 bps           │
│   − inter-stock      │ 0.0244   │ 0.62    │ 36.8% │ 14 bps           │
├──────────────────────────────────────────────────────────────────────┤
│  Full MASTER · 5 seeds                                               │
│  seed 0  0.0389   seed 1  0.0301   seed 2  0.0355                    │
│  seed 3  0.0268   seed 4  0.0392                                     │
│                                                                      │
│  [plotext: per-seed equity curves, gross dashed / net solid]         │
└──────────────────────────────────────────────────────────────────────┘
```

Selecting a row expands per-seed detail below. The per-seed spread being visible on selection is the point — it makes dispersion impossible to ignore while browsing.

Keep it to four views. A fifth is scope creep.

---

## 5. HTML Report

### 5.1 What it is

One self-contained `reports/index.html` — no build step, no server, opens from a file path. This is what goes in a portfolio link.

### 5.2 Design direction

**Subject:** a quantitative research report. **Audience:** someone who reads research notes for a living and has thirty seconds before deciding whether to keep reading. **The page's one job:** make the headline sentence findable in three seconds and defensible in three minutes.

The reference vernacular is the sell-side research note and the academic working paper — not a SaaS landing page. That means: no hero section, no gradient, no feature cards, no call-to-action. It opens the way a research note opens.

**Tokens:**

```
Palette      ink        #14161A   body text, near-black but not black
             paper      #FBFAF7   background, warm white like paper stock
             rule       #D8D4CC   hairlines and table borders
             gross      #8B8680   muted — gross figures are context
             net        #1A4D8F   deep blue — net figures are the finding
             flag       #A63D2F   negative results, failed gates, warnings

Type         Display    a slab or transitional serif with real character —
                        set only at the report title and section numbers
             Body       a text serif at 17px/1.65 — this is a document to be
                        read, not scanned; sans-serif body would fight that
             Data       a monospace with tabular figures for every table,
                        metric, and inline number

Layout       Single column, 68ch measure, wide left margin holding section
             numbers and figure references — the working-paper margin note.
             Tables break the measure and run full width.
```

**Signature element:** the **cost cascade**. A single figure at the top of the report, before any prose, showing the headline number stepping down: gross Sharpe → net of 10bps → net of realistic costs → after factor neutralization. Four bars, each annotated with what was removed. It's the whole project in one graphic, and it's honest — it shows the number getting worse, which is exactly the credibility move.

Spend the boldness there. Everything else is quiet: hairline rules, no shadows, no radius, no animation beyond a respectful reduced-motion-safe fade on scroll.

### 5.3 Structure

```
§      Cost cascade figure                     ← the signature, before any words
§      Headline paragraph                      ← the filled-in sentence from spec §10
§ 01   Framing: why a port, not a replication
§ 02   Data & its limitations                  ← survivorship report lives here
§ 03   Architecture
§ 04   Baselines
§ 05   Ablations                               ← the full grid, noise-marked
§ 06   β sweep                                 ← with no-gating reference line
§ 07   Costs & breakeven
§ 08   Risk model & bias tests
§ 09   Attribution & neutralized alpha
§ 10   Gate interpretation
§ 11   Limitations & next steps
§      Reproduction
```

Section numbers live in the left margin, set in the display face. They're numbered because the report *is* a sequence — each section depends on the prior gate having passed. That's real structure, not decoration.

### 5.4 Generation

```python
# src/master_us/reporting/html.py
def build_report(results: dict[int, PhaseResult], cfg) -> Path:
    """Jinja2 template + matplotlib figures inlined as base64 SVG.
    Single output file, no external assets, no CDN dependencies.
    Every number pulled from PhaseResult — nothing hardcoded in the template."""
```

Matplotlib styling matches the HTML tokens: same palette, same monospace for tick labels, hairline spines, no gridlines except a single zero line where sign matters.

---

## 6. Copy Rules

The report's writing is a graded part of the artifact.

- **Name the finding in the first sentence of each section**, then support it. Never build to a reveal.
- **Negative results stated plainly**, in the flag color, without hedging. "The gating ablation is not distinguishable from the full model at 5 seeds" — not "results were mixed."
- **No adjectives on your own work.** Not "robust," not "comprehensive," not "novel." The tables carry it.
- **Every figure caption states what to look at**, not what it is. "Net Sharpe crosses zero at 24bps for the full model, 14bps without inter-stock attention" — not "Figure 7: cost sensitivity."
- **Limitations get magnitudes.** "12% of 2011 constituents unretrievable" beats "some survivorship bias may be present."

---

## 7. Build Order

Slot into the phase ladder:

| When | Build |
|---|---|
| Phase 0 | `results.py`, `theme.py`, phase panel + `make status` |
| Phase 1 | Metric tables, plotext curves |
| Phase 3 | Live training display |
| Phase 4 | Ablation table with noise marking, then the Textual dashboard |
| Phase 8 | HTML report |

Build `make status` first. It costs an hour and you'll look at it a hundred times.
