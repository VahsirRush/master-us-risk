# Research Terminal — Dashboard Specification

**Replaces §5 (HTML Report) of the output layer spec.** §1–4 and §6–7 stand: the terminal renderers and the `MetricValue` / `PhaseResult` contract remain the single source of truth. This document specifies the web surface that consumes them.

---

## 0. The Conceit

Not a report. A **research terminal** where the instruments being tracked are your own models.

Every model variant is a ticker:

```
MSTR.US     Full MASTER
MSTR.NG     No gating
MSTR.NX     No inter-stock attention
MSTR.SH     Shuffled market vector
LGBM.BL     LightGBM baseline
RIDG.BL     Ridge floor
BARR.RM     Risk model (bias stat as the tracked quantity)
```

This is the organizing idea and it should be visible in the first two seconds. A visitor lands on something that looks like it's monitoring live positions, then realizes it's monitoring *research artifacts* — model variants, their net Sharpe, their turnover, their breakeven, their drift across seeds. The frame does real work: it forces every result into an instrument-like shape with a price (net Sharpe), a volatility (seed dispersion), and a carrying cost (turnover).

**Why this beats a report:** a research note is read once. A terminal is *operated*. The visitor clicks, sorts, switches panels, pulls up a variant — and in doing so reads far more of your work than a scrolling document would ever get them to.

---

## 1. Two Modes, One Codebase

| Mode | Use | Data source |
|---|---|---|
| **Live** | During builds — training runs, ablation grids | Local FastAPI + WebSocket, reading `PhaseResult` cache as it's written |
| **Static** | Portfolio link | Same frontend, results baked into a single JSON blob at export |

The frontend cannot tell the difference. It talks to `/api/results` and `/ws/live`; the static export ships a service-worker shim that serves the frozen JSON from the same paths. Build the live mode first — you'll use it constantly — and the portfolio artifact falls out for free.

```
src/master_us/terminal/
├── server.py           # FastAPI: /api/results, /api/variant/{id}, /ws/live
├── serialize.py        # PhaseResult / AblationResult -> JSON
├── export.py           # freeze to reports/terminal/ as static bundle
└── web/
    ├── index.html
    ├── app.js
    ├── theme.css
    └── panels/{monitor,quote,chart,grid,risk,attrib}.js
```

---

## 2. Visual System

The brief is FactSet × Bloomberg, so the direction is a professional terminal — but built from actual terminal conventions rather than a generic dark theme.

### Palette

```css
--void:      #06070A;   /* page ground */
--panel:     #0D0F14;   /* panel fill, one step up from void */
--rule:      #1C212B;   /* hairline borders, 1px, everywhere */
--amber:     #FFA028;   /* primary data — the Bloomberg inheritance */
--bone:      #C8CDD6;   /* labels, secondary text */
--dim:       #5A626F;   /* headers, units, inactive */
--up:        #3FB950;   /* positive delta */
--down:      #E5484D;   /* negative delta */
--net:       #4EA8DE;   /* net-of-cost figures — the project's thesis color */
--flag:      #D97757;   /* gate failures, warnings */
```

Amber is the terminal inheritance and carries the primary numbers. **Net figures render in `--net` blue** — the one place the palette departs from Bloomberg convention, and it departs deliberately: this project is about the gap between gross and net, so net gets its own channel. Gross figures sit in `--bone`, quieter than net. That inversion is the design argument.

### Type

```css
--display: 'IBM Plex Sans Condensed';  /* panel headers, command bar — condensed
                                          reads as instrumentation, not editorial */
--data:    'IBM Plex Mono';            /* every number, tabular figures on */
--label:   'IBM Plex Sans';            /* the rare sentence of prose */
```

`font-variant-numeric: tabular-nums` on every numeric cell, no exceptions. Columns of figures must align on the decimal or the whole illusion collapses.

Type scale is tight: 11px labels, 12px data, 13px panel headers, 28px for the one hero figure. Terminals are dense. Resist the urge to breathe.

### Structure

- **Zero border radius. Zero shadows. Zero gradients.** Hairline `--rule` borders separate everything.
- 4px grid. Panel padding 8px. Row height 22px.
- No animation except: value-change flash (120ms background pulse in `--up`/`--down`), and a 1px caret blink in the command bar. Nothing else moves. Respect `prefers-reduced-motion` by dropping the flash.

### The command bar

Top of viewport, full width, always present. Bloomberg's function-key line, adapted:

```
MSTR.US  <MONITOR>  <QUOTE>  <ABLA>  <COST>  <RISK>  <ATTR>  <DATA>       BUILD 8/8 ✓
```

Keyboard-driven: type a ticker + Enter to load it into the quote panel; F1–F7 switch panels; `/` focuses the command line. Mouse works too, but the keyboard path exists and is discoverable via a `?` overlay. This is the signature element — it's what makes it feel operated rather than viewed, and it's cheap to build.

---

## 3. Panels

### 3.1 MONITOR — the landing view

Four-quadrant grid, no scrolling at 1440×900.

```
┌─ VARIANTS ─────────────────────────┬─ COST CASCADE ──────────────────┐
│ TICKER    NET SH   Δ     TURN  BE  │                                 │
│ MSTR.US    0.94  ─      38.2%  24  │  gross      ████████████ 1.61   │
│ MSTR.NG    0.81  -0.13  41.0%  19  │  −10bps     ████████     0.94   │
│ MSTR.NX    0.62  -0.32  36.8%  14  │  −realistic ██████       0.71   │
│ MSTR.SH    0.79  -0.15  40.4%  18  │  −neutral   ███          0.38   │
│ LGBM.BL    0.88  -0.06  22.1%  41  │                                 │
│ RIDG.BL    0.41  -0.53  18.4%  29  │  survives: 24% of gross         │
├─ EQUITY ───────────────────────────┼─ BUILD ─────────────────────────┤
│ [lightweight-charts: cumulative    │ 0 DATA      ✓  3,142d · 498/day │
│  net return, selected variants,    │ 1 ENGINE    ✓  mom Sh 0.61      │
│  benchmark dashed, regime bands    │ 2 BASELINE  ✓  LGBM RIC .0312   │
│  shaded]                           │ 3 MASTER    ✓  RIC .0341 ±.0058 │
│                                    │ 4 ABLATION  ✓  7 variants       │
│                                    │ 5 FACTORS   ✓                   │
│                                    │ 6 RISK      ✓  bias 0.97        │
│                                    │ 7 JOIN      ✓  neut Sh 0.38     │
└────────────────────────────────────┴─────────────────────────────────┘
```

The VARIANTS panel is a live-sortable blotter. Click a column header to sort; click a row to load it into QUOTE. Rows whose delta is inside pooled seed dispersion render in `--dim` with the delta struck through — the noise marking from the terminal spec, carried through.

The cost cascade survives from the earlier spec but becomes a panel rather than a hero. Same argument: the headline number visibly degrading is the credibility move.

### 3.2 QUOTE — single variant detail

Bloomberg's `DES` page for a model.

- **Header block:** ticker, full name, what it isolates, parameter count, training wall-clock
- **Metrics grid:** IC, RankIC, ICIR, RankICIR, net Sharpe, max DD, turnover, breakeven — each with `± std` and seed count, gross and net paired
- **Seed strip:** five small sparklines, one per seed, equity curves overlaid. Dispersion visible at a glance — this is the panel that makes seed variance impossible to ignore
- **Regime table:** 2018Q4, 2020Q1, 2022, 2023–25, each with net Sharpe and drawdown

### 3.3 ABLA — the grid

Full ablation matrix, dense, sortable, with a heatmap overlay on the delta column (ECharts). Toggle between metrics with number keys. The β sweep sits below as a line chart with the no-gating horizontal reference — same figure the paper published, on your data.

### 3.4 COST — the contribution panel

- Net Sharpe vs. assumed bps, all variants, one axis, zero-crossing marked per line
- Turnover distribution histogram per variant
- Breakeven table, sorted
- Toggle: flat-bps model vs. Corwin-Schultz + impact model

This panel is the thing the original paper didn't publish. Give it the most chart real estate.

### 3.5 RISK — the Barra half

- Factor return cumulative lines, selectable
- Factor correlation heatmap
- Bias statistic time series with the [0.9, 1.1] band shaded, plus benchmark lines for sample covariance and Ledoit-Wolf
- Specific vs. factor risk share, stacked area over time

### 3.6 ATTR — the join

- Style exposure of MSTR.US over the test period, stacked area
- Return attribution waterfall: total → each factor's contribution → specific alpha
- Neutralized vs. raw equity curves overlaid
- **Gate interpretation:** learned gate activations regressed on Barra factor returns, shown as a rolling coefficient heatmap. If the gate does regime-conditional factor timing, it shows up here as visible banding — and that's the most interesting single image in the project

### 3.7 DATA — provenance

Not decorative. This panel is where the free-data-path honesty lives:
- Universe count over time
- Survivorship retrieval rate by year, with unretrievable names counted
- XBRL tag resolution coverage by concept and year
- Feature NaN rates
- Data pin hash and last pull timestamp

A visitor who opens this panel and sees measured limitations rather than silence learns more about you than any other screen.

---

## 4. Charting Libraries

```
lightweight-charts@4    equity curves, cumulative returns — TradingView's
                        library, financial-chart-native, correct affordances
                        (crosshair, time axis, log scale) for free
echarts@5               heatmaps, stacked areas, waterfall, correlation matrix
uPlot@1                 sparklines and the seed strip — sub-millisecond redraw,
                        matters when five seeds × six variants are on screen
```

No React needed. Vanilla JS with a small panel-registry pattern keeps the bundle under 300KB and the static export trivially portable. If a framework creeps in, the portability requirement in §1 is the thing that breaks.

---

## 5. Live Mode

```python
# server.py
@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """Push on every PhaseResult write and every epoch boundary during training.
    Payload: {phase, variant, seed, epoch, metrics, status}.
    Frontend updates in place with a value-change flash. No polling."""
```

During a Phase 4 run, MONITOR becomes a live board: variants populate as they finish, deltas recompute, the build ladder advances. That's genuinely useful — you'll catch a diverging seed from across the room.

---

## 6. Copy Rules (amended)

Terminals don't write prose. The rules from the report spec compress:

- **Labels are abbreviations.** `NET SH`, `TURN`, `BE`, `RIC`. Full names live in tooltips and the QUOTE header.
- **Every number carries its units and its uncertainty** in the cell or immediately beside it. Never a bare float.
- **Failures render, they don't hide.** A failed gate shows in `--flag` on the BUILD panel with the assertion text. A variant that lost to baseline stays on the board.
- **The one prose block** is a single paragraph in the QUOTE header of MSTR.US, stating the headline finding in plain language. Everything else is instrumentation.

---

## 7. Build Order

| When | Build |
|---|---|
| Phase 1 | `serialize.py`, FastAPI skeleton, BUILD panel only |
| Phase 3 | Live WebSocket, seed strip |
| Phase 4 | MONITOR, VARIANTS blotter, QUOTE, ABLA |
| Phase 4 | COST panel — the contribution |
| Phase 6 | RISK |
| Phase 7 | ATTR, gate interpretation heatmap |
| Phase 8 | DATA panel, static export, command bar polish |

Build BUILD-panel-only first, in an afternoon. It replaces `make status` and you'll leave it open on a second monitor for the entire project.
