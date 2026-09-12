# MASTER-US research terminal

The results dashboard for the MASTER-US port. Structure follows
`../docs/research-terminal-spec.md` §3+; the visual language is Apple/iOS
dark, replacing that document's §2.

Every model variant is tracked as an instrument — `MSTR.US` (full MASTER),
`MSTR.NG` (no gating), `MSTR.NX` (no inter-stock attention), `MSTR.SH`
(market vector shuffled), `LGBM.BL`, `RIDG.BL` — each carrying a net Sharpe,
a seed dispersion, and a turnover.

## Run it

```bash
# 1. generate the payload from committed artifacts (needs the python venv)
~/.venvs/master-us/bin/python ../scripts/50_export_terminal.py

# 2. dev server with hot reload
npm install
npm run dev                 # http://localhost:5173

# 3. production build + local preview
npm run build
npm run preview             # http://localhost:4173

# 4. headless render check — mounts the real build, 20 assertions
npm run smoke

# 5. one self-contained .html that opens with no server
npm run build:single
~/.venvs/master-us/bin/python ../scripts/51_bundle_single.py
```

## Deploy

`npm run build` emits `dist/`, fully static with `base: "./"`, so it works
from a GitHub Pages project subpath without a rebuild:

```bash
npm run build
npx gh-pages -d dist        # or commit dist/ to a gh-pages branch
```

## Two things that will bite you

**Invoke npm binaries as `node node_modules/<pkg>/...`.** The project
directory name contains a colon. `PATH` is colon-separated, so npm's
`node_modules/.bin` entry splits into two broken paths and every binary
resolves as "command not found". Every script in `package.json` already does
this; keep it that way.

**The single-file target must stay IIFE.** Chrome fetches
`<script type="module">` with CORS, and a `file://` page has a null origin,
so an ESM single-file bundle opens to a blank page. That is what
`vite.config.single.ts` exists for.

## Layout

```
src/
  App.tsx              panel registry, payload fetch (static / inlined / live)
  data.ts              payload types + shared formatters
  theme.css            design tokens; gross vs net colour split lives here
  components/
    CommandBar.tsx     ticker input + F-key panel switching
    DataTable.tsx      sortable dense table; noise-dimming of indistinguishable deltas
    Panel.tsx          frosted card shell + the not-yet-run empty state
    Charts.tsx         inline-SVG equity, β sweep, cost curves, seed strips, histograms
  panels/              Monitor, Quote, Abla, Cost, Data
public/results.json    the payload — regenerate, never hand-edit
```

## Rules this UI enforces

- **The frontend computes no statistic.** Every panel is a pure function of
  the payload, so any number on screen traces to an array in
  `data/processed/`. Anything needing a calculation belongs in
  `src/master_us/terminal/export.py`.
- **A gap inside pooled seed dispersion renders struck-through and grey**, so
  noise cannot be scanned as a result. That is
  `MetricValue.distinguishable_from` made visual.
- **Gross and net get different colours** — the gross-to-net gap is the
  project's thesis.
- **Panels for phases that have not run stay empty** and say so. RISK and
  ATTR are wired and waiting on Phases 5-7.
- **The lookback and head sweeps appear only as deferred entries** with their
  projected compute. No chart or table implies data exists for either.
