/** Shape of `public/results.json`, plus the formatters every panel shares.
 *
 * The frontend computes no statistics. Everything below either reads a field
 * the Python exporter wrote or formats one for display — so any number on
 * screen traces back to an array in `data/processed/`. */

export interface Metric {
  gross: number;
  net: number;
  std: number | null;
  net_std: number | null;
  n_seeds: number | null;
}

export interface Hist {
  edges: number[];
  counts: number[];
  mean: number | null;
  p50: number | null;
}

export interface Variant {
  key: string;
  ticker: string;
  name: string;
  isolates: string;
  rank_ic: Metric;
  icir: Metric;
  ls_sharpe: Metric;
  breakeven_bps: Metric;
  turnover_mean: number;
  seeds: number[];
  per_seed: Record<string, number[]>;
  turnover_hist: Hist;
  cost_curve: number[];
  d_net_ls: number;
  d_rank_ic: number;
  net_distinguishable: boolean;
  gross_distinguishable: boolean;
}

export interface GateMeasure {
  metric: string;
  basis: "gross" | "net";
  gated: number;
  gated_sd: number | null;
  ungated: number;
  ungated_sd: number | null;
  gap: number;
  pooled: number;
  ratio: number | null;
  distinguishable: boolean;
}

export interface Payload {
  generated_at: string;
  provenance: {
    bundle: string;
    scores: string;
    phases: string;
    survivorship: string;
    cost_model: { baseline_bps: number; spread: string; impact: string };
    splits: { train: string; valid: string; test: string; embargo_days: number };
    required_seeds: number;
  };
  variants: Variant[];
  gate_null: { budget: string; gated: string; ungated: string; measures: GateMeasure[] }[];
  ablations: {
    key: string; ticker: string; name: string; isolates: string;
    rank_ic: Metric; ls_sharpe: Metric;
    d_rank_ic: number; gross_distinguishable: boolean;
    d_net_ls: number; net_distinguishable: boolean;
  }[];
  beta_sweep: {
    reference: { ticker: string; rank_ic: number; sd: number; net_ls: number; net_sd: number };
    points: {
      beta: number; is_default: boolean; rank_ic: number; sd: number;
      net_ls: number; net_sd: number; breakeven: number; distinguishable: boolean;
    }[];
  };
  cost: {
    bps_grid: number[];
    rows: {
      key: string; ticker: string; name: string; curve: number[];
      breakeven: number; turnover_hist: Hist; turnover_mean: number;
    }[];
  };
  equity: {
    dates: string[];
    series: { key: string; ticker: string; curve: number[] }[];
    basis: string;
  };
  stress: Record<string, {
    regimes: { name: string; ic: number | null; dates: number }[];
    cap_tiers: { tier: string; ic: number | null }[];
    decay: { h: number; ic: number }[];
    sector_neutral: number;
  }>;
  phases: {
    phase: number; label: string; status: string;
    gate_passed: boolean | null; gate: string | null;
    notes: string[]; metrics: Record<string, Metric | null>;
  }[];
  pending_panels: { id: string; label: string; spec: string; phases: string; needs: string }[];
  survivorship: {
    headline: { retrieved: number; total: number; rate: number; bias_direction: string };
    by_year: { year: number; constituents: number; retrieved: number; rate: number }[];
  };
  deferred: { row: string; name: string; status: string; compute: string; detail: string }[];
}

/* ── formatting ───────────────────────────────────────────────────── */

export const sig = (v: number | null | undefined, d = 4): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d);

export const abs = (v: number | null | undefined, d = 2): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d);

export const pct = (v: number | null | undefined, d = 1): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : (v * 100).toFixed(d) + "%";

/** ± dispersion, rendered only when it exists. A metric with no seed
 *  dispersion is a different object from one whose dispersion is zero. */
export const pm = (v: number | null | undefined, d = 4): string =>
  v === null || v === undefined ? "" : "±" + v.toFixed(d);

/** The line colours for multi-series charts, in assignment order. Net-cyan
 *  leads because the first series is always the reference variant. */
export const SERIES_COLORS = [
  "var(--net)",
  "var(--gross)",
  "var(--accent)",
  "var(--warn)",
  "var(--pos)",
  "var(--neg)",
];
