import { SERIES_COLORS, abs, sig } from "../data";

/** Inline-SVG charts.
 *
 * The spec's §4 suggests lightweight-charts / echarts / uPlot. These are
 * hand-drawn instead for one reason: the whole point of a static export is
 * that it opens anywhere with no network, and three chart libraries is
 * ~250 KB of dependency for six small plots with no interaction beyond a
 * hover. If live mode (§5) later needs streaming updates, uPlot slots in
 * behind the same props.
 *
 * Every chart here takes already-computed arrays. None of them derives a
 * statistic. */

const nice = (lo: number, hi: number, pad = 0.08) => {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo - 1, hi + 1] as const;
  const p = (hi - lo) * pad;
  return [lo - p, hi + p] as const;
};

/** End-of-line label positions, pushed apart so converging curves stay
 *  legible: a downward pass separates them, an upward pass keeps the lowest
 *  inside the plot. Positions only — the curve endpoints are not moved. */
const spreadLabels = (ys: number[], gap: number, maxY: number) => {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y);
  let last = -Infinity;
  for (const o of order) {
    o.y = Math.max(o.y, last + gap);
    last = o.y;
  }
  let limit = maxY;
  for (let k = order.length - 1; k >= 0; k--) {
    order[k].y = Math.min(order[k].y, limit);
    limit = order[k].y - gap;
  }
  const out: number[] = [];
  for (const o of order) out[o.i] = o.y;
  return out;
};

/* ── equity curves ────────────────────────────────────────────────── */

export function EquityChart({
  dates,
  series,
  basis,
}: {
  dates: string[];
  series: { key: string; ticker: string; curve: number[] }[];
  basis: string;
}) {
  const W = 880, H = 300, L = 56, R = 78, T = 16, B = 34;
  const all = series.flatMap((s) => s.curve);
  const [y0, y1] = nice(Math.min(...all), Math.max(...all));
  const X = (i: number) => L + (i / Math.max(1, dates.length - 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const yticks = [y0, (y0 + y1) / 2, 0, y1].filter((v, i, a) => a.indexOf(v) === i && v >= y0 && v <= y1);
  const years = dates
    .map((d, i) => ({ y: d.slice(0, 4), i }))
    .filter((d, i, a) => i === 0 || d.y !== a[i - 1].y);
  const labelY = spreadLabels(series.map((s) => Y(s.curve[s.curve.length - 1])), 12, H - B);

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cumulative net return by variant over the 2019-2025 test period">
        {yticks.map((v) => (
          <g key={v}>
            <line className="gl" x1={L} y1={Y(v)} x2={W - R} y2={Y(v)} />
            <text className="axl" x={L - 7} y={Y(v) + 3} textAnchor="end">{pctLabel(v)}</text>
          </g>
        ))}
        {y0 < 0 && y1 > 0 ? (
          <line x1={L} y1={Y(0)} x2={W - R} y2={Y(0)} stroke="var(--hair-2)" strokeWidth={1} />
        ) : null}
        {years.map((y) => (
          <text key={y.y} className="axl" x={X(y.i)} y={H - 12} textAnchor="middle">{y.y}</text>
        ))}
        {series.map((s, si) => {
          const c = SERIES_COLORS[si % SERIES_COLORS.length];
          const d = s.curve.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
          return (
            <g key={s.key}>
              <path d={d} fill="none" stroke={c} strokeWidth={si === 0 ? 2.2 : 1.5}
                    strokeLinejoin="round" opacity={si === 0 ? 1 : 0.75} />
              <circle cx={X(s.curve.length - 1)} cy={Y(s.curve[s.curve.length - 1])} r={3.5}
                      fill={c} stroke="var(--ground)" strokeWidth={1.5} />
              <text className="axl" x={W - R + 7} y={labelY[si] + 3} style={{ fill: c }}>
                {s.ticker}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="note" style={{ marginTop: 10 }}>{basis}</p>
    </>
  );
}

const pctLabel = (v: number) => (v * 100).toFixed(0) + "%";

/* ── beta sweep — the project's headline chart ────────────────────── */

export function BetaChart({
  points,
  reference,
}: {
  points: { beta: number; rank_ic: number; sd: number; is_default: boolean }[];
  reference: { ticker: string; rank_ic: number; sd: number };
}) {
  const W = 880, H = 320, L = 62, R = 24, T = 22, B = 44;
  const xs = points.map((p) => Math.log10(p.beta));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const vals = points.flatMap((p) => [p.rank_ic - p.sd, p.rank_ic + p.sd])
    .concat([reference.rank_ic - reference.sd, reference.rank_ic + reference.sd]);
  const [y0, y1] = nice(Math.min(...vals), Math.max(...vals), 0.14);
  const X = (v: number) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const bandTop = Y(reference.rank_ic + reference.sd);
  const bandH = Math.max(1, Y(reference.rank_ic - reference.sd) - bandTop);

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Gate temperature sweep: every beta sits inside the no-gating reference band">
        {[y0, (y0 + y1) / 2, y1].map((v) => (
          <g key={v}>
            <line className="gl" x1={L} y1={Y(v)} x2={W - R} y2={Y(v)} />
            <text className="axl" x={L - 8} y={Y(v) + 3} textAnchor="end">{v.toFixed(4)}</text>
          </g>
        ))}

        {/* the no-gating reference and its ±1 SD band — the line every
            point has to escape to mean anything */}
        <rect x={L} y={bandTop} width={W - L - R} height={bandH} fill="var(--band-gross)" />
        <line x1={L} y1={Y(reference.rank_ic)} x2={W - R} y2={Y(reference.rank_ic)}
              stroke="var(--gross)" strokeWidth={1.6} strokeDasharray="6 4" />
        {/* left end: the curve starts below the reference, so the label sits
            clear of it; at the right the points ride the line */}
        <text className="axl em" x={L + 10} y={Y(reference.rank_ic) - 8} textAnchor="start"
              style={{ fill: "var(--gross)" }}>
          {reference.ticker} no gating · ±1 SD
        </text>

        <path d={points.map((p, i) => `${i ? "L" : "M"}${X(Math.log10(p.beta))},${Y(p.rank_ic)}`).join(" ")}
              fill="none" stroke="var(--net)" strokeWidth={2} strokeLinejoin="round" />

        {points.map((p) => {
          const x = X(Math.log10(p.beta));
          return (
            <g key={p.beta}>
              <line x1={x} y1={Y(p.rank_ic - p.sd)} x2={x} y2={Y(p.rank_ic + p.sd)}
                    stroke="var(--net)" strokeWidth={1.6} opacity={0.55} />
              <line x1={x - 3} y1={Y(p.rank_ic + p.sd)} x2={x + 3} y2={Y(p.rank_ic + p.sd)}
                    stroke="var(--net)" strokeWidth={1.4} opacity={0.55} />
              <line x1={x - 3} y1={Y(p.rank_ic - p.sd)} x2={x + 3} y2={Y(p.rank_ic - p.sd)}
                    stroke="var(--net)" strokeWidth={1.4} opacity={0.55} />
              <circle cx={x} cy={Y(p.rank_ic)} r={p.is_default ? 6 : 4.5}
                      fill={p.is_default ? "var(--accent)" : "var(--net)"}
                      stroke="var(--ground)" strokeWidth={2} />
              <text className={p.is_default ? "axl em" : "axl"} x={x} y={H - 22} textAnchor="middle">
                {p.beta}
              </text>
              {p.is_default ? (
                <text className="axl" x={x} y={H - 10} textAnchor="middle" style={{ fill: "var(--accent)" }}>
                  default
                </text>
              ) : null}
            </g>
          );
        })}
        <text className="axl" x={(L + W - R) / 2} y={T - 8} textAnchor="middle">
          gross RankIC vs gate temperature β (log scale)
        </text>
      </svg>
    </>
  );
}

/* ── net Sharpe vs assumed cost ───────────────────────────────────── */

export function CostChart({
  bps,
  rows,
  baseline,
}: {
  bps: number[];
  rows: { key: string; ticker: string; curve: number[] }[];
  baseline: number;
}) {
  const W = 880, H = 320, L = 58, R = 86, T = 18, B = 42;
  const all = rows.flatMap((r) => r.curve);
  const [y0, y1] = nice(Math.min(...all), Math.max(...all));
  const X = (v: number) => L + (v / Math.max(...bps)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const labelY = spreadLabels(rows.map((r) => Y(r.curve[r.curve.length - 1])), 12, H - B);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Net Sharpe against assumed round-trip cost for every variant">
      {[y0, (y0 + y1) / 2, y1].map((v) => (
        <g key={v}>
          <line className="gl" x1={L} y1={Y(v)} x2={W - R} y2={Y(v)} />
          <text className="axl" x={L - 8} y={Y(v) + 3} textAnchor="end">{v.toFixed(2)}</text>
        </g>
      ))}

      {y0 < 0 && y1 > 0 ? (
        <>
          <line x1={L} y1={Y(0)} x2={W - R} y2={Y(0)} stroke="var(--hair-2)" strokeWidth={1.2} />
          <text className="axl" x={W - R - 5} y={Y(0) - 5} textAnchor="end">net Sharpe = 0</text>
        </>
      ) : null}

      <line x1={X(baseline)} y1={T} x2={X(baseline)} y2={H - B}
            stroke="var(--warn)" strokeWidth={1} strokeDasharray="4 4" opacity={0.75} />
      <text className="axl" x={X(baseline) + 5} y={T + 11} style={{ fill: "var(--warn)" }}>
        {baseline} bps baseline
      </text>

      {rows.map((r, i) => {
        const c = SERIES_COLORS[i % SERIES_COLORS.length];
        const d = r.curve.map((v, j) => `${j ? "L" : "M"}${X(bps[j])},${Y(v)}`).join(" ");
        return (
          <g key={r.key}>
            <path d={d} fill="none" stroke={c} strokeWidth={2} strokeLinejoin="round" opacity={0.92} />
            <text className="axl" x={W - R + 7} y={labelY[i] + 3} style={{ fill: c }}>
              {r.ticker}
            </text>
          </g>
        );
      })}

      {bps.map((b) => (
        <text key={b} className="axl" x={X(b)} y={H - 20} textAnchor="middle">{b}</text>
      ))}
      <text className="axl" x={(L + W - R) / 2} y={H - 5} textAnchor="middle">
        assumed round-trip cost (bps)
      </text>
    </svg>
  );
}

/* ── seed strip ───────────────────────────────────────────────────── */

/** Per-seed values as a dot strip against the mean and ±1 SD band.
 *
 * The point of this is to make dispersion legible without a number: five
 * dots clustered tight reads differently from five dots scattered across
 * the band, and that difference is exactly what the distinguishability
 * calls turn on. */
export function SeedStrip({
  values,
  digits = 4,
}: {
  values: number[];
  digits?: number;
}) {
  const W = 260, H = 34, P = 8;
  if (!values.length) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = values.length > 1
    ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1))
    : 0;
  const lo = Math.min(...values, mean - sd);
  const hi = Math.max(...values, mean + sd);
  const span = hi - lo || 1;
  const X = (v: number) => P + ((v - lo) / span) * (W - 2 * P);
  const mid = H / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: W, maxWidth: "100%" }} role="img"
         aria-label={`${values.length} seeds, mean ${mean.toFixed(digits)}, SD ${sd.toFixed(digits)}`}>
      <line x1={P} y1={mid} x2={W - P} y2={mid} stroke="var(--hair)" strokeWidth={1} />
      <rect x={X(mean - sd)} y={mid - 7} width={Math.max(1, X(mean + sd) - X(mean - sd))} height={14}
            rx={2} fill="var(--band-net)" />
      <line x1={X(mean)} y1={mid - 9} x2={X(mean)} y2={mid + 9} stroke="var(--net)" strokeWidth={1.8} />
      {values.map((v, i) => (
        <circle key={i} cx={X(v)} cy={mid} r={3.2} fill="var(--ink)" opacity={0.82}
                stroke="var(--ground)" strokeWidth={1} />
      ))}
    </svg>
  );
}

/* ── turnover distribution ────────────────────────────────────────── */

export function TurnoverHist({
  edges,
  counts,
  mean,
  p50,
}: {
  edges: number[];
  counts: number[];
  mean: number | null;
  p50: number | null;
}) {
  const W = 420, H = 142, L = 8, R = 8, T = 20, B = 24;
  if (!counts.length) return null;
  const max = Math.max(...counts);
  const lo = edges[0], hi = edges[edges.length - 1];
  const X = (v: number) => L + ((v - lo) / (hi - lo || 1)) * (W - L - R);
  const bw = (W - L - R) / counts.length;

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Daily turnover distribution, mean ${abs((mean ?? 0) * 100, 0)} percent`}>
        {counts.map((c, i) => {
          const h = (c / max) * (H - T - B);
          return (
            <rect key={i} x={L + i * bw + 0.5} y={H - B - h} width={Math.max(1, bw - 1)} height={h}
                  fill="var(--net)" opacity={0.5} rx={1.5} />
          );
        })}
        <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="var(--hair)" strokeWidth={1} />
        {p50 !== null ? (
          <>
            <line x1={X(p50)} y1={T} x2={X(p50)} y2={H - B} stroke="var(--warn)" strokeWidth={1.4} />
            <text className="axl" x={X(p50)} y={T - 6} textAnchor="middle" style={{ fill: "var(--warn)" }}>
              median
            </text>
          </>
        ) : null}
        <text className="axl" x={L} y={H - 8}>{(lo * 100).toFixed(0)}%</text>
        <text className="axl" x={W - R} y={H - 8} textAnchor="end">{(hi * 100).toFixed(0)}%</text>
      </svg>
      <dl className="kv" style={{ marginTop: 6 }}>
        <dt>mean daily turnover</dt>
        <dd className="net">{((mean ?? 0) * 100).toFixed(1)}%</dd>
        <dt>median</dt>
        <dd className="gross">{((p50 ?? 0) * 100).toFixed(1)}%</dd>
      </dl>
    </>
  );
}

/* ── cost cascade ─────────────────────────────────────────────────── */

/** Gross → net at the baseline, as a diverging bar around zero.
 *
 * A plain left-anchored bar chart would hide the sign change, which is the
 * whole content of this panel: the bar has to cross zero to show that costs
 * do not merely shrink the Sharpe, they invert it. */
export function Cascade({
  items,
}: {
  items: { label: string; value: number; kind: "gross" | "net" }[];
}) {
  const vals = items.map((i) => i.value);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const span = hi - lo || 1;
  const zero = ((0 - lo) / span) * 100;

  return (
    <div className="bars">
      {items.map((it) => {
        const v = ((it.value - lo) / span) * 100;
        const left = Math.min(zero, v), width = Math.max(0.8, Math.abs(v - zero));
        const color = it.kind === "net" ? "var(--net)" : "var(--gross)";
        return (
          <div className="row-b" key={it.label}>
            <div className="lb">{it.label}</div>
            <div className="track">
              <div className="zero" style={{ left: `${zero}%` }} />
              <div className="fill" style={{ left: `${left}%`, width: `${width}%`, background: color }} />
            </div>
            <div className="num" style={{ color }}>{abs(it.value)}</div>
          </div>
        );
      })}
    </div>
  );
}

export { sig };
