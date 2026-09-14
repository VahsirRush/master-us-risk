import { Panel } from "../components/Panel";
import { pct, type Payload } from "../data";

/** DATA — spec §3.7: provenance. Survivorship, the free-tier limits stated
 *  plainly, the split protocol, and the deferred sweeps. */
export function Data({ data }: { data: Payload }) {
  const { survivorship, provenance, deferred } = data;
  const h = survivorship.headline;
  const yrs = survivorship.by_year;

  const W = 880, H = 220, L = 48, R = 20, T = 16, B = 32;
  const x0 = yrs[0]?.year ?? 0, x1 = yrs[yrs.length - 1]?.year ?? 1;
  const X = (v: number) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - 0.7) / 0.3) * (H - T - B);
  const path = yrs.map((r, i) => `${i ? "L" : "M"}${X(r.year)},${Y(r.rate)}`).join(" ");
  const last = yrs[yrs.length - 1];

  return (
    <div className="stack">
      <Panel
        title="Survivorship — what the free-data path costs"
        sub={`${h.retrieved} / ${h.total} historical constituents retrievable · ${pct(h.rate, 1)}`}
      >
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label={`Retrieval rate by year, climbing from ${pct(yrs[0]?.rate, 0)} to ${pct(last?.rate, 0)}`}>
          {[0.7, 0.8, 0.9, 1.0].map((v) => (
            <g key={v}>
              <line className="gl" x1={L} y1={Y(v)} x2={W - R} y2={Y(v)} />
              <text className="axl" x={L - 7} y={Y(v) + 3} textAnchor="end">{(v * 100) | 0}%</text>
            </g>
          ))}
          <path d={`${path} L${X(x1)},${H - B} L${X(x0)},${H - B} Z`} fill="var(--band-net)" />
          <path d={path} fill="none" stroke="var(--net)" strokeWidth={2} />
          {yrs.filter((_, i) => i % 3 === 0 || i === yrs.length - 1).map((r) => (
            <text key={r.year} className="axl" x={X(r.year)} y={H - 12} textAnchor="middle">
              {r.year}
            </text>
          ))}
          {last ? (
            <circle cx={X(last.year)} cy={Y(last.rate)} r={4} fill="var(--net)"
                    stroke="var(--ground)" strokeWidth={2} />
          ) : null}
        </svg>
        <p className="note" style={{ marginTop: 12 }}>
          Retrieval climbs monotonically from <strong>{pct(yrs[0]?.rate, 1)}</strong> in {yrs[0]?.year}{" "}
          to <strong>{pct(last?.rate, 1)}</strong> in {last?.year}. The early sample is the most
          contaminated — and the early sample is the training set. Direction of bias:{" "}
          <strong className="warn">{h.bias_direction}</strong>. Measured, not corrected; the free
          path cannot fix it, so it is stated instead.
        </p>
      </Panel>

      <div className="grid g-2e">
        <Panel title="Coverage by year" flush>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Year</th>
                  <th className="r">Constituents</th>
                  <th className="r">Retrieved</th>
                  <th className="r">Rate</th>
                </tr>
              </thead>
              <tbody>
                {yrs.map((r) => (
                  <tr key={r.year}>
                    <td className="mono">{r.year}</td>
                    <td className="num dim">{r.constituents}</td>
                    <td className="num gross">{r.retrieved}</td>
                    <td className="num net">{pct(r.rate, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Protocol">
            <dl className="kv">
              <dt>Train</dt><dd>{provenance.splits.train}</dd>
              <dt>Valid</dt><dd>{provenance.splits.valid}</dd>
              <dt>Test</dt><dd>{provenance.splits.test}</dd>
              <dt>Embargo</dt><dd>{provenance.splits.embargo_days} d</dd>
              <dt>Seeds required</dt><dd>{provenance.required_seeds}</dd>
              <dt>Cost baseline</dt><dd>{provenance.cost_model.baseline_bps} bps</dd>
            </dl>
            <p className="note" style={{ marginTop: 11 }}>
              Chronological splits, no shuffling and no k-fold. Fundamentals join on the SEC{" "}
              <code>filed</code> date, never <code>period_end</code>. Normalization statistics are
              fit on train only and borrowed by valid and test.
            </p>
          </Panel>

          <Panel title="Sources">
            <dl className="kv">
              <dt>Metrics bundle</dt><dd style={{ fontSize: 10 }}>{provenance.bundle}</dd>
              <dt>Scores</dt><dd style={{ fontSize: 10 }}>{provenance.scores}</dd>
              <dt>Phase results</dt><dd style={{ fontSize: 10 }}>{provenance.phases}</dd>
              <dt>Survivorship</dt><dd style={{ fontSize: 10 }}>{provenance.survivorship}</dd>
            </dl>
          </Panel>
        </div>
      </div>

      <Panel title="Deferred — not run, not estimated" sub="see methodology">
        <div className="stack">
          {deferred.map((d) => (
            <div className="deferred" key={d.row}>
              <div className="hd">{d.row} · {d.status} · {d.compute}</div>
              <div className="nm2">{d.name}</div>
              <p className="note">{d.detail}</p>
            </div>
          ))}
          <p className="note">
            Neither affects the gate-null conclusion, which was established at the default
            configuration with 10 confirmatory seeds at the spec's full budget, a 25-run β sweep,
            and a market-shuffle control. A sweep could only show the gate helps at some{" "}
            <em>other</em> hyperparameter setting — a new claim carrying its own evidential burden.
          </p>
        </div>
      </Panel>
    </div>
  );
}
