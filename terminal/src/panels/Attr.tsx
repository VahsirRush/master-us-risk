import { Panel } from "../components/Panel";
import { Verdict } from "../components/DataTable";
import { abs, sig, type Payload } from "../data";

/** ATTR — gate factor-timing null and the eigenfactor prediction (spec §3.6 / §10.5).
 *
 * The method note lives in this panel, not only in the docs: a viewer of ATTR
 * alone must understand that this measures the gate's consequence, not an
 * activation regression. */
export function Attr({ data }: { data: Payload }) {
  const attr = data.attr;
  if (!attr) return null;

  const { timing, timing_summary: s, eigen } = attr;

  return (
    <div className="stack">
      <Panel
        title="§10.5 Gate interpretation — factor timing"
        sub={`${s.n_significant}/${s.n_factors} factors significant · ${attr.n_seeds} seeds`}
      >
        <div className="deferred" style={{ marginBottom: 14 }}>
          <div className="hd">method note · substitute, not implementation</div>
          <p className="note" style={{ marginTop: 6 }}>{s.method_note}</p>
        </div>

        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Factor</th>
                <th className="r">MASTER</th>
                <th className="r">Ungated</th>
                <th className="r">Gate Δ</th>
                <th className="r">t (gated)</th>
                <th>Significant</th>
              </tr>
            </thead>
            <tbody>
              {timing.map((r) => (
                <tr key={r.factor}>
                  <td className="nm">{r.factor}</td>
                  <td className="num gross">{sig(r.gated, 4)}</td>
                  <td className="num dim">{sig(r.ungated, 4)}</td>
                  <td className="num net">{sig(r.delta, 4)}</td>
                  <td className="num dim">{sig(r.t_stat, 2)}</td>
                  <td>
                    <span className={r.significant ? "chip real" : "chip nd"}>
                      {r.significant ? "yes" : "no"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="hero" style={{ marginTop: 16 }}>
          <div className="figure">
            <div className="l">significant</div>
            <div className="v" style={{ color: "var(--noise)" }}>
              {s.n_significant}/{s.n_factors}
            </div>
          </div>
          <div className="figure">
            <div className="l">largest Δ · {s.largest_delta_factor}</div>
            <div className="v sm warn">{sig(s.largest_delta, 4)}</div>
          </div>
          <div className="figure">
            <div className="l">seed dispersion</div>
            <div className="v sm dim">{abs(s.seed_dispersion, 4)}</div>
          </div>
        </div>

        <p className="note" style={{ marginTop: 14 }}>
          <strong>
            {s.n_significant}/{s.n_factors} factors show significant timing.
          </strong>{" "}
          Largest gate delta {sig(s.largest_delta, 4)} ({s.largest_delta_factor})
          against seed dispersion {abs(s.seed_dispersion, 4)} — inside the noise.
          The gate does no detectable regime-conditional factor timing. Together
          with Phases 3–4 (the gate does not predict returns distinguishably
          better than no gating), this completes the null: a mechanism that
          neither improves prediction nor times factors.
        </p>
      </Panel>

      <Panel
        title="Eigenfactor prediction check"
        sub={eigen.prediction_confirmed ? "confirmed" : "NOT confirmed"}
      >
        <div className="hero">
          <div className="figure">
            <div className="l">net Sharpe gap · on vs off</div>
            <div className="v sm net">{sig(eigen.gap, 4)}</div>
          </div>
          <div className="figure">
            <div className="l">verdict</div>
            <div className="v sm" style={{ color: "var(--ink-2)" }}>
              {eigen.distinguishable ? "DISTINGUISHABLE" : "NOT DISTINGUISHABLE"}
            </div>
          </div>
          <div className="figure" style={{ marginLeft: "auto" }}>
            <Verdict real={eigen.distinguishable} />
            {!eigen.prediction_confirmed ? (
              <span className="chip warn" style={{ marginLeft: 8 }}>failed prediction</span>
            ) : (
              <span className="chip ok" style={{ marginLeft: 8 }}>confirmed</span>
            )}
          </div>
        </div>
        <p className="note" style={{ marginTop: 14 }}>{eigen.caveat}</p>
      </Panel>
    </div>
  );
}
