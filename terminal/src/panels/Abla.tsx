import { Panel } from "../components/Panel";
import { Verdict } from "../components/DataTable";
import { BetaChart } from "../components/Charts";
import { abs, pm, sig, type Payload } from "../data";

/** ABLA — spec §3.3. This panel carries the project's actual finding, so it
 *  gets the most visual weight: the β sweep runs full width with the
 *  no-gating reference drawn as the line every point must escape. */
export function Abla({ data }: { data: Payload }) {
  const { ablations, beta_sweep, gate_null } = data;
  const nx = ablations.find((a) => a.key === "no_inter_stock");
  const sh = ablations.find((a) => a.key === "market_shuffled");
  const anyReal = ablations.some((a) => a.gross_distinguishable || a.net_distinguishable);

  return (
    <div className="stack">
      <Panel
        title="β sweep — gate temperature against the no-gating reference"
        sub="§8.2 · 6 temperatures × 5 seeds · the experiment that settles the question"
      >
        <BetaChart points={beta_sweep.points} reference={beta_sweep.reference} />
        <p className="note" style={{ marginTop: 12 }}>
          No temperature separates from no-gating, and the curve is flat above β=0.5. Two coherence
          checks make that flatness read as measurement rather than noise: the{" "}
          <strong>high-β limit converges to the reference</strong> exactly as theory requires
          (β→∞ approaches no gating), and <strong>hard selection actively hurts</strong> — β=0.1
          falls below the reference with several times the seed dispersion. The gate's only clearly
          detectable effect is harm when forced toward hard selection.
        </p>
      </Panel>

      <div className="grid g-2">
        <Panel title="Architectural ablations" sub="each cell differs from MSTR.US by exactly one thing" flush>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Isolates</th>
                  <th className="r">ΔRankIC</th>
                  <th>Gross</th>
                  <th className="r">Δnet L/S</th>
                  <th>Net</th>
                </tr>
              </thead>
              <tbody>
                {ablations.map((a) => (
                  <tr key={a.key}>
                    <td className="tk">{a.ticker}</td>
                    <td className="dim wrap-cell" style={{ fontSize: 11 }}>{a.isolates}</td>
                    <td className="num gross">{sig(a.d_rank_ic)}</td>
                    <td><Verdict real={a.gross_distinguishable} /></td>
                    <td className="num net">{sig(a.d_net_ls, 3)}</td>
                    <td><Verdict real={a.net_distinguishable} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="β sweep · figures" sub="vs the no-gating reference" flush>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th className="r sym">β</th>
                  <th className="r">RankIC</th>
                  <th className="r">vs ref</th>
                  <th className="r">net L/S</th>
                  <th className="r">BE</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {beta_sweep.points.map((p) => (
                  <tr key={p.beta}>
                    <td className="num mono" style={{ fontWeight: 700 }}>
                      {p.beta}
                      {p.is_default ? <span className="sd"> def</span> : null}
                    </td>
                    <td className="num">
                      <span className="gross">{sig(p.rank_ic)}</span>{" "}
                      <span className="sd">{pm(p.sd)}</span>
                    </td>
                    <td className="num dim">{sig(p.rank_ic - beta_sweep.reference.rank_ic)}</td>
                    <td className="num net">{abs(p.net_ls)}</td>
                    <td className="num warn">{abs(p.breakeven, 1)}</td>
                    <td><Verdict real={p.distinguishable} /></td>
                  </tr>
                ))}
                <tr className="rule-top">
                  <td className="num dim">ref</td>
                  <td className="num">
                    <span className="gross">{sig(beta_sweep.reference.rank_ic)}</span>{" "}
                    <span className="sd">{pm(beta_sweep.reference.sd)}</span>
                  </td>
                  <td className="num dim">—</td>
                  <td className="num net">{abs(beta_sweep.reference.net_ls)}</td>
                  <td className="num dim">—</td>
                  <td><span className="chip off">{beta_sweep.reference.ticker} no gating</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel title="Gate null · both training budgets" sub="ratio < 1.0 ⇒ gap inside pooled seed dispersion" flush>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Budget</th>
                <th>Measure</th>
                <th className="r">Gated</th>
                <th className="r">Ungated</th>
                <th className="r">Gap</th>
                <th className="r">Pooled SD</th>
                <th className="r">Ratio</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {gate_null.flatMap((g) =>
                g.measures.map((m, i) => (
                  <tr key={`${g.budget}-${m.metric}`} className={i === 0 ? "rule-top" : undefined}>
                    <td className="mono dim">{i === 0 ? g.budget : ""}</td>
                    <td className="nm">{m.metric}</td>
                    <td className="num">
                      <span className={m.basis === "net" ? "net" : "gross"}>{abs(m.gated, 4)}</span>{" "}
                      <span className="sd">{pm(m.gated_sd)}</span>
                    </td>
                    <td className="num">
                      <span className={m.basis === "net" ? "net" : "gross"}>{abs(m.ungated, 4)}</span>{" "}
                      <span className="sd">{pm(m.ungated_sd)}</span>
                    </td>
                    <td className="num">{sig(m.gap)}</td>
                    <td className="num dim">{m.pooled.toFixed(4)}</td>
                    <td
                      className="num"
                      style={{ fontWeight: 700, color: (m.ratio ?? 0) < 1 ? "var(--noise)" : "var(--warn)" }}
                    >
                      {m.ratio?.toFixed(3) ?? "—"}
                    </td>
                    <td><Verdict real={m.distinguishable} /></td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Reading the grid">
        <p className="note">
          {anyReal ? (
            <>At least one ablation separates from full MASTER; see the verdict column.</>
          ) : (
            <>
              <strong>None of MASTER's three structural mechanisms is distinguishable from its
              ablation.</strong>{" "}
              Removing inter-stock attention changes RankIC by {sig(nx?.d_rank_ic ?? NaN)}.
            </>
          )}{" "}
          And <code>{sh?.ticker}</code> — the gate fed a date-permuted market vector, its entire
          input destroyed while every marginal distribution and cross-feature correlation is
          preserved — costs {sig(sh?.d_rank_ic ?? NaN)}. The gate is not reading market structure.
        </p>
        <p className="note">
          What would legitimately re-open this: an architectural change to the gating mechanism
          itself, or a materially different experimental setup — different universe, label horizon,
          or feature bank. Not more seeds and not more epochs; both were tried, and the ratio moved
          the wrong way.
        </p>
      </Panel>
    </div>
  );
}
