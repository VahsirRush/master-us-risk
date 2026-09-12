import { Panel } from "../components/Panel";
import { CostChart, TurnoverHist } from "../components/Charts";
import { abs, pct, type Payload } from "../data";

/** COST — spec §3.4: net Sharpe vs assumed bps for every variant on one
 *  axis, the breakeven table, and turnover distributions. */
export function Cost({ data }: { data: Payload }) {
  const { cost, provenance } = data;
  const baseline = provenance.cost_model.baseline_bps;
  const bes = cost.rows.map((r) => r.breakeven).filter(Number.isFinite);
  const lo = Math.min(...bes), hi = Math.max(...bes);
  const byBe = [...cost.rows].sort((a, b) => b.breakeven - a.breakeven);
  const shown = cost.rows.slice(0, 6);

  return (
    <div className="stack">
      <Panel
        title="Net Sharpe vs assumed cost"
        sub="dollar-neutral decile book · the analysis the original paper omitted"
      >
        <CostChart bps={cost.bps_grid} rows={shown} baseline={baseline} />
        <div className="legend">
          {shown.map((r, i) => (
            <span key={r.key}>
              <span
                className="sw"
                style={{ background: ["var(--net)", "var(--gross)", "var(--accent)", "var(--warn)", "var(--pos)", "var(--neg)"][i % 6] }}
              />
              {r.ticker} <span className="dim">{r.name}</span>
            </span>
          ))}
        </div>
      </Panel>

      <Panel title="Breakeven" sub="bps at which net alpha reaches zero" flush>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Variant</th>
                <th className="r">Breakeven</th>
                <th className="r">Turnover</th>
                {cost.bps_grid.map((b) => (
                  <th className="r" key={b}>@{b}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {byBe.map((r) => (
                <tr key={r.key}>
                  <td className="tk">{r.ticker}</td>
                  <td className="nm">{r.name}</td>
                  <td className="num warn" style={{ fontWeight: 700 }}>{abs(r.breakeven, 1)}</td>
                  <td className="num dim">{pct(r.turnover_mean, 0)}</td>
                  {r.curve.map((v, i) => (
                    <td className="num" key={i}>
                      <span className={v > 0 ? "net" : "neg"}>{abs(v)}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Why this panel exists">
        <p className="note">
          Every variant breaks even between <strong>{abs(lo, 1)} and {abs(hi, 1)} bps</strong>{" "}
          against a <strong>{baseline} bps</strong> baseline assumption. The whole family straddles
          the cost line: at {baseline} bps the net Sharpes cluster around zero, and by 20 bps
          everything is decisively underwater. A gross-only table would have shown a set of
          plausible alpha models.
        </p>
        <p className="note">
          Cost model: {provenance.cost_model.spread}, plus {provenance.cost_model.impact}. Turnover
          is a permanent column throughout this project, never an optional one — the two orderings
          genuinely differ, and a model can rank last on gross RankIC while ranking first on net.
        </p>
      </Panel>

      <Panel title="Turnover distributions" sub="daily one-way turnover, pooled across seeds">
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
          {shown.map((r) => (
            <div key={r.key}>
              <div style={{ fontSize: 11, marginBottom: 4 }}>
                <span className="tk">{r.ticker}</span> <span className="dim">{r.name}</span>
              </div>
              <TurnoverHist {...r.turnover_hist} />
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
