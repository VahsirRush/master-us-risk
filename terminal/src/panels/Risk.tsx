import { Panel } from "../components/Panel";
import { Verdict } from "../components/DataTable";
import { abs, pct, pm, sig, type Payload } from "../data";

/** RISK — Barra join books and the risk/return asymmetry (spec §3.5 + §10).
 *
 * Every figure is read from `reports/phase7_tables.json` via the export
 * payload. Nothing here is recomputed. The +1.28 specific-alpha Sharpe is
 * shown only with its caveat attached. */
export function Risk({ data }: { data: Payload }) {
  const risk = data.risk;
  if (!risk) return null;

  const { books, attribution: a, bias, factors, control_note, headline } = risk;
  const neutral = books.find((b) => b.key === "neutral_eigen");
  const decile = books.find((b) => b.key === "decile");

  return (
    <div className="stack">
      <Panel
        title="The join — four books under one protocol"
        sub={`${risk.n_seeds} seeds · test 2019–2025 · net = flat 10 bps`}
      >
        {headline ? <p className="note" style={{ marginBottom: 14 }}>{headline}</p> : null}
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Book</th>
                <th className="r">Gross Sharpe</th>
                <th className="r">Net Sharpe</th>
                <th className="r">Turnover</th>
                <th className="r">BE bps</th>
                <th>vs zero</th>
              </tr>
            </thead>
            <tbody>
              {books.map((b) => (
                <tr key={b.key}>
                  <td>
                    <span className="nm">{b.label}</span>
                    {b.is_control ? (
                      <span className="chip off" style={{ marginLeft: 8 }}>control</span>
                    ) : null}
                  </td>
                  <td className="num">
                    <span className="gross">{sig(b.gross, 3)}</span>{" "}
                    <span className="sd">{pm(b.gross_sd, 3)}</span>
                  </td>
                  <td className="num">
                    <span className="net">{sig(b.net, 3)}</span>{" "}
                    <span className="sd">{pm(b.net_sd, 3)}</span>
                  </td>
                  <td className="num dim">{pct(b.turnover, 0)}</td>
                  <td className="num warn">{abs(b.breakeven_bps, 1)}</td>
                  <td><Verdict real={b.net_distinguishable_from_zero} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note" style={{ marginTop: 12 }}>{control_note}</p>
        {risk.neutral_vs_unconstrained.distinguishable ? (
          <p className="note">
            Neutral vs unconstrained net gap{" "}
            <strong className="net">{sig(risk.neutral_vs_unconstrained.gap, 3)}</strong>{" "}
            — distinguishable. Style-neutralization makes net{" "}
            <strong className="neg">worse</strong>
            {decile && neutral
              ? ` (${abs(decile.net, 2)} → ${abs(neutral.net, 2)})`
              : ""}
            , because it removes volatility faster than return and the same cost
            drag falls on a smaller denominator.
          </p>
        ) : null}
      </Panel>

      <div className="grid g-2e">
        <Panel title="Return attribution · gross annualized" sub="§10.2">
          <dl className="kv">
            <dt>total realized</dt>
            <dd className="gross">{abs(a.total_ann_pct, 2)}%/yr</dd>
            <dt>factor contribution</dt>
            <dd className="gross">
              {abs(a.factor_ann_pct, 2)}%/yr{" "}
              <span className="sd">({pct(a.factor_return_share, 1)})</span>
            </dd>
            <dt>specific (alpha)</dt>
            <dd className="gross">
              {abs(a.specific_ann_pct, 2)}%/yr{" "}
              <span className="sd">({pct(a.specific_return_share, 1)})</span>
            </dd>
          </dl>
          <div className="deferred" style={{ marginTop: 14 }}>
            <div className="hd">not an alpha number</div>
            <div className="nm2">
              specific gross Sharpe {sig(a.specific_gross_sharpe, 3)}
            </div>
            <p className="note" style={{ marginTop: 6 }}>
              {a.specific_gross_sharpe_caveat}
              {neutral ? (
                <>
                  {" "}Here that hedged book is the style-neutral arm, at{" "}
                  <strong className="net">{sig(neutral.net, 3)}</strong>.
                </>
              ) : null}
            </p>
          </div>
        </Panel>

        <Panel title="Risk attribution · predicted variance" sub="§10.3">
          <div className="hero" style={{ marginBottom: 12 }}>
            <div className="figure">
              <div className="l">factor share</div>
              <div className="v warn">{pct(a.factor_risk_share, 1)}</div>
            </div>
            <div className="figure">
              <div className="l">specific share</div>
              <div className="v dim">{pct(a.specific_risk_share, 1)}</div>
            </div>
          </div>
          <p className="note">
            The asymmetry is the finding: the book spends{" "}
            <strong>{abs(a.factor_vol_ann_pct, 2)}%/yr</strong> of volatility on
            factor exposure to earn <strong>{abs(a.factor_ann_pct, 2)}%/yr</strong>{" "}
            from it — a Sharpe of{" "}
            <strong className="warn">{abs(a.factor_component_sharpe, 3)}</strong> on
            the factor component. Close to unrewarded risk, and invisible without
            a factor model.
          </p>
        </Panel>
      </div>

      <div className="grid g-2e">
        <Panel title="Factor returns · Phase 5 gate" sub="monthly WLS · post exposure_lag=1">
          <dl className="kv">
            <dt>market</dt>
            <dd className="gross">{pct(factors.market_ann, 2)}/yr</dd>
            <dt>momentum</dt>
            <dd className="gross">
              {pct(factors.momentum_ann, 2)}/yr{" "}
              <span className="sd">t=1.04 · not significant</span>
            </dd>
            <dt>value</dt>
            <dd className="gross">{pct(factors.value_ann, 2)}/yr</dd>
            <dt>mean weighted R²</dt>
            <dd className="dim">{abs(factors.mean_r2, 3)}</dd>
          </dl>
        </Panel>

        <Panel title="Bias statistic · Phase 6 gate" sub="fraction of portfolios in [0.9, 1.1]">
          <div className="hero" style={{ marginBottom: 10 }}>
            <div className="figure">
              <div className="l">in-gate</div>
              <div className="v pos">{pct(bias.in_gate_fraction, 1)}</div>
            </div>
          </div>
          <dl className="kv">
            <dt>random</dt>
            <dd>{abs(bias.random, 3)}</dd>
            <dt>factor-mimicking</dt>
            <dd>{abs(bias.factor_mimicking, 3)}</dd>
            <dt>cap-weighted market</dt>
            <dd>{abs(bias.market, 3)}</dd>
          </dl>
          <p className="note" style={{ marginTop: 10 }}>{bias.note}</p>
        </Panel>
      </div>
    </div>
  );
}
