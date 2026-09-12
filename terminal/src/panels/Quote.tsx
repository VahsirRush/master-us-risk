import { Panel } from "../components/Panel";
import { SeedStrip, TurnoverHist } from "../components/Charts";
import { abs, pct, pm, sig, type Metric, type Payload, type Variant } from "../data";

/** QUOTE — spec §3.2: header block, metrics grid with gross/net pairs and
 *  seed counts, seed strip, regime table. */
export function Quote({ data, variant }: { data: Payload; variant: Variant }) {
  const stress = data.stress[variant.key];
  const master = data.variants.find((v) => v.key === "master")!;
  const isRef = variant.key === "master";

  const rows: { label: string; m: Metric; digits: number; seedKey?: string; netOnly?: boolean }[] = [
    { label: "RankIC", m: variant.rank_ic, digits: 4, seedKey: "rank_ic" },
    { label: "ICIR", m: variant.icir, digits: 3, seedKey: "icir" },
    { label: "L/S Sharpe", m: variant.ls_sharpe, digits: 2, seedKey: "ls_net" },
    { label: "Breakeven bps", m: variant.breakeven_bps, digits: 1, seedKey: "breakeven" },
  ];

  return (
    <div className="stack">
      <Panel title={variant.ticker} sub={`${variant.name} · isolates: ${variant.isolates}`}>
        <div className="hero">
          <div className="figure">
            <div className="l">net L/S Sharpe</div>
            <div className="v net">{abs(variant.ls_sharpe.net)}</div>
          </div>
          <div className="figure">
            <div className="l">gross</div>
            <div className="v sm gross">{abs(variant.ls_sharpe.gross)}</div>
          </div>
          <div className="figure">
            <div className="l">seed dispersion</div>
            <div className="v sm" style={{ color: "var(--ink-2)" }}>
              {pm(variant.ls_sharpe.net_std, 3)}
            </div>
          </div>
          <div className="figure">
            <div className="l">turnover</div>
            <div className="v sm" style={{ color: "var(--ink-2)" }}>{pct(variant.turnover_mean, 0)}</div>
          </div>
          <div className="figure" style={{ marginLeft: "auto", textAlign: "right" }}>
            <div className="l">vs {master.ticker}</div>
            <div className="v sm" style={{ color: isRef ? "var(--ink-3)" : variant.net_distinguishable ? "var(--warn)" : "var(--noise)" }}>
              {isRef ? "reference" : variant.net_distinguishable ? "distinguishable" : "not distinguishable"}
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid g-2e">
        <Panel title="Metrics" sub="gross / net pair · every figure over 5 seeds" flush>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="r">Gross</th>
                  <th className="r">Net</th>
                  <th className="r">n</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.label}>
                    <td className="nm">{r.label}</td>
                    <td className="num">
                      <span className="gross">{abs(r.m.gross, r.digits)}</span>{" "}
                      <span className="sd">{pm(r.m.std, r.digits)}</span>
                    </td>
                    <td className="num">
                      {r.m.net === r.m.gross ? (
                        <span className="dim">—</span>
                      ) : (
                        <>
                          <span className="net">{abs(r.m.net, r.digits)}</span>{" "}
                          <span className="sd">{pm(r.m.net_std, r.digits)}</span>
                        </>
                      )}
                    </td>
                    <td className="num dim">{r.m.n_seeds ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Seed dispersion" sub={`seeds ${variant.seeds.join(", ")}`}>
          {rows
            .filter((r) => r.seedKey && variant.per_seed[r.seedKey])
            .map((r) => {
              const vals = variant.per_seed[r.seedKey!];
              return (
                <div className="seedrow" key={r.label}>
                  <div className="lb">{r.label}</div>
                  <SeedStrip values={vals} digits={r.digits} />
                  <div className="num sd">
                    {Math.min(...vals).toFixed(r.digits)} … {Math.max(...vals).toFixed(r.digits)}
                  </div>
                </div>
              );
            })}
          <p className="note" style={{ marginTop: 11 }}>
            Each dot is one initialization; the band is ±1 SD around the mean. Tight clusters and
            wide scatters both average to a point estimate — the strip is what keeps that difference
            visible, and it is the difference every distinguishability call turns on.
          </p>
        </Panel>
      </div>

      <div className="grid g-2e">
        <Panel title="Turnover distribution" sub="daily, pooled across seeds">
          <TurnoverHist {...variant.turnover_hist} />
        </Panel>

        {stress ? (
          <Panel title="Stress — §8.5" sub="seed-averaged scores" flush>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>Slice</th>
                    <th className="r">RankIC</th>
                    <th className="r">Test dates</th>
                  </tr>
                </thead>
                <tbody>
                  {stress.regimes.map((r) => (
                    <tr key={r.name}>
                      <td className="nm">{r.name}</td>
                      <td className="num">
                        {r.ic === null ? (
                          <span className="dim">outside test split</span>
                        ) : (
                          <span className="gross">{sig(r.ic)}</span>
                        )}
                      </td>
                      <td className="num dim">{r.dates}</td>
                    </tr>
                  ))}
                  {stress.cap_tiers.map((c) => (
                    <tr key={c.tier}>
                      <td className="nm">cap tier · {c.tier}</td>
                      <td className="num gross">{sig(c.ic)}</td>
                      <td className="num dim">—</td>
                    </tr>
                  ))}
                  <tr>
                    <td className="nm">sector-neutralized</td>
                    <td className="num gross">{sig(stress.sector_neutral)}</td>
                    <td className="num dim">—</td>
                  </tr>
                  {stress.decay.map((d) => (
                    <tr key={d.h}>
                      <td className="nm">signal decay · h={d.h}d</td>
                      <td className="num net">{sig(d.ic)}</td>
                      <td className="num dim">—</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        ) : (
          <Panel title="Stress — §8.5" sub="not computed for this variant">
            <p className="note">
              The §8.5 slices were run for {master.ticker} and the ungated arm, the two cells the
              headline depends on. They have not been computed for {variant.ticker}.
            </p>
          </Panel>
        )}
      </div>
    </div>
  );
}
