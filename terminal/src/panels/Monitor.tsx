import { Panel } from "../components/Panel";
import { Column, DataTable, Noise } from "../components/DataTable";
import { Cascade, EquityChart } from "../components/Charts";
import { abs, pct, pm, sig, type Payload, type Variant } from "../data";

/** MONITOR — spec §3.1, four quadrants: blotter, cost cascade, equity, build. */
export function Monitor({ data, onPick }: { data: Payload; onPick: (key: string) => void }) {
  const master = data.variants.find((v) => v.key === "master")!;
  const short = data.gate_null.find((g) => g.budget.startsWith("short"));
  const full = data.gate_null.find((g) => g.budget.startsWith("full"));
  const rShort = short?.measures.find((m) => m.basis === "gross")?.ratio ?? null;
  const rFull = full?.measures.find((m) => m.basis === "gross")?.ratio ?? null;

  const columns: Column<Variant>[] = [
    {
      key: "ticker", head: "Ticker", sortable: true,
      render: (v) => <span className="tk">{v.ticker}</span>,
      sortValue: (v) => v.ticker,
    },
    {
      key: "name", head: "Variant", sortable: true,
      render: (v) => <span className="nm">{v.name}</span>,
      sortValue: (v) => v.name,
    },
    {
      key: "rank_ic", head: "RankIC", align: "r", sortable: true,
      render: (v) => (
        <>
          <span className="gross">{sig(v.rank_ic.gross)}</span>{" "}
          <span className="sd">{pm(v.rank_ic.std)}</span>
        </>
      ),
      sortValue: (v) => v.rank_ic.gross,
    },
    {
      key: "icir", head: "ICIR", align: "r", sortable: true,
      render: (v) => <span className="gross">{abs(v.icir.gross, 3)}</span>,
      sortValue: (v) => v.icir.gross,
    },
    {
      key: "ls_g", head: "L/S gross", align: "r", sortable: true,
      render: (v) => <span className="gross">{abs(v.ls_sharpe.gross)}</span>,
      sortValue: (v) => v.ls_sharpe.gross,
    },
    {
      key: "ls_n", head: "L/S net", align: "r", sortable: true,
      render: (v) => (
        <>
          <span className="net">{abs(v.ls_sharpe.net)}</span>{" "}
          <span className="sd">{pm(v.ls_sharpe.net_std, 2)}</span>
        </>
      ),
      sortValue: (v) => v.ls_sharpe.net,
    },
    {
      key: "delta", head: "Δ net vs MSTR.US", align: "r", sortable: true,
      render: (v) =>
        v.key === "master" ? (
          <span className="dim">reference</span>
        ) : v.net_distinguishable ? (
          <span className={v.d_net_ls > 0 ? "pos" : "neg"}>{abs(v.d_net_ls, 3)}</span>
        ) : (
          <Noise>{abs(v.d_net_ls, 3)}</Noise>
        ),
      sortValue: (v) => v.d_net_ls,
    },
    {
      key: "turn", head: "Turnover", align: "r", sortable: true,
      render: (v) => <span className="dim">{pct(v.turnover_mean, 0)}</span>,
      sortValue: (v) => v.turnover_mean,
    },
    {
      key: "be", head: "BE bps", align: "r", sortable: true,
      render: (v) => <span className="warn">{abs(v.breakeven_bps.gross, 1)}</span>,
      sortValue: (v) => v.breakeven_bps.gross,
    },
    {
      key: "seeds", head: "n", align: "r",
      render: (v) => <span className="dim">{v.seeds.length}</span>,
    },
  ];

  return (
    <div className="stack">
      <Panel
        title="Gate null — the project's headline"
        sub={`${short?.gated ?? ""} vs ${short?.ungated ?? ""} · 5 seeds per arm per budget`}
      >
        <div className="hero">
          <div className="figure">
            <div className="l">ratio · short 12/4</div>
            <div className="v warn">{rShort?.toFixed(3) ?? "—"}</div>
          </div>
          <div className="arrow">→</div>
          <div className="figure">
            <div className="l">ratio · full 100/10</div>
            <div className="v" style={{ color: "var(--noise)" }}>{rFull?.toFixed(3) ?? "—"}</div>
          </div>
          <div className="figure" style={{ marginLeft: "auto", textAlign: "right" }}>
            <div className="l">verdict</div>
            <div className="v sm" style={{ color: "var(--ink-2)" }}>NOT DISTINGUISHABLE</div>
          </div>
        </div>
        <p className="note" style={{ marginTop: 15 }}>
          A ratio below 1.0 means the gap sits inside pooled seed dispersion. Phase 3's{" "}
          <strong>{rShort?.toFixed(3)}</strong> was a 5% near-miss — the shape of a real effect
          masked by too little power. Training both arms to the spec's full schedule{" "}
          <strong>doubled the gap</strong> and <strong>more than doubled the dispersion</strong>,
          moving it further inside the noise. A real effect behaves the opposite way.
        </p>
      </Panel>

      <div className="grid g-2">
        <Panel title="Variants" sub="click a row → QUOTE · headers sort" flush>
          <DataTable
            columns={columns}
            rows={data.variants}
            rowKey={(v) => v.key}
            onRowClick={(v) => onPick(v.key)}
          />
        </Panel>

        <div className="stack">
          <Panel title="Cost cascade" sub={master.ticker}>
            <Cascade
              items={[
                { label: "gross L/S", value: master.ls_sharpe.gross, kind: "gross" },
                { label: "net @10 bps", value: master.ls_sharpe.net, kind: "net" },
              ]}
            />
            <p className="note" style={{ marginTop: 13 }}>
              At the {data.provenance.cost_model.baseline_bps} bps baseline the dollar-neutral book
              is <strong className="neg">net-negative</strong> for every variant measured. Gross
              alpha exists; daily rebalancing at {pct(master.turnover_mean, 0)} turnover consumes all
              of it.
            </p>
          </Panel>

          <Panel title="Build" sub="from cached PhaseResults">
            <Ladder phases={data.phases} />
          </Panel>
        </div>
      </div>

      <Panel title="Equity — cumulative net return" sub="test split 2019–2025 · seed-averaged">
        <EquityChart dates={data.equity.dates} series={data.equity.series} basis={data.equity.basis} />
      </Panel>
    </div>
  );
}

function Ladder({ phases }: { phases: Payload["phases"] }) {
  const detail = (p: Payload["phases"][number]): string => {
    const m = p.metrics ?? {};
    if (p.status !== "pass") return "not started";
    if (p.phase === 0 && m.tests) return `${m.tests.gross | 0} tests · leakage gates pass`;
    if (p.phase === 1 && m.sharpe) return `momentum Sharpe ${abs(m.sharpe.gross)} · 2009 reproduced`;
    if (p.phase === 2 && m.rankic) return `LGBM RankIC ${sig(m.rankic.gross)}`;
    if (p.phase === 3 && m.rankic) return `RankIC ${sig(m.rankic.gross)} ${pm(m.rankic.std)}`;
    if (p.phase === 4 && m.ratio_full && m.ratio_short)
      return `gate null settled · ratio ${m.ratio_short.gross.toFixed(3)} → ${m.ratio_full.gross.toFixed(3)}`;
    return p.gate ?? "";
  };

  return (
    <div className="lad">
      {phases.map((p) => {
        const done = p.status === "pass";
        return (
          <div className="it" key={p.phase}>
            <div className={done ? "st pos" : "st dim"}>{done ? "✓" : "○"}</div>
            <div className="ph">{p.phase}</div>
            <div className={done ? "nmm" : "nmm dim"}>{p.label}</div>
            <div className="dt" title={detail(p)}>{detail(p)}</div>
          </div>
        );
      })}
    </div>
  );
}
