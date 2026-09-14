import { useEffect, useState } from "react";
import { CommandBar, type PanelDef } from "./components/CommandBar";
import { PendingPanel } from "./components/Panel";
import { Monitor } from "./panels/Monitor";
import { Quote } from "./panels/Quote";
import { Abla } from "./panels/Abla";
import { Cost } from "./panels/Cost";
import { Risk } from "./panels/Risk";
import { Attr } from "./panels/Attr";
import { Data } from "./panels/Data";
import type { Payload } from "./data";

/** Static mode (spec §1): fetch the committed blob once. Live mode replaces
 *  this one call with `/api/results` + a `/ws/live` subscription; nothing
 *  below it needs to change, which is the point of keeping every panel a
 *  pure function of the payload.
 *
 *  A third path exists for the single-file build: `scripts/51_bundle_single.py`
 *  inlines the payload onto `window.__RESULTS__` so the page opens over
 *  `file://`, where a relative fetch is blocked by CORS.
 *
 *  Print mode (`?print=1` or `window.__PRINT__`) stacks every panel for a
 *  static PDF mirror — same components and payload as the interactive UI. */
const RESULTS_URL = `${import.meta.env.BASE_URL}results.json`;

declare global {
  interface Window {
    __RESULTS__?: Payload;
    __PRINT__?: boolean;
  }
}

function isPrintMode(): boolean {
  if (typeof window === "undefined") return false;
  if (window.__PRINT__) return true;
  try {
    return new URLSearchParams(window.location.search).has("print");
  } catch {
    return false;
  }
}

export default function App() {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState("monitor");
  const [quoteKey, setQuoteKey] = useState("master");
  const [printMode] = useState(isPrintMode);

  useEffect(() => {
    if (window.__RESULTS__) {
      setData(window.__RESULTS__);
      return;
    }
    fetch(RESULTS_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(setData)
      .catch((e: Error) =>
        setError(
          `Could not load ${RESULTS_URL} — ${e.message}. Regenerate it with ` +
            `\`python scripts/50_export_terminal.py\`.`,
        ),
      );
  }, []);

  useEffect(() => {
    if (printMode) {
      document.documentElement.classList.add("print-mode");
      document.title = "MASTER-US Terminal — static export";
      return () => document.documentElement.classList.remove("print-mode");
    }
    return;
  }, [printMode]);

  useEffect(() => {
    if (printMode) return;
    if (active !== "monitor") window.scrollTo({ top: 0, behavior: "smooth" });
  }, [active, printMode]);

  if (error) {
    return (
      <div className="wrap">
        <div className="panel">
          <header><h2>Results unavailable</h2></header>
          <div className="body"><p className="note">{error}</p></div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="wrap">
        <div className="panel">
          <div className="body"><p className="note dim">Loading results…</p></div>
        </div>
      </div>
    );
  }

  // Fixed F-key order matches the research-terminal spec §3. RISK / ATTR are
  // never dropped from the bar; they render PendingPanel only when their
  // phases have not passed (derived from phase_ladder via the export).
  const panels: PanelDef[] = [
    { id: "monitor", label: "MONITOR", fn: "F1" },
    { id: "quote", label: "QUOTE", fn: "F2" },
    { id: "abla", label: "ABLA", fn: "F3" },
    { id: "cost", label: "COST", fn: "F4" },
    { id: "risk", label: "RISK", fn: "F5", pending: data.risk == null },
    { id: "attr", label: "ATTR", fn: "F6", pending: data.attr == null },
    { id: "data", label: "DATA", fn: "F7" },
  ];

  const variant = data.variants.find((v) => v.key === quoteKey) ?? data.variants[0];
  const pendingMeta = data.pending_panels.find((p) => p.id === active);
  const passed = data.phases.filter((p) => p.status === "pass").length;

  const pick = (key: string) => {
    setQuoteKey(key);
    setActive("quote");
  };

  if (printMode) {
    const riskPending = data.pending_panels.find((p) => p.id === "risk");
    const attrPending = data.pending_panels.find((p) => p.id === "attr");
    return (
      <div className="print-root">
        <header className="print-masthead">
          <div className="print-brand">MASTER-US research terminal</div>
          <div className="print-meta">
            static PDF mirror · BUILD {passed}/{data.phases.length} · generated{" "}
            {data.generated_at}
          </div>
        </header>

        <section className="print-section" data-panel="monitor">
          <h1 className="print-section-title">F1 · MONITOR</h1>
          <Monitor data={data} onPick={() => undefined} />
        </section>

        <section className="print-section" data-panel="quote">
          <h1 className="print-section-title">F2 · QUOTE · {variant.ticker}</h1>
          <Quote data={data} variant={variant} />
        </section>

        <section className="print-section" data-panel="abla">
          <h1 className="print-section-title">F3 · ABLA</h1>
          <Abla data={data} />
        </section>

        <section className="print-section" data-panel="cost">
          <h1 className="print-section-title">F4 · COST</h1>
          <Cost data={data} />
        </section>

        <section className="print-section" data-panel="risk">
          <h1 className="print-section-title">F5 · RISK</h1>
          {data.risk ? <Risk data={data} /> : riskPending ? <PendingPanel {...riskPending} /> : null}
        </section>

        <section className="print-section" data-panel="attr">
          <h1 className="print-section-title">F6 · ATTR</h1>
          {data.attr ? <Attr data={data} /> : attrPending ? <PendingPanel {...attrPending} /> : null}
        </section>

        <section className="print-section" data-panel="data">
          <h1 className="print-section-title">F7 · DATA</h1>
          <Data data={data} />
        </section>

        <footer className="print-footer">
          Every figure is computed from committed artifacts — per-seed score arrays,{" "}
          <code>reports/status/phase*.json</code>, <code>reports/phase7_tables.json</code>, and{" "}
          <code>reports/survivorship.md</code>. Nothing on this page is illustrative.
        </footer>
      </div>
    );
  }

  return (
    <>
      <CommandBar
        panels={panels}
        active={active}
        onSelect={setActive}
        tickers={data.variants.map((v) => ({ ticker: v.ticker, name: v.name }))}
        onTicker={(t) => {
          const hit = data.variants.find((v) => v.ticker === t);
          if (hit) pick(hit.key);
        }}
        generatedAt={data.generated_at}
        phasesPassed={passed}
        phasesTotal={data.phases.length}
      />

      <div className="wrap">
        <div
          className="panel-enter"
          key={active === "quote" ? `quote-${quoteKey}` : active}
          role="tabpanel"
          id={`panel-${active}`}
          aria-labelledby={`key-${active}`}
        >
          {active === "monitor" ? <Monitor data={data} onPick={pick} /> : null}
          {active === "quote" ? <Quote data={data} variant={variant} /> : null}
          {active === "abla" ? <Abla data={data} /> : null}
          {active === "cost" ? <Cost data={data} /> : null}
          {active === "risk" && data.risk ? <Risk data={data} /> : null}
          {active === "attr" && data.attr ? <Attr data={data} /> : null}
          {active === "data" ? <Data data={data} /> : null}
          {pendingMeta && ((active === "risk" && !data.risk) || (active === "attr" && !data.attr)) ? (
            <PendingPanel {...pendingMeta} />
          ) : null}
        </div>

        <footer>
          Every figure is computed from committed artifacts — per-seed score arrays,{" "}
          <code>reports/status/phase*.json</code>, <code>reports/phase7_tables.json</code>, and{" "}
          <code>reports/survivorship.md</code>. Nothing on this page is illustrative.
          <br />
          Keyboard: <code>F1</code>–<code>F{panels.length}</code> switch panels · <code>/</code>{" "}
          searches tickers and variant names · <code>↑</code> <code>↓</code> <code>Enter</code> open
          one in QUOTE.
        </footer>
      </div>
    </>
  );
}
