import { useEffect, useState } from "react";
import { CommandBar, type PanelDef } from "./components/CommandBar";
import { PendingPanel } from "./components/Panel";
import { Monitor } from "./panels/Monitor";
import { Quote } from "./panels/Quote";
import { Abla } from "./panels/Abla";
import { Cost } from "./panels/Cost";
import { Data } from "./panels/Data";
import type { Payload } from "./data";

/** Static mode (spec §1): fetch the committed blob once. Live mode replaces
 *  this one call with `/api/results` + a `/ws/live` subscription; nothing
 *  below it needs to change, which is the point of keeping every panel a
 *  pure function of the payload.
 *
 *  A third path exists for the single-file build: `scripts/51_bundle_single.py`
 *  inlines the payload onto `window.__RESULTS__` so the page opens over
 *  `file://`, where a relative fetch is blocked by CORS. */
const RESULTS_URL = `${import.meta.env.BASE_URL}results.json`;

declare global {
  interface Window {
    __RESULTS__?: Payload;
  }
}

export default function App() {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState("monitor");
  const [quoteKey, setQuoteKey] = useState("master");

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
    if (active !== "monitor") window.scrollTo({ top: 0, behavior: "smooth" });
  }, [active]);

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

  const panels: PanelDef[] = [
    { id: "monitor", label: "MONITOR", fn: "F1" },
    { id: "quote", label: "QUOTE", fn: "F2" },
    { id: "abla", label: "ABLA", fn: "F3" },
    { id: "cost", label: "COST", fn: "F4" },
    ...data.pending_panels.map((p, i) => ({
      id: p.id,
      label: p.label,
      fn: `F${5 + i}`,
      pending: true,
    })),
    { id: "data", label: "DATA", fn: `F${5 + data.pending_panels.length}` },
  ];

  const variant = data.variants.find((v) => v.key === quoteKey) ?? data.variants[0];
  const pending = data.pending_panels.find((p) => p.id === active);
  const passed = data.phases.filter((p) => p.status === "pass").length;

  const pick = (key: string) => {
    setQuoteKey(key);
    setActive("quote");
  };

  return (
    <>
      <CommandBar
        panels={panels}
        active={active}
        onSelect={setActive}
        tickers={data.variants.map((v) => v.ticker)}
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
          {active === "data" ? <Data data={data} /> : null}
          {pending ? <PendingPanel {...pending} /> : null}
        </div>

        <footer>
          Every figure is computed from committed artifacts — per-seed score arrays,{" "}
          <code>reports/status/phase*.json</code>, and <code>reports/survivorship.md</code>. Nothing
          on this page is illustrative.
          <br />
          Keyboard: <code>F1</code>–<code>F{panels.length}</code> switch panels · <code>/</code>{" "}
          focuses the ticker input · Enter jumps to QUOTE.
        </footer>
      </div>
    </>
  );
}
