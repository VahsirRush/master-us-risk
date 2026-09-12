import { useEffect, useRef, useState } from "react";

/** Ticker input + panel shortcuts — spec §2's command bar.
 *
 * Hand-built; the 21st.dev command-palette search was unavailable. It is
 * deliberately not a modal palette: the spec calls for a persistent bar with
 * the panel row always visible, because on a terminal the available panels
 * are part of the reading, not something you summon.
 *
 * Keyboard: type a ticker + Enter to jump to QUOTE, F1-F7 select a panel,
 * `/` focuses the input, Escape blurs it. */

export interface PanelDef {
  id: string;
  label: string;
  fn: string;
  pending?: boolean;
}

export function CommandBar({
  panels,
  active,
  onSelect,
  tickers,
  onTicker,
  generatedAt,
  phasesPassed,
  phasesTotal,
}: {
  panels: PanelDef[];
  active: string;
  onSelect: (id: string) => void;
  tickers: string[];
  onTicker: (ticker: string) => void;
  generatedAt: string;
  phasesPassed: number;
  phasesTotal: number;
}) {
  const [entry, setEntry] = useState("");
  const [bad, setBad] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = document.activeElement === input.current;
      if (e.key === "/" && !typing) {
        e.preventDefault();
        input.current?.focus();
        return;
      }
      if (e.key === "Escape" && typing) input.current?.blur();
      if (typing) return;
      const m = /^F([1-7])$/.exec(e.key);
      if (m) {
        const p = panels[Number(m[1]) - 1];
        if (p) {
          e.preventDefault();
          onSelect(p.id);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [panels, onSelect]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = entry.trim().toUpperCase();
    const hit = tickers.find((t) => t === q || t.split(".")[0] === q || t.startsWith(q));
    if (hit && q) {
      onTicker(hit);
      setEntry("");
      setBad(false);
      input.current?.blur();
    } else {
      setBad(true);
      window.setTimeout(() => setBad(false), 900);
    }
  };

  return (
    <div className="bar">
      <div className="brand">
        <span className="tick">MASTER-US</span>
        <span className="sep">research terminal</span>
      </div>

      <form onSubmit={submit}>
        <input
          ref={input}
          className={bad ? "ticker-input bad" : "ticker-input"}
          value={entry}
          onChange={(e) => setEntry(e.target.value)}
          placeholder="ticker /"
          aria-label="Go to ticker"
          spellCheck={false}
          autoComplete="off"
        />
      </form>

      <nav className="keys" role="tablist" aria-label="Panels">
        {panels.map((p) => (
          <button
            key={p.id}
            role="tab"
            id={`key-${p.id}`}
            aria-selected={active === p.id}
            aria-controls={`panel-${p.id}`}
            className={p.pending ? "key pending" : "key"}
            onClick={() => onSelect(p.id)}
          >
            <span className="fn">{p.fn}</span>
            {p.label}
          </button>
        ))}
      </nav>

      <div className="right">
        <span className="pill">
          <span className="dot" /> BUILD {phasesPassed}/{phasesTotal}
        </span>
        <span className="mono">{generatedAt.replace("T", " ").replace("+00:00", "Z")}</span>
      </div>
    </div>
  );
}
