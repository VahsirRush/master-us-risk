import { useEffect, useMemo, useRef, useState } from "react";

/** Ticker search + panel shortcuts — spec §2's command bar.
 *
 * Deliberately a persistent bar rather than a summoned palette: the spec
 * calls for the panel row always visible, because on a terminal the available
 * panels are part of the reading, not something you summon. The search does
 * drop a short suggestion list under the input — attached to it, never modal.
 *
 * Matching is plain string filtering over the variant list — the full ticker,
 * the code after the dot ("NG"), or the variant's name ("ridge") — ranked by
 * how specific the hit is, so an exact ticker always comes first.
 *
 * Keyboard: `/` focuses the search, ↑/↓ move through suggestions, Enter opens
 * the highlighted variant in QUOTE, Escape closes, F1-F7 select a panel. */

export interface PanelDef {
  id: string;
  label: string;
  fn: string;
  pending?: boolean;
}

export interface TickerEntry {
  ticker: string;
  name: string;
}

/** Lower is a better match; -1 is no match. An empty query lists everything. */
function rank(q: string, t: TickerEntry): number {
  if (!q) return 5;
  const tk = t.ticker.toUpperCase();
  const nm = t.name.toUpperCase();
  const code = tk.split(".")[1] ?? "";
  if (tk === q) return 0;
  if (tk.startsWith(q)) return 1;
  if (code.startsWith(q)) return 2;
  if (nm.startsWith(q)) return 3;
  if (nm.includes(q) || tk.includes(q)) return 4;
  return -1;
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
  tickers: TickerEntry[];
  onTicker: (ticker: string) => void;
  generatedAt: string;
  phasesPassed: number;
  phasesTotal: number;
}) {
  const [entry, setEntry] = useState("");
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const [bad, setBad] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const q = entry.trim().toUpperCase();
  const matches = useMemo(
    () =>
      tickers
        .map((t) => ({ t, r: rank(q, t) }))
        .filter((m) => m.r >= 0)
        .sort((a, b) => a.r - b.r)
        .map((m) => m.t),
    [q, tickers],
  );

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

  const choose = (ticker: string) => {
    onTicker(ticker);
    setEntry("");
    setOpen(false);
    setBad(false);
    input.current?.blur();
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const hit = matches[hi];
    if (hit) {
      choose(hit.ticker);
    } else {
      setBad(true);
      window.setTimeout(() => setBad(false), 900);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setHi((h) => Math.min(h + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHi((h) => Math.max(h - 1, 0));
    }
  };

  const optId = (t: string) => `opt-${t}`;

  return (
    <div className="bar">
      <div className="brand">
        <span className="tick">MASTER-US</span>
        <span className="sep">research terminal</span>
      </div>

      <form className="tsearch" role="search" onSubmit={submit}>
        <input
          ref={input}
          className={bad ? "ticker-input bad" : "ticker-input"}
          value={entry}
          onChange={(e) => {
            setEntry(e.target.value);
            setOpen(true);
            setHi(0);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onKeyDown={onKeyDown}
          placeholder="ticker or variant   /"
          aria-label="Search tickers and variants"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={open}
          aria-controls="ticker-list"
          aria-activedescendant={open && matches[hi] ? optId(matches[hi].ticker) : undefined}
          spellCheck={false}
          autoComplete="off"
        />
        {open ? (
          <ul className="suggest" id="ticker-list" role="listbox" aria-label="Variants">
            {matches.length ? (
              matches.map((t, i) => (
                <li
                  key={t.ticker}
                  id={optId(t.ticker)}
                  role="option"
                  aria-selected={i === hi}
                  // mousedown, not click: it fires before the input's blur
                  // closes the list
                  onMouseDown={(e) => {
                    e.preventDefault();
                    choose(t.ticker);
                  }}
                  onMouseEnter={() => setHi(i)}
                >
                  <span className="tk">{t.ticker}</span>
                  <span className="nm">{t.name}</span>
                </li>
              ))
            ) : (
              <li className="none" role="presentation">
                No ticker or variant matches “{entry.trim()}”
              </li>
            )}
          </ul>
        ) : null}
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
