import type { ReactNode } from "react";

/** The frosted card every piece of content sits in.
 *
 * Hand-built: the 21st.dev/Magic extension was not available in this
 * environment, so no component search informed this. It is a plain
 * header/body shell — the visual weight is entirely in `theme.css`. */
export function Panel({
  title,
  sub,
  children,
  flush = false,
}: {
  title: string;
  sub?: ReactNode;
  children: ReactNode;
  flush?: boolean;
}) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
        {sub ? <span className="sub">{sub}</span> : null}
      </header>
      <div className={flush ? "body flush" : "body"}>{children}</div>
    </section>
  );
}

/** The empty state for a panel whose phase has not run (spec §3.5, §3.6).
 *
 * This exists so RISK and ATTR can be present and honest rather than absent
 * or fabricated. It renders from the exporter's `pending_panels` record. */
export function PendingPanel({
  label,
  spec,
  phases,
  needs,
}: {
  label: string;
  spec: string;
  phases: string;
  needs: string;
}) {
  return (
    <Panel title={label} sub={`spec ${spec}`}>
      <div className="empty">
        <span className="badge">PHASE {phases} · NOT YET RUN</span>
        <h3>Nothing measured here yet</h3>
        <p className="note">{needs}</p>
        <p className="note dim">
          The panel is specified and wired; it stays empty until the phase produces
          numbers. Populating it with anything else would make this terminal less
          useful, not more.
        </p>
      </div>
    </Panel>
  );
}
