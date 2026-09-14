import type { ReactNode } from "react";

/** A report section: hairline rule, heading, content.
 *
 * Plain header/body shell — the visual weight is entirely in `theme.css`.
 * Kept deliberately simple so gross and net can stay visually distinct; a
 * single-figure/stat pattern would fold net into a percentage of gross and
 * hide the gap this project reports. */
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
 * RISK and ATTR render from this only while `phase_ladder` says their phases
 * have not passed. Once the PhaseResults are pass and the join tables exist,
 * the real panels take over — this component must not invent figures. */
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
        <span className="badge">PHASE {phases} · AWAITING RESULTS</span>
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
