import { useMemo, useState, type ReactNode } from "react";

/** Sortable dense data table.
 *
 * Hand-built — the 21st.dev component search was unavailable, so this is not
 * adapted from a found pattern. Two things it does that a generic table
 * component would not, both required by the spec:
 *
 *   1. `dim` marks a cell whose value is inside seed noise. Those render
 *      struck through and greyed, so a reader scanning the delta column
 *      cannot mistake a noise-level difference for a real one. This is the
 *      `MetricValue.distinguishable_from` contract made visual.
 *   2. Sorting is by `sortValue`, never by the formatted string, so "−0.12"
 *      and "+0.9" order numerically. */

export interface Column<T> {
  key: string;
  head: string;
  align?: "l" | "r";
  sortable?: boolean;
  width?: string;
  render: (row: T) => ReactNode;
  sortValue?: (row: T) => number | string;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  selectedKey,
  initialSort,
  footer,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  selectedKey?: string;
  initialSort?: { key: string; dir: "asc" | "desc" };
  footer?: ReactNode;
}) {
  const [sort, setSort] = useState(initialSort ?? null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const x = col.sortValue!(a);
      const y = col.sortValue!(b);
      if (typeof x === "number" && typeof y === "number") {
        // NaN sorts last regardless of direction — an unmeasured cell is not
        // the smallest value, it is an absent one.
        if (!Number.isFinite(x)) return 1;
        if (!Number.isFinite(y)) return -1;
        return (x - y) * dir;
      }
      return String(x).localeCompare(String(y)) * dir;
    });
  }, [rows, sort, columns]);

  const toggle = (key: string) =>
    setSort((s) =>
      s?.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" },
    );

  return (
    <div className="tw">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className={[c.align === "r" ? "r" : "", c.sortable ? "sortable" : ""]
                  .filter(Boolean)
                  .join(" ")}
                style={c.width ? { width: c.width } : undefined}
                onClick={c.sortable ? () => toggle(c.key) : undefined}
                aria-sort={
                  sort?.key === c.key
                    ? sort.dir === "asc"
                      ? "ascending"
                      : "descending"
                    : c.sortable
                      ? "none"
                      : undefined
                }
              >
                {c.head}
                {sort?.key === c.key ? (
                  <span className="caret">{sort.dir === "asc" ? "▲" : "▼"}</span>
                ) : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const k = rowKey(row);
            return (
              <tr
                key={k}
                className={[onRowClick ? "click" : "", selectedKey === k ? "sel" : ""]
                  .filter(Boolean)
                  .join(" ")}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((c) => (
                  <td key={c.key} className={c.align === "r" ? "num" : undefined}>
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
        {footer ? <tfoot>{footer}</tfoot> : null}
      </table>
    </div>
  );
}

/** A figure that is inside seed dispersion. Struck through and dimmed —
 *  the contract says a gap smaller than pooled dispersion is reported as
 *  "not distinguishable", so the UI must not let it read as a result. */
export function Noise({ children }: { children: ReactNode }) {
  return <span className="noise strike">{children}</span>;
}

export function Verdict({ real }: { real: boolean }) {
  return (
    <span className={real ? "chip real" : "chip nd"}>
      {real ? "distinguishable" : "not distinguishable"}
    </span>
  );
}
