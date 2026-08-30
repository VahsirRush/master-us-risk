"""Phase panels and the build-status ladder — output-layer-spec sections 3.2 and 3.5.

`render_status()` is the spec's "single most useful thing": the full phase
ladder from cached `PhaseResult` JSONs, one line per phase, real numbers only.
A phase with no cached result renders as ○ pending — never as a placeholder
metric, which would be a fabricated number wearing a real one's clothes.

Run as a module for `make status`:

    python -m master_us.reporting.status
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.text import Text

from master_us.reporting.results import PHASE_NAMES, MetricValue, PhaseResult
from master_us.reporting.theme import STATUS_GLYPH, THEME


def _fmt_metric(name: str, value: MetricValue) -> str:
    """Compact single-line rendering for ladder rows."""
    pct = {"turnover", "rebalance_turnover", "hit_rate", "crash_2009_return", "max_drawdown"}
    if name in pct:
        out = f"{value.gross:+.1%}" if value.gross < 0 else f"{value.gross:.1%}"
    elif abs(value.gross) >= 100:
        out = f"{value.gross:,.0f}"
    else:
        out = f"{value.gross:.2f}"
    if value.net is not None and value.net != value.gross:
        out += f" (net {value.net:.2f})"
    if value.std is not None:
        out += f" ±{value.std:.2f}"
    return out


# Which cached metrics each phase surfaces on its ladder line, in order.
LADDER_METRICS: dict[int, list[tuple[str, str]]] = {
    0: [("tests", "tests pass"), ("n_dates", "dates"), ("names_per_day", "names/day avg")],
    1: [
        ("sharpe", "momentum Sharpe"),
        ("crash_2009_return", "2009 crash"),
        ("rebalance_turnover", "rebal turnover"),
    ],
    2: [("rankic", "RankIC")],
    3: [("rankic", "RankIC")],
}


def ladder_line(phase: int, result: PhaseResult | None) -> Text:
    """One `✓ 1  Backtest engine   …numbers…` row."""
    text = Text()
    if result is None:
        text.append(f"  {STATUS_GLYPH['skipped']} {phase}  ", style="skipped")
        text.append(f"{PHASE_NAMES[phase]:<22}", style="skipped")
        return text

    style = result.status if result.status in ("pass", "fail", "running") else "skipped"
    text.append(f"  {STATUS_GLYPH[result.status]} {phase}  ", style=style)
    text.append(f"{result.name:<22}", style="metric")

    parts: list[str] = []
    for key, label in LADDER_METRICS.get(phase, []):
        if key in result.metrics:
            parts.append(f"{label} {_fmt_metric(key, result.metrics[key])}")
    if not parts:  # fall back to the first two metrics rather than nothing
        for key, mv in list(result.metrics.items())[:2]:
            parts.append(f"{key} {_fmt_metric(key, mv)}")
    text.append(" · ".join(parts), style="label")
    if result.status == "pass":
        text.append(" ✓", style="pass")
    elif result.status == "fail":
        text.append(" ✗", style="fail")
    return text


def render_status(console: Console | None = None) -> None:
    """The `make status` ladder."""
    console = console or Console(theme=THEME)
    results = PhaseResult.load_all()

    console.print()
    console.print("MASTER-US · build status", style="bold")
    console.print()
    for phase in sorted(PHASE_NAMES):
        console.print(ladder_line(phase, results.get(phase)))
    console.print()

    failed = [r for r in results.values() if r.status == "fail"]
    for r in failed:
        console.print(f"  phase {r.phase} gate: {r.gate}", style="fail")
        for note in r.notes:
            console.print(f"    {note}", style="warn")


def phase_panel(result: PhaseResult, console: Console | None = None) -> None:
    """The section 3.2 completion panel for one phase."""
    console = console or Console(theme=THEME)
    style = "pass" if result.gate_passed else "fail"
    glyph = STATUS_GLYPH["pass" if result.gate_passed else "fail"]

    body = Text()
    body.append(f"Gate: {result.gate}\n\n", style="label")
    for name, mv in result.metrics.items():
        body.append(f"  {name:<24}", style="label")
        rendered = mv.render()
        body.append(rendered + "\n", style="degraded" if mv.degraded else "metric")
    if result.notes:
        body.append("\n")
        for note in result.notes:
            body.append(f"  ▸ {note}\n", style="label")
    body.append(
        f"\n{glyph} Phase {result.phase} {'complete' if result.gate_passed else 'FAILED'}"
        f" · {result.duration_sec:.0f}s · {len(result.artifacts)} artifacts",
        style=style,
    )
    console.print(
        RichPanel(body, title=f"Phase {result.phase} · {result.name}", border_style=style)
    )


if __name__ == "__main__":
    render_status()
