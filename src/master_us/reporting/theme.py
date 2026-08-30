"""The one visual language — output-layer-spec section 3.1.

Defined once; every renderer imports from here. The opinionated choice the spec
mandates: net is brighter than gross, and anything within seed dispersion is
dimmed to `noise`. The terminal's hierarchy encodes the thesis — costs and
dispersion are the point.
"""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "pass": "bold green",
        "fail": "bold red",
        "running": "bold yellow",
        "skipped": "dim white",
        "gross": "cyan",
        "net": "bold cyan",  # net is emphasized — it's what matters
        "degraded": "yellow",  # net materially below gross
        "noise": "dim italic",  # difference within seed dispersion
        "metric": "white",
        "label": "dim cyan",
        "warn": "bold yellow",
    }
)

STATUS_GLYPH: dict[str, str] = {
    "pass": "✓",
    "fail": "✗",
    "running": "⠹",
    "skipped": "○",
}
