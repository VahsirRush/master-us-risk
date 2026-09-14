"""The result contract — output-layer-spec section 2.

Every phase emits a `PhaseResult`; renderers consume them; nothing renders raw
dicts. The contract encodes the project's two theses directly in the type
system:

* A number that does not know its cost basis is not a result, so `MetricValue`
  carries gross and net together and `render()` shows both.
* A gap smaller than seed dispersion is not a result either, so
  `distinguishable_from()` compares against POOLED std — renderers dim anything
  it rejects, and docs/project-conventions.md rule 4 requires reporting such gaps as "not
  distinguishable" in those words.

DEVIATION from the spec listing: section 2 writes `distinguishable_from` as a
`@property` taking an argument, which is not valid Python — a property receives
only `self`. It is a method here. See NOTES.md, Session 4.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from master_us.data.sources import REPO_ROOT

STATUS_DIR = REPO_ROOT / "reports" / "status"

PhaseStatus = Literal["pass", "fail", "running", "skipped"]

PHASE_NAMES: dict[int, str] = {
    0: "Data layer",
    1: "Backtest engine",
    2: "Baselines",
    3: "MASTER",
    4: "Ablations",
    5: "Barra factors",
    6: "Risk model",
    7: "Join",
    8: "Reports",
}


@dataclass(frozen=True)
class MetricValue:
    """A number that always knows its own uncertainty and cost basis.

    `gross` is required. `net` is None only for metrics where the distinction
    is meaningless (turnover, counts); a performance metric with `net=None`
    should be treated as unfinished, not as gross-only by choice.

    `std` is dispersion ACROSS SEEDS, not across time. A single-seed or
    deterministic quantity carries `std=None`, and comparisons against it are
    honest about that: with no dispersion estimate on either side,
    `distinguishable_from` refuses to bless a gap as real.
    """

    gross: float
    net: float | None = None
    std: float | None = None
    n_seeds: int | None = None
    net_std: float | None = None

    def render(self, fmt: str = ".4f") -> str:
        """'0.0342 ± 0.0061' or '0.0342 (net 0.0198) ± 0.0061'.

        When `net_std` is present the net side carries its own error bar:
        '0.0342 ± 0.0061 (net 0.0198 ± 0.0044)'. The two-argument forms above
        are preserved exactly when it is absent, so nothing that predates the
        Session-7 contract fix changes appearance.
        """
        if self.net is not None and self.net_std is not None:
            out = f"{self.gross:{fmt}}"
            if self.std is not None:
                out += f" ± {self.std:{fmt}}"
            return out + f" (net {self.net:{fmt}} ± {self.net_std:{fmt}})"

        out = f"{self.gross:{fmt}}"
        if self.net is not None:
            out += f" (net {self.net:{fmt}})"
        if self.std is not None:
            out += f" ± {self.std:{fmt}}"
        return out

    def distinguishable_from(self, other: MetricValue) -> bool:
        """True iff the gross gap exceeds pooled seed dispersion.

        Pooled as sqrt(std_a² + std_b²) — the std of the DIFFERENCE of two
        independent estimates, which is the quantity the gap must clear. A
        one-sided std (the other side deterministic) pools against zero.
        If NEITHER side has a dispersion estimate there is nothing to test
        against, and the honest answer is False: an uncertainty-free gap is
        exactly the kind of number this project refuses to call a finding.
        """
        gap = abs(self.gross - other.gross)
        if self.std is None and other.std is None:
            return False
        pooled = math.sqrt((self.std or 0.0) ** 2 + (other.std or 0.0) ** 2)
        return gap > pooled

    def net_distinguishable_from(self, other: MetricValue) -> bool:
        """The same test as `distinguishable_from`, applied to the NET figures.

        A separate method rather than a `basis=` flag on the original, so that
        every call site says on its face which basis it is testing, and so
        that code written before this existed keeps meaning exactly what it
        meant. Net dispersion is NOT derivable from gross dispersion: costs
        scale with each seed's own turnover, so a model whose seeds agree on
        gross return can still disagree on what survives costs.

        Raises if either side has no net figure — silently falling back to
        gross would be the dishonest failure, since the caller asked about net.
        """
        if self.net is None or other.net is None:
            raise ValueError(
                "net_distinguishable_from needs a net figure on both sides; "
                f"got self.net={self.net}, other.net={other.net}"
            )
        gap = abs(self.net - other.net)
        if self.net_std is None and other.net_std is None:
            return False
        pooled = math.sqrt((self.net_std or 0.0) ** 2 + (other.net_std or 0.0) ** 2)
        return gap > pooled

    @property
    def degraded(self) -> bool:
        """Net materially below gross — the renderer's 'degraded' style.

        Material = costs eat more than a third of the gross, with signs
        agreeing. Sign flips (positive gross, negative net) are always material.
        """
        if self.net is None or self.gross == 0.0:
            return False
        if self.gross > 0 > self.net:
            return True
        return (self.gross - self.net) / abs(self.gross) > 1.0 / 3.0


@dataclass(frozen=True)
class PhaseResult:
    """One phase's outcome. `make status` renders the ladder from these."""

    phase: int
    name: str
    status: PhaseStatus
    gate: str  # human-readable gate condition
    gate_passed: bool
    metrics: dict[str, MetricValue]
    duration_sec: float
    artifacts: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.phase not in PHASE_NAMES:
            raise ValueError(f"phase must be 0-8, got {self.phase}")
        if self.status == "pass" and not self.gate_passed:
            raise ValueError(
                f"phase {self.phase} marked 'pass' with gate_passed=False — a phase "
                "does not pass unless its gate does"
            )

    # ------------------------------------------------------------------ #
    # Persistence — the ladder is rendered from these files               #
    # ------------------------------------------------------------------ #

    def save(self, directory: Path = STATUS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["artifacts"] = [str(p) for p in self.artifacts]
        path = directory / f"phase{self.phase}.json"
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: Path) -> PhaseResult:
        with path.open() as fh:
            payload: dict[str, Any] = json.load(fh)
        payload["metrics"] = {
            name: MetricValue(**mv) for name, mv in payload["metrics"].items()
        }
        payload["artifacts"] = [Path(p) for p in payload["artifacts"]]
        return cls(**payload)

    @classmethod
    def load_all(cls, directory: Path = STATUS_DIR) -> dict[int, PhaseResult]:
        """Every cached phase result, keyed by phase number."""
        if not directory.exists():
            return {}
        out: dict[int, PhaseResult] = {}
        for path in sorted(directory.glob("phase*.json")):
            result = cls.load(path)
            out[result.phase] = result
        return out


@dataclass(frozen=True)
class AblationResult:
    """One ablation cell. Built in Phase 4; the contract lands now."""

    variant: str
    isolates: str  # what this cell tests
    metrics: dict[str, MetricValue]
    baseline_delta: dict[str, float]  # signed gap vs. full model
