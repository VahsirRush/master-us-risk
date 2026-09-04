"""Liveness heartbeat for long training runs.

Session 6 lost three background waiters at once while the training process
they were watching kept going. The waiters died silently, which meant the
only thing that would have told us a multi-hour run had died was itself
capable of dying without saying so.

The design rule that follows: **a watcher must not be the only evidence.**
This module writes a small JSON artifact on a fixed cadence, and the artifact
is self-dating. Three states are then distinguishable from the file alone,
with no live process required to interpret it:

    running   the file is fresh AND the pid is alive AND the log is growing
    stalled   the file is fresh AND the pid is alive BUT the log has not
              grown for `stall_after` seconds — a hung job, which a plain
              pid check would call healthy
    dead      the file is fresh and the pid is gone
    unknown   the file itself is STALE — the heartbeat died, or the machine
              slept. This is the case that matters: staleness of the
              evidence is visible in the evidence, so a dead watcher can
              never masquerade as a healthy job.

The writer is meant to be launched detached (`nohup ... &`), the same way the
training run is, so that whatever reaps harness-tracked tasks does not reap
it. Even if it is reaped, the `unknown` state above is the safety net.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from master_us.data.sources import REPO_ROOT

HEARTBEAT_PATH = REPO_ROOT / "reports" / "status" / "heartbeat.json"

DEFAULT_INTERVAL = 60.0
DEFAULT_STALL_AFTER = 900.0  # 15 min without log growth = stalled
DEFAULT_MAX_AGE = 300.0  # heartbeat older than this is not trustworthy


@dataclass(frozen=True)
class Heartbeat:
    """One observation of a training job."""

    checked_at: str  # ISO8601 UTC
    label: str
    pattern: str
    pid: int | None
    alive: bool
    log_path: str
    log_bytes: int
    log_age_sec: float  # seconds since the log last grew
    stalled: bool
    last_progress: str
    note: str = ""

    @property
    def checked_age_sec(self) -> float:
        stamp = datetime.fromisoformat(self.checked_at)
        return (datetime.now(UTC) - stamp).total_seconds()

    def state(self, max_age: float = DEFAULT_MAX_AGE) -> str:
        """running | stalled | dead | unknown — see the module docstring."""
        if self.checked_age_sec > max_age:
            return "unknown"
        if not self.alive:
            return "dead"
        return "stalled" if self.stalled else "running"


def find_pid(pattern: str) -> int | None:
    """First pid whose command line matches `pattern`, or None."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pids = [int(line) for line in out.stdout.split() if line.isdigit()]
    # Exclude ourselves: the heartbeat's own command line contains the pattern.
    pids = [p for p in pids if p != os.getpid()]
    return pids[0] if pids else None


def probe(
    pattern: str,
    log_path: Path,
    label: str = "training",
    stall_after: float = DEFAULT_STALL_AFTER,
    progress_marker: str = "seed",
) -> Heartbeat:
    """Observe a job once: is it alive, and is its log still growing?

    Growth, not mere existence, is what separates a working job from a hung
    one. A process wedged on a deadlocked allocator stays 'alive' forever.
    """
    pid = find_pid(pattern)
    log_bytes = 0
    log_age = float("inf")
    last_progress = ""

    if log_path.exists():
        stat = log_path.stat()
        log_bytes = stat.st_size
        log_age = max(time.time() - stat.st_mtime, 0.0)
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        hits = [ln.strip() for ln in lines if progress_marker in ln]
        last_progress = hits[-1][:200] if hits else ""

    alive = pid is not None
    return Heartbeat(
        checked_at=datetime.now(UTC).isoformat(),
        label=label,
        pattern=pattern,
        pid=pid,
        alive=alive,
        log_path=str(log_path),
        log_bytes=log_bytes,
        log_age_sec=round(log_age, 1) if log_age != float("inf") else -1.0,
        stalled=bool(alive and log_age > stall_after),
        last_progress=last_progress,
    )


def write_heartbeat(hb: Heartbeat, path: Path = HEARTBEAT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(asdict(hb), fh, indent=2, sort_keys=True)
    tmp.replace(path)  # atomic: a reader never sees a half-written file
    return path


def read_heartbeat(path: Path = HEARTBEAT_PATH) -> Heartbeat | None:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return Heartbeat(**json.load(fh))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def format_heartbeat(hb: Heartbeat | None, max_age: float = DEFAULT_MAX_AGE) -> str:
    """One line for `make status`."""
    if hb is None:
        return "no training heartbeat on file"
    state = hb.state(max_age)
    detail = {
        "running": f"pid {hb.pid} · log grew {hb.log_age_sec:.0f}s ago",
        "stalled": f"pid {hb.pid} ALIVE BUT log has not grown for {hb.log_age_sec:.0f}s",
        "dead": "process is gone",
        "unknown": (
            f"heartbeat itself is {hb.checked_age_sec / 60:.0f} min stale — "
            "the watcher died or the machine slept; state is NOT known"
        ),
    }[state]
    line = f"{hb.label}: {state.upper()} — {detail}"
    if hb.last_progress and state in ("running", "stalled"):
        line += f"\n    last: {hb.last_progress}"
    return line


def watch(
    pattern: str,
    log_path: Path,
    label: str = "training",
    interval: float = DEFAULT_INTERVAL,
    stall_after: float = DEFAULT_STALL_AFTER,
    path: Path = HEARTBEAT_PATH,
    exit_when_done: bool = True,
) -> Heartbeat:
    """Write a heartbeat every `interval` seconds until the job disappears.

    Returns the final observation, which is also the last one written — so
    the artifact records the ending, not just silence.
    """
    while True:
        hb = probe(pattern, log_path, label=label, stall_after=stall_after)
        write_heartbeat(hb, path)
        if exit_when_done and not hb.alive:
            return hb
        time.sleep(interval)
