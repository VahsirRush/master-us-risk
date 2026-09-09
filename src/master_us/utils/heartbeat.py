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
import re
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


# Command-line fragments that mark a process as a WATCHER of the job rather
# than the job. `pgrep -f X` matches anything mentioning X, which includes
# every tool built to watch for X — measured the hard way: a waiter running
# `until ! pgrep -f 20_baselines` matched itself, so it never exited (it spun
# for five days) and the heartbeat then reported the finished job as alive.
WATCHER_MARKERS = ("pgrep", "09_heartbeat", "--pattern")


def _is_watcher(command: str) -> bool:
    """True if this command line is watching the job rather than being it.

    Shell wrappers (`/bin/zsh -c '...'`) carry the whole watched command as an
    argument, so they match every pattern their payload mentions.
    """
    if any(marker in command for marker in WATCHER_MARKERS):
        return True
    return bool(re.match(r"^\S*/(?:ba|z|d?a)?sh\s+-c\b", command))


def find_pid(pattern: str, pidfile: Path | None = None) -> int | None:
    """The job's pid: from `pidfile` if given, else the best `pgrep` match.

    A pidfile is unambiguous and is preferred wherever the launcher can write
    one. The pgrep path is the fallback, and it filters out watchers — see
    `_is_watcher`.
    """
    if pidfile is not None and pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(pid, 0)  # signal 0: existence check, no effect
        except (OSError, ProcessLookupError):
            return None
        return pid

    # `ps`, not `pgrep`: BSD/macOS pgrep has no `-a`, so `pgrep -af` returns
    # bare pids with NO command line — every entry then looks like a non-watcher
    # and the filter below silently passes everything. `ps -eo pid=,command=`
    # is portable across macOS and Linux and gives us the text to filter on.
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for raw in out.stdout.splitlines():
        pid_str, _, command = raw.strip().partition(" ")
        if not pid_str.isdigit() or pattern not in command:
            continue
        pid = int(pid_str)
        if pid == os.getpid() or _is_watcher(command):
            continue
        return pid
    return None


def probe(
    pattern: str,
    log_path: Path,
    label: str = "training",
    stall_after: float = DEFAULT_STALL_AFTER,
    progress_marker: str = "seed",
    pidfile: Path | None = None,
) -> Heartbeat:
    """Observe a job once: is it alive, and is its log still growing?

    Growth, not mere existence, is what separates a working job from a hung
    one. A process wedged on a deadlocked allocator stays 'alive' forever.
    """
    pid = find_pid(pattern, pidfile)
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
    pidfile: Path | None = None,
) -> Heartbeat:
    """Write a heartbeat every `interval` seconds until the job disappears.

    Returns the final observation, which is also the last one written — so
    the artifact records the ending, not just silence.
    """
    while True:
        hb = probe(pattern, log_path, label=label, stall_after=stall_after, pidfile=pidfile)
        write_heartbeat(hb, path)
        if exit_when_done and not hb.alive:
            return hb
        time.sleep(interval)
