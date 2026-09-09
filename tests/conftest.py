"""Session-level guards for the test suite.

THE GUARD THAT MATTERS: refuse to run while a training job is live.

Session 8 ran the full suite alongside a training run — the suite loads a
1.2 GB panel, the machine has 8 GB, and earlier sessions had already lost two
training runs to the macOS OOM killer with no traceback. It survived only
because the feature tensor happened to be memmapped by then. "It was luck" is
not a state to leave in place for Phase 6-7, where a run may be much longer
and much more expensive to lose.

A rule written in CLAUDE.md did not prevent it — the rule was two messages
old at the time. So the check is mechanical and lives in the path that would
do the damage. Escape hatch is deliberate and loud:

    MASTER_US_ALLOW_CONCURRENT=1 pytest ...
"""

from __future__ import annotations

import os

import pytest

from master_us.utils.heartbeat import read_heartbeat

ALLOW_ENV = "MASTER_US_ALLOW_CONCURRENT"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Abort before collection if a training heartbeat reads RUNNING."""
    if os.environ.get(ALLOW_ENV) == "1":
        return

    hb = read_heartbeat()
    if hb is None:
        return

    state = hb.state()
    if state != "running":
        return

    raise pytest.UsageError(
        f"\n\nREFUSING TO RUN: a training job is live.\n"
        f"  {hb.label}: pid {hb.pid}, log grew {hb.log_age_sec:.0f}s ago\n"
        f"  last progress: {hb.last_progress or '(none recorded)'}\n\n"
        f"This suite loads a ~1.2 GB panel. On this 8 GB machine that has "
        f"OOM-killed training runs with no traceback.\n"
        f"Wait for the run, or override deliberately:\n"
        f"  {ALLOW_ENV}=1 pytest ...\n"
    )
