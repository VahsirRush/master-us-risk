"""Session-level guards for the test suite.

THE GUARD THAT MATTERS: do not load the real panel while a training job is
live.

Session 8 ran the full suite alongside a training run — the suite loads a
1.2 GB panel, the machine has 8 GB, and earlier sessions had already lost two
training runs to the macOS OOM killer with no traceback. It survived only
because the feature tensor happened to be memmapped by then. "It was luck" is
not a state to leave in place for Phase 6-7, where a run may be much longer
and much more expensive to lose.

A rule written in docs/project-conventions.md did not prevent it — the rule was
newly written at the time and still did not stop the mistake. So the check is
mechanical and lives in the path that would do the damage.

SCOPE, narrowed in Session 13. The first version aborted the entire session.
That was too blunt in a way that mattered: only `test_real_panel_gates.py`
actually loads the panel, so 344 memory-safe tests were being refused to
protect against one module. The cost is not hypothetical — Phase 6 estimates
covariances on live jobs, which is exactly when a guard would fire, and
`test_recovers_known_factor_returns` / `test_industry_constraint_holds` are
the two tests that would catch a regression in the factor-return estimation
path. Tests that would catch a regression need to run during the phase that
might cause it.

So the guard now skips only tests marked `heavy`, and everything else runs.
Both escape hatches remain, and both are loud:

    MASTER_US_ALLOW_CONCURRENT=1 pytest ...   # run the heavy tests anyway
    pytest -m heavy                            # run only them, deliberately
"""

from __future__ import annotations

import os

import pytest

from master_us.utils.heartbeat import read_heartbeat

ALLOW_ENV = "MASTER_US_ALLOW_CONCURRENT"
_LIVE_RUN: str | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Record whether a training job is live, once per session."""
    global _LIVE_RUN
    _LIVE_RUN = None

    if os.environ.get(ALLOW_ENV) == "1":
        return

    hb = read_heartbeat()
    if hb is None or hb.state() != "running":
        return

    _LIVE_RUN = (
        f"{hb.label}: pid {hb.pid}, log grew {hb.log_age_sec:.0f}s ago; "
        f"last progress: {hb.last_progress or '(none recorded)'}"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the panel-loading tests while a training job is live.

    Reported as skips rather than a silent pass: a suite that quietly drops
    its heaviest gates reads identically to one where they passed.
    """
    if _LIVE_RUN is None:
        return

    reason = (
        f"training job live ({_LIVE_RUN}) — this test loads the ~1.2 GB panel, "
        f"which has OOM-killed training runs on this 8 GB machine. "
        f"Override with {ALLOW_ENV}=1."
    )
    marker = pytest.mark.skip(reason=reason)
    skipped = 0
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(marker)
            skipped += 1

    # Written straight to the terminal reporter rather than returned from
    # `pytest_report_header`, because the project's addopts set `-q` and the
    # header is suppressed there — which is precisely the run `make test`
    # performs. A guard whose explanation is invisible in the default
    # invocation leaves a reader looking at "9 skipped" with no reason.
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"TRAINING JOB LIVE — skipping {skipped} test(s) marked 'heavy'. "
            f"{_LIVE_RUN}. All other tests run normally; "
            f"override with {ALLOW_ENV}=1.",
            bold=True,
            yellow=True,
        )
