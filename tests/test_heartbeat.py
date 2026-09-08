"""Tests for the training heartbeat — Session 6.

The states that matter are the two a naive check gets wrong: a job that is
alive but hung, and a WATCHER that died while the job kept running. The
second is the failure this module exists for, so it gets the closest test.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from master_us.utils.heartbeat import (
    Heartbeat,
    find_pid,
    format_heartbeat,
    probe,
    read_heartbeat,
    write_heartbeat,
)


def _hb(**over) -> Heartbeat:
    base = dict(
        checked_at=datetime.now(UTC).isoformat(),
        label="phase2",
        pattern="trainer",
        pid=123,
        alive=True,
        log_path="/tmp/x.log",
        log_bytes=100,
        log_age_sec=5.0,
        stalled=False,
        last_progress="lstm seed 2 epoch 6",
    )
    base.update(over)
    return Heartbeat(**base)


# ------------------------------------------------------------------ #
# The four states                                                     #
# ------------------------------------------------------------------ #


def test_running_state():
    assert _hb().state() == "running"
    assert "RUNNING" in format_heartbeat(_hb())


def test_dead_state_when_process_is_gone():
    hb = _hb(alive=False, pid=None)
    assert hb.state() == "dead"
    assert "DEAD" in format_heartbeat(hb)


def test_stalled_beats_a_bare_pid_check():
    """Alive but not progressing. A pid check calls this healthy; we don't."""
    hb = _hb(alive=True, stalled=True, log_age_sec=1800.0)
    assert hb.state() == "stalled"
    text = format_heartbeat(hb)
    assert "STALLED" in text
    assert "has not grown" in text


def test_stale_heartbeat_reports_unknown_not_healthy():
    """THE test. A dead watcher must never look like a healthy job.

    The Session-6 failure: three watchers died while the training run kept
    going. If a stale artifact still read 'running', that silence would be
    indistinguishable from health. Staleness of the evidence has to be
    visible in the evidence.
    """
    old = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    hb = _hb(checked_at=old, alive=True, stalled=False)
    assert hb.state(max_age=300.0) == "unknown"
    text = format_heartbeat(hb, max_age=300.0)
    assert "UNKNOWN" in text
    assert "NOT known" in text
    # And it must not claim the job is fine.
    assert "RUNNING" not in text


def test_fresh_window_is_configurable():
    old = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    hb = _hb(checked_at=old)
    assert hb.state(max_age=300.0) == "running"
    assert hb.state(max_age=60.0) == "unknown"


# ------------------------------------------------------------------ #
# Probing and persistence                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def live_job():
    """A real subprocess to stand in for a training run.

    Not our own interpreter: `find_pid` excludes its own pid by design, and a
    test that matched the test runner would pass for the wrong reason.
    """
    marker = "master-us-heartbeat-fixture"
    proc = subprocess.Popen(["sleep", "45"], env={**os.environ, "JOB": marker})
    # `sleep 45` is what appears in the command line; match on that.
    try:
        yield "sleep 45"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_probe_detects_a_live_process_and_growing_log(tmp_path, live_job):
    log = tmp_path / "run.log"
    log.write_text("seed 0 epoch 1\n")
    hb = probe(live_job, log, label="job", stall_after=600.0)
    assert hb.alive
    assert hb.pid is not None
    assert hb.log_bytes > 0
    assert hb.last_progress == "seed 0 epoch 1"
    assert not hb.stalled


def test_probe_marks_stalled_when_log_is_old(tmp_path, live_job):
    log = tmp_path / "run.log"
    log.write_text("seed 0 epoch 1\n")
    old = time.time() - 3600
    os.utime(log, (old, old))
    hb = probe(live_job, log, stall_after=60.0)
    assert hb.alive and hb.stalled
    assert hb.log_age_sec > 600


def test_probe_reports_dead_for_an_absent_process(tmp_path):
    hb = probe("a-process-that-does-not-exist-xyzzy", tmp_path / "none.log")
    assert not hb.alive
    assert hb.pid is None
    assert hb.state() == "dead"


def test_probe_ignores_its_own_pid(tmp_path):
    """The watcher's own command line contains the pattern it searches for."""
    hb = probe(str(os.getpid()), tmp_path / "none.log")
    assert hb.pid != os.getpid()


def test_write_is_atomic_and_roundtrips(tmp_path):
    path = tmp_path / "hb.json"
    hb = _hb()
    write_heartbeat(hb, path)
    assert read_heartbeat(path) == hb
    # No partial file is left behind for a reader to trip on.
    assert not (tmp_path / "hb.tmp").exists()
    assert json.loads(path.read_text())["pid"] == 123


def test_read_missing_or_corrupt_returns_none(tmp_path):
    assert read_heartbeat(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_heartbeat(bad) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"unexpected": 1}')
    assert read_heartbeat(wrong) is None


def test_format_handles_no_heartbeat():
    assert "no training heartbeat" in format_heartbeat(None)


def test_find_pid_returns_none_for_nonsense():
    assert find_pid("zzz-no-such-process-zzz") is None


def test_replace_keeps_the_dataclass_frozen():
    hb = _hb()
    assert replace(hb, alive=False).state() == "dead"
    assert hb.alive is True  # original untouched


# ------------------------------------------------------------------ #
# watcher self-matching — the five-day bug                            #
# ------------------------------------------------------------------ #


def test_watcher_command_lines_are_not_mistaken_for_the_job():
    """`pgrep -f X` matches every tool built to WATCH for X.

    The real failure: a waiter running `until ! pgrep -f 20_baselines` matched
    its own command line, so it never exited (it spun for five days), and the
    heartbeat then adopted that shell as the job's pid and reported a
    finished run as alive.
    """
    from master_us.utils.heartbeat import _is_watcher

    assert _is_watcher("/bin/zsh -c until ! pgrep -f 20_baselines; do sleep 120; done")
    assert _is_watcher("/bin/bash -c tail -f /tmp/x.log")
    assert _is_watcher("python scripts/09_heartbeat.py --pattern 20_baselines")
    assert _is_watcher("pgrep -af 20_baselines")

    # The job itself must NOT be filtered out.
    assert not _is_watcher(
        "/opt/homebrew/.../Python -u scripts/20_baselines.py --models lstm ungated"
    )
    assert not _is_watcher("caffeinate -dimsu python scripts/20_baselines.py")


def test_find_pid_prefers_a_pidfile(tmp_path):
    """A pidfile is unambiguous; pgrep is the fallback."""
    from master_us.utils.heartbeat import find_pid

    pidfile = tmp_path / "job.pid"
    pidfile.write_text(str(os.getpid()))
    assert find_pid("anything-at-all", pidfile) == os.getpid()


def test_find_pid_pidfile_for_dead_process_returns_none(tmp_path):
    from master_us.utils.heartbeat import find_pid

    pidfile = tmp_path / "job.pid"
    pidfile.write_text("999999")  # not a live pid
    assert find_pid("x", pidfile) is None

    pidfile.write_text("not-a-number")
    assert find_pid("x", pidfile) is None


def test_probe_with_only_watchers_running_reports_dead(tmp_path):
    """The exact end state: job finished, watchers still up -> DEAD, not alive."""
    log = tmp_path / "run.log"
    log.write_text("ungated seed 4 done\n")
    # 'pgrep' appears in the pattern itself, so every match is a watcher.
    hb = probe("pgrep-only-watchers-match-this", log)
    assert not hb.alive
    assert hb.state() == "dead"
