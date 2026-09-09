#!/usr/bin/env python
"""Watch a long training run and record its liveness.

    reports/status/heartbeat.json

Launch DETACHED so that whatever reaps harness-tracked background tasks does
not reap the watcher too:

    nohup python scripts/09_heartbeat.py \\
        --pattern 20_baselines --log /tmp/phase2_deep.log --label phase2 &

`--once` prints the current state and exits, which is the cheap way to check
in without holding any process open.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from master_us.utils.heartbeat import (
    DEFAULT_INTERVAL,
    DEFAULT_STALL_AFTER,
    format_heartbeat,
    probe,
    read_heartbeat,
    watch,
    write_heartbeat,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="20_baselines", help="pgrep -f pattern")
    parser.add_argument("--log", type=Path, default=Path("/tmp/phase2_deep.log"))
    parser.add_argument("--label", default="phase2")
    parser.add_argument("--pidfile", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--stall-after", type=float, default=DEFAULT_STALL_AFTER)
    parser.add_argument("--once", action="store_true", help="probe, print, exit")
    parser.add_argument("--read", action="store_true", help="print the file, probe nothing")
    args = parser.parse_args()

    if args.read:
        print(format_heartbeat(read_heartbeat()))
        return 0

    if args.once:
        hb = probe(
            args.pattern, args.log, label=args.label,
            stall_after=args.stall_after, pidfile=args.pidfile,
        )
        write_heartbeat(hb)
        print(format_heartbeat(hb))
        return 0 if hb.alive else 1

    final = watch(
        args.pattern,
        args.log,
        label=args.label,
        interval=args.interval,
        stall_after=args.stall_after,
        pidfile=args.pidfile,
    )
    print(format_heartbeat(final))
    return 0


if __name__ == "__main__":
    sys.exit(main())
