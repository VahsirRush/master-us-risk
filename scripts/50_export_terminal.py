#!/usr/bin/env python
"""Export the research terminal's static payload — spec §1.

    terminal/public/results.json

Thin caller. All logic is in `master_us.terminal.export`.
"""

from __future__ import annotations

import argparse
import sys

from master_us.data.sources import REPO_ROOT
from master_us.terminal.export import build, write

DEFAULT_OUT = REPO_ROOT / "terminal" / "public" / "results.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    payload = build()
    path = write(__import__("pathlib").Path(args.out), payload)
    kb = path.stat().st_size / 1024

    print(f"wrote {path.relative_to(REPO_ROOT)}  ({kb:.0f} KB)")
    print(f"  variants      {len(payload['variants'])}")
    print(f"  beta points   {len(payload['beta_sweep']['points'])}")
    print(f"  equity series {len(payload['equity']['series'])} x {len(payload['equity']['dates'])} pts")
    print(f"  phases        {sum(p['status'] == 'pass' for p in payload['phases'])}/9 pass")
    for g in payload["gate_null"]:
        r = next(m["ratio"] for m in g["measures"] if m["basis"] == "gross")
        print(f"  gate null     {g['budget']:<12} RankIC ratio {r:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
