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
COMMITTED_OUT = REPO_ROOT / "reports" / "terminal" / "results.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    payload = build()
    path = write(__import__("pathlib").Path(args.out), payload)
    # Keep the committed mirror in sync so `reports/terminal/` is the
    # self-contained artifact set (HTML + JSON) for the portfolio.
    if path.resolve() == DEFAULT_OUT.resolve():
        write(COMMITTED_OUT, payload)
    kb = path.stat().st_size / 1024

    print(f"wrote {path.relative_to(REPO_ROOT)}  ({kb:.0f} KB)")
    if path.resolve() == DEFAULT_OUT.resolve():
        print(f"wrote {COMMITTED_OUT.relative_to(REPO_ROOT)}")
    print(f"  variants      {len(payload['variants'])}")
    print(f"  beta points   {len(payload['beta_sweep']['points'])}")
    print(f"  equity series {len(payload['equity']['series'])} x {len(payload['equity']['dates'])} pts")
    print(f"  phases        {sum(p['status'] == 'pass' for p in payload['phases'])}/9 pass")
    print(f"  pending       {[p['id'] for p in payload['pending_panels']] or 'none'}")
    print(f"  risk books    {len(payload['risk']['books']) if payload['risk'] else 'pending'}")
    print(f"  attr factors  {len(payload['attr']['timing']) if payload['attr'] else 'pending'}")
    for g in payload["gate_null"]:
        r = next(m["ratio"] for m in g["measures"] if m["basis"] == "gross")
        print(f"  gate null     {g['budget']:<12} RankIC ratio {r:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
