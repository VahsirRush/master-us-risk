#!/usr/bin/env python
"""Fold the built terminal into one self-contained HTML file.

    terminal/dist-single/  ->  reports/terminal/master-us-terminal.html

The Vite build is the deployment target (GitHub Pages, `vite preview`). This
produces the same app as a single file that opens over `file://` with no
server, for attaching to an email or a portfolio submission. The payload is
inlined onto `window.__RESULTS__`, which `App.tsx` prefers over `fetch` —
a relative fetch is blocked by CORS on `file://`.

Thin caller: no data is transformed here, only concatenated.
"""

from __future__ import annotations

import json
import re
import sys

from master_us.data.sources import REPO_ROOT

DIST = REPO_ROOT / "terminal" / "dist-single"
PAYLOAD = REPO_ROOT / "terminal" / "public" / "results.json"
OUT = REPO_ROOT / "reports" / "terminal" / "master-us-terminal.html"


def main() -> int:
    index = DIST / "index.html"
    if not index.exists():
        print(f"error: {index} missing — run `npm run build:single` in terminal/ first", file=sys.stderr)
        return 1

    html = index.read_text()
    payload = json.loads(PAYLOAD.read_text())

    css = "\n".join(p.read_text() for p in sorted((DIST / "assets").glob("*.css")))
    js = "\n".join(p.read_text() for p in sorted((DIST / "assets").glob("*.js")))
    if not js:
        print("error: no JS asset found in dist/assets", file=sys.stderr)
        return 1

    # Drop the tags that point at external files; everything goes inline.
    html = re.sub(r'<script[^>]*src="[^"]*"[^>]*></script>', "", html)
    html = re.sub(r'<link[^>]*rel="stylesheet"[^>]*href="\./assets[^"]*"[^>]*>', "", html)

    # `</script>` inside the JSON would close the tag early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    if css:
        html = html.replace("</head>", f"<style>{css}</style>\n</head>")
    # A plain <script>, never type="module": Chrome fetches module scripts
    # with CORS, and a file:// page has a null origin, so a module bundle
    # opens to a blank page.
    html = html.replace(
        "</body>",
        f"<script>window.__RESULTS__={blob};</script>\n<script>{js}</script>\n</body>",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
