#!/usr/bin/env python
"""Render the research terminal to a static PDF that mirrors every panel.

    reports/terminal/master-us-terminal.pdf

Prints the stacked print layout (`?print=1`) — same React panels and the same
`results.json` payload as the interactive dashboard. Prefers a running
`vite preview` at localhost:4173; falls back to the single-file HTML with
`window.__PRINT__` forced on.

Requires: `pip install playwright && playwright install chromium`
"""

from __future__ import annotations

import argparse
import http.client
import sys
from pathlib import Path
from urllib.parse import urlparse

from master_us.data.sources import REPO_ROOT

HTML = REPO_ROOT / "reports" / "terminal" / "master-us-terminal.html"
OUT = REPO_ROOT / "reports" / "terminal" / "master-us-terminal.pdf"
DEFAULT_URL = "http://localhost:4173/?print=1"


def _preview_up(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        conn = http.client.HTTPConnection(host, port, timeout=1.5)
        conn.request("GET", "/results.json")
        return conn.getresponse().status == 200
    except OSError:
        return False


def _print_html(src: Path) -> str:
    raw = src.read_text()
    if "window.__PRINT__" not in raw:
        raw = raw.replace(
            "<script>window.__RESULTS__",
            "<script>window.__PRINT__=true;window.__RESULTS__",
            1,
        )
    if "window.__PRINT__" not in raw:
        raw = raw.replace("</head>", "<script>window.__PRINT__=true;</script>\n</head>", 1)
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", type=str, default=DEFAULT_URL, help="print-mode URL")
    ap.add_argument("--html", type=Path, default=HTML, help="fallback single-file HTML")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "error: playwright not installed. "
            "`pip install playwright && playwright install chromium`",
            file=sys.stderr,
        )
        return 1

    if _preview_up(args.url):
        target = args.url
        cleanup = None
    elif args.html.exists():
        import tempfile

        tmp = tempfile.TemporaryDirectory(prefix="master-us-pdf-")
        src = Path(tmp.name) / "terminal-print.html"
        src.write_text(_print_html(args.html))
        target = src.resolve().as_uri()
        cleanup = tmp
        print(f"preview not up; falling back to {args.html.name}", file=sys.stderr)
    else:
        print(
            f"error: neither {args.url} nor {args.html} is available. "
            "Start `npm run preview` in terminal/, or build the single-file HTML.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1600})
            page.goto(target, wait_until="networkidle")
            page.wait_for_selector(".print-root", timeout=30000)
            page.wait_for_selector(".print-section[data-panel='data']", timeout=30000)
            page.wait_for_timeout(800)
            n = page.locator(".print-section").count()
            page.pdf(
                path=str(args.out.resolve()),
                format="Letter",
                print_background=True,
                margin={
                    "top": "0.45in",
                    "bottom": "0.55in",
                    "left": "0.45in",
                    "right": "0.45in",
                },
            )
            browser.close()
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out.relative_to(REPO_ROOT)}  ({kb:.0f} KB, {n} panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
