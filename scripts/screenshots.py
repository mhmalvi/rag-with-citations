"""Capture README screenshots of the running web UI and of an eval report with headless Chromium.

Usage (server already running on --base-url):
    python scripts/screenshots.py --report docs/eval/sonnet-5-5/report.html --out docs/screenshots

Set CHROMIUM_PATH to use an existing Chromium/headless-shell binary instead of the
one Playwright downloads.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1440, "height": 960}


def capture_answer(page: Page, base_url: str, question: str, citation: int, target: Path) -> None:
    """Ask a question through the UI, open one citation and save a screenshot.

    Args:
        page: Playwright page.
        base_url: Server root, e.g. http://127.0.0.1:8000.
        question: Question to ask.
        citation: Citation number to click (its passage is shown highlighted).
        target: PNG path.
    """
    page.goto(f"{base_url}/?q={quote(question)}")
    page.wait_for_selector(".cite, .abstain", timeout=240_000)
    button = page.locator(f'button.cite[data-n="{citation}"]')
    if button.count():
        button.first.click()
        page.wait_for_selector("mark#hl", timeout=5_000)
    page.screenshot(path=str(target), full_page=True)


def capture_report(page: Page, report: Path, target: Path) -> None:
    """Save a screenshot of the top of an HTML eval report.

    Args:
        page: Playwright page.
        report: Path of report.html.
        target: PNG path.
    """
    page.goto(report.resolve().as_uri())
    page.wait_for_load_state("load")
    page.screenshot(path=str(target), full_page=False)


def main() -> None:
    """Parse arguments and capture all screenshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shots = [
        ("ui-answer-holiday.png", "How many days of holiday can I carry over into next year?", 2),
        ("ui-answer-nist-tiers.png", "Name the four CSF Tiers and what they describe.", 1),
        ("ui-abstain.png", "What is the National Minimum Wage for workers aged 21 and over?", 1),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        for name, question, citation in shots:
            capture_answer(page, args.base_url, question, citation, args.out / name)
            print(f"saved {args.out / name}")
        capture_report(page, args.report, args.out / "eval-report.png")
        print(f"saved {args.out / 'eval-report.png'}")
        browser.close()


if __name__ == "__main__":
    main()
