import argparse
import sys
import time
from pathlib import Path


SCREENSHOTS = [
    {
        "name": "dataset_intelligence",
        "tab": "Dataset Profile · Overview",
        "text": "Dataset Intelligence Summary",
        "filename": "dataset_intelligence.png",
    },
    {
        "name": "capability_matrix",
        "tab": "Dataset Profile · Overview",
        "text": "Analytics Capability Matrix",
        "filename": "capability_matrix.png",
    },
    {
        "name": "healthcare_intelligence",
        "tab": "Healthcare Analytics · Healthcare Intelligence",
        "text": "Healthcare Intelligence",
        "filename": "healthcare_intelligence.png",
    },
    {
        "name": "modeling_studio",
        "tab": "Healthcare Analytics · Healthcare Intelligence",
        "text": "Predictive Modeling Studio",
        "filename": "modeling_studio.png",
    },
    {
        "name": "export_center",
        "tab": "Insights & Export · Export Center",
        "text": "Export Center",
        "filename": "export_center.png",
    },
]


def _require_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is not installed. Install optional dependencies with "
            "`pip install -r requirements-optional.txt` and then run "
            "`python -m playwright install chromium`."
        ) from exc
    return sync_playwright, PlaywrightTimeoutError


def _capture_section(page, target_text: str, output_path: Path, timeout_error):
    locator = page.get_by_text(target_text, exact=False).first
    try:
        locator.wait_for(timeout=30000)
        locator.scroll_into_view_if_needed(timeout=15000)
        time.sleep(1.0)
        locator.screenshot(path=str(output_path))
        return True
    except timeout_error:
        return False


def _activate_tab(page, tab_text: str, timeout_error):
    tab = page.get_by_text(tab_text, exact=False).first
    try:
        tab.wait_for(timeout=30000)
        tab.scroll_into_view_if_needed(timeout=15000)
        tab.click(timeout=15000)
        time.sleep(1.5)
        return True
    except timeout_error:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Capture key Smart Dataset Analyzer screenshots from a running local Streamlit app."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8501",
        help="URL of the running Streamlit app.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/screenshots",
        help="Directory to store generated screenshots.",
    )
    parser.add_argument(
        "--full-page-fallback",
        action="store_true",
        help="Save a full-page fallback screenshot if a target section cannot be located.",
    )
    args = parser.parse_args()

    sync_playwright, timeout_error = _require_playwright()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captured = []
    missing = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 2200})
        page.goto(args.url, wait_until="networkidle", timeout=120000)
        time.sleep(3.0)

        for item in SCREENSHOTS:
            target_path = output_dir / item["filename"]
            _activate_tab(page, item["tab"], timeout_error)
            ok = _capture_section(page, item["text"], target_path, timeout_error)
            if ok:
                captured.append(target_path.name)
            else:
                missing.append(item["text"])
                if args.full_page_fallback:
                    fallback_path = output_dir / f"{item['name']}_fullpage.png"
                    page.screenshot(path=str(fallback_path), full_page=True)

        browser.close()

    print("Captured screenshots:")
    for name in captured:
        print(f" - {name}")

    if missing:
        print("Missing sections:")
        for name in missing:
            print(f" - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
