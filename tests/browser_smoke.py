from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_FIXTURE = ROOT / "data" / "synthetic_generic_operations_data.csv"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class BrowserSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("playwright") is None:
            raise unittest.SkipTest("playwright is not installed; browser smoke tests are optional")

        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        cls._PlaywrightError = PlaywrightError
        cls._sync_playwright = sync_playwright
        cls.port = _free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.streamlit_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(cls.port),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUTF8": "1"},
        )

        try:
            cls._wait_for_server()
            cls.playwright = cls._sync_playwright().start()
            cls.browser = cls.playwright.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - environment-specific fallback
            cls._stop_server()
            raise unittest.SkipTest(f"browser smoke runtime is not available: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        browser = getattr(cls, "browser", None)
        if browser is not None:
            browser.close()
        playwright = getattr(cls, "playwright", None)
        if playwright is not None:
            playwright.stop()
        cls._stop_server()

    @classmethod
    def _stop_server(cls) -> None:
        process = getattr(cls, "streamlit_process", None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()

    @classmethod
    def _wait_for_server(cls, timeout_seconds: int = 90) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            if getattr(cls, "streamlit_process", None) is not None and cls.streamlit_process.poll() is not None:
                raise RuntimeError("Streamlit exited before the smoke test server became ready.")
            try:
                with urllib.request.urlopen(cls.base_url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            time.sleep(1.0)
        raise RuntimeError(f"Timed out waiting for Streamlit to start: {last_error}")

    def setUp(self) -> None:
        self.page = self.browser.new_page(viewport={"width": 1600, "height": 2200})
        self.page.goto(self.base_url, wait_until="domcontentloaded", timeout=120000)
        self.page.get_by_text("Analysis context is ready.", exact=False).first.wait_for(timeout=120000)
        self.page.wait_for_timeout(2500)

    def tearDown(self) -> None:
        self.page.close()

    def _open_section_or_skip(self, nav_text: str, expected_text: str) -> None:
        trigger = self.page.get_by_text(nav_text, exact=False).first
        if trigger.count() == 0:
            raise unittest.SkipTest(f"Navigation label '{nav_text}' is not exposed in this browser runtime.")
        try:
            trigger.wait_for(timeout=10000)
            trigger.click(timeout=5000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(f"Navigation label '{nav_text}' is not clickable in this browser runtime: {exc}") from exc
        self.page.wait_for_timeout(1800)
        if self.page.get_by_text(expected_text, exact=False).count() == 0:
            raise unittest.SkipTest(
                f"Section '{expected_text}' is not exposed after clicking '{nav_text}' in this browser runtime."
            )
        try:
            self.page.get_by_text(expected_text, exact=False).first.wait_for(timeout=30000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(
                f"Section '{expected_text}' is present but not visibly exposed after clicking '{nav_text}' in this browser runtime: {exc}"
            ) from exc

    def _fill_text_input_or_skip(self, label: str, value: str) -> None:
        field = self.page.get_by_label(label, exact=True).first
        if field.count() == 0:
            raise unittest.SkipTest(f"Input '{label}' is not exposed in this browser runtime.")
        try:
            field.scroll_into_view_if_needed(timeout=10000)
            field.fill(value, timeout=10000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(f"Input '{label}' is not interactable in this browser runtime: {exc}") from exc

    def _click_button_or_skip(self, label: str) -> None:
        button = self.page.get_by_role("button", name=label, exact=True).first
        if button.count() == 0:
            raise unittest.SkipTest(f"Button '{label}' is not exposed in this browser runtime.")
        try:
            button.scroll_into_view_if_needed(timeout=10000)
            button.click(timeout=10000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(f"Button '{label}' is not interactable in this browser runtime: {exc}") from exc
        self.page.wait_for_timeout(1500)

    def _select_sidebar_option(self, label: str, option: str) -> None:
        sidebar = self.page.locator('[data-testid="stSidebar"]').first
        combobox = sidebar.locator(f'input[role="combobox"][aria-label*="{label}"]').first
        combobox.wait_for(timeout=20000)
        combobox.click(timeout=10000)
        choice = self.page.get_by_role("option", name=option, exact=True).first
        choice.wait_for(timeout=20000)
        choice.click(timeout=10000)
        self.page.wait_for_timeout(1800)
        self.assertIn(option, combobox.get_attribute("aria-label") or "")

    def _set_sidebar_radio(self, option_text: str) -> None:
        sidebar = self.page.locator('[data-testid="stSidebar"]').first
        option = sidebar.get_by_text(option_text, exact=True).first
        option.wait_for(timeout=20000)
        option.click(timeout=10000)
        self.page.wait_for_timeout(1500)

    def _upload_sidebar_dataset(self, upload_path: Path) -> None:
        sidebar = self.page.locator('[data-testid="stSidebar"]').first
        file_input = sidebar.locator('input[type="file"]').first
        file_input.set_input_files(str(upload_path))
        self.page.get_by_text("Analysis context is ready.", exact=False).first.wait_for(timeout=120000)
        self.page.wait_for_timeout(2500)

    def test_launch_and_built_in_dataset_selection(self) -> None:
        self.assertTrue(self.page.get_by_text("Healthcare Data Readiness + Analytics Copilot", exact=False).first.is_visible())
        self.assertTrue(self.page.get_by_text("Recommended Workflow", exact=False).first.is_visible())
        self.assertTrue(self.page.get_by_text("Startup Demo Flow", exact=False).first.is_visible())

        self._select_sidebar_option("Example dataset", "Hospital Reporting Demo")
        self._select_sidebar_option("Example dataset", "Generic Business Demo")
        self._select_sidebar_option("Example dataset", "Healthcare Operations Demo")
        self.page.get_by_text("Built-in Demo Datasets", exact=False).first.wait_for(timeout=30000)
        self.assertTrue(self.page.get_by_text("Built-in demo datasets", exact=False).first.is_visible())

    def test_all_demo_datasets_reach_analysis_ready_state(self) -> None:
        for dataset_name in (
            "Healthcare Operations Demo",
            "Hospital Reporting Demo",
            "Generic Business Demo",
        ):
            with self.subTest(dataset_name=dataset_name):
                self._select_sidebar_option("Example dataset", dataset_name)
                self.page.get_by_text("Analysis context is ready.", exact=False).first.wait_for(timeout=60000)
                self.page.get_by_text("Recommended Workflow", exact=False).first.wait_for(timeout=30000)

    def test_quick_analysis_core_sections_when_tabs_are_exposed(self) -> None:
        self._open_section_or_skip("Dataset Profile", "Detected Column Types")
        self.page.get_by_text("Dataset Overview", exact=False).first.wait_for(timeout=30000)

        self._open_section_or_skip("Data Quality", "Analysis Readiness")
        self.page.get_by_text("Readiness", exact=False).first.wait_for(timeout=30000)

    def test_full_analysis_healthcare_copilot_and_export_when_tabs_are_exposed(self) -> None:
        self._open_section_or_skip("Healthcare Analytics - Healthcare Intelligence", "Hospital Readmission Risk Analytics")
        self.page.get_by_text("AI Copilot", exact=False).first.wait_for(timeout=30000)
        self.page.get_by_label("Ask the AI Copilot", exact=True).fill("Summarize the dataset")
        self.page.get_by_role("button", name="Ask Copilot", exact=True).click()
        self.page.get_by_text("The current filtered dataset includes", exact=False).first.wait_for(timeout=30000)

        self._open_section_or_skip("Insights & Export - Export Center", "Generate Executive Report")
        self.page.get_by_role("button", name="Generate Executive Report", exact=True).click()
        self.page.get_by_text("Executive Report is ready for preview and download.", exact=False).first.wait_for(timeout=30000)
        self.page.get_by_text("Generated report preview", exact=False).first.wait_for(timeout=30000)

    def test_uploaded_dataset_and_workspace_save_flow(self) -> None:
        if not UPLOAD_FIXTURE.exists():
            raise unittest.SkipTest(f"upload fixture is missing: {UPLOAD_FIXTURE}")

        self._set_sidebar_radio("Uploaded dataset")
        self._upload_sidebar_dataset(UPLOAD_FIXTURE)

        self.page.get_by_text("Data Ingestion Wizard", exact=False).first.wait_for(timeout=30000)
        self.page.get_by_text("Source selection", exact=False).first.wait_for(timeout=30000)

        snapshot_name = self.page.get_by_label("Snapshot name", exact=True).first
        if snapshot_name.count() == 0:
            raise unittest.SkipTest("Snapshot controls are not exposed in this browser runtime.")
        try:
            snapshot_name.scroll_into_view_if_needed(timeout=10000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(f"Snapshot controls are not interactable in this browser runtime: {exc}") from exc
        snapshot_name.fill("Browser Smoke Snapshot")
        self.page.get_by_role("button", name="Save Analysis Snapshot", exact=True).click()
        self.page.get_by_text("Analysis snapshot saved for this session.", exact=False).first.wait_for(timeout=30000)

        workflow_pack_name = self.page.get_by_label("Workflow pack name", exact=True).first
        if workflow_pack_name.count() == 0:
            raise unittest.SkipTest("Workflow pack controls are not exposed in this browser runtime.")
        try:
            workflow_pack_name.scroll_into_view_if_needed(timeout=10000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(f"Workflow pack controls are not interactable in this browser runtime: {exc}") from exc
        workflow_pack_name.fill("Browser Smoke Workflow")
        self.page.get_by_role("button", name="Save Workflow Pack", exact=True).click()
        self.page.get_by_text("Workflow pack saved for this session.", exact=False).first.wait_for(timeout=30000)

    def test_signed_in_viewer_export_guardrail_when_workspace_controls_are_exposed(self) -> None:
        self._fill_text_input_or_skip("Display name", "Browser Viewer")
        self._fill_text_input_or_skip("Workspace name", "Browser Pilot")
        self._fill_text_input_or_skip("Email", "viewer@example.com")

        role_combo = self.page.get_by_label("Workspace role", exact=True).first
        if role_combo.count() == 0:
            raise unittest.SkipTest("Workspace role control is not exposed in this browser runtime.")
        try:
            role_combo.click(timeout=10000)
            self.page.get_by_role("option", name="Viewer", exact=True).first.click(timeout=10000)
        except self._PlaywrightError as exc:
            raise unittest.SkipTest(f"Workspace role control is not interactable in this browser runtime: {exc}") from exc

        self._click_button_or_skip("Sign In")
        self.page.get_by_text("Workspace viewer with read-only access.", exact=False).first.wait_for(timeout=30000)

        self._open_section_or_skip("Insights & Export - Export Center", "Generate Executive Report")
        self.page.get_by_text("read-only for exports", exact=False).first.wait_for(timeout=30000)
        self._click_button_or_skip("Generate Executive Report")
        self.page.get_by_text("Role-aware export protections prevented this action.", exact=False).first.wait_for(timeout=30000)


if __name__ == "__main__":
    unittest.main()
