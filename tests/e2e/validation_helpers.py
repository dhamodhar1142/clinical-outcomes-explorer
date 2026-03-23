from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import pandas as pd

from src.data_loader import load_uploaded_file_bundle
from src.services.export_service import generate_export_report_output
from src.pipeline import run_analysis_pipeline
from src.schema_detection import detect_structure
from tests.e2e.fixture_registry import (
    AMBIGUOUS_MAPPING_FIXTURE,
    DatasetFixture,
    MALFORMED_FIXTURE,
    SECONDARY_HEALTHCARE_FIXTURE,
    SMALL_HEALTHCARE_FIXTURE,
    get_default_fixture,
)

try:
    from PIL import Image, ImageChops
except Exception:  # pragma: no cover - optional dependency path
    Image = None
    ImageChops = None


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / 'artifacts' / 'validation'
DEFAULT_TIMEOUT_SECONDS = 120
LARGE_DATASET_TIMEOUT_SECONDS = 360
VISUAL_REGRESSION_SCREENS = [
    'data_intake.png',
    'overview.png',
    'column_detection.png',
    'quality_review.png',
    'readiness.png',
    'healthcare_intelligence.png',
    'trend_analysis.png',
    'cohort_analysis.png',
    'key_insights.png',
    'export_center.png',
]


@dataclass
class ValidationCheck:
    area: str
    status: str
    details: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    screenshot: str = ''


@dataclass
class ValidationReport:
    dataset_path: str
    dataset_name: str
    mode: str
    used_browser: bool
    command: str
    commands_run: list[str]
    checks: list[ValidationCheck] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    remaining_risks: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    root_cause_summary: str = ''
    artifact_dir: str = ''
    degraded_browser_reason: str = ''
    sections: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str = ''

    def finalize(self) -> None:
        self.completed_at = datetime.now(UTC).isoformat()

    @property
    def passed(self) -> bool:
        return all(check.status == 'pass' for check in self.checks)


def browser_environment_readiness() -> dict[str, Any]:
    readiness = {
        'playwright_installed': False,
        'browser_launchable': False,
        'usable': False,
        'reason': '',
    }
    if importlib.util.find_spec('playwright') is None:
        readiness['reason'] = 'Playwright is not installed.'
        return readiness
    readiness['playwright_installed'] = True
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
        readiness['browser_launchable'] = True
        readiness['usable'] = True
    except Exception as error:  # pragma: no cover - environment-specific
        readiness['reason'] = f'Playwright is installed, but Chromium could not launch: {error}'
    return readiness


def ensure_artifact_dir(mode: str, dataset_name: str) -> Path:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '-', dataset_name).strip('-') or 'dataset'
    path = ARTIFACT_ROOT / f'{stamp}-{mode}-{safe_name}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(5):
        temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            temporary_path.write_text(content, encoding='utf-8')
            temporary_path.replace(path)
            return
        except PermissionError as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            time.sleep(0.2 * (attempt + 1))
    if last_error is not None:
        raise last_error


def write_partial_state(artifact_dir: Path, payload: dict[str, Any]) -> Path:
    state_path = artifact_dir / 'validation_state.json'
    _atomic_write_text(state_path, json.dumps(payload, indent=2, default=str))
    return state_path


def _artifact_evidence(artifact_dir: Path) -> dict[str, list[str]]:
    screenshots = sorted(str(path) for path in artifact_dir.glob('*.png'))
    logs = sorted(str(path) for path in artifact_dir.glob('*.log'))
    diffs = sorted(str(path) for path in artifact_dir.rglob('diff_*.png'))
    return {
        'screenshots': screenshots,
        'logs': logs,
        'diff_artifacts': diffs,
    }


def build_failure_report(
    *,
    dataset_path: Path,
    mode: str,
    artifact_dir: Path,
    command: str,
    commands_run: list[str],
    last_completed_step: str,
    error: Exception,
    diagnostics: dict[str, Any] | None = None,
) -> ValidationReport:
    evidence = _artifact_evidence(artifact_dir)
    traceback_summary = ''.join(traceback.format_exception(type(error), error, error.__traceback__))[-12000:]
    report = ValidationReport(
        dataset_path=str(dataset_path),
        dataset_name=dataset_path.name,
        mode=mode,
        used_browser=False,
        command=command,
        commands_run=commands_run,
        diagnostics=diagnostics or {},
        artifact_dir=str(artifact_dir),
    )
    report.checks.append(
        ValidationCheck(
            'validation runtime',
            'fail',
            f'Validation aborted during {last_completed_step}: {error}',
            {
                'last_completed_step': last_completed_step,
                'exception_type': type(error).__name__,
                'exception_message': str(error),
                'traceback_summary': traceback_summary,
                **evidence,
            },
            screenshot=evidence['screenshots'][-1] if evidence['screenshots'] else '',
        )
    )
    report.root_cause_summary = f'Validation failed during {last_completed_step}: {type(error).__name__}: {error}'
    report.sections['failure'] = {
        'overall_result': 'FAIL',
        'last_completed_step': last_completed_step,
        'exception_type': type(error).__name__,
        'exception_message': str(error),
        'traceback_summary': traceback_summary,
        **evidence,
        'commands_run': commands_run,
    }
    report.remaining_risks.append('Review the failure section, screenshots, and Streamlit logs in the artifact directory.')
    report.finalize()
    return report


def read_dataset_shape(dataset_path: Path) -> dict[str, Any]:
    suffix = dataset_path.suffix.lower()
    if suffix == '.csv':
        header = pd.read_csv(dataset_path, nrows=0)
        with dataset_path.open('rb') as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)
        return {
            'dataset_name': dataset_path.name,
            'row_count': int(row_count),
            'column_count': int(len(header.columns)),
            'columns': [str(column) for column in header.columns],
        }
    else:
        data = pd.read_excel(dataset_path)
        return {
            'dataset_name': dataset_path.name,
            'row_count': int(len(data)),
            'column_count': int(len(data.columns)),
            'columns': [str(column) for column in data.columns],
        }


def load_fixture_bundle(dataset_path: Path, fixture: DatasetFixture) -> tuple[bytes, dict[str, Any]]:
    file_bytes = dataset_path.read_bytes()
    return file_bytes, {
        'fixture_key': fixture.key,
        'dataset_type': fixture.dataset_type,
        'large_file_fixture': fixture.large_file,
        'expected_approximate_row_count': fixture.expected_approximate_row_count,
    }


def inspect_fixture_schema(dataset_path: Path, fixture: DatasetFixture | None = None) -> dict[str, Any]:
    active_fixture = fixture or get_default_fixture()
    file_bytes, fixture_meta = load_fixture_bundle(dataset_path, active_fixture)
    data, _original_lookup, load_stats = load_uploaded_file_bundle(dataset_path.name, file_bytes, sampling_override='auto')
    structure = detect_structure(data)
    detected_dates = {str(column).upper() for column in structure.date_columns}
    columns = {str(column).upper() for column in data.columns}
    inferred_identifier_like = {
        column
        for column in columns
        if column.endswith('_ID') or column in {'REFR_NO', 'ROM_ID', 'PAT_ID', 'MEDT_ID'}
    }
    identifier_like_columns = sorted(set(str(column).upper() for column in structure.identifier_columns).union(inferred_identifier_like))
    return {
        'fixture_key': active_fixture.key,
        'dataset_type': active_fixture.dataset_type,
        'large_file_fixture': active_fixture.large_file,
        'dataset_cache_key': str(load_stats.get('dataset_cache_key', '')),
        'ingestion_strategy': str(load_stats.get('ingestion_strategy', 'standard')),
        'sampling_mode': str(load_stats.get('sampling_mode', 'full')),
        'source_row_count': int(load_stats.get('source_row_count', len(data))),
        'analyzed_row_count': int(load_stats.get('analyzed_row_count', len(data))),
        'expected_row_count': int(active_fixture.expected_approximate_row_count),
        'actual_row_count': int(load_stats.get('source_row_count', len(data))),
        'row_count_delta': int(load_stats.get('source_row_count', len(data))) - int(active_fixture.expected_approximate_row_count),
        'expected_columns_present': sorted(columns.intersection(active_fixture.expected_key_source_columns)),
        'missing_expected_columns': sorted(set(active_fixture.expected_key_source_columns) - columns),
        'date_columns_detected': sorted(detected_dates),
        'expected_date_columns_detected': sorted(detected_dates.intersection(active_fixture.expected_date_columns)),
        'missing_expected_date_columns': sorted(set(active_fixture.expected_date_columns) - detected_dates),
        'identifier_columns_detected': identifier_like_columns,
        'categorical_columns_detected': [str(column).upper() for column in structure.categorical_columns],
        'fixture_meta': fixture_meta,
    }


def _validation_baseline_dir(fixture: DatasetFixture) -> Path:
    return ROOT / 'tests' / 'e2e' / 'baselines' / fixture.key


def _compare_images(reference_path: Path, candidate_path: Path, diff_path: Path, *, mismatch_threshold: float = 0.01) -> dict[str, Any]:
    if Image is None or ImageChops is None:
        return {'available': False, 'reason': 'Pillow is not available for image diffing.'}
    if not reference_path.exists():
        return {'available': False, 'reason': f'Baseline image is missing: {reference_path}'}
    reference = Image.open(reference_path).convert('RGB')
    candidate = Image.open(candidate_path).convert('RGB')
    if reference.size != candidate.size:
        candidate = candidate.resize(reference.size)
    diff = ImageChops.difference(reference, candidate)
    bbox = diff.getbbox()
    changed_pixels = 0
    if bbox is not None:
        grayscale_diff = diff.convert('L')
        binary_diff = grayscale_diff.point(lambda value: 255 if value > 8 else 0)
        histogram = binary_diff.histogram()
        changed_pixels = int(histogram[255] if len(histogram) > 255 else 0)
        diff.save(diff_path)
    total_pixels = max(reference.size[0] * reference.size[1], 1)
    mismatch_ratio = changed_pixels / total_pixels
    return {
        'available': True,
        'reference': str(reference_path),
        'candidate': str(candidate_path),
        'diff': str(diff_path) if diff_path.exists() else '',
        'mismatch_ratio': mismatch_ratio,
        'threshold': mismatch_threshold,
        'passed': mismatch_ratio <= mismatch_threshold,
    }


def _build_export_validation(
    dataset_path: Path,
    *,
    fixture: DatasetFixture,
    artifact_dir: Path,
) -> dict[str, Any]:
    file_bytes, _fixture_meta = load_fixture_bundle(dataset_path, fixture)
    data, _original_lookup, load_stats = load_uploaded_file_bundle(dataset_path.name, file_bytes, sampling_override='auto')
    source_meta = {
        'source_mode': 'Uploaded dataset',
        'description': 'Automated export validation upload',
        'dataset_cache_key': str(load_stats.get('dataset_cache_key', '')),
        'ingestion_strategy': str(load_stats.get('ingestion_strategy', 'standard')),
        'sampling_mode': str(load_stats.get('sampling_mode', 'full')),
        'source_row_count': int(load_stats.get('source_row_count', len(data))),
        'analyzed_row_count': int(load_stats.get('analyzed_row_count', len(data))),
        'upload_status': 'ready',
    }
    pipeline = run_analysis_pipeline(
        data,
        dataset_path.name,
        source_meta,
        demo_config={'synthetic_helper_mode': 'Auto'},
        active_control_values={'report_mode': 'Executive Summary'},
    )
    pipeline = {**pipeline, 'source_meta': source_meta}
    session_state: dict[str, Any] = {'workspace_identity': {'workspace_id': 'validation-workspace'}}
    workspace_identity = {
        'workspace_id': 'validation-workspace',
        'workspace_name': 'Validation Workspace',
        'auth_mode': 'local',
        'role': 'owner',
        'membership_validated': True,
    }
    report_labels = ['Executive Report', 'Analyst Report', 'Data Readiness Report']
    results: list[dict[str, Any]] = []
    export_dir = artifact_dir / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    for label in report_labels:
        result = generate_export_report_output(
            session_state,
            job_runtime={},
            report_label=label,
            dataset_name=dataset_path.name,
            pipeline=pipeline,
            workspace_identity=workspace_identity,
            role='Owner',
            policy_name='Internal Review',
            privacy_review=pipeline.get('privacy_review', {}),
            governance_config={},
        )
        zip_bytes = result.get('zip_bytes', b'') or b''
        txt_bytes = result.get('report_bytes', b'') or b''
        json_bytes_payload = result.get('json_bytes', b'') or b''
        pdf_bytes = result.get('pdf_bytes', b'') or b''
        export_stem = label.replace(' ', '_').lower()
        txt_path = export_dir / f'{export_stem}.txt'
        json_path = export_dir / f'{export_stem}.json'
        pdf_path = export_dir / f'{export_stem}.pdf'
        zip_path = export_dir / f'{export_stem}.zip'
        txt_path.write_bytes(txt_bytes)
        json_path.write_bytes(json_bytes_payload)
        pdf_path.write_bytes(pdf_bytes)
        zip_path.write_bytes(zip_bytes)
        txt_text = txt_bytes.decode('utf-8', errors='replace')
        json_text = json_bytes_payload.decode('utf-8', errors='replace')
        contamination = any(token in txt_text for token in ('Healthcare Operations Demo', 'Hospital Reporting Demo', 'Generic Business Demo'))
        results.append(
            {
                'report_label': label,
                'txt_path': str(txt_path),
                'json_path': str(json_path),
                'pdf_path': str(pdf_path),
                'zip_path': str(zip_path),
                'txt_non_empty': bool(txt_bytes),
                'json_non_empty': bool(json_bytes_payload),
                'pdf_non_empty': bool(pdf_bytes),
                'zip_non_empty': bool(zip_bytes),
                'dataset_name_present': dataset_path.name in txt_text or dataset_path.name in json_text,
                'source_mode_present': 'Uploaded dataset' in json_text or 'Uploaded dataset' in txt_text,
                'demo_contamination': contamination,
            }
        )
    return {
        'reports': results,
        'all_passed': all(
            item['txt_non_empty']
            and item['json_non_empty']
            and item['pdf_non_empty']
            and item['zip_non_empty']
            and item['dataset_name_present']
            and not item['demo_contamination']
            for item in results
        ),
    }


def run_cross_dataset_cache_validation(*, python_executable: str) -> ValidationReport:
    fixture_a = SMALL_HEALTHCARE_FIXTURE
    fixture_b = SECONDARY_HEALTHCARE_FIXTURE
    artifact_dir = ensure_artifact_dir('cache', 'cross_dataset')
    inspection_a = inspect_fixture_schema(fixture_a.path, fixture_a)
    inspection_b = inspect_fixture_schema(fixture_b.path, fixture_b)
    report = ValidationReport(
        dataset_path=f'{fixture_a.path};{fixture_b.path}',
        dataset_name='cross-dataset-cache-validation',
        mode='cache',
        used_browser=False,
        command='python scripts/run_cross_dataset_cache_validation.py',
        commands_run=[f'{python_executable} scripts/run_cross_dataset_cache_validation.py'],
        artifact_dir=str(artifact_dir),
    )
    distinct_keys = inspection_a.get('dataset_cache_key') != inspection_b.get('dataset_cache_key')
    no_column_leak = inspection_a.get('actual_row_count') != inspection_b.get('actual_row_count') or fixture_a.dataset_name != fixture_b.dataset_name
    report.sections['cross_dataset_cache'] = {
        'fixture_a': inspection_a,
        'fixture_b': inspection_b,
    }
    report.checks.append(
        ValidationCheck(
            'cross-dataset cache identity',
            'pass' if distinct_keys else 'fail',
            'Distinct uploaded datasets produced distinct cache identifiers.',
            {'fixture_a_cache_key': inspection_a.get('dataset_cache_key'), 'fixture_b_cache_key': inspection_b.get('dataset_cache_key')},
        )
    )
    report.checks.append(
        ValidationCheck(
            'cross-dataset contamination',
            'pass' if no_column_leak else 'fail',
            'Second uploaded fixture did not reuse the first fixture context.',
            {'fixture_a_rows': inspection_a.get('actual_row_count'), 'fixture_b_rows': inspection_b.get('actual_row_count')},
        )
    )
    report.root_cause_summary = 'No cross-dataset cache contamination detected.' if report.passed else 'Cross-dataset cache ownership drift was detected.'
    report.finalize()
    write_report_files(report, artifact_dir)
    return report


def write_report_files(report: ValidationReport, artifact_dir: Path) -> tuple[Path, Path]:
    payload = asdict(report)
    json_path = artifact_dir / 'validation_report.json'
    markdown_path = artifact_dir / 'validation_report.md'
    _atomic_write_text(json_path, json.dumps(payload, indent=2, default=str))

    lines = [
        '# Dataset Validation Report',
        '',
        f"- Dataset: `{report.dataset_name}`",
        f"- Mode: `{report.mode}`",
        f"- Browser workflow used: `{report.used_browser}`",
        f"- Overall result: `{'PASS' if report.passed else 'FAIL'}`",
        f"- Artifact directory: `{artifact_dir}`",
        '',
        '## Commands Run',
    ]
    lines.extend(f'- `{command}`' for command in report.commands_run)
    lines.extend(['', '## Diagnostics'])
    for key, value in sorted(report.diagnostics.items()):
        lines.append(f'- `{key}`: `{value}`')
    if report.sections:
        lines.extend(['', '## Detailed Sections'])
        for section_name, section_payload in report.sections.items():
            lines.append(f'### {section_name}')
            if isinstance(section_payload, dict):
                for key, value in sorted(section_payload.items()):
                    lines.append(f'- `{key}`: `{value}`')
            elif isinstance(section_payload, list):
                lines.extend(f'- {value}' for value in section_payload)
            else:
                lines.append(f'- {section_payload}')
    lines.extend(['', '## Checks'])
    for check in report.checks:
        lines.append(f"- `{check.status.upper()}` {check.area}: {check.details}")
        if check.screenshot:
            lines.append(f"  screenshot: `{check.screenshot}`")
    lines.extend(['', '## Root Cause Summary', report.root_cause_summary or 'No failures detected.'])
    lines.extend(['', '## Remaining Risks'])
    if report.remaining_risks:
        lines.extend(f'- {risk}' for risk in report.remaining_risks)
    else:
        lines.append('- No additional risks recorded.')
    _atomic_write_text(markdown_path, '\n'.join(lines))
    return json_path, markdown_path


def _stable_payload_identity(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class StreamlitAppServer:
    def __init__(self, root: Path, artifact_dir: Path) -> None:
        self.root = root
        self.artifact_dir = artifact_dir
        self.port = _free_port()
        self.base_url = f'http://127.0.0.1:{self.port}'
        self.stdout_path = artifact_dir / 'streamlit_stdout.log'
        self.stderr_path = artifact_dir / 'streamlit_stderr.log'
        self.process: subprocess.Popen[str] | None = None

    def start(self, python_executable: str) -> None:
        stdout_handle = self.stdout_path.open('w', encoding='utf-8')
        stderr_handle = self.stderr_path.open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            [
                python_executable,
                '-m',
                'streamlit',
                'run',
                'app.py',
                '--server.headless',
                'true',
                '--server.address',
                '127.0.0.1',
                '--server.port',
                str(self.port),
            ],
            cwd=self.root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env={**os.environ, 'PYTHONUTF8': '1'},
            text=True,
        )
        self.wait_ready()

    def wait_ready(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError('Streamlit exited before becoming ready.')
            try:
                with urllib.request.urlopen(self.base_url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            time.sleep(1.0)
        raise RuntimeError(f'Timed out waiting for Streamlit: {last_error}')

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _contrast_ratio(foreground: tuple[int, int, int], background: tuple[int, int, int]) -> float:
    def _channel(value: int) -> float:
        normalized = value / 255.0
        if normalized <= 0.03928:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    def _luminance(rgb: tuple[int, int, int]) -> float:
        r, g, b = rgb
        return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)

    lighter = max(_luminance(foreground), _luminance(background))
    darker = min(_luminance(foreground), _luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def _parse_rgb(value: str) -> tuple[int, int, int]:
    if value.strip().lower() == 'transparent':
        return (255, 255, 255)
    rgba_match = re.search(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', value)
    if rgba_match:
        red, green, blue, alpha = rgba_match.groups()
        if float(alpha) == 0:
            return (255, 255, 255)
        return (int(red), int(green), int(blue))
    match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)', value)
    if not match:
        return (255, 255, 255)
    return tuple(int(group) for group in match.groups())


class ProgrammaticDatasetValidator:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir

    def validate(self, dataset_path: Path, mode: str, fixture: DatasetFixture | None = None) -> ValidationReport:
        active_fixture = fixture or get_default_fixture()
        shape = read_dataset_shape(dataset_path)
        file_bytes, fixture_meta = load_fixture_bundle(dataset_path, active_fixture)
        data, original_lookup, source_meta = self._load_uploaded_dataset(dataset_path.name, file_bytes)
        structure = detect_structure(data)
        columns_upper = {str(column).upper() for column in data.columns}
        date_columns_upper = {str(column).upper() for column in structure.date_columns}
        pipeline = run_analysis_pipeline(
            data,
            dataset_path.name,
            source_meta,
            demo_config={'synthetic_helper_mode': 'Auto'},
            active_control_values={'report_mode': 'Executive Summary'},
        )
        report = ValidationReport(
            dataset_path=str(dataset_path),
            dataset_name=dataset_path.name,
            mode=mode,
            used_browser=False,
            command=f'python scripts/run_dataset_validation.py --dataset "{dataset_path}" --mode {mode}',
            commands_run=['programmatic pipeline validation'],
            artifact_dir=str(self.artifact_dir),
        )
        report.diagnostics.update(
            {
                'active_dataset_source': source_meta.get('source_mode', ''),
                'dataset_name': dataset_path.name,
                'dataset_identifier': pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_identifier', ''),
                'dataset_cache_key': source_meta.get('dataset_cache_key', ''),
                'row_count': shape['row_count'],
                'column_count': shape['column_count'],
                'fixture_key': active_fixture.key,
                'large_file_fixture': active_fixture.large_file,
                'ingestion_strategy': source_meta.get('ingestion_strategy', ''),
                'sampling_mode': source_meta.get('sampling_mode', ''),
            }
        )
        report.checks.extend(
            [
                ValidationCheck('uploaded dataset authority', 'pass' if source_meta.get('source_mode') == 'Uploaded dataset' else 'fail', 'Programmatic upload flow resolved as uploaded dataset.', report.diagnostics.copy()),
                ValidationCheck('row/column stability', 'pass' if int(source_meta.get('source_row_count', len(data))) == shape['row_count'] and len(data.columns) == shape['column_count'] else 'fail', 'Loaded dataset shape matches fixture.', {'expected_rows': shape['row_count'], 'actual_rows': int(source_meta.get('source_row_count', len(data))), 'expected_columns': shape['column_count'], 'actual_columns': len(data.columns), 'analyzed_rows_in_memory': len(data)}),
                ValidationCheck('expected source columns', 'pass' if not (set(active_fixture.expected_key_source_columns) - columns_upper) else 'fail', 'Expected healthcare source columns are present in the uploaded dataset.', {'missing_expected_columns': sorted(set(active_fixture.expected_key_source_columns) - columns_upper)}),
                ValidationCheck('date field detection', 'pass' if set(active_fixture.expected_date_columns).issubset(date_columns_upper) else 'fail', 'Expected date-like visit fields are recognized.', {'detected_date_columns': [str(column).upper() for column in structure.date_columns], 'missing_expected_date_columns': sorted(set(active_fixture.expected_date_columns) - date_columns_upper)}),
                ValidationCheck('pipeline context', 'pass' if pipeline.get('dataset_runtime_diagnostics', {}).get('dataset_identifier') else 'fail', 'Pipeline runtime diagnostics were produced for the uploaded dataset.', pipeline.get('dataset_runtime_diagnostics', {})),
                ValidationCheck('quality + analytics surfaces', 'pass' if pipeline.get('quality') and pipeline.get('healthcare') and pipeline.get('insights') else 'fail', 'Major downstream surfaces were materialized from the uploaded dataset.'),
            ]
        )
        if active_fixture.large_file:
            report.checks.append(
                ValidationCheck(
                    'large dataset handling',
                    'pass' if source_meta.get('ingestion_strategy') not in {'', 'standard'} else 'fail',
                    'Large fixture used an explicit large-file ingestion path instead of silently falling back.',
                    {'ingestion_strategy': source_meta.get('ingestion_strategy', ''), 'sampling_mode': source_meta.get('sampling_mode', '')},
                )
            )
        report.root_cause_summary = 'Programmatic fallback completed. Browser automation was not available, so UI visibility was not directly audited.'
        report.remaining_risks.append('UI visibility and real upload interaction were not validated because Playwright was unavailable.')
        report.finalize()
        return report

    def _load_uploaded_dataset(self, file_name: str, file_bytes: bytes) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
        data, original_lookup, load_stats = load_uploaded_file_bundle(file_name, file_bytes, sampling_override='auto')
        source_meta = {
            'source_mode': 'Uploaded dataset',
            'description': 'Programmatic validation upload',
            'best_for': 'Automated dataset validation',
            'file_size_mb': len(file_bytes) / (1024 ** 2),
            'dataset_cache_key': str(load_stats.get('dataset_cache_key', '')),
            'ingestion_strategy': str(load_stats.get('ingestion_strategy', 'standard')),
            'sampling_mode': str(load_stats.get('sampling_mode', 'full')),
            'source_row_count': int(load_stats.get('source_row_count', len(data))),
            'analyzed_row_count': int(load_stats.get('analyzed_row_count', len(data))),
            'upload_status': 'ready',
        }
        return data, original_lookup, source_meta


class BrowserDatasetValidator:
    TAB_EXPECTATIONS = [
        ('Data Intake', 'Data Ingestion Wizard'),
        ('Overview', 'Clinical Readiness Workspace'),
        ('Column Detection', 'Schema Detection Confidence'),
        ('Field Profiling', 'Data profiling'),
        ('Quality Review', 'Quality Review'),
        ('Readiness', 'Analysis Readiness'),
        ('Healthcare Intelligence', 'Healthcare Intelligence'),
        ('Trend Analysis', 'Trend Analysis'),
        ('Cohort Analysis', 'Cohort Analysis'),
        ('Key Insights', 'Key Insights'),
        ('Export Center', 'Export Center'),
    ]

    QUICK_TABS = TAB_EXPECTATIONS[:6]

    def __init__(self, artifact_dir: Path) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        self._PlaywrightError = PlaywrightError
        self._sync_playwright = sync_playwright
        self.artifact_dir = artifact_dir
        self._active_fixture: DatasetFixture | None = None

    def validate(
        self,
        dataset_path: Path,
        mode: str,
        python_executable: str,
        fixture: DatasetFixture | None = None,
        *,
        visual_regression: bool = False,
        update_baselines: bool = False,
        soak_seconds: int = 0,
        rerun_cycles: int = 0,
    ) -> ValidationReport:
        active_fixture = fixture or get_default_fixture()
        self._active_fixture = active_fixture
        shape = read_dataset_shape(dataset_path)
        fixture_inspection = inspect_fixture_schema(dataset_path, active_fixture)
        report = ValidationReport(
            dataset_path=str(dataset_path),
            dataset_name=dataset_path.name,
            mode=mode,
            used_browser=True,
            command=f'python scripts/run_dataset_validation.py --dataset "{dataset_path}" --mode {mode}',
            commands_run=[],
            artifact_dir=str(self.artifact_dir),
        )
        server = StreamlitAppServer(ROOT, self.artifact_dir)
        browser = None
        page = None
        playwright = None
        try:
            server.start(python_executable)
            report.commands_run.append(f'{python_executable} -m streamlit run app.py --server.port {server.port}')
            playwright = self._sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1600, 'height': 2200})
            page.goto(server.base_url, wait_until='domcontentloaded', timeout=DEFAULT_TIMEOUT_SECONDS * 1000)
            self._set_uploaded_source(page)
            self._upload_dataset(page, dataset_path)
            self._wait_for_analysis(page)
            diagnostics = self._collect_diagnostics(page)
            report.diagnostics.update({**fixture_inspection, **diagnostics})
            report.sections['browser_environment'] = {'mode': 'browser', 'usable': True}
            report.sections['large_dataset_fidelity'] = {
                'source_row_count': fixture_inspection.get('source_row_count'),
                'profiled_row_count': fixture_inspection.get('analyzed_row_count'),
                'sampled_row_count': fixture_inspection.get('analyzed_row_count'),
                'analyzed_row_count': fixture_inspection.get('analyzed_row_count'),
                'authoritative_metrics': ['source_row_count', 'actual_row_count', 'dataset_identifier', 'dataset_cache_key'],
                'sampled_metrics': ['profiled_row_count', 'sampled_row_count', 'analyzed_row_count'] if active_fixture.large_file else [],
                'sampling_mode': fixture_inspection.get('sampling_mode'),
                'ingestion_strategy': fixture_inspection.get('ingestion_strategy'),
            }
            last_known_diagnostics = diagnostics.copy()
            report.checks.append(
                ValidationCheck(
                    'uploaded dataset authority',
                    'pass' if diagnostics.get('active_dataset_source') == 'Uploaded dataset' else 'fail',
                    'Uploaded dataset remained the active source after UI upload.',
                    diagnostics.copy(),
                )
            )
            report.checks.append(
                ValidationCheck(
                    'row/column stability',
                    'pass' if (report.diagnostics.get('actual_row_count') == shape['row_count'] and diagnostics.get('column_count') == shape['column_count']) else 'fail',
                    'Developer diagnostics and fixture inspection counts match the uploaded dataset.',
                    {'expected_rows': shape['row_count'], 'actual_rows': report.diagnostics.get('actual_row_count'), 'expected_columns': shape['column_count'], 'actual_columns': diagnostics.get('column_count'), 'analyzed_rows_in_ui': diagnostics.get('row_count')},
                )
            )
            report.checks.append(
                ValidationCheck(
                    'expected source columns',
                    'pass' if not fixture_inspection.get('missing_expected_columns') else 'fail',
                    'Expected healthcare source columns are present in the uploaded dataset.',
                    fixture_inspection,
                )
            )
            report.checks.append(
                ValidationCheck(
                    'date field detection',
                    'pass' if not fixture_inspection.get('missing_expected_date_columns') else 'fail',
                    'Expected visit date fields are recognized as date-like.',
                    fixture_inspection,
                )
            )
            if active_fixture.large_file:
                report.checks.append(
                    ValidationCheck(
                        'large dataset handling',
                        'pass' if fixture_inspection.get('ingestion_strategy') not in {'', 'standard'} else 'fail',
                        'Large fixture used an explicit large-file ingestion path instead of silently falling back.',
                        fixture_inspection,
                    )
                )
            if mode == 'full':
                persistence_diagnostics = self._wait_for_idle_persistence(page, seconds=90)
                diagnostics_after_wait = persistence_diagnostics.get('diagnostics', {}) or last_known_diagnostics.copy()
                if diagnostics_after_wait:
                    last_known_diagnostics = diagnostics_after_wait.copy()
                report.checks.append(
                    ValidationCheck(
                        'persistence after 90s idle wait',
                        'pass' if diagnostics_after_wait.get('active_dataset_source') == 'Uploaded dataset' else 'fail',
                        'Uploaded dataset remained active after a 90-second idle wait during the long-running validation flow.',
                        persistence_diagnostics,
                    )
                )
            if soak_seconds > 0 or rerun_cycles > 0 or mode == 'soak':
                soak_summary = self._run_stability_cycles(
                    page,
                    soak_seconds=soak_seconds or (300 if mode == 'soak' else 0),
                    rerun_cycles=rerun_cycles or (3 if mode == 'soak' else 1),
                )
                report.sections['soak_stability'] = soak_summary
                report.checks.append(
                    ValidationCheck(
                        'soak stability',
                        'pass' if soak_summary.get('stable') else 'fail',
                        'Uploaded dataset stayed authoritative through the configured soak/rerun cycle.' if soak_summary.get('stable') else 'Uploaded dataset drifted during the configured soak/rerun cycle.',
                        soak_summary,
                    )
                )
            tabs = self.QUICK_TABS if mode == 'quick' else self.TAB_EXPECTATIONS
            for tab_name, expected_text in tabs:
                screenshot_path = self._validate_tab(page, tab_name, expected_text)
                tab_diagnostics = self._collect_diagnostics(page) or last_known_diagnostics.copy()
                if tab_diagnostics:
                    last_known_diagnostics = tab_diagnostics.copy()
                status = 'pass'
                if tab_diagnostics.get('active_dataset_source') != 'Uploaded dataset':
                    status = 'fail'
                if 'Trend Analysis' in tab_name and tab_diagnostics.get('trend_cache_belongs_to_current_dataset') is False:
                    status = 'fail'
                if 'Cohort Analysis' in tab_name and tab_diagnostics.get('cohort_cache_belongs_to_current_dataset') is False:
                    status = 'fail'
                if self._demo_context_visible(page):
                    status = 'fail'
                report.checks.append(
                    ValidationCheck(
                        tab_name,
                        status,
                        f'{tab_name} loaded without reverting to demo context.',
                        tab_diagnostics,
                        screenshot=str(screenshot_path),
                    )
                )
            if active_fixture.key == AMBIGUOUS_MAPPING_FIXTURE.key:
                manual_mapping = self._validate_manual_mapping_profile_flow(page)
                report.sections['manual_mapping_profile_flow'] = manual_mapping
                report.checks.append(
                    ValidationCheck(
                        'manual mapping profile flow',
                        'pass' if manual_mapping.get('passed') else 'fail',
                        manual_mapping.get('summary', 'Manual mapping profile flow executed.'),
                        manual_mapping,
                        screenshot=manual_mapping.get('screenshot', ''),
                    )
                )
            accessibility = self._audit_accessibility(page)
            report.sections['accessibility'] = accessibility
            report.checks.append(
                ValidationCheck(
                    'accessibility audit',
                    'pass' if not accessibility.get('issues') else 'fail',
                    'Critical workflow areas passed practical accessibility checks.' if not accessibility.get('issues') else 'Accessibility-oriented checks found issues in critical workflow areas.',
                    accessibility,
                )
            )
            visibility = self._audit_visibility(page)
            report.sections['ui_visibility'] = visibility
            report.checks.append(
                ValidationCheck(
                    'UI visibility audit',
                    'pass' if not visibility['issues'] else 'fail',
                    'Critical controls remain visible and readable.' if not visibility['issues'] else 'Detected low-visibility controls in the critical workflow.',
                    visibility,
                )
            )
            if visibility['issues']:
                report.remaining_risks.extend(visibility['issues'])
            if visual_regression:
                visual = self._run_visual_regression(active_fixture, update_baselines=update_baselines)
                report.sections['visual_regression'] = visual
                report.checks.append(
                    ValidationCheck(
                        'visual regression',
                        'pass' if visual.get('passed', False) else 'fail',
                        visual.get('summary', 'Visual regression completed.'),
                        visual,
                    )
                )
            export_validation = _build_export_validation(dataset_path, fixture=active_fixture, artifact_dir=self.artifact_dir)
            report.sections['export_validation'] = export_validation
            report.checks.append(
                ValidationCheck(
                    'export validation',
                    'pass' if export_validation.get('all_passed') else 'fail',
                    'Export artifacts were generated and matched the uploaded dataset context.' if export_validation.get('all_passed') else 'One or more generated export artifacts were missing or contaminated.',
                    export_validation,
                )
            )
            report.root_cause_summary = 'No functional regressions detected in the uploaded dataset workflow.' if report.passed else 'One or more uploaded-dataset or UI visibility checks failed. Review the per-tab diagnostics and screenshots.'
            return report
        except Exception as error:
            failure_path = self.artifact_dir / 'browser_validation_failure.png'
            if page is not None:
                try:
                    page.screenshot(path=str(failure_path), full_page=True)
                except Exception:
                    pass
            report.checks.append(
                ValidationCheck(
                    'browser validation runtime',
                    'fail',
                    f'Browser validation failed before completion: {error}',
                    {'exception_type': type(error).__name__, 'exception_message': str(error)},
                    screenshot=str(failure_path) if failure_path.exists() else '',
                )
            )
            report.root_cause_summary = f'Browser validation failed: {error}'
            report.remaining_risks.append('Review the browser failure screenshot and Streamlit logs in the artifact directory.')
            return report
        finally:
            if page is not None:
                page.close()
            if browser is not None:
                browser.close()
            if playwright is not None:
                playwright.stop()
            server.stop()
            report.finalize()

    def _set_uploaded_source(self, page) -> None:
        sidebar = page.locator('[data-testid="stSidebar"]').first
        option = sidebar.get_by_text('Uploaded dataset', exact=True).first
        option.wait_for(timeout=20000)
        option.click(timeout=10000)
        page.wait_for_timeout(1200)

    def _upload_dataset(self, page, dataset_path: Path) -> None:
        sidebar = page.locator('[data-testid="stSidebar"]').first
        file_input = sidebar.locator('input[type="file"]').first
        file_input.set_input_files(str(dataset_path))

    def _wait_for_analysis(self, page) -> None:
        timeout_seconds = LARGE_DATASET_TIMEOUT_SECONDS if self._active_fixture and self._active_fixture.large_file else DEFAULT_TIMEOUT_SECONDS
        page.get_by_text('Analysis context is ready.', exact=False).first.wait_for(timeout=timeout_seconds * 1000)
        page.wait_for_timeout(2500)

    def _wait_for_idle_persistence(self, page, *, seconds: int) -> dict[str, Any]:
        page.wait_for_timeout(seconds * 1000)
        diagnostics = self._collect_diagnostics(page)
        return {
            'wait_seconds': seconds,
            'diagnostics': diagnostics,
        }

    def _run_stability_cycles(self, page, *, soak_seconds: int, rerun_cycles: int) -> dict[str, Any]:
        tabs = [name for name, _ in self.TAB_EXPECTATIONS[: min(6, len(self.TAB_EXPECTATIONS))]]
        cycles: list[dict[str, Any]] = []
        for cycle in range(rerun_cycles):
            for tab_name in tabs:
                self._validate_tab(page, tab_name, tab_name)
            diagnostics = self._collect_diagnostics(page)
            cycles.append({'cycle': cycle + 1, 'diagnostics': diagnostics})
        if soak_seconds > 0:
            page.wait_for_timeout(soak_seconds * 1000)
        final_diagnostics = self._collect_diagnostics(page)
        stable = final_diagnostics.get('active_dataset_source') == 'Uploaded dataset'
        return {
            'rerun_cycles': rerun_cycles,
            'soak_seconds': soak_seconds,
            'cycles': cycles,
            'final_diagnostics': final_diagnostics,
            'stable': stable,
        }

    def _run_visual_regression(self, fixture: DatasetFixture, *, update_baselines: bool) -> dict[str, Any]:
        baseline_dir = _validation_baseline_dir(fixture)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        diff_dir = self.artifact_dir / 'visual_diffs'
        diff_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        for name in VISUAL_REGRESSION_SCREENS:
            candidate = self.artifact_dir / name
            baseline = baseline_dir / name
            if update_baselines and candidate.exists():
                baseline.write_bytes(candidate.read_bytes())
            result = _compare_images(baseline, candidate, diff_dir / f'diff_{name}')
            result['name'] = name
            results.append(result)
        available = [item for item in results if item.get('available')]
        passed = bool(available) and all(item.get('passed', False) for item in available)
        if not available:
            return {
                'passed': False,
                'summary': 'Visual regression could not run because baselines or Pillow were unavailable.',
                'results': results,
            }
        return {
            'passed': passed,
            'summary': 'Visual regression matched configured baselines.' if passed else 'Visual regression found screenshot drift beyond the allowed threshold.',
            'results': results,
        }

    def _validate_tab(self, page, tab_name: str, expected_text: str) -> Path:
        locator = page.get_by_role('tab', name=re.compile(tab_name, re.I)).first
        if locator.count() == 0:
            locator = page.get_by_text(tab_name, exact=False).first
        locator.wait_for(timeout=20000)
        locator.click(timeout=10000)
        page.wait_for_timeout(1800)
        aria_selected = locator.get_attribute('aria-selected')
        if aria_selected not in {None, 'true'}:
            raise AssertionError(f"Tab '{tab_name}' did not become active after click.")
        self._wait_for_visible_text(page, expected_text, timeout_ms=12000)
        screenshot_path = self.artifact_dir / f"{re.sub(r'[^a-zA-Z0-9]+', '_', tab_name.lower()).strip('_')}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        return screenshot_path

    def _validate_manual_mapping_profile_flow(self, page) -> dict[str, Any]:
        screenshot_path = self._validate_tab(page, 'Data Intake', 'Field Remapping Studio')
        body = page.locator('body')
        initial_text = body.inner_text(timeout=12000)
        if 'Encounter Raw Feed Template' not in initial_text:
            return {
                'passed': False,
                'summary': 'Expected suggested mapping profile was not visible for the ambiguous mapping fixture.',
                'screenshot': str(screenshot_path),
            }
        page.get_by_role('button', name='Apply Suggested Profile').first.click(timeout=10000)
        page.wait_for_timeout(1800)
        self._wait_for_visible_text(page, 'Field Remapping Studio', timeout_ms=12000)
        page.get_by_role('button', name='Save Remap Overrides').first.click(timeout=10000)
        confirmation_text = 'Applied mapping profile:'
        confirmation_visible = self._wait_for_visible_text(page, confirmation_text, timeout_ms=12000)
        page.wait_for_timeout(1200)
        final_text = ''
        try:
            final_text = page.locator('body').inner_text(timeout=5000)
        except Exception:
            final_text = ''
        final_screenshot = self.artifact_dir / 'manual_mapping_profile_flow.png'
        page.screenshot(path=str(final_screenshot), full_page=True)
        return {
            'passed': confirmation_visible,
            'summary': 'Suggested mapping profile was applied and the post-save confirmation state was visible on-screen.' if confirmation_visible else 'Suggested mapping profile flow completed, but the post-save confirmation state was not detected.',
            'confirmation_visible': confirmation_visible,
            'manual_semantic_override_summary': final_text,
            'screenshot': str(final_screenshot),
        }

    def _wait_for_visible_text(self, page, expected_text: str, *, timeout_ms: int) -> bool:
        deadline = time.time() + (timeout_ms / 1000.0)
        body = page.locator('body')
        expected_normalized = ' '.join(expected_text.split()).lower()
        while time.time() < deadline:
            try:
                content = body.inner_text(timeout=5000)
            except Exception:
                content = ''
            if expected_normalized in ' '.join(content.split()).lower():
                return True
            page.wait_for_timeout(500)
        return False

    def _collect_diagnostics(self, page) -> dict[str, Any]:
        sidebar = page.locator('[data-testid="stSidebar"]').first
        expander = sidebar.get_by_text('Developer diagnostics', exact=False).first
        expander.click(timeout=10000)
        page.wait_for_timeout(600)
        sidebar_text = sidebar.inner_text()
        diagnostics: dict[str, Any] = {}
        patterns = {
            'active_dataset_source': r'active_dataset_source["\']?\s*[: ]\s*["\']?([^\n,"\'}]+)',
            'dataset_name': r'dataset_name["\']?\s*[: ]\s*["\']?([^\n,"\'}]+)',
            'dataset_identifier': r'dataset_identifier["\']?\s*[: ]\s*["\']?([^\n,"\'}]+)',
            'dataset_cache_key': r'dataset_cache_key["\']?\s*[: ]\s*["\']?([^\n,"\'}]+)',
            'row_count': r'row_count["\']?\s*[: ]\s*(\d+)',
            'column_count': r'column_count["\']?\s*[: ]\s*(\d+)',
            'manual_semantic_override_summary': r'manual_semantic_override_summary["\']?\s*[: ]\s*([^\n]+)',
            'trend_cache_belongs_to_current_dataset': r'trend_cache_belongs_to_current_dataset["\']?\s*[: ]\s*(True|False|true|false)',
            'cohort_cache_belongs_to_current_dataset': r'cohort_cache_belongs_to_current_dataset["\']?\s*[: ]\s*(True|False|true|false)',
            'manual_semantic_override_count': r'manual_semantic_override_count["\']?\s*[: ]\s*(\d+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, sidebar_text)
            if not match:
                continue
            value = match.group(1)
            if value in {'True', 'False', 'true', 'false'}:
                diagnostics[key] = value.lower() == 'true'
            elif value.isdigit():
                diagnostics[key] = int(value)
            else:
                diagnostics[key] = value.strip()
        return diagnostics

    def _demo_context_visible(self, page) -> bool:
        forbidden = (
            'Guided Demo Mode',
            'Startup Demo Flow',
            'Demo mode is active',
        )
        return any(page.get_by_text(label, exact=False).count() > 0 for label in forbidden)

    def _audit_visibility(self, page) -> dict[str, Any]:
        samples = page.evaluate(
            """
            () => {
              const pick = Array.from(document.querySelectorAll('button, [role="tab"], [role="combobox"], input, textarea'))
                .filter((node) => {
                  const style = window.getComputedStyle(node);
                  const rect = node.getBoundingClientRect();
                  return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
                })
                .slice(0, 40);
              return pick.map((node) => {
                const style = window.getComputedStyle(node);
                return {
                  text: (node.innerText || node.getAttribute('aria-label') || node.getAttribute('placeholder') || node.tagName).trim().slice(0, 80),
                  background: style.backgroundColor,
                  color: style.color,
                  opacity: style.opacity,
                };
              });
            }
            """
        )
        issues: list[str] = []
        evaluated: list[dict[str, Any]] = []
        ignored_labels = {
            'keyboard_double_arrow_left',
            'search',
            'fullscreen',
            'deploy',
            'main menu',
        }
        for sample in samples:
            label = str(sample.get('text', '')).strip()
            if not label or label.lower() in ignored_labels:
                continue
            fg = _parse_rgb(sample.get('color', 'rgb(15, 23, 42)'))
            bg = _parse_rgb(sample.get('background', 'rgb(255, 255, 255)'))
            contrast = round(_contrast_ratio(fg, bg), 2)
            opacity = float(sample.get('opacity', 1) or 1)
            evaluated.append({'text': label, 'contrast_ratio': contrast, 'opacity': opacity})
            if contrast < 4.5 or opacity < 0.7:
                issues.append(f"{label} has low visibility (contrast={contrast}, opacity={opacity}).")
        return {'controls_evaluated': evaluated, 'issues': issues}

    def _audit_accessibility(self, page) -> dict[str, Any]:
        issues: list[str] = []
        focus_samples: list[dict[str, Any]] = []
        ignored_labels = {'download as csv', 'search', 'fullscreen', 'canvas', 'deploy'}
        interactive_roles = {'button', 'link', 'tab', 'textbox', 'combobox', 'option', 'menuitem', 'checkbox', 'radio', 'switch', 'summary'}
        interactive_tags = {'BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA', 'SUMMARY'}
        page.locator('body').focus()
        last_text = ''
        for _ in range(6):
            page.keyboard.press('Tab')
            page.wait_for_timeout(100)
            sample = page.evaluate(
                """
                () => {
                  const node = document.activeElement;
                  if (!node) return {};
                  const style = window.getComputedStyle(node);
                  return {
                    text: (node.innerText || node.getAttribute('aria-label') || node.getAttribute('placeholder') || node.tagName).trim().slice(0, 80),
                    outline: style.outlineStyle,
                    outlineWidth: style.outlineWidth,
                    boxShadow: style.boxShadow,
                    role: node.getAttribute('role') || '',
                    tagName: node.tagName,
                  };
                }
                """
            )
            text = str(sample.get('text', '')).strip()
            if not text:
                continue
            if text.lower() in ignored_labels:
                continue
            focus_samples.append(sample)
            if text == last_text:
                issues.append('Keyboard navigation focus did not advance to a new control.')
            last_text = text
            outline_width = str(sample.get('outlineWidth', '0px'))
            box_shadow = str(sample.get('boxShadow', 'none'))
            role_name = str(sample.get('role', '')).strip().lower()
            tag_name = str(sample.get('tagName', '')).strip().upper()
            requires_focus_indicator = role_name in interactive_roles or tag_name in interactive_tags
            if requires_focus_indicator and outline_width in {'0px', '0'} and box_shadow in {'none', ''}:
                issues.append(f"Focused control '{text}' has no obvious focus indicator.")
        headings_present = {
            'data_intake': page.get_by_text('Data Intake', exact=False).count() > 0,
            'overview': page.get_by_text('Clinical Readiness Workspace', exact=False).count() > 0,
        }
        if not all(headings_present.values()):
            issues.append('Important headings are missing from the active workflow.')
        return {
            'focus_samples': focus_samples,
            'headings_present': headings_present,
            'issues': issues,
        }


def run_validation(
    dataset_path: Path,
    *,
    mode: str,
    python_executable: str,
    fixture: DatasetFixture | None = None,
    visual_regression: bool = False,
    update_baselines: bool = False,
    soak_seconds: int = 0,
    rerun_cycles: int = 0,
) -> ValidationReport:
    artifact_dir = ensure_artifact_dir(mode, dataset_path.name)
    active_fixture = fixture or get_default_fixture()
    command = f'python scripts/run_dataset_validation.py --dataset "{dataset_path}" --mode {mode}'
    commands_run = [command]
    browser_env: dict[str, Any] = {}
    last_completed_step = 'artifact directory created'
    write_partial_state(
        artifact_dir,
        {
            'status': 'started',
            'dataset_path': str(dataset_path),
            'dataset_name': dataset_path.name,
            'mode': mode,
            'last_completed_step': last_completed_step,
            'command': command,
        },
    )
    try:
        browser_env = browser_environment_readiness()
        last_completed_step = 'browser environment readiness'
        write_partial_state(
            artifact_dir,
            {
                'status': 'running',
                'dataset_name': dataset_path.name,
                'mode': mode,
                'last_completed_step': last_completed_step,
                'browser_environment': browser_env,
            },
        )
        if browser_env.get('usable'):
            validator = BrowserDatasetValidator(artifact_dir)
            report = validator.validate(
                dataset_path,
                mode,
                python_executable,
                active_fixture,
                visual_regression=visual_regression,
                update_baselines=update_baselines,
                soak_seconds=soak_seconds,
                rerun_cycles=rerun_cycles,
            )
            last_completed_step = 'browser validation'
        else:
            validator = ProgrammaticDatasetValidator(artifact_dir)
            report = validator.validate(dataset_path, mode, active_fixture)
            report.degraded_browser_reason = str(browser_env.get('reason', 'Browser environment is unavailable.'))
            report.checks.append(
                ValidationCheck(
                    'browser environment readiness',
                    'pass',
                    'Browser validation was degraded and the programmatic fallback path was used.',
                    browser_env,
                )
            )
            report.root_cause_summary = report.root_cause_summary or report.degraded_browser_reason
            last_completed_step = 'programmatic validation'
        report.sections.setdefault('browser_environment', browser_env)
        report.artifact_dir = str(artifact_dir)
        write_report_files(report, artifact_dir)
        write_partial_state(
            artifact_dir,
            {
                'status': 'completed',
                'dataset_name': dataset_path.name,
                'mode': mode,
                'last_completed_step': last_completed_step,
                'passed': report.passed,
                'artifact_dir': str(artifact_dir),
            },
        )
        return report
    except Exception as error:
        failure_report = build_failure_report(
            dataset_path=dataset_path,
            mode=mode,
            artifact_dir=artifact_dir,
            command=command,
            commands_run=commands_run,
            last_completed_step=last_completed_step,
            error=error,
            diagnostics={'browser_environment': browser_env} if browser_env else {},
        )
        failure_report.sections.setdefault('browser_environment', browser_env)
        failure_report.artifact_dir = str(artifact_dir)
        write_report_files(failure_report, artifact_dir)
        write_partial_state(
            artifact_dir,
            {
                'status': 'failed',
                'dataset_name': dataset_path.name,
                'mode': mode,
                'last_completed_step': last_completed_step,
                'exception_type': type(error).__name__,
                'exception_message': str(error),
                'artifact_dir': str(artifact_dir),
            },
        )
        return failure_report
