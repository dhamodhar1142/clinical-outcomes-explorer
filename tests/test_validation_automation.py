from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.run_release_validation as release_runner
import scripts.run_dataset_validation as validation_runner
import scripts.run_validation_task as validation_task
from tests.e2e.fixture_registry import (
    AMBIGUOUS_MAPPING_FIXTURE,
    MALFORMED_FIXTURE,
    PRIMARY_FULL_VALIDATION_FIXTURE,
    SECONDARY_HEALTHCARE_FIXTURE,
    SMALL_HEALTHCARE_FIXTURE,
    get_default_fixture,
)
from tests.e2e.validation_helpers import (
    BrowserDatasetValidator,
    ProgrammaticDatasetValidator,
    ValidationCheck,
    ValidationReport,
    VISUAL_REGRESSION_SCREENS,
    _contrast_ratio,
    _compare_images,
    _stable_payload_identity,
    browser_environment_readiness,
    build_failure_report,
    inspect_fixture_schema,
    read_dataset_shape,
    run_cross_dataset_cache_validation,
    run_validation,
    write_report_files,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'upload_validation_fixture.csv'


class ValidationAutomationTests(unittest.TestCase):
    def test_default_fixture_registry_points_to_primary_healthcare_dataset(self) -> None:
        fixture = get_default_fixture()
        self.assertEqual(fixture.dataset_name, 'STG_EHP__VIST.csv')
        self.assertEqual(fixture.dataset_type, 'healthcare')
        self.assertTrue(fixture.large_file)
        self.assertEqual(fixture.expected_approximate_row_count, 917331)
        self.assertIn('VIS_EN', fixture.expected_date_columns)
        self.assertIn('VIS_EX', fixture.expected_date_columns)

    def test_programmatic_validator_produces_uploaded_dataset_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = ProgrammaticDatasetValidator(Path(temp_dir))
            report = validator.validate(FIXTURE, 'quick')

        self.assertEqual(report.dataset_name, FIXTURE.name)
        self.assertFalse(report.used_browser)
        self.assertEqual(report.diagnostics['active_dataset_source'], 'Uploaded dataset')
        self.assertGreaterEqual(len(report.checks), 4)

    def test_report_writer_emits_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            report = ValidationReport(
                dataset_path=str(FIXTURE),
                dataset_name=FIXTURE.name,
                mode='quick',
                used_browser=False,
                command='python scripts/run_dataset_validation.py --dataset fixture.csv --mode quick',
                commands_run=['command-a'],
                checks=[ValidationCheck('area', 'pass', 'details')],
                artifact_dir=str(artifact_dir),
            )
            report.finalize()
            json_path, markdown_path = write_report_files(report, artifact_dir)

            payload = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['dataset_name'], FIXTURE.name)
            self.assertTrue(markdown_path.read_text(encoding='utf-8').startswith('# Dataset Validation Report'))

    def test_contrast_ratio_flags_dark_on_dark_controls(self) -> None:
        ratio = _contrast_ratio((20, 23, 28), (15, 18, 22))
        self.assertLess(ratio, 1.5)

    def test_fixture_shape_is_stable(self) -> None:
        shape = read_dataset_shape(FIXTURE)
        self.assertEqual(shape['row_count'], 3)
        self.assertEqual(shape['column_count'], 8)

    def test_browser_environment_readiness_reports_structure(self) -> None:
        readiness = browser_environment_readiness()
        self.assertIn('playwright_installed', readiness)
        self.assertIn('browser_launchable', readiness)
        self.assertIn('usable', readiness)
        self.assertIn('reason', readiness)

    def test_fixture_schema_inspection_is_case_normalized(self) -> None:
        inspection = inspect_fixture_schema(SMALL_HEALTHCARE_FIXTURE.resolve_path(), SMALL_HEALTHCARE_FIXTURE)
        self.assertIn('VIS_EN', inspection['date_columns_detected'])
        self.assertIn('REFR_NO', inspection['identifier_columns_detected'])
        self.assertEqual(inspection['missing_expected_columns'], [])

    def test_stable_payload_identity_is_deterministic(self) -> None:
        payload = {'b': 2, 'a': 1}
        self.assertEqual(_stable_payload_identity(payload), _stable_payload_identity({'a': 1, 'b': 2}))

    def test_cross_dataset_cache_validation_report_passes(self) -> None:
        report = run_cross_dataset_cache_validation(python_executable='python')
        self.assertTrue(any(check.area == 'cross-dataset cache identity' for check in report.checks))
        self.assertTrue(report.passed)

    def test_fixture_registry_supports_secondary_and_malformed_fixtures(self) -> None:
        self.assertTrue(SECONDARY_HEALTHCARE_FIXTURE.path.exists())
        self.assertTrue(MALFORMED_FIXTURE.path.exists())
        self.assertTrue(AMBIGUOUS_MAPPING_FIXTURE.path.exists())

    def test_large_fixture_resolution_prefers_env_override(self) -> None:
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            local_copy = Path(temp_dir) / PRIMARY_FULL_VALIDATION_FIXTURE.dataset_name
            local_copy.write_text('REFR_NO,PAT_ID\n1,2\n', encoding='utf-8')
            original = os.environ.get('SMART_DATASET_ANALYZER_LARGE_FIXTURE_PATH')
            os.environ['SMART_DATASET_ANALYZER_LARGE_FIXTURE_PATH'] = str(local_copy)
            try:
                self.assertEqual(PRIMARY_FULL_VALIDATION_FIXTURE.resolve_path(), local_copy)
            finally:
                if original is None:
                    os.environ.pop('SMART_DATASET_ANALYZER_LARGE_FIXTURE_PATH', None)
                else:
                    os.environ['SMART_DATASET_ANALYZER_LARGE_FIXTURE_PATH'] = original

    def test_ambiguous_mapping_fixture_is_registered_for_manual_override_e2e(self) -> None:
        self.assertEqual(AMBIGUOUS_MAPPING_FIXTURE.dataset_name, 'AMBIGUOUS_ENCOUNTER_VISITS.csv')
        self.assertEqual(AMBIGUOUS_MAPPING_FIXTURE.expected_approximate_row_count, 3)

    def test_visual_regression_tracks_required_screens(self) -> None:
        required = {
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
        }
        self.assertTrue(required.issubset(set(VISUAL_REGRESSION_SCREENS)))

    def test_prompt_friendly_task_runner_supports_release(self) -> None:
        self.assertIn('release', validation_task.TASK_COMMANDS)
        self.assertEqual(validation_task.TASK_COMMANDS['release'], ['scripts/run_release_validation.py'])

    def test_release_summary_writer_emits_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            release_dir = Path(temp_dir)
            summary = release_runner.ReleaseValidationSummary(
                overall_status='PASS',
                started_at='2026-03-17T00:00:00+00:00',
                completed_at='2026-03-17T00:10:00+00:00',
                selected_fixture='fixture.csv',
                release_artifact_dir=str(release_dir),
                phases=[
                    release_runner.ReleasePhaseResult(
                        name='phase',
                        command=['python', 'phase.py'],
                        status='pass',
                        exit_code=0,
                        artifact_path='artifact-path',
                    )
                ],
                commands_run=['python phase.py'],
            )
            json_path, markdown_path = release_runner._write_summary(summary, release_dir)
            payload = json.loads(json_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['overall_status'], 'PASS')
            self.assertTrue(markdown_path.read_text(encoding='utf-8').startswith('# Release Validation Summary'))

    def test_release_phase_builder_skips_visual_in_ci_when_baselines_missing(self) -> None:
        phases = release_runner._build_phase_specs(
            python_executable='python',
            fixture_key='small-healthcare',
            ci_mode=True,
            skip_visual=False,
            skip_soak=False,
            soak_seconds=60,
            baseline_ready=False,
        )
        visual_phase = next(phase for phase in phases if phase.name == 'visual regression')
        self.assertIn('Skipped in CI mode because no baselines are available', visual_phase.skip_reason or '')

    def test_release_phase_builder_respects_fixture_override(self) -> None:
        phases = release_runner._build_phase_specs(
            python_executable='python',
            fixture_key='small-healthcare',
            ci_mode=True,
            skip_visual=True,
            skip_soak=True,
            soak_seconds=60,
            baseline_ready=False,
        )
        full_phase = next(phase for phase in phases if phase.name == 'full uploaded-dataset validation')
        self.assertIn('small-healthcare', full_phase.command)

    def test_release_runner_uses_log_artifact_when_phase_does_not_emit_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            release_dir = Path(temp_dir)
            result = release_runner._run_phase(
                'quick framework checks',
                ['python', '-c', "print('phase ok')"],
                release_dir,
                parse_artifacts=False,
            )
            self.assertEqual(result.artifact_path, result.output_path)
            self.assertIn('using the phase log as the phase artifact', ' '.join(result.notes).lower())

    @unittest.skipIf(Image is None, 'Pillow is required for image diff testing.')
    def test_compare_images_reports_small_pixel_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            reference_path = temp_path / 'reference.png'
            candidate_path = temp_path / 'candidate.png'
            diff_path = temp_path / 'diff.png'
            reference = Image.new('RGB', (10, 10), color='white')
            candidate = Image.new('RGB', (10, 10), color='white')
            candidate.putpixel((0, 0), (0, 0, 0))
            reference.save(reference_path)
            candidate.save(candidate_path)
            result = _compare_images(reference_path, candidate_path, diff_path, mismatch_threshold=0.05)

            self.assertTrue(result['available'])
            self.assertAlmostEqual(result['mismatch_ratio'], 0.01, places=6)
            self.assertTrue(result['passed'])

    def test_run_validation_writes_reports_when_browser_validation_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            fake_validator = type(
                'FakeBrowserValidator',
                (),
                {
                    '__init__': lambda self, _artifact_dir: None,
                    'validate': lambda self, *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('browser crash before completion')),
                },
            )
            with patch('tests.e2e.validation_helpers.ensure_artifact_dir', return_value=artifact_dir), patch(
                'tests.e2e.validation_helpers.browser_environment_readiness',
                return_value={'playwright_installed': True, 'browser_launchable': True, 'usable': True, 'reason': ''},
            ), patch('tests.e2e.validation_helpers.BrowserDatasetValidator', fake_validator):
                report = run_validation(FIXTURE, mode='quick', python_executable='python')

            self.assertFalse(report.passed)
            self.assertTrue((artifact_dir / 'validation_report.json').exists())
            self.assertTrue((artifact_dir / 'validation_report.md').exists())
            payload = json.loads((artifact_dir / 'validation_report.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['sections']['failure']['overall_result'], 'FAIL')
            self.assertIn('browser crash before completion', payload['sections']['failure']['exception_message'])

    def test_run_validation_writes_reports_when_startup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            fake_validator = type(
                'FakeBrowserValidator',
                (),
                {
                    '__init__': lambda self, _artifact_dir: None,
                    'validate': lambda self, *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('Streamlit exited before becoming ready.')),
                },
            )
            with patch('tests.e2e.validation_helpers.ensure_artifact_dir', return_value=artifact_dir), patch(
                'tests.e2e.validation_helpers.browser_environment_readiness',
                return_value={'playwright_installed': True, 'browser_launchable': True, 'usable': True, 'reason': ''},
            ), patch('tests.e2e.validation_helpers.BrowserDatasetValidator', fake_validator):
                report = run_validation(FIXTURE, mode='quick', python_executable='python')

            self.assertFalse(report.passed)
            self.assertTrue((artifact_dir / 'validation_report.json').exists())
            self.assertTrue((artifact_dir / 'validation_report.md').exists())
            payload = json.loads((artifact_dir / 'validation_report.json').read_text(encoding='utf-8'))
            self.assertIn('Streamlit exited before becoming ready.', payload['sections']['failure']['exception_message'])

    def test_failure_report_preserves_partial_screenshots_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            screenshot_path = artifact_dir / 'browser_validation_failure.png'
            log_path = artifact_dir / 'streamlit_stdout.log'
            screenshot_path.write_bytes(b'fake-png')
            log_path.write_text('startup log', encoding='utf-8')
            report = build_failure_report(
                dataset_path=FIXTURE,
                mode='full',
                artifact_dir=artifact_dir,
                command='python scripts/run_dataset_validation.py --mode full',
                commands_run=['python scripts/run_dataset_validation.py --mode full'],
                last_completed_step='tab validation',
                error=RuntimeError('validation exception after partial screenshots'),
            )
            write_report_files(report, artifact_dir)
            payload = json.loads((artifact_dir / 'validation_report.json').read_text(encoding='utf-8'))
            self.assertIn(str(screenshot_path), payload['sections']['failure']['screenshots'])
            self.assertIn(str(log_path), payload['sections']['failure']['logs'])

    def test_runner_writes_failure_reports_for_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            with patch.object(validation_runner, 'ensure_artifact_dir', return_value=artifact_dir), patch(
                'sys.argv',
                ['run_dataset_validation.py', '--dataset', str(artifact_dir / 'missing.csv'), '--mode', 'full'],
            ):
                exit_code = validation_runner.main()

            self.assertEqual(exit_code, 1)
            self.assertTrue((artifact_dir / 'validation_report.json').exists())
            self.assertTrue((artifact_dir / 'validation_report.md').exists())


if __name__ == '__main__':
    unittest.main()
