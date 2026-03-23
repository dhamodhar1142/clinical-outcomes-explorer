from __future__ import annotations

import unittest
from unittest import mock

from src.beta_interest import (
    beta_interest_csv_bytes,
    append_beta_interest_submission,
    build_beta_conversion_panel,
    build_beta_interest_summary,
    merge_beta_interest_submissions,
    save_beta_interest_submission,
    update_beta_interest_status,
    validate_beta_interest,
)
from src.product_settings import DEFAULT_PRODUCT_SETTINGS, build_product_settings_summary
from src.plan_awareness import build_plan_awareness


class _FakeApplicationService:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.saved: list[tuple[dict[str, object], dict[str, object]]] = []

    def save_beta_interest_submission(self, identity: dict[str, object], submission: dict[str, object]) -> None:
        self.saved.append((identity, submission))


class BetaInterestTests(unittest.TestCase):
    def test_append_beta_interest_submission_records_local_demo_fields(self) -> None:
        submissions: list[dict[str, object]] = []
        record = append_beta_interest_submission(
            submissions,
            name='Jamie Rivera',
            email='Jamie@Example.com',
            organization='Care Ops',
            use_case='Population health review',
            workspace_name='Workspace A',
            submitted_by='Dana Analyst',
        )
        self.assertEqual(len(submissions), 1)
        self.assertEqual(record['email'], 'jamie@example.com')
        self.assertEqual(record['capture_mode'], 'Local demo')

    def test_build_beta_interest_summary_handles_empty_state(self) -> None:
        summary = build_beta_interest_summary([])
        self.assertEqual(summary['summary_cards'][0]['value'], '0')
        self.assertTrue(summary['history'].empty)

    def test_build_beta_interest_summary_counts_organizations_and_use_cases(self) -> None:
        submissions = [
            {
                'submitted_at': '2026-03-13T00:00:00+00:00',
                'name': 'Jamie Rivera',
                'email': 'jamie@example.com',
                'organization': 'Care Ops',
                'use_case': 'Population health review',
                'workspace_name': 'Workspace A',
                'submitted_by': 'Dana Analyst',
                'capture_mode': 'Local demo',
            },
            {
                'submitted_at': '2026-03-13T00:05:00+00:00',
                'name': 'Sam Lee',
                'email': 'sam@example.com',
                'organization': 'Care Ops',
                'use_case': 'Readmission workflow',
                'workspace_name': 'Workspace A',
                'submitted_by': 'Dana Analyst',
                'capture_mode': 'Local demo',
            },
        ]
        summary = build_beta_interest_summary(submissions)
        self.assertEqual(summary['summary_cards'][0]['value'], '2')
        self.assertEqual(summary['summary_cards'][1]['value'], '1')
        self.assertEqual(summary['summary_cards'][2]['value'], '2')

    def test_product_settings_summary_includes_beta_interest_row(self) -> None:
        plan = build_plan_awareness('Pro', 'Demo-safe', {'file_size_mb': 5.0}, workflow_pack_count=1, snapshot_count=1)
        summary = build_product_settings_summary(DEFAULT_PRODUCT_SETTINGS, plan)
        self.assertTrue((summary['settings_table']['setting_area'] == 'Beta Interest Capture').any())

    def test_build_beta_conversion_panel_exposes_startup_ctas(self) -> None:
        panel = build_beta_conversion_panel(True, [])
        self.assertIn('headline', panel)
        self.assertEqual(len(panel['cta_cards']), 3)
        self.assertEqual(panel['cta_cards'][0]['label'], 'Request Early Access')
        self.assertIn('local', panel['status_note'].lower())
        self.assertTrue(panel['conversion_steps'])
        self.assertIn('reassurance_note', panel)
        self.assertIn('Clinverity', panel['subheadline'])
        self.assertIn('cta_summary', panel['cta_cards'][0])

    def test_validate_beta_interest_rejects_invalid_values(self) -> None:
        errors = validate_beta_interest('!', 'bad-email', 'O' * 201, 'short')
        self.assertGreaterEqual(len(errors), 3)

    def test_save_beta_interest_submission_local_allows_duplicate_email(self) -> None:
        submissions: list[dict[str, object]] = []
        identity = {
            'workspace_id': 'workspace-1',
            'workspace_name': 'Workspace A',
            'display_name': 'Dana Analyst',
        }
        first = save_beta_interest_submission(
            local_submissions=submissions,
            name='Jamie Rivera',
            email='jamie@example.com',
            organization='Care Ops',
            use_case='Population health review for an analytics pilot.',
            workspace_identity=identity,
            dataset_name='Healthcare Demo',
            dataset_source_mode='Demo dataset',
            storage_mode='LOCAL',
        )
        second = save_beta_interest_submission(
            local_submissions=submissions,
            name='Jamie Rivera',
            email='jamie@example.com',
            organization='Care Ops',
            use_case='Readmission analytics workflow for a second business case.',
            workspace_identity=identity,
            dataset_name='Healthcare Demo',
            dataset_source_mode='Demo dataset',
            storage_mode='LOCAL',
        )
        self.assertEqual(len(submissions), 2)
        self.assertNotEqual(first['submission_id'], second['submission_id'])

    def test_save_beta_interest_submission_database_uses_application_service(self) -> None:
        submissions: list[dict[str, object]] = []
        identity = {
            'workspace_id': 'workspace-1',
            'workspace_name': 'Workspace A',
            'display_name': 'Dana Analyst',
        }
        application_service = _FakeApplicationService(enabled=True)
        save_beta_interest_submission(
            local_submissions=submissions,
            name='Jamie Rivera',
            email='jamie@example.com',
            organization='Care Ops',
            use_case='Population health review for an analytics pilot.',
            workspace_identity=identity,
            dataset_name='Healthcare Demo',
            dataset_source_mode='Demo dataset',
            storage_mode='DATABASE',
            application_service=application_service,
        )
        self.assertEqual(len(submissions), 1)
        self.assertEqual(len(application_service.saved), 1)

    @mock.patch('src.beta_interest.urllib_request.urlopen')
    def test_save_beta_interest_submission_api_posts_json(self, mock_urlopen: mock.Mock) -> None:
        response = mock.Mock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response
        submissions: list[dict[str, object]] = []
        identity = {
            'workspace_id': 'workspace-1',
            'workspace_name': 'Workspace A',
            'display_name': 'Dana Analyst',
        }
        save_beta_interest_submission(
            local_submissions=submissions,
            name='Jamie Rivera',
            email='jamie@example.com',
            organization='Care Ops',
            use_case='Population health review for an analytics pilot.',
            workspace_identity=identity,
            dataset_name='Healthcare Demo',
            dataset_source_mode='Demo dataset',
            storage_mode='API',
            api_url='https://crm.example.com/beta-interest',
            api_token='secret-token',
        )
        self.assertEqual(len(submissions), 1)
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, 'https://crm.example.com/beta-interest')
        self.assertEqual(request.headers['Authorization'], 'Bearer secret-token')
        self.assertIn(b'jamie@example.com', request.data)

    def test_beta_interest_csv_and_status_update(self) -> None:
        submissions = [
            {
                'submission_id': 'submission-1',
                'submitted_at': '2026-03-13T00:00:00+00:00',
                'name': 'Jamie Rivera',
                'email': 'jamie@example.com',
                'organization': 'Care Ops',
                'use_case': 'Population health review',
                'workspace_name': 'Workspace A',
                'workspace_id': 'workspace-1',
                'submitted_by': 'Dana Analyst',
                'dataset_name': 'Healthcare Demo',
                'dataset_source_mode': 'Demo dataset',
                'capture_mode': 'Local demo',
                'follow_up_status': 'New',
                'contacted_at': '',
                'completed_at': '',
            }
        ]
        updated = update_beta_interest_status(submissions, 'submission-1', follow_up_status='Completed')
        self.assertIsNotNone(updated)
        self.assertEqual(updated['follow_up_status'], 'Completed')
        self.assertTrue(updated['completed_at'])
        csv_payload = beta_interest_csv_bytes(submissions).decode('utf-8')
        self.assertIn('submission_id,submitted_at,name,email', csv_payload)
        self.assertIn('jamie@example.com', csv_payload)

    def test_merge_beta_interest_submissions_dedupes_by_submission_id(self) -> None:
        submissions = merge_beta_interest_submissions(
            [{'submission_id': 'sub-1', 'email': 'a@example.com', 'submitted_at': '2026-03-13T00:00:00+00:00'}],
            [{'submission_id': 'sub-1', 'email': 'a@example.com', 'submitted_at': '2026-03-13T00:00:00+00:00'}],
        )
        self.assertEqual(len(submissions), 1)


if __name__ == '__main__':
    unittest.main()
