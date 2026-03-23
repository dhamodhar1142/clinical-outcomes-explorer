from __future__ import annotations

import unittest

import pandas as pd

from src.modules.audit import log_audit_event
from src.modules.privacy_security import build_export_governance_summary, run_privacy_security_review


class SecurityGovernanceFoundationTests(unittest.TestCase):
    def test_audit_events_capture_workspace_context(self) -> None:
        events = log_audit_event(
            [],
            'Export Downloaded',
            'Downloaded executive summary.',
            actor_context={
                'workspace_id': 'care-ops::local-demo-user',
                'workspace_name': 'Care Ops',
                'user_id': 'local::dana',
                'role_label': 'Admin',
                'owner_label': 'Dana Analyst',
            },
            resource_context={
                'resource_type': 'generated_report',
                'resource_name': 'Executive Summary',
                'action_outcome': 'success',
            },
        )
        self.assertEqual(events[0]['workspace_id'], 'care-ops::local-demo-user')
        self.assertEqual(events[0]['workspace_role'], 'Admin')
        self.assertEqual(events[0]['owner_label'], 'Dana Analyst')
        self.assertEqual(events[0]['resource_type'], 'generated_report')
        self.assertEqual(events[0]['resource_name'], 'Executive Summary')

    def test_export_governance_summary_returns_controls(self) -> None:
        privacy_review = {
            'sensitive_fields': pd.DataFrame(
                [
                    {'column_name': 'email', 'sensitive_type': 'Email'},
                    {'column_name': 'mrn', 'sensitive_type': 'Medical Record Number'},
                ]
            )
        }
        summary = build_export_governance_summary(
            'HIPAA-style Limited Dataset',
            privacy_review,
            {'role_label': 'Analyst', 'owner_label': 'Dana Analyst'},
        )
        self.assertTrue(summary['summary_cards'])
        self.assertFalse(summary['controls_table'].empty)
        self.assertIn('Workspace ownership boundary', summary['controls_table']['control_area'].tolist())
        self.assertIn('Workspace-level access control', summary['controls_table']['control_area'].tolist())

    def test_privacy_review_builds_data_classification(self) -> None:
        data = pd.DataFrame(
            [
                {'patient_name': 'Dana Analyst', 'mrn': 'MRN1234', 'department': 'ICU'},
            ]
        )
        review = run_privacy_security_review(data)
        classification_table = review['classification_table']
        self.assertFalse(classification_table.empty)
        self.assertIn('classification', classification_table.columns)
        self.assertEqual(
            classification_table.loc[classification_table['column_name'] == 'patient_name', 'classification'].iloc[0],
            'PII',
        )
        self.assertEqual(
            classification_table.loc[classification_table['column_name'] == 'mrn', 'classification'].iloc[0],
            'PHI',
        )

    def test_export_governance_summary_respects_workspace_access_override(self) -> None:
        privacy_review = {
            'sensitive_fields': pd.DataFrame([{'column_name': 'email', 'sensitive_type': 'Email', 'classification': 'PII'}]),
            'classification_summary': {'public_count': 1, 'pii_count': 1, 'phi_count': 0},
        }
        summary = build_export_governance_summary(
            'Research-safe Extract',
            privacy_review,
            {'role': 'viewer', 'role_label': 'Viewer', 'owner_label': 'Dana Analyst', 'auth_mode': 'local'},
            {'workspace_export_access': 'Owner only', 'redaction_level': 'High', 'watermark_sensitive_exports': True},
        )
        controls = summary['controls_table']
        access_row = controls[controls['control_area'] == 'Workspace-level access control'].iloc[0]
        self.assertEqual(access_row['status'], 'Restricted')
        watermark_row = controls[controls['control_area'] == 'Export watermark'].iloc[0]
        self.assertEqual(watermark_row['status'], 'Enabled')


if __name__ == '__main__':
    unittest.main()
