from __future__ import annotations

import json
import unittest
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from src.data_loader import load_demo_dataset
from src.pipeline import run_analysis_pipeline
from src.services.export_service import generate_export_report_output


class ExportGenerationPipelineTests(unittest.TestCase):
    def test_export_generation_pipeline_supports_all_demo_datasets(self) -> None:
        demo_cases = [
            ('Healthcare Operations Demo', 'Executive Report'),
            ('Hospital Reporting Demo', 'Analyst Report'),
            ('Generic Business Demo', 'Data Readiness Report'),
        ]

        for dataset_name, report_label in demo_cases:
            with self.subTest(dataset_name=dataset_name, report_label=report_label):
                data, _ = load_demo_dataset(dataset_name)
                pipeline = run_analysis_pipeline(
                    data,
                    dataset_name,
                    {'source_mode': 'Demo dataset'},
                )
                session_state = {'job_runs': []}
                result = generate_export_report_output(
                    session_state,
                    job_runtime={'backend_configured': False, 'mode': 'sync'},
                    report_label=report_label,
                    dataset_name=dataset_name,
                    pipeline=pipeline,
                    workspace_identity={'workspace_id': 'guest-demo-workspace', 'auth_mode': 'guest', 'role': 'viewer'},
                    role='Analyst',
                    policy_name='Internal Review',
                    privacy_review=pipeline.get('privacy_review', {}),
                )

                self.assertTrue(result['report_bytes'])
                self.assertTrue(result['pdf_bytes'].startswith(b'%PDF'))
                self.assertTrue(result['excel_bytes'])
                self.assertTrue(result['json_bytes'])
                with ZipFile(BytesIO(result['excel_bytes'])) as workbook:
                    names = set(workbook.namelist())
                    workbook_xml = workbook.read('xl/workbook.xml').decode('utf-8')
                self.assertIn('xl/workbook.xml', names)
                self.assertIn('Overview', workbook_xml)
                payload = json.loads(result['json_bytes'].decode('utf-8'))
                self.assertEqual(payload['dataset_name'], dataset_name)
                self.assertEqual(payload['report_label'], report_label)
                self.assertIn('governance', payload)
                with ZipFile(BytesIO(result['zip_bytes'])) as bundle:
                    names = set(bundle.namelist())
                self.assertTrue(any(name.endswith('.txt') for name in names))
                self.assertTrue(any(name.endswith('.pdf') for name in names))
                self.assertTrue(any(name.endswith('.xlsx') for name in names))
                self.assertTrue(any(name.endswith('.json') for name in names))

    def test_export_generation_pipeline_uses_summary_first_strategy_for_large_datasets(self) -> None:
        pipeline = {
            'overview': {'rows': 900000, 'columns': 6, 'analyzed_columns': 6},
            'quality': {'quality_score': 0.82, 'high_missing': pd.DataFrame()},
            'readiness': {'readiness_score': 0.76, 'readiness_table': pd.DataFrame([{'analysis_module': 'Data Quality Review', 'status': 'Available'}])},
            'healthcare': {'healthcare_readiness_score': 0.55},
            'insights': {'summary_lines': ['Large dataset summary.']},
            'action_recommendations': pd.DataFrame([{'recommendation_title': 'Review staged exports', 'priority': 'High', 'rationale': 'Keep exports summary-first.'}]),
            'sample_info': {'sampling_applied': True},
            'privacy_review': {},
        }
        session_state = {'job_runs': []}
        result = generate_export_report_output(
            session_state,
            job_runtime={'backend_configured': False, 'mode': 'sync'},
            report_label='Data Readiness Report',
            dataset_name='Large Dataset',
            pipeline=pipeline,
            workspace_identity={'workspace_id': 'guest-demo-workspace', 'auth_mode': 'guest', 'role': 'viewer'},
            role='Analyst',
        )

        self.assertTrue(result['export_strategy']['large_dataset_mode'])
        self.assertLessEqual(result['export_strategy']['excel_row_cap'], 25000)
        with ZipFile(BytesIO(result['excel_bytes'])) as workbook:
            overview_sheet = workbook.read('xl/worksheets/sheet1.xml').decode('utf-8')
        self.assertIn('Large dataset mode', overview_sheet)
        self.assertIn('Yes', overview_sheet)
        payload = json.loads(result['json_bytes'].decode('utf-8'))
        self.assertTrue(payload['export_strategy']['large_dataset_mode'])
        self.assertIn('governance', payload)

    def test_sensitive_exports_include_watermark_and_governance_metadata(self) -> None:
        pipeline = {
            'overview': {'rows': 100, 'columns': 3, 'analyzed_columns': 3},
            'quality': {'quality_score': 0.82, 'high_missing': pd.DataFrame()},
            'readiness': {'readiness_score': 0.76, 'readiness_table': pd.DataFrame([{'analysis_module': 'Data Quality Review', 'status': 'Available'}])},
            'healthcare': {'healthcare_readiness_score': 0.55},
            'insights': {'summary_lines': ['Sensitive dataset summary.']},
            'action_recommendations': pd.DataFrame([{'recommendation_title': 'Review protected exports', 'priority': 'High', 'rationale': 'Keep exports governed.'}]),
            'sample_info': {'sampling_applied': False},
            'privacy_review': {
                'sensitive_fields': pd.DataFrame([{'column_name': 'mrn', 'sensitive_type': 'Medical Record Number', 'classification': 'PHI'}]),
                'classification_summary': {'public_count': 2, 'pii_count': 0, 'phi_count': 1},
            },
        }
        session_state = {'job_runs': []}
        result = generate_export_report_output(
            session_state,
            job_runtime={'backend_configured': False, 'mode': 'sync'},
            report_label='Executive Report',
            dataset_name='Sensitive Dataset',
            pipeline=pipeline,
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Care Ops', 'auth_mode': 'local', 'role': 'owner', 'membership_validated': True},
            role='Analyst',
            policy_name='Research-safe Extract',
            privacy_review=pipeline['privacy_review'],
            governance_config={'redaction_level': 'High', 'workspace_export_access': 'Owner only', 'watermark_sensitive_exports': True},
        )

        text_payload = result['report_bytes'].decode('utf-8')
        self.assertIn('Watermark:', text_payload)
        self.assertIn('SENSITIVE EXPORT', text_payload)
        json_payload = json.loads(result['json_bytes'].decode('utf-8'))
        self.assertEqual(json_payload['governance']['redaction_level'], 'High')
        self.assertEqual(json_payload['governance']['workspace_export_access'], 'Owner only')
        with ZipFile(BytesIO(result['excel_bytes'])) as workbook:
            workbook_xml = workbook.read('xl/workbook.xml').decode('utf-8')
        self.assertIn('Governance', workbook_xml)


if __name__ == '__main__':
    unittest.main()
