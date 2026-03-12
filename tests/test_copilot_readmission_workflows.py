from __future__ import annotations

import unittest

import pandas as pd

from src.ai_copilot import get_readmission_summary, plan_workflow_action
from src.export_utils import build_readmission_summary_text
from src.healthcare_analysis import readmission_risk_analytics, run_healthcare_analysis
from src.readiness_engine import evaluate_analysis_readiness


def _sample_readmission_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    data = pd.DataFrame(
        {
            'patient_id': [f'P{i:03d}' for i in range(1, 41)],
            'age': [68, 72, 51, 47, 63, 59, 74, 77, 45, 49] * 4,
            'gender': ['F', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M'] * 4,
            'smoking_status': ['Smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Smoker', 'Non-smoker', 'Non-smoker'] * 4,
            'cancer_stage': ['III', 'IV', 'II', 'II', 'III', 'II', 'IV', 'III', 'I', 'II'] * 4,
            'treatment_type': ['Chemo', 'Chemo', 'Radiation', 'Radiation', 'Chemo', 'Radiation', 'Immunotherapy', 'Chemo', 'Radiation', 'Radiation'] * 4,
            'survived': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes'] * 4,
            'diagnosis_date': pd.date_range('2024-01-01', periods=40, freq='7D'),
            'department': ['Oncology', 'Oncology', 'Medicine', 'Medicine', 'Oncology', 'Medicine', 'Oncology', 'Oncology', 'Medicine', 'Medicine'] * 4,
            'diagnosis_code': ['C34', 'C50', 'I10', 'E11', 'C34', 'I10', 'C50', 'C18', 'E11', 'I10'] * 4,
            'length_of_stay': [8, 9, 4, 3, 7, 5, 10, 8, 3, 4] * 4,
            'readmission': [1, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 4,
            'bmi': [34, 36, 24, 26, 33, 28, 37, 35, 25, 27] * 4,
            'comorbidities': [2, 3, 0, 1, 2, 1, 4, 3, 0, 1] * 4,
        }
    )
    canonical = {
        'patient_id': 'patient_id',
        'age': 'age',
        'gender': 'gender',
        'smoking_status': 'smoking_status',
        'cancer_stage': 'cancer_stage',
        'treatment_type': 'treatment_type',
        'survived': 'survived',
        'diagnosis_date': 'diagnosis_date',
        'department': 'department',
        'diagnosis_code': 'diagnosis_code',
        'length_of_stay': 'length_of_stay',
        'readmission': 'readmission',
        'bmi': 'bmi',
        'comorbidities': 'comorbidities',
    }
    return data, canonical


class CopilotReadmissionWorkflowTests(unittest.TestCase):
    def test_readmission_summary_tool_returns_payload(self) -> None:
        data, canonical = _sample_readmission_frame()
        payload = get_readmission_summary(data, {'matched_schema': canonical})
        self.assertEqual(payload['tool'], 'get_readmission_summary')
        self.assertIn('overall readmission rate', payload['answer'].lower())
        self.assertIsInstance(payload['table'], pd.DataFrame)
        self.assertFalse(payload['table'].empty)

    def test_readmission_workflow_action_prepares_export(self) -> None:
        data, canonical = _sample_readmission_frame()
        readiness = evaluate_analysis_readiness(canonical)
        healthcare = run_healthcare_analysis(data, canonical)
        action = plan_workflow_action(
            'Generate a readmission summary report',
            data,
            canonical,
            readiness,
            healthcare,
            remediation=pd.DataFrame(),
        )
        self.assertTrue(action['available'])
        self.assertEqual(action['action_type'], 'readmission_report')
        self.assertEqual(action['widget_updates'].get('report_mode'), 'Operational Report')
        self.assertIn('planned_action', action)

    def test_readmission_summary_export_builds_text(self) -> None:
        data, canonical = _sample_readmission_frame()
        readmission = readmission_risk_analytics(data, canonical)
        export_bytes = build_readmission_summary_text(
            'sample_readmission_dataset.csv',
            readmission,
            pd.DataFrame(
                [
                    {
                        'priority': 'High',
                        'recommendation_title': 'Review oncology discharges',
                        'rationale': 'Oncology cohorts show the highest readmission concentration.',
                    }
                ]
            ),
        )
        export_text = export_bytes.decode('utf-8')
        self.assertIn('Readmission Risk Summary', export_text)
        self.assertIn('Overall readmission rate', export_text)
        self.assertIn('Recommended Next Steps', export_text)

    def test_readmission_summary_export_handles_unavailable_payload(self) -> None:
        export_bytes = build_readmission_summary_text(
            'incomplete_dataset.csv',
            {
                'available': False,
                'readiness': {
                    'missing_fields': ['patient_id', 'admission_date'],
                    'available_analysis': ['Department-level descriptive review'],
                },
            },
            pd.DataFrame(),
        )
        export_text = export_bytes.decode('utf-8')
        self.assertIn('not fully available', export_text.lower())
        self.assertIn('patient id', export_text.lower())


if __name__ == '__main__':
    unittest.main()
