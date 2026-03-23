from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import pandas as pd

if importlib.util.find_spec('plotly') is None:
    raise unittest.SkipTest('plotly is required for readmission copilot workflow tests')

import src.ai_copilot as ai_copilot
from src.ai_copilot import build_copilot_panel_config, get_readmission_summary, run_copilot_question, plan_workflow_action
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
    def setUp(self) -> None:
        for key in ('copilot_messages', 'copilot_response_cache'):
            try:
                ai_copilot.st.session_state.pop(key, None)
            except Exception:
                pass

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
        self.assertIn('Intervention Recommendations', export_text)
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

    def test_readmission_workflow_action_handles_blocker_prompt(self) -> None:
        data, canonical = _sample_readmission_frame()
        readiness = evaluate_analysis_readiness(canonical)
        healthcare = run_healthcare_analysis(data, canonical)
        action = plan_workflow_action(
            'Explain readmission blockers',
            data,
            canonical,
            readiness,
            healthcare,
            remediation=pd.DataFrame(),
        )
        self.assertTrue(action['available'])
        self.assertEqual(action['action_type'], 'readmission_readiness')
        self.assertEqual(action['recommended_section'], 'Healthcare Analytics - Healthcare Intelligence')

    def test_copilot_panel_config_exposes_visible_prompts(self) -> None:
        data, canonical = _sample_readmission_frame()
        panel = build_copilot_panel_config(data, {'matched_schema': canonical})
        self.assertIn('Summarize the dataset', panel['suggested_prompts'])
        self.assertIn('Which patients are high risk?', panel['suggested_prompts'])
        self.assertIn('Show key healthcare insights', panel['suggested_prompts'])
        self.assertIn('Show readmission by department', panel['suggested_prompts'])
        self.assertIn('What drives readmission?', panel['suggested_prompts'])
        self.assertIn('Summarize readmission risk', panel['suggested_prompts'])
        self.assertIn('Generate readmission report', panel['suggested_prompts'])
        self.assertIn('Show highest-risk readmission patients', panel['suggested_prompts'])
        self.assertIn('What interventions could reduce readmissions?', panel['suggested_prompts'])
        self.assertIn('Explain readmission blockers', panel['suggested_prompts'])

    def test_rule_based_copilot_handles_visible_demo_prompts(self) -> None:
        data, canonical = _sample_readmission_frame()
        high_risk = run_copilot_question('Which patients are high risk?', data, {'matched_schema': canonical})
        insights = run_copilot_question('Show key healthcare insights', data, {'matched_schema': canonical})
        self.assertIn('readmission-risk', high_risk['answer'].lower())
        self.assertIsInstance(high_risk['table'], pd.DataFrame)
        self.assertFalse(high_risk['table'].empty)
        self.assertIn('healthcare insights', insights['answer'].lower())
        self.assertIsInstance(insights['table'], pd.DataFrame)
        self.assertFalse(insights['table'].empty)

    def test_rule_based_copilot_handles_readmission_report_prompt(self) -> None:
        data, canonical = _sample_readmission_frame()
        report = run_copilot_question('Generate readmission report', data, {'matched_schema': canonical})
        self.assertEqual(report['tool'], 'get_readmission_report')
        self.assertIn('readmission workflow is ready for reporting', report['answer'].lower())
        self.assertIsInstance(report['table'], pd.DataFrame)
        self.assertFalse(report['table'].empty)

    def test_rule_based_copilot_explains_readmission_blockers_when_dataset_is_weak(self) -> None:
        weak = pd.DataFrame({'department': ['Oncology', 'Medicine'], 'age': [72, 55]})
        blockers = run_copilot_question('Explain readmission blockers', weak, {'matched_schema': {'department': 'department', 'age': 'age'}})
        self.assertEqual(blockers['tool'], 'get_readmission_summary')
        self.assertIn('what still works now', blockers['answer'].lower())
        self.assertIn('add or map these fields next', blockers['answer'].lower())

    def test_copilot_response_includes_grounding_metadata(self) -> None:
        data, canonical = _sample_readmission_frame()
        response = run_copilot_question(
            'Summarize the dataset',
            data,
            {'matched_schema': canonical, 'dataset_name': 'Healthcare Demo'},
        )
        self.assertTrue(response['answer'])
        self.assertTrue(response['explanation'])
        self.assertIsInstance(response['citations'], list)
        self.assertGreaterEqual(len(response['citations']), 2)
        self.assertFalse(response['cached_response'])

    def test_copilot_response_cache_reuses_dataset_scoped_answer(self) -> None:
        data, canonical = _sample_readmission_frame()
        first = run_copilot_question(
            'Summarize the dataset',
            data,
            {'matched_schema': canonical, 'dataset_name': 'Healthcare Demo'},
        )
        second = run_copilot_question(
            'Summarize the dataset',
            data,
            {'matched_schema': canonical, 'dataset_name': 'Healthcare Demo'},
        )
        self.assertFalse(first['cached_response'])
        self.assertTrue(second['cached_response'])
        self.assertEqual(first['answer'], second['answer'])

    def test_copilot_llm_polish_supports_openai_label(self) -> None:
        data, canonical = _sample_readmission_frame()
        with patch('src.ai_copilot._get_llm_backend', return_value={'provider': 'openai', 'label': 'OpenAI', 'model': 'gpt-4.1-mini'}):
            with patch('src.ai_copilot._complete_with_backend', return_value='Polished OpenAI summary.'):
                response = run_copilot_question(
                    'Summarize the dataset',
                    data,
                    {'matched_schema': canonical, 'dataset_name': 'Healthcare Demo'},
                )
        self.assertEqual(response['answer'], 'Polished OpenAI summary.')
        self.assertEqual(response['llm_provider'], 'OpenAI')

    def test_copilot_llm_polish_supports_claude_label(self) -> None:
        data, canonical = _sample_readmission_frame()
        with patch('src.ai_copilot._get_llm_backend', return_value={'provider': 'anthropic', 'label': 'Claude', 'model': 'claude-3-5-sonnet'}):
            with patch('src.ai_copilot._complete_with_backend', return_value='Polished Claude summary.'):
                response = run_copilot_question(
                    'Summarize the dataset',
                    data,
                    {'matched_schema': canonical, 'dataset_name': 'Healthcare Demo'},
                )
        self.assertEqual(response['answer'], 'Polished Claude summary.')
        self.assertEqual(response['llm_provider'], 'Claude')

    def test_copilot_llm_polish_supports_local_llm_label(self) -> None:
        data, canonical = _sample_readmission_frame()
        with patch('src.ai_copilot._get_llm_backend', return_value={'provider': 'ollama', 'label': 'Local LLM', 'model': 'llama3.1'}):
            with patch('src.ai_copilot._complete_with_backend', return_value='Polished local summary.'):
                response = run_copilot_question(
                    'Summarize the dataset',
                    data,
                    {'matched_schema': canonical, 'dataset_name': 'Healthcare Demo'},
                )
        self.assertEqual(response['answer'], 'Polished local summary.')
        self.assertEqual(response['llm_provider'], 'Local LLM')


if __name__ == '__main__':
    unittest.main()
