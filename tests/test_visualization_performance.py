from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.visualization_performance import (
    build_cohort_analysis_payload,
    build_trend_analysis_payload,
    build_visual_cache_key,
    record_tab_render_metric,
    resolve_debounced_filters,
    warm_healthcare_visualization_payloads,
)


class VisualizationPerformanceTests(unittest.TestCase):
    def _build_pipeline(self) -> dict[str, object]:
        data = pd.DataFrame(
            {
                'event_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-02', '2024-03-10']),
                'patient_id': ['P1', 'P2', 'P3', 'P4'],
                'cost_amount': [100.0, 125.0, 130.0, 90.0],
                'readmission_flag': [0, 1, 0, 1],
            }
        )
        data.attrs['dataset_cache_key'] = 'demo-pipeline-key'
        return {
            'data': data,
            'structure': SimpleNamespace(default_date_column='event_date', numeric_columns=['cost_amount', 'readmission_flag']),
            'semantic': {'canonical_map': {'event_date': 'event_date', 'patient_id': 'patient_id'}},
            'healthcare': {
                'survival_outcomes': {'available': True, 'stage_table': pd.DataFrame({'stage': ['I'], 'survival_rate': [0.9]})},
                'readmission': {'available': True, 'trend': pd.DataFrame({'month': pd.to_datetime(['2024-01-01']), 'readmission_rate': [0.2]})},
            },
        }

    def test_build_trend_analysis_payload_summarizes_monthly_volume(self) -> None:
        payload = build_trend_analysis_payload(self._build_pipeline())
        self.assertEqual(payload['date_col'], 'event_date')
        self.assertEqual(payload['monthly']['record_count'].sum(), 4)
        self.assertFalse(payload['correlation'].empty)

    def test_build_cohort_analysis_payload_wraps_domain_builders(self) -> None:
        pipeline = self._build_pipeline()
        cohort_payload = {'available': True, 'summary': {'cohort_size': 2}, 'cohort_frame': pd.DataFrame({'event_date': pd.to_datetime(['2024-01-01'])})}
        with patch('src.visualization_performance.build_cohort_summary', return_value=cohort_payload) as cohort_mock, \
             patch('src.visualization_performance.cohort_monitoring_over_time', return_value={'available': True, 'trend_table': pd.DataFrame({'month': ['2024-01'], 'record_count': [2]})}) as monitoring_mock:
            payload = build_cohort_analysis_payload(pipeline, {'diagnosis': ['DX-1']})

        self.assertTrue(payload['cohort']['available'])
        self.assertTrue(payload['monitoring']['available'])
        cohort_mock.assert_called_once()
        monitoring_mock.assert_called_once()
        self.assertIs(monitoring_mock.call_args.args[2], cohort_payload['cohort_frame'])

    def test_resolve_debounced_filters_applies_first_state_and_debounces_updates(self) -> None:
        session_state: dict[str, object] = {}
        first = resolve_debounced_filters(session_state, 'cohort', {'gender': ['F']}, debounce_seconds=0.5)
        self.assertFalse(first.pending)
        self.assertEqual(first.applied_filters, {'gender': ['F']})

        with patch('src.visualization_performance.time.monotonic', side_effect=[10.0, 10.1, 10.7]):
            resolve_debounced_filters(session_state, 'cohort2', {'gender': ['F']}, debounce_seconds=0.5)
            pending = resolve_debounced_filters(session_state, 'cohort2', {'gender': ['M']}, debounce_seconds=0.5)
            applied = resolve_debounced_filters(session_state, 'cohort2', {'gender': ['M']}, debounce_seconds=0.5)

        self.assertTrue(pending.pending)
        self.assertEqual(pending.applied_filters, {'gender': ['F']})
        self.assertFalse(applied.pending)
        self.assertEqual(applied.applied_filters, {'gender': ['M']})

    def test_warm_healthcare_visualization_payloads_populates_cache_once(self) -> None:
        session_state: dict[str, object] = {}
        pipeline = self._build_pipeline()
        warm_healthcare_visualization_payloads(session_state, pipeline)
        trend_key = build_visual_cache_key(pipeline, 'trend_analysis')
        self.assertIn(trend_key, session_state['visualization_payload_cache'])
        cache_size = len(session_state['visualization_payload_cache'])
        warm_healthcare_visualization_payloads(session_state, pipeline)
        self.assertEqual(len(session_state['visualization_payload_cache']), cache_size)

    def test_record_tab_render_metric_tracks_target_threshold(self) -> None:
        session_state: dict[str, object] = {}
        metric = record_tab_render_metric(session_state, 'Trend Analysis', 0.42)
        self.assertEqual(metric['duration_ms'], 420)
        self.assertTrue(metric['meets_target'])


if __name__ == '__main__':
    unittest.main()
