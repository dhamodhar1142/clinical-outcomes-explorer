from __future__ import annotations

import unittest

import pandas as pd

from src.actionable_fallbacks import (
    build_actionable_fallback_context,
    build_cancellation_timeout_message,
    build_large_dataset_actionable_message,
)


class ActionableFallbacksTests(unittest.TestCase):
    def test_trend_analysis_fallback_lists_missing_date_fields(self) -> None:
        pipeline = {
            'semantic': {
                'canonical_map': {'patient_id': 'member_number'},
                'mapping_table': pd.DataFrame([{'original_column': 'member_number', 'semantic_label': 'patient_id'}]),
                'suggestion_table': pd.DataFrame([
                    {
                        'source_column': 'service_dt',
                        'suggested_field': 'service_date',
                        'confidence_score': 0.84,
                        'suggestion_rank': 1,
                        'reason': 'column-name similarity to service_date',
                    }
                ]),
            },
            'readiness': {},
        }

        context = build_actionable_fallback_context('Trend Analysis', pipeline)

        self.assertIn('event_date', context['missing_fields'])
        self.assertFalse(context['mapping_suggestions'].empty)
        self.assertTrue(context['auto_fix_available'])
        self.assertFalse(context['example_structure'].empty)
        self.assertEqual(context['message'].severity, 'Error')
        self.assertIn('Trend Analysis', context['message'].module)

    def test_generic_section_still_returns_docs_and_example_structure(self) -> None:
        pipeline = {
            'semantic': {
                'canonical_map': {},
                'mapping_table': pd.DataFrame(),
                'suggestion_table': pd.DataFrame(),
            },
            'readiness': {},
        }

        context = build_actionable_fallback_context('Unclassified Section', pipeline)

        self.assertFalse(context['docs_table'].empty)
        self.assertFalse(context['example_structure'].empty)
        self.assertIn('entity_id', context['required_fields_table']['required_field'].astype(str).tolist())

    def test_healthcare_missing_entity_id_returns_specific_issue(self) -> None:
        pipeline = {
            'semantic': {
                'canonical_map': {'event_date': 'service_dt'},
                'mapping_table': pd.DataFrame([{'original_column': 'service_dt', 'semantic_label': 'event_date'}]),
                'suggestion_table': pd.DataFrame([
                    {
                        'source_column': 'id',
                        'suggested_field': 'patient_id',
                        'confidence_score': 0.32,
                        'suggestion_rank': 1,
                        'reason': 'weak identifier similarity',
                    }
                ]),
            },
            'readiness': {'readiness_score': 0.67},
        }
        context = build_actionable_fallback_context('Healthcare Intelligence', pipeline)
        self.assertIn('Entity ID field is missing', context['message'].issue)
        self.assertTrue(context['message'].doc_links)

    def test_large_dataset_message_includes_sampling_guidance(self) -> None:
        message = build_large_dataset_actionable_message(
            {'file_size_mb': 65.8, 'ingestion_strategy': 'sampled_streaming_csv'},
            {'total_rows': 917000, 'profile_sample_rows': 120000, 'sampling_mode': 'sampled'},
        )
        self.assertEqual(message.severity, 'Warning')
        self.assertIn('65.8 MB', message.issue)
        self.assertIn('+/-5%', ' '.join(message.remediations))

    def test_cancellation_timeout_message_is_critical(self) -> None:
        message = build_cancellation_timeout_message(file_size_mb=65.8, row_count=917000, column_count=21)
        self.assertEqual(message.severity, 'Critical')
        self.assertIn('force-terminated', message.issue)
        self.assertIn('30', str(message.details_table.iloc[0]['timeout_seconds']))


if __name__ == '__main__':
    unittest.main()
