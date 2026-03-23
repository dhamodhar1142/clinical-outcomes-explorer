import unittest

import pandas as pd

from src.solution_layers import (
    build_demo_guidance_system,
    build_market_specific_solution_views,
    build_solution_layer_guidance,
    build_solution_packages,
    build_use_case_detection,
)


class SolutionLayerGuidanceTests(unittest.TestCase):
    def test_solution_layers_for_healthcare_ready_dataset(self):
        capability = pd.DataFrame(
            [
                {'analytics_module': 'Data Profiling', 'status': 'enabled'},
                {'analytics_module': 'Data Quality Review', 'status': 'enabled'},
                {'analytics_module': 'Risk Segmentation', 'status': 'enabled'},
                {'analytics_module': 'Cohort Analysis', 'status': 'enabled'},
                {'analytics_module': 'Readmission Analytics', 'status': 'partial'},
                {'analytics_module': 'Predictive Modeling', 'status': 'enabled'},
                {'analytics_module': 'Care Pathway Intelligence', 'status': 'partial'},
                {'analytics_module': 'Standards / Governance Review', 'status': 'enabled'},
                {'analytics_module': 'Export / Executive Reporting', 'status': 'enabled'},
            ]
        )
        result = build_solution_layer_guidance(
            {'analytics_capability_matrix': capability},
            {'healthcare_readiness_score': 0.72},
            {'available_count': 8},
        )
        self.assertEqual(result['summary']['solution_layers'], 5)
        self.assertGreaterEqual(result['summary']['ready_now'], 1)
        self.assertIn('recommended_starting_point', result['summary'])
        self.assertFalse(result['solution_cards'].empty)

    def test_solution_layers_fallback_when_capability_missing(self):
        result = build_solution_layer_guidance({}, {}, {})
        self.assertEqual(result['summary']['needs_stronger_source_support'], 5)
        self.assertEqual(result['summary']['recommended_starting_point'], 'Healthcare Data Readiness')
        self.assertIn('stronger', result['narrative'].lower())


if __name__ == '__main__':
    unittest.main()


class UseCaseDetectionTests(unittest.TestCase):
    def test_detects_patient_level_clinical_use_case(self):
        capability = pd.DataFrame(
            [
                {'analytics_module': 'Risk Segmentation', 'status': 'enabled', 'support': 'native', 'rationale': 'ok'},
                {'analytics_module': 'Cohort Analysis', 'status': 'enabled', 'support': 'native', 'rationale': 'ok'},
                {'analytics_module': 'Predictive Modeling', 'status': 'enabled', 'support': 'native', 'rationale': 'ok'},
                {'analytics_module': 'Care Pathway Intelligence', 'status': 'partial', 'support': 'native', 'rationale': 'missing dates'},
            ]
        )
        result = build_use_case_detection(
            {
                'dataset_type_label': 'Clinical / patient-level dataset',
                'dataset_type_confidence': 0.81,
                'dataset_type_rationale': 'Patient identifiers and clinical fields are present.',
                'analytics_capability_matrix': capability,
            },
            {'available_count': 6},
            {'healthcare_readiness_score': 0.74},
        )
        self.assertEqual(result['detected_use_case'], 'Patient-level clinical dataset')
        self.assertEqual(result['recommended_workflow'], 'Clinical Outcome Analytics')
        self.assertIn('Risk Segmentation', result['relevant_modules'])

    def test_detects_generic_healthcare_fallback(self):
        result = build_use_case_detection(
            {
                'dataset_type_label': 'Healthcare-related dataset',
                'dataset_type_confidence': 0.44,
                'dataset_type_rationale': 'Only limited healthcare structure is present.',
                'analytics_capability_matrix': pd.DataFrame(),
            },
            {'available_count': 2},
            {'healthcare_readiness_score': 0.22},
        )
        self.assertEqual(result['detected_use_case'], 'Generic healthcare dataset')
        self.assertEqual(result['recommended_workflow'], 'Healthcare Data Readiness')
        self.assertIn('healthcare data readiness', result['narrative'].lower())


class SolutionPackageTests(unittest.TestCase):
    def test_solution_packages_recommend_matching_workflow(self):
        capability = pd.DataFrame(
            [
                {'analytics_module': 'Risk Segmentation', 'status': 'enabled'},
                {'analytics_module': 'Cohort Analysis', 'status': 'enabled'},
                {'analytics_module': 'Predictive Modeling', 'status': 'enabled'},
                {'analytics_module': 'Care Pathway Intelligence', 'status': 'partial'},
            ]
        )
        result = build_solution_packages(
            {'analytics_capability_matrix': capability},
            {'recommended_workflow': 'Clinical Outcome Analytics'},
        )
        self.assertEqual(result['recommended_package'], 'Clinical Outcomes Review')
        self.assertFalse(result['packages_table'].empty)
        details = result['package_details']['Clinical Outcomes Review']
        self.assertIn('Generate clinical report', details['suggested_prompts'])

    def test_solution_packages_fallback_cleanly(self):
        result = build_solution_packages({}, {})
        self.assertFalse(result['packages_table'].empty)
        self.assertEqual(result['recommended_package'], 'Healthcare Data Readiness')


class DemoGuidanceSystemTests(unittest.TestCase):
    def test_demo_guidance_recommends_clinical_workflow(self):
        capability = pd.DataFrame(
            [
                {'analytics_module': 'Risk Segmentation', 'status': 'enabled'},
                {'analytics_module': 'Cohort Analysis', 'status': 'enabled'},
                {'analytics_module': 'Predictive Modeling', 'status': 'enabled'},
                {'analytics_module': 'Care Pathway Intelligence', 'status': 'partial', 'rationale': 'Longitudinal detail is limited.'},
            ]
        )
        intelligence = {
            'dataset_type_label': 'Clinical / patient-level dataset',
            'dataset_type_rationale': 'Patient identifiers and clinical support fields are present.',
            'analytics_capability_matrix': capability,
        }
        use_case = build_use_case_detection(
            intelligence,
            {'available_count': 5},
            {'healthcare_readiness_score': 0.78},
        )
        packages = build_solution_packages(intelligence, use_case)
        guidance = build_demo_guidance_system(intelligence, use_case, packages)
        self.assertEqual(guidance['detected_dataset_type'], 'Clinical / patient-level dataset')
        self.assertEqual(guidance['recommended_workflow'], 'Clinical Outcome Analytics')
        self.assertTrue(guidance['recommended_steps'])
        self.assertIn('Recommended workflow', guidance['narrative'])

    def test_demo_guidance_falls_back_cleanly(self):
        guidance = build_demo_guidance_system({}, {}, {})
        self.assertEqual(guidance['recommended_workflow'], 'Healthcare Data Readiness')
        self.assertTrue(guidance['recommended_steps'])


class MarketSpecificSolutionViewTests(unittest.TestCase):
    def test_market_specific_solution_views_highlight_current_best_fit(self):
        capability = pd.DataFrame(
            [
                {'analytics_module': 'Readmission Analytics', 'status': 'enabled'},
                {'analytics_module': 'Provider / Facility Volume', 'status': 'enabled'},
                {'analytics_module': 'Trend Analysis', 'status': 'enabled'},
                {'analytics_module': 'Cost Driver Analysis', 'status': 'partial'},
                {'analytics_module': 'Export / Executive Reporting', 'status': 'enabled'},
            ]
        )
        intelligence = {
            'dataset_type_label': 'Mixed healthcare operational dataset',
            'dataset_type_rationale': 'Operational and encounter-style signals are present.',
            'analytics_capability_matrix': capability,
        }
        use_case = build_use_case_detection(intelligence, {'available_count': 5}, {'healthcare_readiness_score': 0.7})
        packages = build_solution_packages(intelligence, use_case)
        layers = build_solution_layer_guidance(intelligence, {'healthcare_readiness_score': 0.7}, {'available_count': 5})
        views = build_market_specific_solution_views(intelligence, use_case, packages, layers)
        table = views['market_solution_views']
        self.assertFalse(table.empty)
        self.assertIn('Hospital Operations', table['solution_view'].tolist())
        self.assertEqual(views['recommended_solution_view'], 'Hospital Operations')
        self.assertIn('hospital operations analytics', views['narrative'].lower())

    def test_market_specific_solution_views_fallback_cleanly(self):
        views = build_market_specific_solution_views({}, {}, {}, {})
        table = views['market_solution_views']
        self.assertFalse(table.empty)
        self.assertIn('Healthcare Data Readiness', table['solution_view'].tolist())
