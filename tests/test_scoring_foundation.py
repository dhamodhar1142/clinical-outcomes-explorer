from __future__ import annotations

import unittest

import pandas as pd

from src.profiler import build_field_profile, build_quality_checks
from src.readiness_engine import ANALYSIS_RULES, evaluate_analysis_readiness
from src.remediation_engine import apply_remediation_augmentations
from src.schema_detection import detect_structure


class ScoringFoundationTests(unittest.TestCase):
    def test_readiness_score_is_full_when_all_modules_are_available(self) -> None:
        canonical_map = {
            'entity_id': 'patient_id',
            'patient_id': 'patient_id',
            'member_id': 'patient_id',
            'encounter_id': 'encounter_id',
            'event_date': 'admission_date',
            'service_date': 'service_date',
            'admission_date': 'admission_date',
            'discharge_date': 'discharge_date',
            'diagnosis_date': 'diagnosis_date',
            'encounter_status_code': 'encounter_status_code',
            'encounter_status': 'encounter_status',
            'encounter_type_code': 'encounter_type_code',
            'encounter_type': 'encounter_type',
            'cost_amount': 'cost',
            'paid_amount': 'paid_amount',
            'allowed_amount': 'allowed_amount',
            'billed_amount': 'billed_amount',
            'provider_id': 'provider_id',
            'provider_name': 'provider_name',
            'facility': 'facility',
            'diagnosis_code': 'diagnosis_code',
            'procedure_code': 'procedure_code',
            'department': 'department',
            'specialty': 'specialty',
            'survived': 'survived',
            'cancer_stage': 'cancer_stage',
            'treatment_type': 'treatment_type',
        }

        readiness = evaluate_analysis_readiness(canonical_map)

        self.assertEqual(readiness['available_count'], len(ANALYSIS_RULES))
        self.assertEqual(readiness['partial_count'], 0)
        self.assertEqual(readiness['readiness_score'], 1.0)
        self.assertTrue((readiness['readiness_table']['status'] == 'Available').all())

    def test_readiness_score_tracks_partial_and_missing_modules(self) -> None:
        canonical_map = {
            'patient_id': 'patient_id',
            'cost_amount': 'cost',
            'provider_id': 'provider_id',
            'diagnosis_code': 'diagnosis_code',
        }
        partial_map = {
            'patient_id': 'patient_id',
            'cost_amount': 'cost',
            'department': 'department',
        }

        readiness = evaluate_analysis_readiness(partial_map)

        self.assertGreaterEqual(readiness['available_count'], 1)
        self.assertGreater(readiness['partial_count'], 0)
        self.assertLess(readiness['readiness_score'], 1.0)
        self.assertIn('Partial', readiness['readiness_table']['status'].tolist())
        self.assertIn('Unavailable', readiness['readiness_table']['status'].tolist())

    def test_quality_score_penalizes_missing_duplicates_and_invalid_numeric_values(self) -> None:
        data = pd.DataFrame(
            [
                {'patient_id': 'A1', 'length_of_stay': 4, 'cost': 1200.0, 'readmission_flag': 1},
                {'patient_id': 'A1', 'length_of_stay': 0, 'cost': -50.0, 'readmission_flag': 0},
                {'patient_id': 'A3', 'length_of_stay': None, 'cost': None, 'readmission_flag': 0},
                {'patient_id': 'A4', 'length_of_stay': None, 'cost': None, 'readmission_flag': 1},
            ]
        )

        structure = detect_structure(data)
        field_profile = build_field_profile(data, structure)
        quality = build_quality_checks(data, structure, field_profile)
        clean_data = pd.DataFrame(
            [
                {'patient_id': 'A1', 'length_of_stay': 4, 'cost': 1200.0, 'readmission_flag': 1},
                {'patient_id': 'A2', 'length_of_stay': 3, 'cost': 950.0, 'readmission_flag': 0},
                {'patient_id': 'A3', 'length_of_stay': 2, 'cost': 875.0, 'readmission_flag': 0},
            ]
        )
        clean_structure = detect_structure(clean_data)
        clean_field_profile = build_field_profile(clean_data, clean_structure)
        clean_quality = build_quality_checks(clean_data, clean_structure, clean_field_profile)

        self.assertLess(quality['quality_score'], clean_quality['quality_score'])
        self.assertFalse(quality['duplicate_identifiers'].empty)
        self.assertFalse(quality['high_missing'].empty)
        self.assertTrue(quality['near_constant'].empty or isinstance(quality['near_constant'], pd.DataFrame))

    def test_quality_score_is_high_for_clean_dataset(self) -> None:
        data = pd.DataFrame(
            [
                {'patient_id': 'A1', 'length_of_stay': 4, 'cost': 1200.0, 'readmission_flag': 1},
                {'patient_id': 'A2', 'length_of_stay': 3, 'cost': 950.0, 'readmission_flag': 0},
                {'patient_id': 'A3', 'length_of_stay': 2, 'cost': 875.0, 'readmission_flag': 0},
            ]
        )

        structure = detect_structure(data)
        field_profile = build_field_profile(data, structure)
        quality = build_quality_checks(data, structure, field_profile)

        self.assertGreaterEqual(quality['quality_score'], 90)
        self.assertTrue(quality['duplicate_identifiers'].empty)
        self.assertTrue(quality['high_missing'].empty)

    def test_helper_audit_fields_are_suppressed_from_high_missing_severity(self) -> None:
        data = pd.DataFrame(
            [
                {'patient_id': 'A1', 'age': 55, 'gender': 'F', 'height_cm': 165, 'weight_kg': 70, 'bmi': 25.7},
                {'patient_id': 'A2', 'age': 58, 'gender': 'M', 'height_cm': 172, 'weight_kg': 82, 'bmi': 27.7},
                {'patient_id': 'A3', 'age': 61, 'gender': 'F', 'height_cm': 160, 'weight_kg': 120, 'bmi': 99.0},
            ]
        )

        remediated, remediation_context = apply_remediation_augmentations(data, helper_mode='off')
        helper_fields = remediation_context.get('helper_fields', pd.DataFrame())
        remediated.attrs['helper_field_names'] = (
            helper_fields['helper_field'].astype(str).tolist()
            if 'helper_field' in helper_fields.columns
            else []
        )

        structure = detect_structure(remediated)
        field_profile = build_field_profile(remediated, structure)
        quality = build_quality_checks(remediated, structure, field_profile)

        suppressed_helper_missing = quality.get('suppressed_helper_missing', pd.DataFrame())
        self.assertFalse(suppressed_helper_missing.empty)
        self.assertIn('bmi_flag_reason', suppressed_helper_missing['column_name'].astype(str).tolist())
        self.assertNotIn('bmi_flag_reason', quality['high_missing']['column_name'].astype(str).tolist())


if __name__ == '__main__':
    unittest.main()
