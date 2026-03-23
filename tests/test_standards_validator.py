from __future__ import annotations

import unittest

import pandas as pd

from src.standards_validator import validate_healthcare_standards


class StandardsValidatorTests(unittest.TestCase):
    def test_detects_cdisc_sdtm_style_dataset(self) -> None:
        data = pd.DataFrame(
            {
                'STUDYID': ['STUDY-1', 'STUDY-1'],
                'USUBJID': ['SUBJ-001', 'SUBJ-002'],
                'DOMAIN': ['AE', 'AE'],
                'VISIT': ['Screening', 'Baseline'],
                'VISITNUM': [1, 2],
                'AESTDTC': ['2024-01-01', '2024-01-08'],
                'SEX': ['F', 'M'],
                'AGE': [62, 55],
            }
        )
        result = validate_healthcare_standards(data, None, {'canonical_map': {'patient_id': 'USUBJID', 'service_date': 'AESTDTC', 'age': 'AGE', 'gender': 'SEX'}})
        self.assertTrue(result.get('available'))
        self.assertEqual(result.get('detected_standard'), 'CDISC SDTM-style dataset')
        self.assertFalse(result.get('standards_profiles', pd.DataFrame()).empty)
        self.assertIn('required_field_review', result)

    def test_detects_cdisc_adam_style_dataset(self) -> None:
        data = pd.DataFrame(
            {
                'STUDYID': ['STUDY-1', 'STUDY-1'],
                'USUBJID': ['SUBJ-001', 'SUBJ-002'],
                'PARAM': ['Overall Survival', 'Overall Survival'],
                'PARAMCD': ['OS', 'OS'],
                'AVAL': [180, 240],
                'ADT': ['2024-06-01', '2024-06-08'],
                'TRTA': ['Chemo', 'Immunotherapy'],
            }
        )
        result = validate_healthcare_standards(data, None, {'canonical_map': {'patient_id': 'USUBJID', 'survived': 'AVAL', 'treatment_type': 'TRTA'}})
        self.assertTrue(result.get('available'))
        self.assertEqual(result.get('detected_standard'), 'CDISC ADaM-style dataset')
        profiles = result.get('standards_profiles', pd.DataFrame())
        self.assertIn('CDISC ADaM-style dataset', profiles['standard_type'].tolist())

    def test_detects_fhir_like_structure(self) -> None:
        data = pd.DataFrame(
            {
                'resourceType': ['Encounter', 'Condition'],
                'id': ['enc-1', 'cond-1'],
                'subject': ['pat-1', 'pat-1'],
                'code': ['E11', 'I10'],
                'status': ['finished', 'active'],
                'encounter': ['enc-1', 'enc-1'],
                'effectiveDateTime': ['2024-01-01', '2024-01-02'],
            }
        )
        result = validate_healthcare_standards(data, None, {'canonical_map': {'patient_id': 'subject', 'diagnosis_code': 'code', 'service_date': 'effectiveDateTime'}})
        self.assertTrue(result.get('available'))
        self.assertEqual(result.get('detected_standard'), 'FHIR-like structure')
        self.assertFalse(result.get('mapping_suggestions', pd.DataFrame()).empty)

    def test_detects_hl7_like_pattern(self) -> None:
        data = pd.DataFrame(
            {
                'MSH_message_type': ['ADT^A01', 'ADT^A03'],
                'PID_patient_id': ['P1', 'P2'],
                'PV1_visit_number': ['V100', 'V200'],
                'OBX_result_value': ['7.2', '8.5'],
                'admission_date': ['2024-01-01', '2024-01-02'],
            }
        )
        result = validate_healthcare_standards(data, None, {'canonical_map': {'patient_id': 'PID_patient_id', 'admission_date': 'admission_date'}})
        self.assertTrue(result.get('available'))
        self.assertEqual(result.get('detected_standard'), 'HL7-like pattern')
        required_field_review = result.get('required_field_review', pd.DataFrame())
        self.assertIn('HL7-like pattern', required_field_review['standard_type'].tolist())


if __name__ == '__main__':
    unittest.main()
