from __future__ import annotations

import unittest

from src.data_loader import load_demo_dataset


class DemoDatasetQualityTests(unittest.TestCase):
    def test_healthcare_operations_demo_is_rich_enough_for_showcase(self) -> None:
        data, _ = load_demo_dataset('Healthcare Operations Demo')
        self.assertGreaterEqual(len(data), 1000)
        required_columns = {
            'patient_id',
            'age',
            'gender',
            'diagnosis',
            'cancer_stage',
            'treatment_type',
            'length_of_stay',
            'readmission_flag',
            'cost',
            'hospital_department',
            'admission_date',
            'discharge_date',
        }
        self.assertTrue(required_columns.issubset(set(data.columns)))

    def test_healthcare_claims_demo_is_rich_enough_for_claims_showcase(self) -> None:
        data, _ = load_demo_dataset('Healthcare Claims Demo')
        self.assertGreaterEqual(len(data), 20)
        required_columns = {
            'member_id',
            'claim_id',
            'provider_id',
            'service_date',
            'admit_date',
            'discharge_date',
            'diagnosis_code',
            'procedure_code',
            'payer',
            'plan',
            'line_of_business',
            'billed_amount',
            'allowed_amount',
            'paid_amount',
            'encounter_type',
        }
        self.assertTrue(required_columns.issubset(set(data.columns)))
        self.assertGreater(int(data['claim_id'].duplicated().sum()), 0)
        self.assertGreater(int((data['paid_amount'] < 0).sum()), 0)
        self.assertGreater(int(data['service_date'].isna().sum()), 0)


if __name__ == '__main__':
    unittest.main()
