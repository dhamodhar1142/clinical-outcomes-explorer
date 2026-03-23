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


if __name__ == '__main__':
    unittest.main()
