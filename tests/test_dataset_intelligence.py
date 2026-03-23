from __future__ import annotations

import unittest

import pandas as pd

from src.dataset_intelligence import build_dataset_intelligence_report
from src.readiness_engine import evaluate_analysis_readiness
from src.schema_detection import detect_structure
from src.semantic_mapper import build_data_remediation_assistant, infer_semantic_mapping


def _quality_stub(score: float = 80.0) -> dict[str, object]:
    return {"quality_score": score}


def _governance_stub() -> dict[str, object]:
    return {"governance_notes": ["Governance review is ready for the current dataset."]}


class DatasetIntelligenceTests(unittest.TestCase):
    def test_generic_dataset_classification_and_blockers(self) -> None:
        data = pd.DataFrame(
            {
                "order_id": [f"O{i:03d}" for i in range(20)],
                "category": ["A", "B"] * 10,
                "revenue": [100 + i for i in range(20)],
            }
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(data, structure)
        readiness = evaluate_analysis_readiness(semantic["canonical_map"])
        healthcare = {
            "healthcare_readiness_score": 0.05,
            "likely_dataset_type": "General tabular dataset with limited healthcare context",
            "risk_segmentation": {"available": False, "reason": "Healthcare fields are limited."},
            "default_cohort_summary": {"available": False, "reason": "Patient-level fields are missing."},
            "readmission": {"available": False, "reason": "Encounter fields are missing."},
            "cost": {"available": False, "reason": "No healthcare cost support is available."},
            "diagnosis": {"available": False, "reason": "No diagnosis-like fields are available."},
            "provider": {"available": False, "reason": "No provider-like fields are available."},
            "care_pathway": {"available": False, "reason": "No pathway-supporting fields are available."},
        }
        remediation = build_data_remediation_assistant(structure, semantic, readiness)
        report = build_dataset_intelligence_report(
            data,
            structure,
            semantic,
            readiness,
            healthcare,
            _quality_stub(72.0),
            remediation,
            {},
            {},
            {},
            _governance_stub(),
        )
        self.assertEqual(report["dataset_type_label"], "Generic tabular dataset")
        self.assertIn("Trend Analysis", report["blocked_analytics"])
        self.assertFalse(report["blocker_explanations"].empty)

    def test_healthcare_ready_dataset_and_support_disclosure(self) -> None:
        rows = 80
        data = pd.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(rows)],
                "admission_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "discharge_date": pd.date_range("2024-01-03", periods=rows, freq="D"),
                "age": [40 + (i % 25) for i in range(rows)],
                "gender": ["F" if i % 2 == 0 else "M" for i in range(rows)],
                "diagnosis_code": ["E11" if i % 3 == 0 else "I10" for i in range(rows)],
                "procedure_code": ["PROC1" if i % 2 == 0 else "PROC2" for i in range(rows)],
                "department": ["Cardiology" if i % 2 == 0 else "Medicine" for i in range(rows)],
                "length_of_stay": [2 + (i % 5) for i in range(rows)],
                "cost_amount": [2000 + (i * 10) for i in range(rows)],
                "readmission_flag": [1 if i % 6 == 0 else 0 for i in range(rows)],
                "survived": [1 if i % 7 else 0 for i in range(rows)],
                "treatment_type": ["Therapy A" if i % 2 == 0 else "Therapy B" for i in range(rows)],
                "cancer_stage": ["Stage II" if i % 2 == 0 else "Stage III" for i in range(rows)],
            }
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(data, structure)
        readiness = evaluate_analysis_readiness(semantic["canonical_map"])
        healthcare = {
            "healthcare_readiness_score": 0.86,
            "likely_dataset_type": "Claims or encounter-level healthcare dataset",
            "risk_segmentation": {"available": True},
            "default_cohort_summary": {"available": True, "trend_table": pd.DataFrame([{"month": pd.Timestamp("2024-01-01"), "record_count": 10}])},
            "readmission": {"available": True, "overview": {"overall_readmission_rate": 0.16}, "source": "native"},
            "cost": {"available": True, "summary": {"average_cost": 2500}},
            "diagnosis": {"available": True},
            "provider": {"available": True},
            "care_pathway": {"available": True},
        }
        remediation_context = {
            "helper_fields": pd.DataFrame(
                [
                    {"helper_field": "event_date", "helper_type": "derived"},
                    {"helper_field": "estimated_cost", "helper_type": "synthetic"},
                ]
            )
        }
        remediation = build_data_remediation_assistant(structure, semantic, readiness)
        report = build_dataset_intelligence_report(
            data,
            structure,
            semantic,
            readiness,
            healthcare,
            _quality_stub(89.0),
            remediation,
            remediation_context,
            {"available": True, "combined_readiness_score": 65.0, "cdisc_report": {"available": False}},
            {},
            _governance_stub(),
        )
        self.assertIn(report["dataset_type_label"], {"Claims / encounter-level healthcare dataset", "Clinical / patient-level dataset"})
        self.assertFalse(report["analytics_capability_matrix"].empty)
        self.assertTrue(report["enabled_analytics"])
        self.assertTrue(report["synthetic_support_summary"])
        self.assertTrue(report["support_disclosure_note"])

    def test_partial_support_explanations_include_missing_and_unlock_guidance(self) -> None:
        data = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4"],
                "admission_date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "age": [66, 71, 58, 63],
                "department": ["Oncology", "Medicine", "Oncology", "Medicine"],
            }
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(data, structure)
        readiness = evaluate_analysis_readiness(semantic["canonical_map"])
        healthcare = {
            "healthcare_readiness_score": 0.48,
            "likely_dataset_type": "Healthcare-related dataset",
            "risk_segmentation": {"available": False, "reason": "Need stronger clinical severity support."},
            "default_cohort_summary": {"available": True},
            "readmission": {
                "available": True,
                "source": "synthetic",
                "readiness": {
                    "missing_fields": ["readmission_flag", "diagnosis", "length_of_stay"],
                    "additional_fields_to_unlock_full_analysis": ["readmission_flag", "diagnosis", "length_of_stay", "discharge_date"],
                },
            },
            "cost": {"available": False, "reason": "No native cost field."},
            "diagnosis": {"available": False, "reason": "No diagnosis-like fields are available."},
            "provider": {"available": False, "reason": "No provider-like fields are available."},
            "care_pathway": {"available": False, "reason": "Need treatment and outcome timing."},
        }
        remediation = build_data_remediation_assistant(structure, semantic, readiness)
        remediation_context = {"synthetic_readmission": {"available": True}}
        report = build_dataset_intelligence_report(
            data,
            structure,
            semantic,
            readiness,
            healthcare,
            _quality_stub(75.0),
            remediation,
            remediation_context,
            {},
            {},
            _governance_stub(),
        )
        partial = report["partial_support_explanations"]
        self.assertFalse(partial.empty)
        self.assertIn("what_still_works", partial.columns)
        self.assertIn("what_is_missing", partial.columns)
        self.assertIn("fields_to_unlock_full_support", partial.columns)

    def test_recommendations_prioritize_high_impact_fields(self) -> None:
        data = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "age": [70, 60, 55],
            }
        )
        structure = detect_structure(data)
        semantic = infer_semantic_mapping(data, structure)
        readiness = evaluate_analysis_readiness(semantic["canonical_map"])
        healthcare = {
            "healthcare_readiness_score": 0.22,
            "likely_dataset_type": "Healthcare-related dataset",
            "risk_segmentation": {"available": False, "reason": "Need outcome and risk fields."},
            "default_cohort_summary": {"available": False, "reason": "Need diagnosis or treatment context."},
            "readmission": {"available": False, "reason": "Need encounter timestamps."},
            "cost": {"available": False, "reason": "Need cost or payment support."},
            "diagnosis": {"available": False, "reason": "Need diagnosis or procedure fields."},
            "provider": {"available": False, "reason": "Need provider or facility fields."},
            "care_pathway": {"available": False, "reason": "Need treatment and outcome timing."},
        }
        remediation = build_data_remediation_assistant(structure, semantic, readiness)
        report = build_dataset_intelligence_report(
            data,
            structure,
            semantic,
            readiness,
            healthcare,
            _quality_stub(78.0),
            remediation,
            {},
            {},
            {},
            _governance_stub(),
        )
        upgrades = report["highest_impact_upgrades"]
        self.assertFalse(upgrades.empty)
        self.assertIn("recommended_field", upgrades.columns)
        self.assertIn("analytics_unlocked", upgrades.columns)

    def test_synthetic_helpers_do_not_overclassify_generic_dataset(self) -> None:
        data = pd.DataFrame(
            {
                "order_id": ["O1", "O2", "O3"],
                "year": [2020, 2021, 2022],
                "revenue": [100.0, 110.0, 120.0],
            }
        )
        structure = detect_structure(data)
        semantic = {
            "canonical_map": {
                "cost_amount": "cost_amount",
                "event_date": "order_date",
                "provider_id": "order_id",
            },
            "semantic_confidence_score": 0.58,
        }
        readiness = {"readiness_score": 0.74, "readiness_table": pd.DataFrame()}
        healthcare = {"healthcare_readiness_score": 0.40}
        remediation_context = {
            "helper_fields": pd.DataFrame(
                [
                    {"helper_field": "cost_amount", "helper_type": "synthetic"},
                    {"helper_field": "event_date", "helper_type": "synthetic"},
                ]
            )
        }
        report = build_dataset_intelligence_report(
            data,
            structure,
            semantic,
            readiness,
            healthcare,
            _quality_stub(82.0),
            pd.DataFrame(),
            remediation_context,
            {"cdisc_report": {"available": False, "readiness_score": 0.0}},
            {},
            _governance_stub(),
        )
        self.assertEqual(report["dataset_type_label"], "Generic tabular dataset")


if __name__ == "__main__":
    unittest.main()
