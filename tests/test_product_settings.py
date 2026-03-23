from __future__ import annotations

import unittest

from src.plan_awareness import build_plan_awareness
from src.product_settings import DEFAULT_PRODUCT_SETTINGS, LARGE_DATASET_PROFILES, build_product_settings_summary


class ProductSettingsTests(unittest.TestCase):
    def test_default_product_settings_are_stable(self) -> None:
        self.assertIn('product_demo_mode_enabled', DEFAULT_PRODUCT_SETTINGS)
        self.assertIn('product_default_report_mode', DEFAULT_PRODUCT_SETTINGS)
        self.assertIn('Standard', LARGE_DATASET_PROFILES)

    def test_build_product_settings_summary_returns_cards_and_table(self) -> None:
        plan = build_plan_awareness(
            'Pro',
            'Demo-safe',
            {'file_size_mb': 12.5},
            workflow_pack_count=2,
            snapshot_count=3,
        )
        summary = build_product_settings_summary(DEFAULT_PRODUCT_SETTINGS, plan)
        self.assertEqual(len(summary['summary_cards']), 4)
        self.assertFalse(summary['settings_table'].empty)
        self.assertIn('Balanced default', summary['admin_notes'][0])

    def test_strict_plan_note_is_added(self) -> None:
        plan = build_plan_awareness(
            'Free',
            'Strict',
            {'file_size_mb': 5.0},
            workflow_pack_count=0,
            snapshot_count=0,
        )
        summary = build_product_settings_summary(DEFAULT_PRODUCT_SETTINGS, plan)
        self.assertTrue(any('Strict plan enforcement' in note for note in summary['admin_notes']))


if __name__ == '__main__':
    unittest.main()
