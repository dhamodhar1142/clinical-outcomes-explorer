from __future__ import annotations

import unittest

from src.product_story import (
    PRODUCT_ONE_LINER,
    build_core_outcomes_table,
    build_demo_script_text,
    build_demo_walkthrough_table,
    build_professional_export_template,
    build_role_pitch_table,
)


class ProductStoryTests(unittest.TestCase):
    def test_core_outcomes_table_has_expected_rows(self):
        table = build_core_outcomes_table()
        self.assertEqual(len(table), 3)
        self.assertIn('outcome', table.columns)
        self.assertIn('description', table.columns)

    def test_demo_walkthrough_has_four_steps(self):
        table = build_demo_walkthrough_table()
        self.assertEqual(len(table), 4)
        self.assertTrue(table['step'].astype(str).str.startswith(('1.', '2.', '3.', '4.')).all())

    def test_role_pitch_table_contains_three_audiences(self):
        table = build_role_pitch_table()
        self.assertEqual(len(table), 3)
        self.assertIn('pitch', table.columns)

    def test_professional_export_template_changes_for_readiness_review(self):
        template = build_professional_export_template('Data Readiness Review')
        sections = template['sections']
        self.assertIn('4. Quality blockers and remediation priorities', sections)
        self.assertIn('5. Recommended data upgrades', sections)

    def test_product_statement_and_demo_script_are_nonempty(self):
        self.assertIn('Clinverity', PRODUCT_ONE_LINER)
        self.assertIn('1. Open Clinverity', build_demo_script_text())


if __name__ == '__main__':
    unittest.main()
