from __future__ import annotations

import unittest

from src.plan_awareness import build_plan_awareness, is_strict_plan_enforcement, plan_feature_enabled


class PlanAwarenessTests(unittest.TestCase):
    def test_free_plan_flags_limited_features(self) -> None:
        profile = build_plan_awareness(
            'Free',
            'Strict',
            {'file_size_mb': 22.0},
            workflow_pack_count=3,
            snapshot_count=2,
        )
        self.assertTrue(profile['strict_enforcement'])
        self.assertTrue(profile['file_size_exceeded'])
        self.assertTrue(profile['workflow_pack_limit_reached'])
        premium = profile['premium_features']
        self.assertIn('Limited in strict mode', premium['status'].tolist())
        self.assertFalse(profile['entitlement_summary'].empty)
        self.assertIn('Soft limit reached', profile['entitlement_summary']['current_state'].tolist())

    def test_pro_plan_keeps_advanced_exports_available(self) -> None:
        profile = build_plan_awareness(
            'Pro',
            'Demo-safe',
            {'file_size_mb': 8.0},
            workflow_pack_count=2,
            snapshot_count=1,
        )
        self.assertFalse(profile['file_size_exceeded'])
        self.assertFalse(profile['workflow_pack_limit_reached'])
        self.assertTrue(plan_feature_enabled('Pro', 'advanced_exports'))

    def test_team_plan_has_highest_limits(self) -> None:
        profile = build_plan_awareness(
            'Team',
            'Off',
            {'file_size_mb': 140.0},
            workflow_pack_count=12,
            snapshot_count=4,
        )
        self.assertEqual(profile['workflow_pack_limit'], 50)
        self.assertEqual(profile['file_limit_mb'], 250.0)
        self.assertFalse(is_strict_plan_enforcement(profile['enforcement_mode']))

    def test_enterprise_placeholder_plan_is_available(self) -> None:
        profile = build_plan_awareness(
            'Enterprise',
            'Demo-safe',
            {'file_size_mb': 300.0},
            workflow_pack_count=20,
            snapshot_count=5,
        )
        self.assertEqual(profile['active_plan'], 'Enterprise')
        self.assertIn('Enterprise', profile['plan_comparison']['plan'].tolist())
        self.assertIn('Demo-safe mode', ' '.join(profile['guidance_notes']))
        self.assertIn('plan_story', profile)
        self.assertIn('entitlement_summary', profile)


if __name__ == '__main__':
    unittest.main()
