from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_PRODUCT_SETTINGS: dict[str, Any] = {
    'product_demo_mode_enabled': True,
    'product_beta_interest_enabled': True,
    'product_large_dataset_profile': 'Standard',
    'product_sampling_explanation_mode': 'Concise',
    'product_default_report_mode': 'Executive Summary',
    'product_default_export_policy': 'Internal Review',
    'product_copilot_response_style': 'Concise',
    'product_copilot_show_workflow_preview': True,
}


LARGE_DATASET_PROFILES: dict[str, dict[str, Any]] = {
    'Conservative': {
        'recommended_upload_mb': 25,
        'recommended_rows': 50_000,
        'warn_upload_mb': 20.0,
        'block_upload_mb': 75.0,
        'warn_memory_mb': 225.0,
        'block_memory_mb': 700.0,
        'long_task_rows': 40_000,
        'long_task_memory_mb': 125.0,
        'profile_sample_rows': 8_000,
        'profile_large_sample_rows': 12_000,
        'quality_sample_rows': 12_000,
        'quality_large_sample_rows': 18_000,
        'very_large_dataset_rows': 80_000,
        'export_guard_rows': 150_000,
        'note': 'Best for smoother demos on smaller machines and shared browser sessions.',
    },
    'Standard': {
        'recommended_upload_mb': 75,
        'recommended_rows': 100_000,
        'warn_upload_mb': 25.0,
        'block_upload_mb': 100.0,
        'warn_memory_mb': 300.0,
        'block_memory_mb': 900.0,
        'long_task_rows': 50_000,
        'long_task_memory_mb': 150.0,
        'profile_sample_rows': 10_000,
        'profile_large_sample_rows': 20_000,
        'quality_sample_rows': 15_000,
        'quality_large_sample_rows': 25_000,
        'very_large_dataset_rows': 100_000,
        'export_guard_rows': 250_000,
        'note': 'Balanced default for most recruiter demos and analyst walkthroughs.',
    },
    'High Capacity': {
        'recommended_upload_mb': 150,
        'recommended_rows': 250_000,
        'warn_upload_mb': 40.0,
        'block_upload_mb': 150.0,
        'warn_memory_mb': 450.0,
        'block_memory_mb': 1_250.0,
        'long_task_rows': 90_000,
        'long_task_memory_mb': 225.0,
        'profile_sample_rows': 14_000,
        'profile_large_sample_rows': 28_000,
        'quality_sample_rows': 18_000,
        'quality_large_sample_rows': 32_000,
        'very_large_dataset_rows': 180_000,
        'export_guard_rows': 400_000,
        'note': 'Useful when testing broader extracts, while still relying on sampling safeguards.',
    },
}


def get_large_dataset_profile(profile_name: str | None) -> dict[str, Any]:
    return dict(LARGE_DATASET_PROFILES.get(str(profile_name or 'Standard'), LARGE_DATASET_PROFILES['Standard']))


def build_product_settings_summary(
    settings: dict[str, Any],
    plan_awareness: dict[str, Any],
) -> dict[str, Any]:
    profile_name = str(settings.get('product_large_dataset_profile', 'Standard'))
    profile = LARGE_DATASET_PROFILES.get(profile_name, LARGE_DATASET_PROFILES['Standard'])
    report_mode = str(settings.get('product_default_report_mode', 'Executive Summary'))
    export_policy = str(settings.get('product_default_export_policy', 'Internal Review'))
    copilot_style = str(settings.get('product_copilot_response_style', 'Concise'))
    show_previews = bool(settings.get('product_copilot_show_workflow_preview', True))
    demo_enabled = bool(settings.get('product_demo_mode_enabled', True))
    plan_name = str(plan_awareness.get('active_plan', 'Pro'))

    summary_cards = [
        {'label': 'Demo Mode', 'value': 'Enabled' if demo_enabled else 'Muted'},
        {'label': 'Large Dataset Profile', 'value': profile_name},
        {'label': 'Default Report', 'value': report_mode},
        {'label': 'AI Copilot Style', 'value': copilot_style},
    ]

    settings_table = pd.DataFrame(
        [
            {'setting_area': 'Demo Guidance', 'current_setting': 'Enabled' if demo_enabled else 'Muted', 'impact': 'Controls whether guided demo content is prominently surfaced in the onboarding flow.'},
            {'setting_area': 'Beta Interest Capture', 'current_setting': 'Enabled' if bool(settings.get('product_beta_interest_enabled', True)) else 'Muted', 'impact': 'Controls whether the local beta-interest capture form appears in the onboarding flow for demo follow-up.'},
            {'setting_area': 'Large Dataset Guidance', 'current_setting': profile_name, 'impact': f"Targets up to ~{int(profile['recommended_rows']):,} rows and ~{int(profile['recommended_upload_mb'])} MB before stronger warnings appear."},
            {'setting_area': 'Sampling Explanation', 'current_setting': str(settings.get('product_sampling_explanation_mode', 'Concise')), 'impact': 'Controls how much context the app shows when sampling safeguards are active.'},
            {'setting_area': 'Default Export Report', 'current_setting': report_mode, 'impact': 'Used when applying export defaults from the product settings panel.'},
            {'setting_area': 'Default Export Policy', 'current_setting': export_policy, 'impact': 'Sets the default sharing posture applied by the export defaults action.'},
            {'setting_area': 'AI Copilot Output Style', 'current_setting': copilot_style, 'impact': 'Shapes how concise or detailed the copilot guidance appears during demos.'},
            {'setting_area': 'Workflow Preview Visibility', 'current_setting': 'Visible' if show_previews else 'Hidden', 'impact': 'Controls whether workflow action preview tables and charts are shown by default.'},
            {'setting_area': 'Active Plan Context', 'current_setting': plan_name, 'impact': 'Helps explain which limits and premium capabilities are currently being previewed.'},
        ]
    )

    admin_notes = [
        profile['note'],
        'These settings shape presentation, defaults, and guidance; they do not alter the source data or analytics calculations.',
    ]
    if bool(plan_awareness.get('strict_enforcement')):
        admin_notes.append('Strict plan enforcement is enabled, so feature gating may follow the active plan more closely.')

    return {
        'summary_cards': summary_cards,
        'settings_table': settings_table,
        'admin_notes': admin_notes,
        'large_dataset_profile': profile,
    }
