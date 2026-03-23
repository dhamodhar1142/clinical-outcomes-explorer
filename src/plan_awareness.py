from __future__ import annotations

from typing import Any

import pandas as pd


PLAN_CATALOG: dict[str, dict[str, Any]] = {
    'Free': {
        'max_file_mb': 15.0,
        'max_workflow_packs': 3,
        'advanced_exports': False,
        'predictive_modeling': False,
        'model_comparison': False,
        'governance_exports': False,
        'stakeholder_bundle': False,
        'description': 'Entry-level plan for lightweight dataset review and core profiling.',
    },
    'Pro': {
        'max_file_mb': 75.0,
        'max_workflow_packs': 15,
        'advanced_exports': True,
        'predictive_modeling': True,
        'model_comparison': True,
        'governance_exports': True,
        'stakeholder_bundle': True,
        'description': 'Professional plan for full analytics, guided modeling, and advanced exports.',
    },
    'Team': {
        'max_file_mb': 250.0,
        'max_workflow_packs': 50,
        'advanced_exports': True,
        'predictive_modeling': True,
        'model_comparison': True,
        'governance_exports': True,
        'stakeholder_bundle': True,
        'description': 'Team plan for broader collaboration, heavier datasets, and richer handoff workflows.',
    },
    'Enterprise': {
        'max_file_mb': 1000.0,
        'max_workflow_packs': 250,
        'advanced_exports': True,
        'predictive_modeling': True,
        'model_comparison': True,
        'governance_exports': True,
        'stakeholder_bundle': True,
        'description': 'Enterprise placeholder for larger deployments, broader governance needs, and future tenant-aware controls.',
    },
}


PREMIUM_FEATURES = [
    ('predictive_modeling', 'Predictive Modeling Studio', 'Guided model training and scoring'),
    ('model_comparison', 'Model Comparison Studio', 'Side-by-side comparison of supported models'),
    ('advanced_exports', 'Advanced Export Workflows', 'Compliance packs, print-friendly summaries, and extended report assets'),
    ('governance_exports', 'Governance & Audit Packet', 'Governance review TXT, CSV, and JSON handoff assets'),
    ('stakeholder_bundle', 'Stakeholder Export Bundle', 'Bundled stakeholder handoff artifacts and manifests'),
]


def is_strict_plan_enforcement(mode: str | None) -> bool:
    return str(mode or '').strip().lower() == 'strict'


def plan_feature_enabled(plan_name: str, feature_name: str) -> bool:
    plan = PLAN_CATALOG.get(plan_name, PLAN_CATALOG['Pro'])
    return bool(plan.get(feature_name, False))


def build_plan_awareness(
    plan_name: str,
    enforcement_mode: str,
    source_meta: dict[str, Any],
    workflow_pack_count: int,
    snapshot_count: int,
) -> dict[str, Any]:
    active_plan = PLAN_CATALOG.get(plan_name, PLAN_CATALOG['Pro'])
    file_size_mb = float(source_meta.get('file_size_mb', 0.0) or 0.0)
    strict = is_strict_plan_enforcement(enforcement_mode)

    file_limit = float(active_plan['max_file_mb'])
    pack_limit = int(active_plan['max_workflow_packs'])
    file_exceeded = file_size_mb > file_limit
    packs_exceeded = workflow_pack_count >= pack_limit

    support_rows = []
    for feature_key, label, note in PREMIUM_FEATURES:
        included = bool(active_plan.get(feature_key, False))
        support_rows.append(
            {
                'feature': label,
                'included_in_plan': 'Yes' if included else 'No',
                'status': 'Included' if included else ('Limited in strict mode' if strict else 'Demo-safe access'),
                'why_it_matters': note,
            }
        )
    premium_features = pd.DataFrame(support_rows)

    plan_comparison = pd.DataFrame(
        [
            {
                'plan': name,
                'best_for': (
                    'Solo evaluation, lightweight review, and guided profiling'
                    if name == 'Free'
                    else 'Core analytics, reporting, and startup pilot workflows'
                    if name == 'Pro'
                    else 'Team collaboration, heavier datasets, and richer handoff workflows'
                    if name == 'Team'
                    else 'Future enterprise rollout, governance-heavy use cases, and tenant-aware deployment needs'
                ),
                'upload_guidance_mb': f"{float(config['max_file_mb']):.0f}",
                'workflow_pack_guidance': str(int(config['max_workflow_packs'])),
                'advanced_exports': 'Included' if bool(config.get('advanced_exports')) else 'Not included',
                'predictive_modeling': 'Included' if bool(config.get('predictive_modeling')) else 'Not included',
            }
            for name, config in PLAN_CATALOG.items()
        ]
    )

    limits_table = pd.DataFrame(
        [
            {'limit_type': 'Upload guidance', 'value': f'{file_limit:.0f} MB', 'current_usage': f'{file_size_mb:.1f} MB', 'status': 'Above plan guidance' if file_exceeded else 'Within guidance'},
            {'limit_type': 'Workflow packs', 'value': str(pack_limit), 'current_usage': str(workflow_pack_count), 'status': 'At limit' if packs_exceeded else 'Within guidance'},
            {'limit_type': 'Saved snapshots', 'value': 'Session-scoped', 'current_usage': str(snapshot_count), 'status': 'Available'},
        ]
    )

    summary_cards = [
        {'label': 'Active Plan', 'value': plan_name},
        {'label': 'Enforcement', 'value': enforcement_mode},
        {'label': 'Upload Guidance', 'value': f'{file_limit:.0f} MB'},
        {'label': 'Workflow Pack Limit', 'value': str(pack_limit)},
    ]

    guidance_notes: list[str] = []
    guidance_notes.append(
        'Demo-safe mode keeps the full walkthrough visible even when the selected plan would normally have narrower packaging or limits.'
    )
    if file_exceeded:
        guidance_notes.append(
            f"This dataset is larger than the {plan_name} plan upload guidance ({file_limit:.0f} MB). Analysis can still continue in demo-safe mode, but a paid plan would normally be recommended for sustained use."
        )
    if packs_exceeded:
        guidance_notes.append(
            f"The current workspace has reached the {plan_name} plan workflow-pack guidance ({pack_limit}). Save actions can be limited when strict enforcement is enabled."
        )
    if not guidance_notes:
        guidance_notes.append(f'The current dataset and saved assets are within the {plan_name} plan guidance for this session.')

    entitlement_rows = [
        {
            'entitlement_area': 'Dataset size guidance',
            'current_state': 'Soft limit reached' if file_exceeded else 'Within plan guidance',
            'effect_in_demo_safe_mode': 'Analysis can continue with guidance messaging.',
            'effect_in_strict_mode': 'A higher plan would normally be recommended for sustained usage.',
            'recommended_upgrade_path': 'Pro or Team' if plan_name == 'Free' else 'Team or Enterprise' if plan_name == 'Pro' else 'Enterprise',
        },
        {
            'entitlement_area': 'Workflow pack saves',
            'current_state': 'Soft limit reached' if packs_exceeded else 'Within plan guidance',
            'effect_in_demo_safe_mode': 'Saving stays visible so the workflow can still be demonstrated.',
            'effect_in_strict_mode': 'Additional workflow pack saves can pause until the plan or enforcement mode changes.',
            'recommended_upgrade_path': 'Pro or Team' if plan_name == 'Free' else 'Team or Enterprise' if plan_name == 'Pro' else 'Enterprise',
        },
    ]
    for feature_key, label, _note in PREMIUM_FEATURES:
        included = bool(active_plan.get(feature_key, False))
        entitlement_rows.append(
            {
                'entitlement_area': label,
                'current_state': 'Included' if included else 'Premium',
                'effect_in_demo_safe_mode': 'Feature stays visible for walkthrough and packaging preview.' if not included else 'Feature is fully aligned with the active plan.',
                'effect_in_strict_mode': 'Feature can be paused with upgrade guidance.' if not included else 'Feature remains available.',
                'recommended_upgrade_path': 'Pro' if plan_name == 'Free' else 'Team' if plan_name == 'Pro' and feature_key in {'stakeholder_bundle', 'governance_exports'} else 'Enterprise' if plan_name == 'Team' else 'Current plan',
            }
        )
    entitlement_summary = pd.DataFrame(entitlement_rows)

    return {
        'active_plan': plan_name,
        'enforcement_mode': enforcement_mode,
        'strict_enforcement': strict,
        'headline': 'Product plan packaging',
        'plan_story': 'Use Free, Pro, Team, and Enterprise packaging to preview how Clinverity can be positioned for evaluation, pilot, team rollout, and future enterprise deployment.',
        'summary_cards': summary_cards,
        'description': active_plan['description'],
        'limits_table': limits_table,
        'premium_features': premium_features,
        'plan_comparison': plan_comparison,
        'entitlement_summary': entitlement_summary,
        'guidance_notes': guidance_notes,
        'file_limit_mb': file_limit,
        'workflow_pack_limit': pack_limit,
        'file_size_exceeded': file_exceeded,
        'workflow_pack_limit_reached': packs_exceeded,
    }
