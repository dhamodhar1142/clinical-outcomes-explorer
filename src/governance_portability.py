from __future__ import annotations

import json
from typing import Any

import pandas as pd


_DRIFT_FIELD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ('export_policy_name', 'Export policy', 'Medium', 'Governed export behavior changed.'),
    ('active_benchmark_pack_name', 'Active benchmark pack', 'High', 'Benchmark calibration changed.'),
    ('accuracy_reporting_threshold_profile', 'Threshold profile', 'High', 'Reporting interpretation thresholds changed.'),
    ('accuracy_reporting_min_trust_score', 'Minimum trust score', 'High', 'External reporting minimum trust changed.'),
    ('accuracy_allow_directional_external_reporting', 'Directional reporting', 'High', 'Directional external reporting allowance changed.'),
    ('workspace_governance_redaction_level', 'Redaction level', 'Medium', 'Export redaction posture changed.'),
    ('workspace_governance_export_access', 'Export access', 'Medium', 'Export access scope changed.'),
    ('workspace_governance_watermark_sensitive_exports', 'Sensitive watermarking', 'Low', 'Sensitive export watermark posture changed.'),
    ('governance_default_owner', 'Default owner', 'Low', 'Adaptive work routing owner changed.'),
    ('governance_default_reviewer_role', 'Default reviewer role', 'Medium', 'Reviewer routing default changed.'),
    ('governance_release_gate_mode', 'Release gate mode', 'High', 'Release signoff strictness changed.'),
    ('validation_runtime_preference', 'Validation runtime preference', 'Medium', 'Validation runtime posture changed.'),
    ('export_runtime_preference', 'Export runtime preference', 'Medium', 'Export runtime posture changed.'),
)


def _normalize_drift_value(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    if isinstance(value, list):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    if value is None:
        return ''
    return str(value)


def _queue_summary(queue_rows: list[dict[str, Any]]) -> dict[str, int]:
    high_priority_open = [
        row for row in queue_rows
        if str(row.get('priority', '')).strip() == 'High' and str(row.get('status', '')).strip() != 'Completed'
    ]
    return {
        'total_items': len(queue_rows),
        'high_priority_open': len(high_priority_open),
    }


def build_governance_release_bundle(
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    workspace_identity: dict[str, Any],
    policy_pack_name: str,
    policy_pack: dict[str, Any],
    benchmark_packs: dict[str, Any],
    execution_queue: list[dict[str, Any]] | None,
    review_approvals: dict[str, Any],
) -> dict[str, Any]:
    queue_rows = [dict(item) for item in execution_queue or [] if isinstance(item, dict)]
    queue_summary = _queue_summary(queue_rows)
    return {
        'bundle_type': 'clinverity_governance_release_bundle',
        'bundle_version': 1,
        'dataset_context': {
            'dataset_name': dataset_name,
            'source_mode': str(source_meta.get('source_mode', 'Unknown')),
            'dataset_identifier': str(source_meta.get('dataset_identifier', '') or source_meta.get('dataset_cache_key', '')),
        },
        'workspace_context': {
            'workspace_id': str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
            'workspace_name': str(workspace_identity.get('workspace_name', 'Guest Demo Workspace')),
            'display_name': str(workspace_identity.get('display_name', 'Guest User')),
        },
        'policy_pack_name': str(policy_pack_name or 'Current Policy Pack'),
        'policy_pack': dict(policy_pack or {}),
        'organization_benchmark_packs': dict(benchmark_packs or {}),
        'review_approvals': dict(review_approvals or {}),
        'execution_queue_summary': queue_summary,
        'execution_queue': queue_rows[:100],
    }


def build_governance_release_bundle_bytes(bundle: dict[str, Any]) -> bytes:
    return json.dumps(bundle, indent=2).encode('utf-8')


def parse_governance_release_bundle(raw_text: str) -> dict[str, Any]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict) or payload.get('bundle_type') != 'clinverity_governance_release_bundle':
        raise ValueError('The selected file is not a valid Clinverity governance release bundle.')
    if int(payload.get('bundle_version', 1) or 1) > 1:
        raise ValueError('This governance release bundle was created by a newer version of Clinverity and cannot be restored safely here.')
    return payload


def restore_governance_release_bundle(bundle: dict[str, Any], session_state: dict[str, Any]) -> dict[str, Any]:
    applied: list[str] = []
    policy_pack = dict(bundle.get('policy_pack', {}))
    if policy_pack:
        for key, value in policy_pack.items():
            session_state[key] = value
        applied.extend(policy_pack.keys())
    benchmark_packs = dict(bundle.get('organization_benchmark_packs', {}))
    if benchmark_packs:
        session_state['organization_benchmark_packs'] = benchmark_packs
        applied.append('organization_benchmark_packs')
    review_approvals = dict(bundle.get('review_approvals', {}))
    if review_approvals:
        session_state['dataset_review_approvals'] = review_approvals
        applied.append('dataset_review_approvals')
    execution_queue = [dict(item) for item in bundle.get('execution_queue', []) if isinstance(item, dict)]
    if execution_queue:
        session_state['evolution_execution_queue'] = execution_queue
        applied.append('evolution_execution_queue')
    return {
        'applied_keys': applied,
        'policy_pack_name': str(bundle.get('policy_pack_name', 'Imported Policy Pack')),
    }


def build_governance_release_bundle_drift(
    current_bundle: dict[str, Any],
    imported_bundle: dict[str, Any],
) -> pd.DataFrame:
    current_policy = dict(current_bundle.get('policy_pack', {}))
    imported_policy = dict(imported_bundle.get('policy_pack', {}))
    current_routing = dict(current_policy.get('approval_routing_rules', {}))
    imported_routing = dict(imported_policy.get('approval_routing_rules', {}))

    drift_rows: list[dict[str, Any]] = []
    for key, label, impact, detail in _DRIFT_FIELD_SPECS:
        current_value = _normalize_drift_value(current_policy.get(key, ''))
        imported_value = _normalize_drift_value(imported_policy.get(key, ''))
        status = 'Match' if current_value == imported_value else ('Missing' if not imported_value else 'Changed')
        drift_rows.append(
            {
                'drift_area': label,
                'current_value': current_value or 'Not set',
                'bundle_value': imported_value or 'Not set',
                'status': status,
                'release_impact': impact,
                'detail': detail,
            }
        )

    routing_keys = sorted(set(current_routing) | set(imported_routing))
    for routing_key in routing_keys:
        current_value = _normalize_drift_value(current_routing.get(routing_key, ''))
        imported_value = _normalize_drift_value(imported_routing.get(routing_key, ''))
        status = 'Match' if current_value == imported_value else ('Missing' if not imported_value else 'Changed')
        drift_rows.append(
            {
                'drift_area': f"Approval routing: {routing_key}",
                'current_value': current_value or 'Not set',
                'bundle_value': imported_value or 'Not set',
                'status': status,
                'release_impact': 'High' if routing_key == 'release_signoff' else 'Medium',
                'detail': 'Approval routing checkpoint role changed.',
            }
        )

    current_benchmarks = dict(current_bundle.get('organization_benchmark_packs', {}))
    imported_benchmarks = dict(imported_bundle.get('organization_benchmark_packs', {}))
    current_queue_summary = dict(current_bundle.get('execution_queue_summary', _queue_summary(list(current_bundle.get('execution_queue', [])))))
    imported_queue_summary = dict(imported_bundle.get('execution_queue_summary', _queue_summary(list(imported_bundle.get('execution_queue', [])))))

    summary_specs = [
        (
            'Organization benchmark packs',
            len(current_benchmarks),
            len(imported_benchmarks),
            'Medium',
            'Organization benchmark pack inventory changed.',
        ),
        (
            'Execution queue items',
            int(current_queue_summary.get('total_items', 0)),
            int(imported_queue_summary.get('total_items', 0)),
            'Low',
            'Tracked adaptive execution volume changed.',
        ),
        (
            'High-priority open items',
            int(current_queue_summary.get('high_priority_open', 0)),
            int(imported_queue_summary.get('high_priority_open', 0)),
            'High',
            'Open high-priority adaptive work changed.',
        ),
    ]
    for label, current_value, imported_value, impact, detail in summary_specs:
        status = 'Match' if current_value == imported_value else 'Changed'
        drift_rows.append(
            {
                'drift_area': label,
                'current_value': str(current_value),
                'bundle_value': str(imported_value),
                'status': status,
                'release_impact': impact,
                'detail': detail,
            }
        )

    return pd.DataFrame(drift_rows)


def build_governance_release_bundle_gate(drift_table: pd.DataFrame | Any) -> dict[str, Any]:
    drift_df = drift_table if isinstance(drift_table, pd.DataFrame) else pd.DataFrame()
    if drift_df.empty or 'status' not in drift_df.columns:
        return {
            'changed_items_count': 0,
            'high_impact_drift_count': 0,
            'requires_signoff': False,
            'gate_message': 'No release-control drift detected.',
        }
    changed_mask = drift_df['status'].astype(str) != 'Match'
    high_impact_mask = changed_mask & drift_df.get('release_impact', pd.Series(dtype=str)).astype(str).eq('High')
    changed_items_count = int(changed_mask.sum())
    high_impact_drift_count = int(high_impact_mask.sum())
    requires_signoff = high_impact_drift_count > 0
    if requires_signoff:
        gate_message = (
            f'{high_impact_drift_count} high-impact release-control drift item(s) were detected. '
            'Explicit signoff is required before applying this governance bundle.'
        )
    elif changed_items_count:
        gate_message = (
            f'{changed_items_count} release-control drift item(s) were detected. '
            'Review the differences before applying the bundle.'
        )
    else:
        gate_message = 'No release-control drift detected.'
    return {
        'changed_items_count': changed_items_count,
        'high_impact_drift_count': high_impact_drift_count,
        'requires_signoff': requires_signoff,
        'gate_message': gate_message,
    }


def build_governance_release_bundle_runtime_compatibility(
    imported_bundle: dict[str, Any],
    *,
    validation_runtime_profile: dict[str, Any],
    export_runtime_profile: dict[str, Any],
) -> pd.DataFrame:
    imported_policy = dict(imported_bundle.get('policy_pack', {}))
    execution_queue = [dict(item) for item in imported_bundle.get('execution_queue', []) if isinstance(item, dict)]
    compatibility_rows: list[dict[str, Any]] = []

    validation_preference = str(imported_policy.get('validation_runtime_preference', 'Auto'))
    validation_runtime = str(validation_runtime_profile.get('runtime_label', 'Unknown'))
    validation_supports_heavy = bool(validation_runtime_profile.get('supports_heavy_validation', False))
    open_heavy_validation = [
        item for item in execution_queue
        if str(item.get('status', '')).strip() != 'Completed'
        and str(item.get('suggested_validation', '')).strip().lower() in {
            'full validation',
            'release validation',
            'cross-dataset cache validation',
        }
    ]
    validation_status = 'Compatible'
    validation_detail = 'The current runtime posture can support the bundle validation expectations.'
    validation_impact = 'Low'
    if open_heavy_validation and not validation_supports_heavy:
        validation_status = 'Needs local/staging handoff'
        validation_detail = 'The imported bundle carries open heavy validation work that the current runtime should not execute directly.'
        validation_impact = 'High'
    elif validation_preference == 'Prefer local/staging for heavy actions' and not validation_supports_heavy:
        validation_status = 'Compatible with handoff'
        validation_detail = 'The bundle prefers heavier validation in local or staging environments, which matches the current constrained runtime.'
        validation_impact = 'Medium'

    export_preference = str(imported_policy.get('export_runtime_preference', 'Auto'))
    export_runtime = str(export_runtime_profile.get('runtime_label', 'Unknown'))
    export_supports_governed = bool(export_runtime_profile.get('supports_governed_packaging', False))
    open_governed_export = [
        item for item in execution_queue
        if str(item.get('status', '')).strip() != 'Completed'
        and (
            'export' in str(item.get('proposal_title', '')).lower()
            or 'report' in str(item.get('proposal_title', '')).lower()
            or 'external reporting' in str(item.get('proposed_change', '')).lower()
        )
    ]
    export_status = 'Compatible'
    export_detail = 'The current runtime posture can support the bundle export expectations.'
    export_impact = 'Low'
    if open_governed_export and not export_supports_governed:
        export_status = 'Needs local/staging handoff'
        export_detail = 'The imported bundle includes open governed export work that should be executed from a heavier local or staging runtime.'
        export_impact = 'High'
    elif export_preference == 'Prefer local/staging for heavy actions' and not export_supports_governed:
        export_status = 'Compatible with handoff'
        export_detail = 'The bundle expects governed packaging to run outside lighter runtimes, which matches the current environment.'
        export_impact = 'Medium'

    compatibility_rows.extend(
        [
            {
                'compatibility_area': 'Validation runtime posture',
                'current_runtime': validation_runtime,
                'bundle_expectation': validation_preference,
                'status': validation_status,
                'release_impact': validation_impact,
                'detail': validation_detail,
            },
            {
                'compatibility_area': 'Export runtime posture',
                'current_runtime': export_runtime,
                'bundle_expectation': export_preference,
                'status': export_status,
                'release_impact': export_impact,
                'detail': export_detail,
            },
        ]
    )
    return pd.DataFrame(compatibility_rows)


def build_governance_release_bundle_compatibility_gate(
    compatibility_table: pd.DataFrame | Any,
) -> dict[str, Any]:
    compatibility_df = compatibility_table if isinstance(compatibility_table, pd.DataFrame) else pd.DataFrame()
    if compatibility_df.empty or 'status' not in compatibility_df.columns:
        return {
            'mismatch_count': 0,
            'high_impact_mismatch_count': 0,
            'requires_signoff': False,
            'gate_message': 'No runtime compatibility issues detected.',
        }
    mismatch_mask = ~compatibility_df['status'].astype(str).isin(['Compatible'])
    high_impact_mask = mismatch_mask & compatibility_df.get('release_impact', pd.Series(dtype=str)).astype(str).eq('High')
    mismatch_count = int(mismatch_mask.sum())
    high_impact_mismatch_count = int(high_impact_mask.sum())
    requires_signoff = high_impact_mismatch_count > 0
    if requires_signoff:
        gate_message = (
            f'{high_impact_mismatch_count} high-impact runtime compatibility issue(s) were detected. '
            'Explicit signoff is required before promoting this governance bundle here.'
        )
    elif mismatch_count:
        gate_message = (
            f'{mismatch_count} runtime compatibility warning(s) were detected. '
            'Review the required handoff posture before applying the bundle.'
        )
    else:
        gate_message = 'No runtime compatibility issues detected.'
    return {
        'mismatch_count': mismatch_count,
        'high_impact_mismatch_count': high_impact_mismatch_count,
        'requires_signoff': requires_signoff,
        'gate_message': gate_message,
    }


def build_governance_release_bundle_promotion_readiness(
    *,
    drift_gate: dict[str, Any] | None,
    compatibility_gate: dict[str, Any] | None,
    imported_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    drift_gate = dict(drift_gate or {})
    compatibility_gate = dict(compatibility_gate or {})
    imported_bundle = dict(imported_bundle or {})
    queue_summary = dict(imported_bundle.get('execution_queue_summary', {}))
    execution_queue = [dict(item) for item in imported_bundle.get('execution_queue', []) if isinstance(item, dict)]
    open_release_blockers = int(
        sum(
            1
            for item in execution_queue
            if str(item.get('status', '')).strip() != 'Completed'
            and str(item.get('release_gate_status', '')).strip() in {'Required before release', 'Blocked for release'}
        )
    ) or int(queue_summary.get('high_priority_open', 0))
    requires_signoff = bool(drift_gate.get('requires_signoff', False) or compatibility_gate.get('requires_signoff', False))
    if open_release_blockers > 0:
        status = 'Blocked'
        detail = 'Open release blockers are still present in the imported bundle execution queue.'
    elif requires_signoff:
        status = 'Needs signoff'
        detail = 'High-impact promotion checks require explicit signoff before this bundle should be applied.'
    elif int(drift_gate.get('changed_items_count', 0)) > 0 or int(compatibility_gate.get('mismatch_count', 0)) > 0:
        status = 'Review recommended'
        detail = 'Promotion is possible, but drift or runtime compatibility warnings should be reviewed first.'
    else:
        status = 'Ready'
        detail = 'No blocking promotion issues were detected for the current environment.'
    summary_table = pd.DataFrame(
        [
            {'checkpoint': 'Policy drift', 'status': 'Needs signoff' if drift_gate.get('requires_signoff', False) else ('Review' if int(drift_gate.get('changed_items_count', 0)) > 0 else 'Clear'), 'detail': str(drift_gate.get('gate_message', 'No release-control drift detected.'))},
            {'checkpoint': 'Runtime compatibility', 'status': 'Needs signoff' if compatibility_gate.get('requires_signoff', False) else ('Review' if int(compatibility_gate.get('mismatch_count', 0)) > 0 else 'Clear'), 'detail': str(compatibility_gate.get('gate_message', 'No runtime compatibility issues detected.'))},
            {'checkpoint': 'Open release blockers', 'status': 'Blocked' if open_release_blockers > 0 else 'Clear', 'detail': f'{open_release_blockers} open blocker(s) carried by the imported bundle.'},
        ]
    )
    return {
        'status': status,
        'detail': detail,
        'requires_signoff': requires_signoff,
        'open_release_blockers': open_release_blockers,
        'summary_table': summary_table,
    }


__all__ = [
    'build_governance_release_bundle',
    'build_governance_release_bundle_compatibility_gate',
    'build_governance_release_bundle_drift',
    'build_governance_release_bundle_gate',
    'build_governance_release_bundle_promotion_readiness',
    'build_governance_release_bundle_runtime_compatibility',
    'build_governance_release_bundle_bytes',
    'parse_governance_release_bundle',
    'restore_governance_release_bundle',
]
