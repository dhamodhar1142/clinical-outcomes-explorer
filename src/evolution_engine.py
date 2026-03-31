from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.semantic_mapper import build_data_remediation_assistant
from src.schema_detection import StructureSummary


def _safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _priority_label(score: float) -> str:
    if score >= 0.86:
        return 'High'
    if score >= 0.62:
        return 'Medium'
    return 'Low'


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _metadata_json(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    metadata = record.get('metadata_json', {})
    return metadata if isinstance(metadata, dict) else {}


def _metadata_columns(record: dict[str, Any] | None) -> list[str]:
    metadata = _metadata_json(record)
    columns = metadata.get('source_columns', [])
    if isinstance(columns, list):
        return [str(item).strip() for item in columns if str(item).strip()]
    return []


def _metadata_dtypes(record: dict[str, Any] | None) -> dict[str, str]:
    metadata = _metadata_json(record)
    dtypes = metadata.get('source_dtypes', {})
    if not isinstance(dtypes, dict):
        return {}
    return {
        str(column).strip(): str(dtype).strip()
        for column, dtype in dtypes.items()
        if str(column).strip()
    }


def _infer_goal_profile(
    *,
    readiness: dict[str, Any],
    healthcare: dict[str, Any],
    accuracy_summary: dict[str, Any],
    sample_info: dict[str, Any],
) -> tuple[str, str]:
    module_gates = _safe_df(accuracy_summary.get('module_reporting_gates'))
    blocked_names = (
        module_gates.loc[
            module_gates.get('reporting_gate', pd.Series(dtype=str)).astype(str).isin(
                ['Restricted', 'Blocked', 'Internal only']
            )
        ]['analytics_module'].astype(str).tolist()
        if not module_gates.empty and 'analytics_module' in module_gates.columns
        else []
    )
    if bool(sample_info.get('sampling_applied')):
        return 'Release-grade validation', 'The current run is sampled, so the highest-value next goal is promoting this dataset toward full trusted validation.'
    if healthcare.get('readmission', {}).get('available'):
        return 'Readmission reduction analytics', 'The dataset supports encounter-aware readmission analysis, so Clinverity should optimize toward stronger intervention and cohort insight quality.'
    if any(name in blocked_names for name in ['Export Center', 'Governed exports', 'Readmission Analytics']):
        return 'Governed external reporting', 'Current reporting gates are still restrictive, so the product should prioritize trust and export-safe improvements first.'
    if float(readiness.get('readiness_score', 0.0) or 0.0) < 0.8:
        return 'Readiness uplift', 'Module readiness is still mixed, so Clinverity should focus on unlocking more native capabilities from the current schema.'
    return 'Healthcare intelligence expansion', 'The dataset already supports the core workflow, so the best next step is deepening insight quality and reusable intelligence for this family.'


def queue_execution_items(
    existing_queue: list[dict[str, Any]] | None,
    proposals: pd.DataFrame | Any,
    *,
    dataset_name: str,
    dataset_family_key: str,
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in existing_queue or []:
        if not isinstance(item, dict):
            continue
        queue.append(dict(item))
        seen.add((str(item.get('dataset_family_key', '')), str(item.get('proposal_title', ''))))
    proposal_table = _safe_df(proposals)
    if proposal_table.empty:
        return queue
    for _, row in proposal_table.iterrows():
        title = str(row.get('proposal_title', '')).strip()
        key = (dataset_family_key, title)
        if not title or key in seen:
            continue
        queue.append(
            {
                'proposal_title': title,
                'proposal_type': str(row.get('proposal_type', 'Adaptive spec')),
                'priority': str(row.get('priority', 'Medium')),
                'proposed_change': str(row.get('proposed_change', '')),
                'suggested_validation': str(row.get('suggested_validation', '')),
                'dataset_name': dataset_name,
                'dataset_family_key': dataset_family_key,
                'status': 'Queued',
                'owner': 'Unassigned',
                'due_state': 'Next review' if str(row.get('priority', 'Medium')) == 'High' else 'Planned',
                'release_gate_status': (
                    'Required before release'
                    if 'external reporting' in str(row.get('proposed_change', '')).lower()
                    or 'full local or release validation' in str(row.get('proposed_change', '')).lower()
                    else 'Advisory'
                ),
            }
        )
        seen.add(key)
    return queue


def update_execution_item_status(
    existing_queue: list[dict[str, Any]] | None,
    *,
    proposal_title: str,
    status: str,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for item in existing_queue or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        if str(entry.get('proposal_title', '')).strip() == proposal_title:
            entry['status'] = status
        updated.append(entry)
    return updated


def build_execution_autopilot_actions(
    existing_queue: list[dict[str, Any]] | None,
    proposals: pd.DataFrame | Any,
    validation_recommendations: pd.DataFrame | Any,
) -> pd.DataFrame:
    queue_table = pd.DataFrame([dict(item) for item in existing_queue or [] if isinstance(item, dict)])
    proposal_table = _safe_df(proposals)
    validation_table = _safe_df(validation_recommendations)
    rows: list[dict[str, Any]] = []

    if not proposal_table.empty:
        high_priority_proposals = proposal_table.get('priority', pd.Series(dtype=str)).astype(str).eq('High').sum()
        rows.append(
            {
                'action': 'Queue high-priority proposals',
                'impact': f'{int(high_priority_proposals or len(proposal_table))} proposal(s) ready for tracked execution.',
                'safe_result': 'Adds missing high-priority proposals into the adaptive backlog without altering production logic.',
            }
        )
    if not validation_table.empty:
        rows.append(
            {
                'action': 'Queue recommended validation',
                'impact': f'{len(validation_table)} validation follow-up item(s) can be tracked.',
                'safe_result': 'Queues the recommended validation plan so release hardening work is visible and auditable.',
            }
        )
    if not queue_table.empty:
        release_blockers = queue_table.get('release_gate_status', pd.Series(dtype=str)).astype(str).isin(
            ['Required before release', 'Blocked for release']
        )
        open_release_blockers = int(
            (
                release_blockers
                & ~queue_table.get('status', pd.Series(dtype=str)).astype(str).eq('Completed')
            ).sum()
        )
        if open_release_blockers:
            rows.append(
                {
                    'action': 'Start release blockers',
                    'impact': f'{open_release_blockers} blocker(s) can be moved into active review.',
                    'safe_result': 'Advances blocker status to In progress and marks the due state as Before release.',
                }
            )
        unassigned_high_priority = int(
            (
                queue_table.get('priority', pd.Series(dtype=str)).astype(str).eq('High')
                & queue_table.get('owner', pd.Series(dtype=str)).astype(str).isin(['', 'Unassigned'])
                & ~queue_table.get('status', pd.Series(dtype=str)).astype(str).eq('Completed')
            ).sum()
        )
        if unassigned_high_priority:
            rows.append(
                {
                    'action': 'Assign high-priority items',
                    'impact': f'{unassigned_high_priority} high-priority item(s) can be assigned.',
                    'safe_result': 'Assigns open high-priority work to the active role for clearer ownership.',
                }
            )
    return pd.DataFrame(rows)


def apply_execution_autopilot(
    existing_queue: list[dict[str, Any]] | None,
    *,
    action_name: str,
    proposals: pd.DataFrame | Any,
    validation_recommendations: pd.DataFrame | Any,
    dataset_name: str,
    dataset_family_key: str,
    default_owner: str = 'Analyst',
) -> list[dict[str, Any]]:
    queue = [dict(item) for item in existing_queue or [] if isinstance(item, dict)]
    action = str(action_name).strip()
    if action == 'Queue high-priority proposals':
        proposal_table = _safe_df(proposals)
        if 'priority' in proposal_table.columns:
            proposal_table = proposal_table[proposal_table['priority'].astype(str) == 'High']
        return queue_execution_items(
            queue,
            proposal_table,
            dataset_name=dataset_name,
            dataset_family_key=dataset_family_key,
        )
    if action == 'Queue recommended validation':
        rows = []
        validation_table = _safe_df(validation_recommendations)
        for row in validation_table.to_dict(orient='records'):
            rows.append(
                {
                    'proposal_title': f"{row.get('validation', 'Validation')} follow-up",
                    'proposal_type': 'Validation plan',
                    'priority': row.get('priority', 'Medium'),
                    'proposed_change': row.get('why', ''),
                    'suggested_validation': row.get('validation', ''),
                }
            )
        return queue_execution_items(
            queue,
            pd.DataFrame(rows),
            dataset_name=dataset_name,
            dataset_family_key=dataset_family_key,
        )
    updated: list[dict[str, Any]] = []
    for item in queue:
        entry = dict(item)
        status = str(entry.get('status', 'Queued'))
        owner = str(entry.get('owner', '') or 'Unassigned')
        release_gate = str(entry.get('release_gate_status', 'Advisory'))
        priority = str(entry.get('priority', 'Medium'))
        if action == 'Start release blockers':
            if release_gate in {'Required before release', 'Blocked for release'} and status != 'Completed':
                entry['status'] = 'In progress'
                entry['due_state'] = 'Before release'
        elif action == 'Assign high-priority items':
            if priority == 'High' and status != 'Completed' and owner in {'', 'Unassigned'}:
                entry['owner'] = default_owner
        updated.append(entry)
    return updated


def build_dataset_version_diff(
    current_version: dict[str, Any] | None,
    previous_version: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(current_version or {})
    previous = dict(previous_version or {})
    current_columns = set(_metadata_columns(current))
    previous_columns = set(_metadata_columns(previous))
    current_dtypes = _metadata_dtypes(current)
    previous_dtypes = _metadata_dtypes(previous)

    summary_table = pd.DataFrame(
        [
            {
                'metric': 'Row count delta',
                'current': current.get('row_count', 0),
                'previous': previous.get('row_count', 0),
                'delta': _as_int(current.get('row_count', 0)) - _as_int(previous.get('row_count', 0)),
            },
            {
                'metric': 'Column count delta',
                'current': current.get('column_count', 0),
                'previous': previous.get('column_count', 0),
                'delta': _as_int(current.get('column_count', 0)) - _as_int(previous.get('column_count', 0)),
            },
            {
                'metric': 'File size delta (MB)',
                'current': current.get('file_size_mb', 0.0),
                'previous': previous.get('file_size_mb', 0.0),
                'delta': round(_as_float(current.get('file_size_mb', 0.0)) - _as_float(previous.get('file_size_mb', 0.0)), 3),
            },
            {
                'metric': 'Added columns',
                'current': len(current_columns - previous_columns),
                'previous': '-',
                'delta': len(current_columns - previous_columns),
            },
            {
                'metric': 'Removed columns',
                'current': len(previous_columns - current_columns),
                'previous': '-',
                'delta': len(previous_columns - current_columns),
            },
            {
                'metric': 'Type changes',
                'current': sum(
                    1 for column in sorted(current_columns & previous_columns)
                    if current_dtypes.get(column, '') != previous_dtypes.get(column, '')
                ),
                'previous': '-',
                'delta': sum(
                    1 for column in sorted(current_columns & previous_columns)
                    if current_dtypes.get(column, '') != previous_dtypes.get(column, '')
                ),
            },
        ]
    )
    added_columns_table = pd.DataFrame(
        [{'column_name': column, 'current_dtype': current_dtypes.get(column, '')} for column in sorted(current_columns - previous_columns)]
    )
    removed_columns_table = pd.DataFrame(
        [{'column_name': column, 'previous_dtype': previous_dtypes.get(column, '')} for column in sorted(previous_columns - current_columns)]
    )
    dtype_changes_table = pd.DataFrame(
        [
            {
                'column_name': column,
                'previous_dtype': previous_dtypes.get(column, ''),
                'current_dtype': current_dtypes.get(column, ''),
            }
            for column in sorted(current_columns & previous_columns)
            if current_dtypes.get(column, '') != previous_dtypes.get(column, '')
        ]
    )
    return {
        'summary_table': summary_table,
        'added_columns_table': added_columns_table,
        'removed_columns_table': removed_columns_table,
        'dtype_changes_table': dtype_changes_table,
    }


def build_validation_recommendations(
    *,
    readiness: dict[str, Any],
    trust_summary: dict[str, Any],
    sample_info: dict[str, Any],
    module_reporting_gates: pd.DataFrame | Any,
) -> pd.DataFrame:
    recommendations: list[dict[str, Any]] = [
        {
            'validation': 'Quick validation',
            'priority': 'Baseline',
            'why': 'Confirms the active workflow still behaves correctly on the CI-friendly healthcare fixture.',
        }
    ]
    if bool(sample_info.get('sampling_applied')):
        recommendations.append(
            {
                'validation': 'Full validation',
                'priority': 'High',
                'why': 'This dataset ran in sampled scope, so a full run is recommended before strong external use.',
            }
        )
    if float(readiness.get('readiness_score', 0.0) or 0.0) < 0.8:
        recommendations.append(
            {
                'validation': 'Accessibility / UI audit',
                'priority': 'Medium',
                'why': 'Lower readiness often coincides with heavier remediation workflows that need visible controls and clear messaging.',
            }
        )
    if float(trust_summary.get('trust_score', 0.0) or 0.0) < 0.82:
        recommendations.append(
            {
                'validation': 'Release validation',
                'priority': 'High',
                'why': 'Trust is not yet strong, so the broader release gate is the safest final confirmation path.',
            }
        )
    gates = _safe_df(module_reporting_gates)
    if not gates.empty and gates.get('reporting_gate', pd.Series(dtype=str)).astype(str).isin(['Restricted', 'Blocked']).any():
        recommendations.append(
            {
                'validation': 'Cross-dataset cache validation',
                'priority': 'Medium',
                'why': 'Restricted modules warrant a stronger check that runtime context and dataset identity stay isolated.',
            }
        )
    return pd.DataFrame(recommendations)


def build_drift_alerts(
    *,
    family_memory_entry: dict[str, Any],
    readiness_score: float,
    trust_score: float,
    helper_columns_added: int,
    goal_profile: str,
) -> pd.DataFrame:
    previous_snapshot = dict(family_memory_entry.get('last_snapshot', {}))
    alerts: list[dict[str, Any]] = []
    previous_readiness = _as_float(previous_snapshot.get('readiness_score', readiness_score), readiness_score)
    previous_trust = _as_float(previous_snapshot.get('trust_score', trust_score), trust_score)
    previous_helper = _as_int(previous_snapshot.get('helper_columns_added', helper_columns_added), helper_columns_added)
    previous_goal = str(previous_snapshot.get('goal_profile', goal_profile))
    previous_columns = _as_int(previous_snapshot.get('total_source_columns', 0), 0)
    previous_strongly_mapped = _as_int(previous_snapshot.get('strongly_mapped_columns', 0), 0)
    previous_blocked = _as_int(previous_snapshot.get('blocked_reporting_module_count', 0), 0)
    previous_benchmark = str(previous_snapshot.get('benchmark_profile', ''))
    if abs(readiness_score - previous_readiness) >= 0.08:
        alerts.append(
            {
                'drift_area': 'Readiness score',
                'severity': 'High' if readiness_score < previous_readiness else 'Medium',
                'detail': f'Readiness shifted from {previous_readiness:.2f} to {readiness_score:.2f} for this dataset family.',
            }
        )
    if abs(trust_score - previous_trust) >= 0.08:
        alerts.append(
            {
                'drift_area': 'Trust score',
                'severity': 'High' if trust_score < previous_trust else 'Medium',
                'detail': f'Trust shifted from {previous_trust:.2f} to {trust_score:.2f}.',
            }
        )
    if abs(helper_columns_added - previous_helper) >= 2:
        alerts.append(
            {
                'drift_area': 'Helper dependence',
                'severity': 'Medium',
                'detail': f'Helper-backed field count changed from {previous_helper} to {helper_columns_added}.',
            }
        )
    if previous_columns and abs(previous_columns - _as_int(family_memory_entry.get('current_total_source_columns', previous_columns), previous_columns)) >= 3:
        alerts.append(
            {
                'drift_area': 'Schema breadth',
                'severity': 'Medium',
                'detail': f'Source column count shifted from {previous_columns} to {_as_int(family_memory_entry.get("current_total_source_columns", previous_columns), previous_columns)}.',
            }
        )
    current_strongly_mapped = _as_int(family_memory_entry.get('current_strongly_mapped_columns', previous_strongly_mapped), previous_strongly_mapped)
    if previous_strongly_mapped and abs(previous_strongly_mapped - current_strongly_mapped) >= 2:
        alerts.append(
            {
                'drift_area': 'Native mapping coverage',
                'severity': 'High' if current_strongly_mapped < previous_strongly_mapped else 'Medium',
                'detail': f'Strongly mapped columns changed from {previous_strongly_mapped} to {current_strongly_mapped}.',
            }
        )
    current_blocked = _as_int(family_memory_entry.get('current_blocked_reporting_module_count', previous_blocked), previous_blocked)
    if abs(previous_blocked - current_blocked) >= 1:
        alerts.append(
            {
                'drift_area': 'Reporting gate coverage',
                'severity': 'High' if current_blocked > previous_blocked else 'Medium',
                'detail': f'Blocked or restricted reporting modules changed from {previous_blocked} to {current_blocked}.',
            }
        )
    if previous_goal and previous_goal != goal_profile:
        alerts.append(
            {
                'drift_area': 'Primary optimization goal',
                'severity': 'Medium',
                'detail': f"The learned goal changed from '{previous_goal}' to '{goal_profile}'.",
            }
        )
    current_benchmark = str(family_memory_entry.get('current_benchmark_profile', previous_benchmark))
    if previous_benchmark and current_benchmark and previous_benchmark != current_benchmark:
        alerts.append(
            {
                'drift_area': 'Benchmark profile',
                'severity': 'Low',
                'detail': f"Benchmark context changed from '{previous_benchmark}' to '{current_benchmark}'.",
            }
        )
    return pd.DataFrame(alerts)


def build_family_intelligence_table(
    *,
    dataset_family_label: str,
    family_key: str,
    suggested_benchmark: str,
    current_benchmark: str,
    suggested_profile_name: str,
    accounting: dict[str, Any],
    family_memory: dict[str, Any],
    semantic: dict[str, Any],
) -> pd.DataFrame:
    healthcare_hits = [str(item) for item in semantic.get('healthcare_field_hits', [])]
    unresolved = [str(item) for item in accounting.get('unresolved_columns', []) if str(item).strip()]
    rows = [
        {'signal': 'Dataset family', 'value': dataset_family_label, 'detail': family_key},
        {'signal': 'Suggested mapping profile', 'value': suggested_profile_name or 'No saved profile suggestion yet', 'detail': 'Uses built-in templates plus approved family learning.'},
        {'signal': 'Suggested benchmark', 'value': suggested_benchmark or 'Generic Healthcare', 'detail': f'Current benchmark: {current_benchmark or "Auto"}'},
        {'signal': 'Healthcare concepts recognized', 'value': f'{len(healthcare_hits)} native hits', 'detail': ', '.join(healthcare_hits[:8]) or 'No strong healthcare-native concepts yet.'},
        {'signal': 'Unresolved source columns', 'value': f'{len(unresolved)} unresolved', 'detail': ', '.join(unresolved[:6]) or 'No unresolved columns in the current top accounting set.'},
        {'signal': 'Prior family observations', 'value': str(_as_int(family_memory.get('observation_count', 0))), 'detail': 'Counts prior runs for this learned dataset family.'},
    ]
    return pd.DataFrame(rows)


def build_semantic_learning_table(
    *,
    active_control_values: dict[str, Any],
    family_key: str,
    unresolved_columns: list[str],
) -> pd.DataFrame:
    profiles = dict(active_control_values.get('semantic_mapping_profiles', {}))
    rows: list[dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        if str(profile.get('family_key', '')).strip() != family_key:
            continue
        resolved_mappings = dict(profile.get('resolved_mappings', {}))
        rows.append(
            {
                'profile_name': profile_name,
                'learning_source': 'Approved family profile',
                'resolved_fields': len(resolved_mappings),
                'suggested_next_focus': ', '.join(unresolved_columns[:4]) or 'Expand native support only if new columns appear.',
                'detail': ', '.join(
                    f"{field} <- {column}"
                    for field, column in list(resolved_mappings.items())[:4]
                ) or 'No stored resolved mappings.',
            }
        )
    return pd.DataFrame(rows)


def build_execution_backlog_summary(
    execution_queue: list[dict[str, Any]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    queue_rows = [dict(item) for item in execution_queue or [] if isinstance(item, dict)]
    queue_table = pd.DataFrame(queue_rows)
    if queue_table.empty:
        return queue_table, {
            'total_items': 0,
            'release_blockers': 0,
            'completed_items': 0,
            'open_items': 0,
            'high_priority_open': 0,
        }
    release_blockers = int(
        queue_table.get('release_gate_status', pd.Series(dtype=str))
        .astype(str)
        .isin(['Required before release', 'Blocked for release'])
        .sum()
    )
    completed_items = int(queue_table.get('status', pd.Series(dtype=str)).astype(str).eq('Completed').sum())
    open_items = int(len(queue_table) - completed_items)
    high_priority_open = int(
        (
            queue_table.get('priority', pd.Series(dtype=str)).astype(str).eq('High')
            & ~queue_table.get('status', pd.Series(dtype=str)).astype(str).eq('Completed')
        ).sum()
    )
    summary_table = pd.DataFrame(
        [
            {'signal': 'Tracked backlog items', 'value': len(queue_table)},
            {'signal': 'Release blockers', 'value': release_blockers},
            {'signal': 'Completed items', 'value': completed_items},
            {'signal': 'Open items', 'value': open_items},
            {'signal': 'High-priority open', 'value': high_priority_open},
        ]
    )
    return summary_table, {
        'total_items': int(len(queue_table)),
        'release_blockers': release_blockers,
        'completed_items': completed_items,
        'open_items': open_items,
        'high_priority_open': high_priority_open,
    }


def build_release_readiness_summary(
    *,
    execution_queue: list[dict[str, Any]] | None,
    review_history: list[dict[str, Any]] | None,
    approval_workflow: dict[str, Any] | None,
) -> pd.DataFrame:
    _, backlog_stats = build_execution_backlog_summary(execution_queue)
    approval_workflow = dict(approval_workflow or {})
    review_history = [dict(item) for item in review_history or [] if isinstance(item, dict)]
    last_review = review_history[-1] if review_history else {}
    rows = [
        {'checkpoint': 'Mapping approval', 'status': approval_workflow.get('mapping_status', 'Pending'), 'detail': 'Confirms semantic mapping is approved for this dataset context.'},
        {'checkpoint': 'Trust gate', 'status': approval_workflow.get('trust_gate_status', 'Pending'), 'detail': 'Confirms reporting thresholds and trust posture are acceptable.'},
        {'checkpoint': 'Export eligibility', 'status': approval_workflow.get('export_eligibility_status', 'Pending'), 'detail': 'Confirms outward-facing export posture for the active dataset.'},
        {'checkpoint': 'Release signoff', 'status': approval_workflow.get('release_signoff_status', 'Pending'), 'detail': 'Tracks whether the current dataset review is release-ready.'},
        {'checkpoint': 'Adaptive backlog', 'status': 'Blocked' if backlog_stats['release_blockers'] else 'Clear', 'detail': f"{backlog_stats['release_blockers']} release blockers and {backlog_stats['high_priority_open']} high-priority open items remain."},
        {'checkpoint': 'Latest governance action', 'status': last_review.get('status', 'No review history'), 'detail': str(last_review.get('action', 'No governance note recorded yet.'))},
    ]
    return pd.DataFrame(rows)


def build_drift_history_table(family_memory: dict[str, Any]) -> pd.DataFrame:
    history_rows = [dict(item) for item in family_memory.get('run_history', []) if isinstance(item, dict)]
    return pd.DataFrame(history_rows)


def build_outcome_feedback_summary(
    *,
    family_memory: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feedback_entries = [
        dict(item)
        for item in family_memory.get('outcome_feedback', [])
        if isinstance(item, dict)
    ]
    feedback_table = pd.DataFrame(feedback_entries)
    helpful_count = sum(1 for item in feedback_entries if str(item.get('feedback', '')).lower() == 'helpful')
    rejected_count = sum(1 for item in feedback_entries if str(item.get('feedback', '')).lower() in {'not_helpful', 'rejected'})
    return feedback_table, {
        'feedback_count': len(feedback_entries),
        'helpful_count': helpful_count,
        'rejected_count': rejected_count,
        'helpful_rate': round(helpful_count / max(len(feedback_entries), 1), 3) if feedback_entries else 0.0,
    }


def record_outcome_feedback(
    existing_memory: dict[str, Any] | None,
    *,
    dataset_family_key: str,
    dataset_name: str,
    feedback: str,
    surface: str,
    notes: str = '',
    reviewer: str = '',
) -> dict[str, Any]:
    memory = dict(existing_memory or {})
    family_memory = dict(memory.get('family_memory', {}))
    family_entry = dict(family_memory.get(dataset_family_key, {}))
    feedback_entries = [
        dict(item)
        for item in family_entry.get('outcome_feedback', [])
        if isinstance(item, dict)
    ]
    feedback_entries.append(
        {
            'recorded_at': _utcnow(),
            'dataset_name': dataset_name,
            'feedback': feedback,
            'surface': surface,
            'notes': str(notes).strip(),
            'reviewer': str(reviewer).strip() or 'Analyst',
        }
    )
    family_entry['outcome_feedback'] = feedback_entries[-50:]
    family_memory[dataset_family_key] = family_entry
    memory['family_memory'] = family_memory
    return memory


def append_review_history(
    review_store: dict[str, Any] | None,
    *,
    dataset_identifier: str,
    reviewer: str,
    reviewer_role: str,
    action: str,
    status: str,
    notes: str,
) -> dict[str, Any]:
    updated_store = dict(review_store or {})
    review_entry = dict(updated_store.get(dataset_identifier, {}))
    history = [dict(item) for item in review_entry.get('review_history', []) if isinstance(item, dict)]
    history.append(
        {
            'recorded_at': _utcnow(),
            'reviewer': str(reviewer).strip() or 'Analyst',
            'reviewer_role': str(reviewer_role).strip() or 'Analyst',
            'action': action,
            'status': status,
            'notes': str(notes).strip(),
        }
    )
    review_entry['review_history'] = history[-50:]
    updated_store[dataset_identifier] = review_entry
    return updated_store


def build_evolution_summary(
    *,
    dataset_name: str,
    source_meta: dict[str, Any],
    structure: StructureSummary,
    semantic: dict[str, Any],
    readiness: dict[str, Any],
    healthcare: dict[str, Any],
    trust_summary: dict[str, Any],
    accuracy_summary: dict[str, Any],
    overview: dict[str, Any],
    sample_info: dict[str, Any],
    active_control_values: dict[str, Any] | None = None,
    existing_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_control_values = active_control_values or {}
    existing_memory = existing_memory or {}
    dataset_family = dict(semantic.get('dataset_family', {}))
    family_key = str(dataset_family.get('family_key', 'generic-healthcare'))
    family_label = str(dataset_family.get('family_label', 'Generic Healthcare Feed'))
    suggested_benchmark = str(dataset_family.get('benchmark_profile', 'Generic Healthcare'))
    suggested_profile_name = str(dataset_family.get('profile_name', '') or '')
    current_benchmark = str(
        accuracy_summary.get('benchmark_profile', {}).get('profile_name')
        or active_control_values.get('accuracy_benchmark_profile', 'Auto')
    )
    manual_overrides = dict(semantic.get('manual_overrides_applied', {}))
    accounting = dict(semantic.get('column_accounting_summary', {}))
    unresolved_columns = [str(item) for item in accounting.get('unresolved_columns', []) if str(item).strip()]
    helper_columns_added = _as_int(overview.get('helper_columns_added', 0))
    readiness_score = _as_float(readiness.get('readiness_score', 0.0))
    trust_score = _as_float(trust_summary.get('trust_score', 0.0))
    sampling_active = bool(sample_info.get('sampling_applied') or trust_summary.get('sampling_active'))
    module_gates = _safe_df(accuracy_summary.get('module_reporting_gates'))
    blocked_modules = (
        module_gates.loc[module_gates.get('reporting_gate', pd.Series(dtype=str)).astype(str).isin(['Restricted', 'Blocked'])]
        if not module_gates.empty and 'reporting_gate' in module_gates.columns
        else pd.DataFrame()
    )
    goal_profile, goal_rationale = _infer_goal_profile(
        readiness=readiness,
        healthcare=healthcare,
        accuracy_summary=accuracy_summary,
        sample_info=sample_info,
    )

    opportunities: list[dict[str, Any]] = []

    if unresolved_columns:
        opportunities.append(
            {
                'improvement_area': 'Semantic coverage',
                'priority_score': round(min(0.94, 0.68 + (len(unresolved_columns) * 0.03)), 3),
                'why_it_matters': 'Source columns are still unresolved, so the app cannot fully unlock native downstream analysis.',
                'recommended_action': 'Confirm the important unresolved fields and save them as a reusable dataset-family mapping profile.',
                'auto_evolution_path': 'Promote approved manual mappings into future auto-suggestions.',
                'signal_count': len(unresolved_columns),
                'goal_alignment': goal_profile,
            }
        )
    if manual_overrides:
        opportunities.append(
            {
                'improvement_area': 'Reusable mapping intelligence',
                'priority_score': 0.79,
                'why_it_matters': 'Manual overrides already improved this run, which means the platform can learn a stronger family template.',
                'recommended_action': 'Persist the current approved mappings as a reusable profile for similar uploads.',
                'auto_evolution_path': 'Auto-suggest this profile whenever a similar dataset family appears.',
                'signal_count': len(manual_overrides),
                'goal_alignment': goal_profile,
            }
        )
    if helper_columns_added > 0:
        opportunities.append(
            {
                'improvement_area': 'Native-source dependence',
                'priority_score': round(min(0.88, 0.54 + (helper_columns_added * 0.05)), 3),
                'why_it_matters': 'Derived or synthetic helper fields are still supporting some modules.',
                'recommended_action': 'Target helper-backed concepts first when refining schema mapping or upstream extracts.',
                'auto_evolution_path': 'Keep ranking future work by helper dependence and trust impact.',
                'signal_count': helper_columns_added,
                'goal_alignment': goal_profile,
            }
        )
    if readiness_score < 0.8:
        opportunities.append(
            {
                'improvement_area': 'Readiness uplift',
                'priority_score': round(min(0.91, 0.58 + ((0.8 - readiness_score) * 0.7)), 3),
                'why_it_matters': 'Important analysis modules are still blocked or only partially available.',
                'recommended_action': 'Address the top blockers before relying on broader analytics and export workflows.',
                'auto_evolution_path': 'Build a ranked backlog of blockers by module unlock value.',
                'signal_count': _as_int(readiness.get('blocked_count', 0)) + _as_int(readiness.get('partial_count', 0)),
                'goal_alignment': goal_profile,
            }
        )
    if trust_score < 0.82:
        opportunities.append(
            {
                'improvement_area': 'Trust strengthening',
                'priority_score': round(min(0.93, 0.62 + ((0.82 - trust_score) * 0.9)), 3),
                'why_it_matters': 'The current run still has trust constraints that should influence reporting posture.',
                'recommended_action': 'Use the trust notes and support-type gaps as the next hardening backlog.',
                'auto_evolution_path': 'Feed repeated trust gaps into future benchmark and reporting defaults.',
                'signal_count': len(list(trust_summary.get('trust_notes', []))),
                'goal_alignment': goal_profile,
            }
        )
    if sampling_active:
        opportunities.append(
            {
                'improvement_area': 'Full-run promotion',
                'priority_score': 0.71,
                'why_it_matters': 'This run used sampled interactive scope, which is faster but weaker for final external reporting.',
                'recommended_action': 'Promote this dataset to a full local or release validation run before final distribution.',
                'auto_evolution_path': 'Automatically flag sampled datasets for later full-run confirmation.',
                'signal_count': _as_int(sample_info.get('analyzed_rows', 0)),
                'goal_alignment': goal_profile,
            }
        )
    if current_benchmark in {'Auto', 'Generic Healthcare'} and suggested_benchmark not in {'', 'Generic Healthcare'}:
        opportunities.append(
            {
                'improvement_area': 'Benchmark specialization',
                'priority_score': 0.74,
                'why_it_matters': 'The dataset family already suggests a more specific benchmark pack than the current baseline.',
                'recommended_action': f'Adopt the {suggested_benchmark} benchmark profile for tighter calibration.',
                'auto_evolution_path': 'Recommend the family-specific benchmark pack whenever the active profile is generic.',
                'signal_count': 1,
                'goal_alignment': goal_profile,
            }
        )
    if not blocked_modules.empty:
        opportunities.append(
            {
                'improvement_area': 'External reporting safety',
                'priority_score': round(min(0.9, 0.6 + (len(blocked_modules) * 0.04)), 3),
                'why_it_matters': 'Some modules are still restricted for external reporting.',
                'recommended_action': 'Review the module-level reporting gates before approving outward-facing deliverables.',
                'auto_evolution_path': 'Turn repeated gate failures into approval defaults and checklist policy.',
                'signal_count': len(blocked_modules),
                'goal_alignment': goal_profile,
            }
        )

    opportunities = sorted(opportunities, key=lambda row: float(row.get('priority_score', 0.0)), reverse=True)
    for row in opportunities:
        row['priority'] = _priority_label(float(row.get('priority_score', 0.0)))
    opportunities_table = pd.DataFrame(opportunities)

    family_memory = dict(existing_memory.get('family_memory', {})).get(family_key, {}) if isinstance(existing_memory, dict) else {}
    execution_queue = list(existing_memory.get('execution_queue', [])) if isinstance(existing_memory, dict) else []
    pattern_counts = dict(family_memory.get('pattern_counts', {}))
    goal_counts = dict(family_memory.get('goal_counts', {}))
    family_memory_with_current = {
        **family_memory,
        'current_total_source_columns': _as_int(accounting.get('total_source_columns', 0), 0),
        'current_strongly_mapped_columns': _as_int(accounting.get('strongly_mapped_columns', 0), 0),
        'current_blocked_reporting_module_count': len(blocked_modules),
        'current_benchmark_profile': current_benchmark,
    }
    validation_recommendations = build_validation_recommendations(
        readiness=readiness,
        trust_summary=trust_summary,
        sample_info=sample_info,
        module_reporting_gates=module_gates,
    )
    drift_alerts = build_drift_alerts(
        family_memory_entry=family_memory_with_current,
        readiness_score=readiness_score,
        trust_score=trust_score,
        helper_columns_added=helper_columns_added,
        goal_profile=goal_profile,
    )
    field_remediation = build_data_remediation_assistant(structure, semantic, readiness)
    feedback_table, feedback_summary = build_outcome_feedback_summary(family_memory=family_memory)
    backlog_summary_table, backlog_summary = build_execution_backlog_summary(execution_queue)
    drift_history_table = build_drift_history_table(family_memory)
    family_intelligence_table = build_family_intelligence_table(
        dataset_family_label=family_label,
        family_key=family_key,
        suggested_benchmark=suggested_benchmark,
        current_benchmark=current_benchmark,
        suggested_profile_name=suggested_profile_name,
        accounting=accounting,
        family_memory=family_memory,
        semantic=semantic,
    )
    semantic_learning_table = build_semantic_learning_table(
        active_control_values=active_control_values,
        family_key=family_key,
        unresolved_columns=unresolved_columns,
    )
    recurring_patterns = [
        {
            'pattern': area,
            'current_hits': count,
            'historical_hits': _as_int(pattern_counts.get(area, 0)),
        }
        for area, count in Counter(str(row.get('improvement_area', '')) for row in opportunities).items()
    ]
    recurring_patterns = sorted(recurring_patterns, key=lambda row: (row['historical_hits'], row['current_hits']), reverse=True)

    auto_actions: list[dict[str, Any]] = []
    if manual_overrides:
        auto_actions.append(
            {
                'action': 'Promote approved overrides',
                'status': 'Ready',
                'automation': 'Persist current manual mappings as a reusable dataset-family profile.',
            }
        )
    if current_benchmark in {'Auto', 'Generic Healthcare'} and suggested_benchmark not in {'', 'Generic Healthcare'}:
        auto_actions.append(
            {
                'action': 'Specialize benchmark pack',
                'status': 'Ready',
                'automation': f'Suggest {suggested_benchmark} whenever this dataset family appears.',
            }
        )
    if sampling_active:
        auto_actions.append(
            {
                'action': 'Queue full validation follow-up',
                'status': 'Watch',
                'automation': 'Flag sampled runs for later full validation before final external reporting.',
            }
        )

    proposal_queue: list[dict[str, Any]] = []
    for row in opportunities[:5]:
        priority = str(row.get('priority', 'Medium'))
        proposal_queue.append(
            {
                'proposal_title': f"{row.get('improvement_area', 'Improvement')} upgrade",
                'proposal_type': 'Adaptive spec',
                'priority': priority,
                'severity': 'Critical' if priority == 'High' else 'Moderate' if priority == 'Medium' else 'Low',
                'proposed_change': row.get('recommended_action', ''),
                'suggested_validation': row.get('auto_evolution_path', ''),
                'release_impact': 'Release blocker' if 'report' in str(row.get('improvement_area', '')).lower() or 'validation' in str(row.get('recommended_action', '')).lower() else 'Release hardening',
                'suggested_owner': 'Data Steward' if 'mapping' in str(row.get('recommended_action', '')).lower() else 'Analyst',
                'suggested_test': 'Quick validation' if priority != 'High' else 'Release validation',
            }
        )

    summary_text = (
        f"Clinverity is actively learning from the {family_label} pattern for {dataset_name}. "
        f"Its primary goal for this dataset is {goal_profile.lower()}. "
        f"It found {len(opportunities)} ranked improvement opportunities, {len(auto_actions)} safe adaptive actions, "
        f"and {len(proposal_queue)} draft upgrade proposals."
    )

    return {
        'available': True,
        'dataset_family_key': family_key,
        'dataset_family_label': family_label,
        'goal_profile': goal_profile,
        'goal_rationale': goal_rationale,
        'suggested_benchmark_profile': suggested_benchmark,
        'current_benchmark_profile': current_benchmark,
        'total_source_columns': _as_int(accounting.get('total_source_columns', 0), 0),
        'strongly_mapped_columns': _as_int(accounting.get('strongly_mapped_columns', 0), 0),
        'readiness_score': readiness_score,
        'trust_score': trust_score,
        'summary_text': summary_text,
        'opportunity_count': len(opportunities),
        'high_priority_count': int(sum(1 for row in opportunities if row.get('priority') == 'High')),
        'helper_gap_count': helper_columns_added,
        'recurring_pattern_count': len(recurring_patterns),
        'prior_observation_count': _as_int(family_memory.get('observation_count', 0)),
        'historical_goal_hits': _as_int(goal_counts.get(goal_profile, 0)),
        'opportunities_table': opportunities_table,
        'recurring_patterns_table': pd.DataFrame(recurring_patterns),
        'auto_actions_table': pd.DataFrame(auto_actions),
        'proposal_queue_table': pd.DataFrame(proposal_queue),
        'validation_recommendations_table': validation_recommendations,
        'drift_alerts_table': drift_alerts,
        'field_remediation_table': field_remediation,
        'family_intelligence_table': family_intelligence_table,
        'semantic_learning_table': semantic_learning_table,
        'outcome_feedback_table': feedback_table,
        'outcome_feedback_summary': feedback_summary,
        'backlog_summary_table': backlog_summary_table,
        'backlog_summary': backlog_summary,
        'drift_history_table': drift_history_table,
        'manual_override_count': len(manual_overrides),
        'unresolved_column_count': len(unresolved_columns),
        'blocked_reporting_module_count': len(blocked_modules),
    }


def merge_evolution_memory(
    existing_memory: dict[str, Any] | None,
    summary: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    memory = dict(existing_memory or {})
    family_memory = dict(memory.get('family_memory', {}))
    family_key = str(summary.get('dataset_family_key', 'generic-healthcare'))
    family_entry = dict(family_memory.get(family_key, {}))
    pattern_counter = Counter(dict(family_entry.get('pattern_counts', {})))
    opportunities = _safe_df(summary.get('opportunities_table'))
    if not opportunities.empty and 'improvement_area' in opportunities.columns:
        pattern_counter.update(opportunities['improvement_area'].astype(str).tolist())
    family_entry['family_label'] = str(summary.get('dataset_family_label', family_key))
    family_entry['last_dataset_name'] = dataset_name
    family_entry['observation_count'] = _as_int(family_entry.get('observation_count', 0)) + 1
    family_entry['pattern_counts'] = dict(pattern_counter)
    goal_counter = Counter(dict(family_entry.get('goal_counts', {})))
    goal_counter.update([str(summary.get('goal_profile', 'Healthcare intelligence expansion'))])
    family_entry['goal_counts'] = dict(goal_counter)
    run_history = [dict(item) for item in family_entry.get('run_history', []) if isinstance(item, dict)]
    run_history.append(
        {
            'recorded_at': _utcnow(),
            'dataset_name': dataset_name,
            'goal_profile': str(summary.get('goal_profile', 'Healthcare intelligence expansion')),
            'readiness_score': _as_float(summary.get('readiness_score', 0.0), 0.0),
            'trust_score': _as_float(summary.get('trust_score', 0.0), 0.0),
            'helper_columns_added': _as_int(summary.get('helper_gap_count', 0), 0),
            'blocked_reporting_module_count': _as_int(summary.get('blocked_reporting_module_count', 0), 0),
            'benchmark_profile': str(summary.get('current_benchmark_profile', '')),
        }
    )
    family_entry['run_history'] = run_history[-50:]
    family_entry['last_snapshot'] = {
        'readiness_score': _as_float(summary.get('readiness_score', 0.0), 0.0),
        'trust_score': _as_float(summary.get('trust_score', 0.0), 0.0),
        'helper_columns_added': _as_int(summary.get('helper_gap_count', 0), 0),
        'goal_profile': str(summary.get('goal_profile', 'Healthcare intelligence expansion')),
        'total_source_columns': _as_int(summary.get('total_source_columns', 0), 0),
        'strongly_mapped_columns': _as_int(summary.get('strongly_mapped_columns', 0), 0),
        'blocked_reporting_module_count': _as_int(summary.get('blocked_reporting_module_count', 0), 0),
        'benchmark_profile': str(summary.get('current_benchmark_profile', '')),
    }
    family_entry['last_summary_text'] = str(summary.get('summary_text', ''))
    family_memory[family_key] = family_entry
    memory['family_memory'] = family_memory
    memory.setdefault('execution_queue', [])
    memory['last_dataset_name'] = dataset_name
    memory['last_family_key'] = family_key
    return memory
