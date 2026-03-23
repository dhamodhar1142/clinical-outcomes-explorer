from __future__ import annotations

import json
from typing import Any


CORE_STATE_KEYS = [
    'dataset_source_mode',
    'demo_dataset_name',
    'analysis_template',
    'report_mode',
    'export_policy_name',
    'active_role',
    'accuracy_benchmark_profile',
    'active_benchmark_pack_name',
    'accuracy_reporting_threshold_profile',
    'accuracy_reporting_min_trust_score',
    'accuracy_allow_directional_external_reporting',
    'organization_benchmark_packs',
    'semantic_mapping_profiles',
    'dataset_review_approvals',
    'active_plan',
    'plan_enforcement_mode',
    'workflow_action_prompt',
    'selected_workflow_pack',
    'selected_snapshot',
    'demo_synthetic_helper_mode',
    'demo_bmi_remediation_mode',
    'demo_synthetic_cost_mode',
    'demo_synthetic_readmission_mode',
    'demo_executive_summary_verbosity',
    'demo_scenario_simulation_mode',
]

STATE_PREFIXES = [
    'cohort_',
    'readmit_',
    'modeling_',
]

MAX_PORTABLE_COLLECTION_ITEMS = 100


def _coerce_string(value: Any, *, max_len: int = 200) -> str:
    return str(value or '').strip()[:max_len]


def _restore_prefixed_state(key: str, value: Any) -> tuple[bool, Any, str | None]:
    if key == 'cohort_age_range':
        if isinstance(value, list) and len(value) == 2:
            try:
                low = int(float(value[0]))
                high = int(float(value[1]))
            except (TypeError, ValueError):
                return False, None, 'Cohort age range was skipped because it did not contain valid numeric values.'
            low = max(0, min(low, 120))
            high = max(low, min(high, 120))
            return True, (low, high), None
        return False, None, 'Cohort age range was skipped because it was not saved in a valid range format.'
    if key in {'readmit_follow_up', 'readmit_case_mgmt', 'readmit_early_follow_up'}:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            return False, None, f'{key.replace("_", " ").title()} was skipped because it was not a valid percentage.'
        return True, max(0, min(numeric, 50)), None
    if key == 'readmit_los_reduction':
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False, None, 'Readmission LOS reduction was skipped because it was not a valid number.'
        return True, max(0.0, min(numeric, 3.0)), None
    if key in {'workflow_action_prompt', 'modeling_target_column'}:
        return True, _coerce_string(value, max_len=250), None
    if key.startswith('modeling_') and isinstance(value, list):
        cleaned = [_coerce_string(item, max_len=120) for item in value if _coerce_string(item, max_len=120)]
        return True, cleaned[:MAX_PORTABLE_COLLECTION_ITEMS], None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True, value, None
    return False, None, f'{key.replace("_", " ").title()} was skipped because the saved value was not portable.'


def _restore_collection(bundle_value: Any, expected_type: type, label: str) -> tuple[bool, Any, str | None]:
    if not isinstance(bundle_value, expected_type):
        return False, None, f'{label} was skipped because the imported bundle did not contain the expected structure.'
    if expected_type is dict:
        items = list(bundle_value.items())[:MAX_PORTABLE_COLLECTION_ITEMS]
        return True, {str(key): value for key, value in items}, None
    items = list(bundle_value)[:MAX_PORTABLE_COLLECTION_ITEMS]
    return True, items, None


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _looks_portable_key(key: str) -> bool:
    return key in CORE_STATE_KEYS or any(key.startswith(prefix) for prefix in STATE_PREFIXES)


def build_session_export_bundle(
    session_state: dict[str, Any],
    dataset_name: str,
    source_meta: dict[str, Any],
    workspace_identity: dict[str, Any],
) -> dict[str, Any]:
    portable_state = {
        key: _json_safe(value)
        for key, value in session_state.items()
        if _looks_portable_key(key)
    }
    return {
        'bundle_type': 'smart_dataset_analyzer_session',
        'bundle_version': 1,
        'dataset_context': {
            'dataset_name': dataset_name,
            'source_mode': source_meta.get('source_mode', 'Unknown'),
            'description': source_meta.get('description', ''),
        },
        'workspace_context': {
            'workspace_name': workspace_identity.get('workspace_name', 'Guest Demo Workspace'),
            'display_name': workspace_identity.get('display_name', 'Guest User'),
        },
        'portable_state': portable_state,
        'saved_snapshots': _json_safe(session_state.get('saved_snapshots', {})),
        'workflow_packs': _json_safe(session_state.get('workflow_packs', {})),
        'collaboration_notes': _json_safe(session_state.get('collaboration_notes', [])),
        'beta_interest_submissions': _json_safe(session_state.get('beta_interest_submissions', [])),
    }


def build_session_export_text(bundle: dict[str, Any]) -> bytes:
    return json.dumps(bundle, indent=2).encode('utf-8')


def parse_session_import(raw_text: str) -> dict[str, Any]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict) or payload.get('bundle_type') != 'smart_dataset_analyzer_session':
        raise ValueError('The selected file is not a valid Clinverity session bundle.')
    if int(payload.get('bundle_version', 1) or 1) > 1:
        raise ValueError('This session bundle was created by a newer version of Clinverity and cannot be restored safely here.')
    return payload


def restore_session_bundle(
    bundle: dict[str, Any],
    session_state: dict[str, Any],
    valid_options: dict[str, list[str]],
) -> dict[str, Any]:
    portable_state = bundle.get('portable_state', {})
    applied_keys: list[str] = []
    skipped_keys: list[str] = []
    notes: list[str] = []

    workspace_context = bundle.get('workspace_context', {})
    if workspace_context.get('display_name'):
        session_state['workspace_user_name'] = workspace_context['display_name']
        applied_keys.append('workspace_user_name')
    if workspace_context.get('workspace_name'):
        session_state['workspace_name'] = workspace_context['workspace_name']
        applied_keys.append('workspace_name')

    for key, value in portable_state.items():
        if key == 'dataset_source_mode':
            if value in valid_options.get('dataset_source_mode', []):
                session_state[key] = value
                applied_keys.append(key)
            else:
                skipped_keys.append(key)
        elif key == 'demo_dataset_name':
            if value in valid_options.get('demo_dataset_name', []):
                session_state[key] = value
                applied_keys.append(key)
            else:
                skipped_keys.append(key)
                notes.append('The saved demo dataset is not available in this environment, so the current dataset selection was kept.')
        elif key in {'analysis_template', 'report_mode', 'export_policy_name', 'active_role', 'active_plan', 'plan_enforcement_mode'}:
            if value in valid_options.get(key, []):
                session_state[key] = value
                applied_keys.append(key)
            else:
                skipped_keys.append(key)
        elif key in {'selected_snapshot', 'selected_workflow_pack'}:
            skipped_keys.append(key)
        elif any(key.startswith(prefix) for prefix in STATE_PREFIXES) or key == 'workflow_action_prompt':
            restored, restored_value, note = _restore_prefixed_state(key, value)
            if restored:
                session_state[key] = restored_value
                applied_keys.append(key)
            else:
                skipped_keys.append(key)
                if note:
                    notes.append(note)
        else:
            session_state[key] = value
            applied_keys.append(key)

    snapshots_ok, snapshots_value, snapshots_note = _restore_collection(bundle.get('saved_snapshots', {}), dict, 'Saved snapshots')
    if snapshots_ok:
        session_state['saved_snapshots'] = snapshots_value
        applied_keys.append('saved_snapshots')
    elif snapshots_note:
        notes.append(snapshots_note)
    packs_ok, packs_value, packs_note = _restore_collection(bundle.get('workflow_packs', {}), dict, 'Workflow packs')
    if packs_ok:
        session_state['workflow_packs'] = packs_value
        applied_keys.append('workflow_packs')
    elif packs_note:
        notes.append(packs_note)
    notes_ok, collaboration_value, collaboration_note = _restore_collection(bundle.get('collaboration_notes', []), list, 'Collaboration notes')
    if notes_ok:
        session_state['collaboration_notes'] = collaboration_value
        applied_keys.append('collaboration_notes')
    elif collaboration_note:
        notes.append(collaboration_note)
    beta_ok, beta_value, beta_note = _restore_collection(bundle.get('beta_interest_submissions', []), list, 'Beta interest submissions')
    if beta_ok:
        session_state['beta_interest_submissions'] = beta_value
        applied_keys.append('beta_interest_submissions')
    elif beta_note:
        notes.append(beta_note)

    selected_snapshot = portable_state.get('selected_snapshot')
    if selected_snapshot in session_state.get('saved_snapshots', {}):
        session_state['selected_snapshot'] = selected_snapshot
        applied_keys.append('selected_snapshot')
    elif selected_snapshot:
        notes.append('The saved snapshot selection was not restored because that snapshot is not available in the imported bundle.')

    selected_pack = portable_state.get('selected_workflow_pack')
    if selected_pack in session_state.get('workflow_packs', {}):
        session_state['selected_workflow_pack'] = selected_pack
        applied_keys.append('selected_workflow_pack')
    elif selected_pack:
        notes.append('The saved workflow pack selection was not restored because that workflow pack is not available in the imported bundle.')

    dataset_context = bundle.get('dataset_context', {})
    if dataset_context.get('source_mode') == 'Uploaded dataset':
        notes.append('Uploaded file contents are not embedded in the session bundle, so upload-based datasets still need to be provided again before running analysis.')

    return {
        'applied_keys': applied_keys,
        'skipped_keys': skipped_keys,
        'notes': notes,
    }
