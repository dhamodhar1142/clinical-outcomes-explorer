from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.governance_portability import (
    build_governance_release_bundle,
    build_governance_release_bundle_compatibility_gate,
    build_governance_release_bundle_drift,
    build_governance_release_bundle_gate,
    build_governance_release_bundle_promotion_readiness,
    build_governance_release_bundle_runtime_compatibility,
    build_governance_release_bundle_bytes,
    parse_governance_release_bundle,
    restore_governance_release_bundle,
)
from src.evolution_engine import append_review_history
from src.export_orchestrator import build_export_execution_plan, build_export_runtime_profile
from src.modules.privacy_security import (
    REDACTION_LEVEL_OPTIONS,
    WORKSPACE_EXPORT_ACCESS_OPTIONS,
    get_export_policy_presets,
)
from src.result_accuracy import BENCHMARK_PROFILES, REPORTING_THRESHOLD_PRESETS
from src.ui_components import render_section_intro, render_subsection_header, render_surface_panel
from src.validation_orchestrator import build_validation_execution_plan, build_validation_runtime_profile
from ui.common import info_or_table, safe_df


ROLE_OPTIONS = ['Admin', 'Analyst', 'Executive', 'Clinician', 'Researcher', 'Data Steward', 'Viewer']
RELEASE_GATE_OPTIONS = ['Strict signoff', 'Standard signoff', 'Advisory']
RUNTIME_PREFERENCE_OPTIONS = ['Auto', 'Prefer local/staging for heavy actions', 'Allow lightweight cloud actions only']
ROUTING_CHECKPOINTS = [
    ('mapping_approval', 'Mapping approval'),
    ('trust_gate', 'Trust gate'),
    ('export_eligibility', 'Export eligibility'),
    ('release_signoff', 'Release signoff'),
    ('governance_note', 'Governance note'),
]


def _persist(session_state: dict[str, Any]) -> None:
    application_service = session_state.get('application_service')
    if application_service is not None:
        application_service.persist_user_settings(session_state)


def _current_policy_pack(session_state: dict[str, Any]) -> dict[str, Any]:
    return {
        'export_policy_name': str(session_state.get('export_policy_name', 'Internal Review')),
        'active_benchmark_pack_name': str(session_state.get('active_benchmark_pack_name', 'None')),
        'accuracy_reporting_threshold_profile': str(session_state.get('accuracy_reporting_threshold_profile', 'Role default')),
        'accuracy_reporting_min_trust_score': float(session_state.get('accuracy_reporting_min_trust_score', 0.76)),
        'accuracy_allow_directional_external_reporting': bool(session_state.get('accuracy_allow_directional_external_reporting', False)),
        'workspace_governance_redaction_level': str(session_state.get('workspace_governance_redaction_level', 'Low')),
        'workspace_governance_export_access': str(session_state.get('workspace_governance_export_access', 'Editors and owners')),
        'workspace_governance_watermark_sensitive_exports': bool(session_state.get('workspace_governance_watermark_sensitive_exports', False)),
        'governance_default_owner': str(session_state.get('governance_default_owner', 'Analyst')),
        'governance_default_reviewer_role': str(session_state.get('governance_default_reviewer_role', 'Data Steward')),
        'governance_release_gate_mode': str(session_state.get('governance_release_gate_mode', 'Standard signoff')),
        'validation_runtime_preference': str(session_state.get('validation_runtime_preference', 'Auto')),
        'export_runtime_preference': str(session_state.get('export_runtime_preference', 'Auto')),
        'approval_routing_rules': dict(session_state.get('approval_routing_rules', {})),
    }


def _apply_policy_pack(session_state: dict[str, Any], pack: dict[str, Any]) -> None:
    for key, value in _current_policy_pack(session_state).items():
        if key in pack:
            session_state[key] = pack[key]


def _selected_policy_pack_name(policy_packs: dict[str, Any], selected_policy_pack: str) -> str:
    if policy_packs and selected_policy_pack in policy_packs:
        return selected_policy_pack
    return 'Current Policy Pack'


def render_policy_center(pipeline: dict[str, Any], dataset_name: str, source_meta: dict[str, Any]) -> None:
    render_section_intro(
        'Policy Center',
        'Manage benchmark packs, reporting thresholds, approval routing, and runtime gate posture from one governed control surface.',
    )
    application_service = st.session_state.get('application_service')
    role = str(st.session_state.get('active_role') or st.session_state.get('workspace_role') or 'Analyst')
    plan_awareness = pipeline.get('plan_awareness', {})
    strict_plan = bool(plan_awareness.get('strict_enforcement'))
    active_plan = str(plan_awareness.get('active_plan', 'Pro'))
    export_allowed = bool(pipeline.get('export_summary', {}).get('available', True))
    advanced_exports_allowed = export_allowed and (active_plan != 'Starter' or not strict_plan)
    governance_exports_allowed = export_allowed and (active_plan in {'Pro', 'Enterprise'} or not strict_plan)
    stakeholder_bundle_allowed = export_allowed and (active_plan != 'Starter' or not strict_plan)

    render_surface_panel(
        'Governed controls',
        'These settings shape how Clinverity calibrates benchmarks, gates reporting, routes approvals, and explains what should run here versus locally.',
        tone='info',
    )

    render_subsection_header('Benchmark pack manager')
    org_packs = dict(st.session_state.get('organization_benchmark_packs', {}))
    policy_packs = dict(st.session_state.get('governance_policy_packs', {}))
    combined_profiles = {**BENCHMARK_PROFILES, **org_packs}
    info_or_table(
        pd.DataFrame(
            [
                {
                    'benchmark_pack': name,
                    'scope': 'Organization pack' if name in org_packs else 'Built-in',
                    'detail': str(config.get('detail_note', '')),
                }
                for name, config in combined_profiles.items()
            ]
        ),
        'Benchmark packs will appear here when built-in or organization-specific packs are available.',
    )
    current_pack = str(st.session_state.get('active_benchmark_pack_name', 'None'))
    selected_pack = st.selectbox(
        'Active benchmark pack',
        ['None'] + list(combined_profiles.keys()),
        index=(['None'] + list(combined_profiles.keys())).index(current_pack) if current_pack in ['None'] + list(combined_profiles.keys()) else 0,
        key='policy_center_active_benchmark_pack',
    )
    if st.button('Apply Benchmark Pack', key='policy_center_apply_benchmark_pack'):
        st.session_state['active_benchmark_pack_name'] = selected_pack
        _persist(st.session_state)
        st.success(f"Active benchmark pack set to '{selected_pack}'.")
        st.rerun()

    with st.expander('Create organization benchmark pack', expanded=False):
        template_name = st.selectbox('Starting template', list(BENCHMARK_PROFILES.keys()), key='policy_center_benchmark_template')
        pack_name = st.text_input('New benchmark pack name', key='policy_center_new_pack_name', placeholder='Example: North Texas Encounter Benchmarks')
        template = dict(BENCHMARK_PROFILES.get(template_name, {}))
        rate_bands = dict(template.get('rate_bands', {}))
        numeric_bands = dict(template.get('numeric_bands', {}))
        col_a, col_b = st.columns(2)
        readmit_low = col_a.number_input('Readmission rate low', min_value=0.0, max_value=1.0, value=float(rate_bands.get('Readmission rate', (0.05, 0.35))[0]), step=0.01, key='policy_center_readmit_low')
        readmit_high = col_b.number_input('Readmission rate high', min_value=0.0, max_value=1.0, value=float(rate_bands.get('Readmission rate', (0.05, 0.35))[1]), step=0.01, key='policy_center_readmit_high')
        los_low = col_a.number_input('Average LOS low', min_value=0.0, value=float(numeric_bands.get('Average length of stay', (1.0, 12.0))[0]), step=0.5, key='policy_center_los_low')
        los_high = col_b.number_input('Average LOS high', min_value=0.0, value=float(numeric_bands.get('Average length of stay', (1.0, 12.0))[1]), step=0.5, key='policy_center_los_high')
        detail_note = st.text_area('Benchmark pack note', value=str(template.get('detail_note', '')), key='policy_center_pack_note', height=90)
        if st.button('Save Benchmark Pack', key='policy_center_save_pack', disabled=not pack_name.strip()):
            org_packs[pack_name.strip()] = {
                'profile_family': ''.join(ch.lower() if ch.isalnum() else '-' for ch in pack_name.strip()).strip('-') or 'custom-benchmark-pack',
                'rate_bands': {
                    'Readmission rate': (float(readmit_low), float(readmit_high)),
                    'High-risk share': tuple(rate_bands.get('High-risk share', (0.05, 0.50))),
                },
                'numeric_bands': {
                    'Average length of stay': (float(los_low), float(los_high)),
                    'Average cost': tuple(numeric_bands.get('Average cost', (500.0, 100000.0))),
                },
                'detail_note': detail_note.strip() or f'Organization-specific benchmark pack derived from {template_name}.',
            }
            st.session_state['organization_benchmark_packs'] = org_packs
            _persist(st.session_state)
            st.success(f"Saved organization benchmark pack '{pack_name.strip()}'.")
            st.rerun()

    render_subsection_header('Policy pack library')
    policy_pack_table = pd.DataFrame(
        [
            {
                'policy_pack': name,
                'export_policy': str(config.get('export_policy_name', 'Internal Review')),
                'benchmark_pack': str(config.get('active_benchmark_pack_name', 'None')),
                'threshold_profile': str(config.get('accuracy_reporting_threshold_profile', 'Role default')),
                'default_owner': str(config.get('governance_default_owner', 'Analyst')),
            }
            for name, config in policy_packs.items()
            if isinstance(config, dict)
        ]
    )
    info_or_table(
        policy_pack_table,
        'Saved policy packs will appear here once the current governance posture is captured for reuse.',
    )
    policy_pack_names = list(policy_packs.keys())
    selected_policy_pack = st.selectbox(
        'Saved policy pack',
        policy_pack_names or ['No saved packs'],
        key='policy_center_selected_policy_pack',
        disabled=not policy_pack_names,
    )
    pack_cols = st.columns(3)
    if pack_cols[0].button('Apply Policy Pack', key='policy_center_apply_policy_pack', disabled=not policy_pack_names):
        _apply_policy_pack(st.session_state, dict(policy_packs.get(selected_policy_pack, {})))
        _persist(st.session_state)
        st.success(f"Applied policy pack '{selected_policy_pack}'.")
        st.rerun()
    if pack_cols[1].download_button(
        'Export Policy Pack JSON',
        data=json.dumps(dict(policy_packs.get(selected_policy_pack, {})), indent=2).encode('utf-8') if policy_pack_names else b'',
        file_name=f"{str(selected_policy_pack).replace(' ', '_').lower()}_policy_pack.json",
        mime='application/json',
        disabled=not policy_pack_names,
        key='policy_center_export_policy_pack',
    ):
        pass
    if pack_cols[2].button('Delete Policy Pack', key='policy_center_delete_policy_pack', disabled=not policy_pack_names):
        policy_packs.pop(selected_policy_pack, None)
        st.session_state['governance_policy_packs'] = policy_packs
        _persist(st.session_state)
        st.success(f"Deleted policy pack '{selected_policy_pack}'.")
        st.rerun()
    with st.expander('Create or import policy pack', expanded=False):
        new_policy_pack_name = st.text_input('Policy pack name', key='policy_center_new_policy_pack_name', placeholder='Example: Enterprise Release Controls')
        if st.button('Save Current Settings as Policy Pack', key='policy_center_save_policy_pack', disabled=not new_policy_pack_name.strip()):
            policy_packs[new_policy_pack_name.strip()] = _current_policy_pack(st.session_state)
            st.session_state['governance_policy_packs'] = policy_packs
            _persist(st.session_state)
            st.success(f"Saved policy pack '{new_policy_pack_name.strip()}'.")
            st.rerun()
        imported_json = st.text_area(
            'Import policy pack JSON',
            key='policy_center_import_policy_pack_json',
            height=140,
            placeholder='Paste a previously exported policy pack JSON payload here.',
        )
        if st.button('Import Policy Pack', key='policy_center_import_policy_pack', disabled=not imported_json.strip()):
            try:
                imported_pack = json.loads(imported_json)
                pack_name = str(imported_pack.get('policy_pack_name', '')).strip() or new_policy_pack_name.strip() or 'Imported Policy Pack'
                sanitized_pack = dict(_current_policy_pack(st.session_state))
                for key in sanitized_pack:
                    if key in imported_pack:
                        sanitized_pack[key] = imported_pack[key]
                policy_packs[pack_name] = sanitized_pack
                st.session_state['governance_policy_packs'] = policy_packs
                _persist(st.session_state)
                st.success(f"Imported policy pack '{pack_name}'.")
                st.rerun()
            except json.JSONDecodeError:
                st.error('The policy pack JSON could not be parsed. Check the copied payload and try again.')

    render_subsection_header('Reporting threshold policy')
    threshold_profile = str(st.session_state.get('accuracy_reporting_threshold_profile', 'Role default'))
    threshold_options = ['Role default'] + list(REPORTING_THRESHOLD_PRESETS.keys())
    selected_threshold = st.selectbox(
        'Threshold profile',
        threshold_options,
        index=threshold_options.index(threshold_profile) if threshold_profile in threshold_options else 0,
        key='policy_center_threshold_profile',
    )
    min_trust = st.slider(
        'Minimum trust score for external reporting',
        min_value=0.50,
        max_value=0.95,
        value=float(st.session_state.get('accuracy_reporting_min_trust_score', 0.76)),
        step=0.01,
        key='policy_center_min_trust',
    )
    directional = st.toggle(
        'Allow directional external reporting',
        value=bool(st.session_state.get('accuracy_allow_directional_external_reporting', False)),
        key='policy_center_directional_external',
    )
    if st.button('Apply Reporting Policy', key='policy_center_apply_reporting_policy'):
        st.session_state['accuracy_reporting_threshold_profile'] = selected_threshold
        st.session_state['accuracy_reporting_min_trust_score'] = float(min_trust)
        st.session_state['accuracy_allow_directional_external_reporting'] = bool(directional)
        _persist(st.session_state)
        st.success('Reporting threshold policy updated.')
        st.rerun()

    render_subsection_header('Approval routing and governance defaults')
    default_owner = st.selectbox(
        'Default adaptive work owner',
        ROLE_OPTIONS,
        index=ROLE_OPTIONS.index(str(st.session_state.get('governance_default_owner', 'Analyst'))) if str(st.session_state.get('governance_default_owner', 'Analyst')) in ROLE_OPTIONS else ROLE_OPTIONS.index('Analyst'),
        key='policy_center_default_owner',
    )
    default_reviewer = st.selectbox(
        'Default reviewer role',
        ROLE_OPTIONS,
        index=ROLE_OPTIONS.index(str(st.session_state.get('governance_default_reviewer_role', 'Data Steward'))) if str(st.session_state.get('governance_default_reviewer_role', 'Data Steward')) in ROLE_OPTIONS else ROLE_OPTIONS.index('Data Steward'),
        key='policy_center_default_reviewer',
    )
    release_gate_mode = st.selectbox(
        'Release gate mode',
        RELEASE_GATE_OPTIONS,
        index=RELEASE_GATE_OPTIONS.index(str(st.session_state.get('governance_release_gate_mode', 'Standard signoff'))) if str(st.session_state.get('governance_release_gate_mode', 'Standard signoff')) in RELEASE_GATE_OPTIONS else RELEASE_GATE_OPTIONS.index('Standard signoff'),
        key='policy_center_release_gate_mode',
    )
    export_policy_name = st.selectbox(
        'Default export policy',
        list(get_export_policy_presets()['policy_name'].tolist()),
        index=list(get_export_policy_presets()['policy_name'].tolist()).index(str(st.session_state.get('export_policy_name', 'Internal Review'))) if str(st.session_state.get('export_policy_name', 'Internal Review')) in list(get_export_policy_presets()['policy_name'].tolist()) else 0,
        key='policy_center_export_policy_name',
    )
    redaction_level = st.selectbox(
        'Workspace redaction level',
        REDACTION_LEVEL_OPTIONS,
        index=REDACTION_LEVEL_OPTIONS.index(str(st.session_state.get('workspace_governance_redaction_level', 'Low'))) if str(st.session_state.get('workspace_governance_redaction_level', 'Low')) in REDACTION_LEVEL_OPTIONS else 0,
        key='policy_center_redaction_level',
    )
    export_access = st.selectbox(
        'Workspace export access',
        WORKSPACE_EXPORT_ACCESS_OPTIONS,
        index=WORKSPACE_EXPORT_ACCESS_OPTIONS.index(str(st.session_state.get('workspace_governance_export_access', 'Editors and owners'))) if str(st.session_state.get('workspace_governance_export_access', 'Editors and owners')) in WORKSPACE_EXPORT_ACCESS_OPTIONS else WORKSPACE_EXPORT_ACCESS_OPTIONS.index('Editors and owners'),
        key='policy_center_export_access',
    )
    watermark_sensitive = st.toggle(
        'Watermark sensitive exports',
        value=bool(st.session_state.get('workspace_governance_watermark_sensitive_exports', False)),
        key='policy_center_watermark_sensitive',
    )
    if st.button('Apply Governance Defaults', key='policy_center_apply_governance_defaults'):
        st.session_state['governance_default_owner'] = default_owner
        st.session_state['governance_default_reviewer_role'] = default_reviewer
        st.session_state['governance_release_gate_mode'] = release_gate_mode
        st.session_state['export_policy_name'] = export_policy_name
        st.session_state['workspace_governance_redaction_level'] = redaction_level
        st.session_state['workspace_governance_export_access'] = export_access
        st.session_state['workspace_governance_watermark_sensitive_exports'] = bool(watermark_sensitive)
        _persist(st.session_state)
        st.success('Governance defaults updated.')
        st.rerun()

    routing_defaults = dict(st.session_state.get('approval_routing_rules', {}))
    routing_cols = st.columns(2)
    updated_routing: dict[str, str] = {}
    for idx, (checkpoint_key, checkpoint_label) in enumerate(ROUTING_CHECKPOINTS):
        target_col = routing_cols[idx % len(routing_cols)]
        default_role = str(
            routing_defaults.get(checkpoint_key, st.session_state.get('governance_default_reviewer_role', 'Data Steward'))
        )
        updated_routing[checkpoint_key] = target_col.selectbox(
            checkpoint_label,
            ROLE_OPTIONS,
            index=ROLE_OPTIONS.index(default_role) if default_role in ROLE_OPTIONS else ROLE_OPTIONS.index('Data Steward'),
            key=f'policy_center_routing_{checkpoint_key}',
        )
    if st.button('Apply Approval Routing Rules', key='policy_center_apply_routing_rules'):
        st.session_state['approval_routing_rules'] = updated_routing
        _persist(st.session_state)
        st.success('Approval routing rules updated.')
        st.rerun()
    info_or_table(
        pd.DataFrame(
            [
                {'checkpoint': label, 'routed_role': updated_routing.get(key, '')}
                for key, label in ROUTING_CHECKPOINTS
            ]
        ),
        'Approval routing rules will appear here once reviewer routing is configured.',
    )

    render_subsection_header('Runtime gate posture')
    validation_pref = st.selectbox(
        'Validation runtime preference',
        RUNTIME_PREFERENCE_OPTIONS,
        index=RUNTIME_PREFERENCE_OPTIONS.index(str(st.session_state.get('validation_runtime_preference', 'Auto'))) if str(st.session_state.get('validation_runtime_preference', 'Auto')) in RUNTIME_PREFERENCE_OPTIONS else 0,
        key='policy_center_validation_pref',
    )
    export_pref = st.selectbox(
        'Export runtime preference',
        RUNTIME_PREFERENCE_OPTIONS,
        index=RUNTIME_PREFERENCE_OPTIONS.index(str(st.session_state.get('export_runtime_preference', 'Auto'))) if str(st.session_state.get('export_runtime_preference', 'Auto')) in RUNTIME_PREFERENCE_OPTIONS else 0,
        key='policy_center_export_pref',
    )
    if st.button('Apply Runtime Preferences', key='policy_center_apply_runtime_prefs'):
        st.session_state['validation_runtime_preference'] = validation_pref
        st.session_state['export_runtime_preference'] = export_pref
        _persist(st.session_state)
        st.success('Runtime gate preferences updated.')
        st.rerun()

    validation_plan = safe_df(
        build_validation_execution_plan(safe_df(pipeline.get('evolution_summary', {}).get('validation_recommendations_table')))
    )
    export_plan = safe_df(
        build_export_execution_plan(
            pipeline,
            role=role,
            export_allowed=export_allowed,
            advanced_exports_allowed=advanced_exports_allowed,
            governance_exports_allowed=governance_exports_allowed,
            stakeholder_bundle_allowed=stakeholder_bundle_allowed,
        )
    )
    render_subsection_header('Runtime gate summary')
    info_or_table(
        pd.DataFrame(
            [
                {'surface': 'Validation', 'runtime': build_validation_runtime_profile().get('runtime_label', 'Unknown'), 'preference': validation_pref},
                {'surface': 'Export', 'runtime': build_export_runtime_profile(pipeline).get('runtime_label', 'Unknown'), 'preference': export_pref},
                {'surface': 'Dataset', 'runtime': dataset_name, 'preference': str(source_meta.get('source_mode', 'Unknown'))},
            ]
        ),
        'Runtime gate summary will appear here once policy context is available.',
    )
    info_or_table(validation_plan, 'Validation execution guidance will appear here once recommendations are available.')
    info_or_table(export_plan, 'Export execution guidance will appear here once export posture is available.')

    render_subsection_header('Environment promotion and release bundles')
    selected_bundle_pack_name = _selected_policy_pack_name(policy_packs, selected_policy_pack if policy_pack_names else '')
    selected_bundle_pack = dict(policy_packs.get(selected_bundle_pack_name, _current_policy_pack(st.session_state)))
    governance_bundle = build_governance_release_bundle(
        dataset_name=dataset_name,
        source_meta=source_meta,
        workspace_identity=st.session_state.get('workspace_identity', {}),
        policy_pack_name=selected_bundle_pack_name,
        policy_pack=selected_bundle_pack,
        benchmark_packs=org_packs,
        execution_queue=st.session_state.get('evolution_execution_queue', []),
        review_approvals=st.session_state.get('dataset_review_approvals', {}),
    )
    governance_bundle_bytes = build_governance_release_bundle_bytes(governance_bundle)
    info_or_table(
        pd.DataFrame(
            [
                {'signal': 'Bundle type', 'value': governance_bundle.get('bundle_type', '')},
                {'signal': 'Policy pack', 'value': governance_bundle.get('policy_pack_name', '')},
                {'signal': 'Execution items', 'value': governance_bundle.get('execution_queue_summary', {}).get('total_items', 0)},
                {'signal': 'High-priority open', 'value': governance_bundle.get('execution_queue_summary', {}).get('high_priority_open', 0)},
            ]
        ),
        'Governance release bundle details will appear here when policy and adaptive execution context are available.',
    )
    bundle_cols = st.columns(2)
    bundle_cols[0].download_button(
        'Download Governance Release Bundle',
        data=governance_bundle_bytes,
        file_name=f"{dataset_name.replace(' ', '_').lower()}_governance_release_bundle.json",
        mime='application/json',
        key='policy_center_download_governance_bundle',
    )
    storage_service = st.session_state.get('storage_service')
    if bundle_cols[1].button('Store Release Bundle Artifact', key='policy_center_store_governance_bundle'):
        if storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
            artifact = storage_service.save_runtime_state(
                st.session_state.get('workspace_identity', {}),
                state_name=f"{dataset_name}_governance_release_bundle",
                payload=governance_bundle_bytes,
            )
            st.session_state['last_governance_bundle_artifact'] = artifact
            st.success(f"Stored governance release bundle artifact at {artifact.get('artifact_path', '')}.")
        else:
            st.info('Persistent artifact storage is not enabled in this runtime, so download the bundle JSON directly instead.')
    if st.session_state.get('last_governance_bundle_artifact'):
        artifact = dict(st.session_state.get('last_governance_bundle_artifact', {}))
        st.caption(f"Latest stored release bundle artifact: {artifact.get('artifact_path', '')}")
    imported_bundle_json = st.text_area(
        'Import governance release bundle JSON',
        key='policy_center_import_governance_bundle_json',
        height=140,
        placeholder='Paste a governance release bundle JSON payload here to promote settings from another environment.',
    )
    parsed_bundle: dict[str, Any] | None = None
    drift_gate: dict[str, Any] | None = None
    compatibility_gate: dict[str, Any] | None = None
    promotion_readiness: dict[str, Any] | None = None
    if imported_bundle_json.strip():
        try:
            parsed_bundle = parse_governance_release_bundle(imported_bundle_json)
            drift_table = safe_df(build_governance_release_bundle_drift(governance_bundle, parsed_bundle))
            drift_gate = build_governance_release_bundle_gate(drift_table)
            changed_items = int(drift_gate.get('changed_items_count', 0))
            high_impact_drift = int(drift_gate.get('high_impact_drift_count', 0))
            summary_cols = st.columns(3)
            summary_cols[0].metric('Drift items', changed_items)
            summary_cols[1].metric('High-impact drift', high_impact_drift)
            summary_cols[2].metric('Bundle policy pack', str(parsed_bundle.get('policy_pack_name', 'Imported Policy Pack')))
            info_or_table(
                drift_table,
                'Release-control drift will appear here when an importable governance bundle is provided.',
            )
            if bool(drift_gate.get('requires_signoff', False)):
                st.warning(str(drift_gate.get('gate_message', 'High-impact drift requires explicit signoff.')))
            else:
                st.info(str(drift_gate.get('gate_message', 'Review release-control drift before promoting this governance bundle into the current environment.')))
            compatibility_table = safe_df(
                build_governance_release_bundle_runtime_compatibility(
                    parsed_bundle,
                    validation_runtime_profile=build_validation_runtime_profile(),
                    export_runtime_profile=build_export_runtime_profile(pipeline),
                )
            )
            compatibility_gate = build_governance_release_bundle_compatibility_gate(compatibility_table)
            compatibility_cols = st.columns(2)
            compatibility_cols[0].metric('Runtime mismatches', int(compatibility_gate.get('mismatch_count', 0)))
            compatibility_cols[1].metric('High-impact runtime issues', int(compatibility_gate.get('high_impact_mismatch_count', 0)))
            info_or_table(
                compatibility_table,
                'Runtime compatibility checks will appear here when an importable governance bundle is provided.',
            )
            if bool(compatibility_gate.get('requires_signoff', False)):
                st.warning(str(compatibility_gate.get('gate_message', 'High-impact runtime compatibility issues require explicit signoff.')))
            else:
                st.info(str(compatibility_gate.get('gate_message', 'Review runtime compatibility before promoting this governance bundle into the current environment.')))
            promotion_readiness = build_governance_release_bundle_promotion_readiness(
                drift_gate=drift_gate,
                compatibility_gate=compatibility_gate,
                imported_bundle=parsed_bundle,
            )
            readiness_cols = st.columns(3)
            readiness_cols[0].metric('Promotion status', str(promotion_readiness.get('status', 'Unknown')))
            readiness_cols[1].metric('Open release blockers', int(promotion_readiness.get('open_release_blockers', 0)))
            readiness_cols[2].metric('Explicit signoff required', 'Yes' if bool(promotion_readiness.get('requires_signoff', False)) else 'No')
            if str(promotion_readiness.get('status', 'Unknown')) == 'Blocked':
                st.error(str(promotion_readiness.get('detail', 'Promotion is blocked.')))
            elif bool(promotion_readiness.get('requires_signoff', False)):
                st.warning(str(promotion_readiness.get('detail', 'Promotion requires signoff.')))
            else:
                st.success(str(promotion_readiness.get('detail', 'Promotion looks ready.')))
            info_or_table(
                safe_df(promotion_readiness.get('summary_table')),
                'Promotion readiness checkpoints will appear here when an importable governance bundle is provided.',
            )
        except ValueError as error:
            st.error(str(error))
    signoff_ack = st.checkbox(
        'I acknowledge the release-control drift and want to promote this bundle.',
        key='policy_center_promotion_signoff_ack',
        disabled=not imported_bundle_json.strip(),
    )
    signoff_reason = st.text_area(
        'Promotion signoff reason',
        key='policy_center_promotion_signoff_reason',
        height=90,
        placeholder='Explain why this drift is acceptable for promotion or release.',
        disabled=not imported_bundle_json.strip(),
    )
    apply_disabled = not imported_bundle_json.strip()
    if (drift_gate and bool(drift_gate.get('requires_signoff', False))) or (compatibility_gate and bool(compatibility_gate.get('requires_signoff', False))):
        apply_disabled = apply_disabled or (not signoff_ack) or (not signoff_reason.strip())
    if promotion_readiness and str(promotion_readiness.get('status', '')) == 'Blocked':
        apply_disabled = True
    if st.button('Apply Governance Release Bundle', key='policy_center_apply_governance_bundle', disabled=apply_disabled):
        try:
            if parsed_bundle is None:
                parsed_bundle = parse_governance_release_bundle(imported_bundle_json)
            restore_result = restore_governance_release_bundle(parsed_bundle, st.session_state)
            dataset_identifier = str(source_meta.get('dataset_identifier', '') or source_meta.get('dataset_cache_key', '')).strip()
            if dataset_identifier:
                review_store = dict(st.session_state.get('dataset_review_approvals', {}))
                review_entry = dict(review_store.get(dataset_identifier, {}))
                if drift_gate and bool(drift_gate.get('requires_signoff', False)):
                    review_entry['release_signoff_status'] = 'Approved with drift signoff'
                    review_entry['release_signoff_note'] = signoff_reason.strip()
                elif compatibility_gate and bool(compatibility_gate.get('requires_signoff', False)):
                    review_entry['release_signoff_status'] = 'Approved with compatibility signoff'
                    review_entry['release_signoff_note'] = signoff_reason.strip()
                else:
                    review_entry['release_signoff_status'] = str(review_entry.get('release_signoff_status', 'Approved') or 'Approved')
                review_entry['release_bundle_policy_pack'] = str(parsed_bundle.get('policy_pack_name', 'Imported Policy Pack'))
                review_entry['release_bundle_drift_summary'] = (
                    dict(drift_gate) if isinstance(drift_gate, dict) else {'changed_items_count': 0, 'high_impact_drift_count': 0}
                )
                review_entry['release_bundle_runtime_compatibility'] = (
                    dict(compatibility_gate) if isinstance(compatibility_gate, dict) else {'mismatch_count': 0, 'high_impact_mismatch_count': 0}
                )
                review_store[dataset_identifier] = review_entry
                review_store = append_review_history(
                    review_store,
                    dataset_identifier=dataset_identifier,
                    reviewer=str(st.session_state.get('workspace_display_name', st.session_state.get('display_name', 'Guest User'))),
                    reviewer_role=str(st.session_state.get('active_role') or st.session_state.get('workspace_role') or role),
                    action='Governance bundle promotion',
                    status=str(review_entry.get('release_signoff_status', 'Approved')),
                    notes=signoff_reason.strip() or f"Applied governance bundle '{restore_result.get('policy_pack_name', 'Imported Policy Pack')}'.",
                )
                st.session_state['dataset_review_approvals'] = review_store
            _persist(st.session_state)
            st.success(f"Applied governance release bundle '{restore_result.get('policy_pack_name', 'Imported Policy Pack')}'.")
            st.rerun()
        except ValueError as error:
            st.error(str(error))


__all__ = ['render_policy_center']
