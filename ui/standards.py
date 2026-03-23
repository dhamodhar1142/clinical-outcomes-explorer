from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.modules.rbac import can_access
from src.ui_components import metric_row

from ui.common import fmt, info_or_table, safe_df

def compliance_snapshot(pipeline: dict[str, Any]) -> pd.DataFrame:
    standards = pipeline['standards']
    privacy = pipeline['privacy_review']
    lineage = pipeline['lineage']
    return pd.DataFrame([
        {'dimension': 'Standards readiness', 'status': standards.get('badge_text', 'Not assessed')},
        {'dimension': 'Privacy posture', 'status': privacy.get('hipaa', {}).get('risk_level', 'Low')},
        {'dimension': 'Governance depth', 'status': f"{len(lineage.get('transformation_steps', []))} tracked steps"},
        {'dimension': 'Export context', 'status': f"{st.session_state.get('active_role', 'Analyst')} · {st.session_state.get('export_policy_name', 'Internal Review')}"},
    ])

def _normalize_choice(value: Any) -> str:
    if value in (None, '', 'Auto-detect'):
        return ''
    return str(value)

def _override_key(prefix: str, target: str) -> str:
    safe = ''.join(char.lower() if char.isalnum() else '_' for char in str(target))
    return f'{prefix}_{safe}'

def _apply_mapping_suggestions(rows: pd.DataFrame, key_column: str, source_column: str, prefix: str, target_filter=None) -> int:
    if rows.empty:
        return 0
    applied = 0
    for _, row in rows.iterrows():
        target = str(row.get(key_column, ''))
        source = str(row.get(source_column, ''))
        if not target or not source:
            continue
        if target_filter and not target_filter(target):
            continue
        st.session_state[_override_key(prefix, target)] = source
        applied += 1
    return applied

def _render_standards_mapping_overrides(pipeline: dict[str, Any], key_prefix: str) -> None:
    standards = pipeline['standards']
    if not standards.get('available'):
        return
    columns = sorted(str(column) for column in pipeline['data'].columns)
    options = ['Auto-detect'] + columns
    override_catalog = build_standards_override_catalog(standards)
    terminology_catalog = build_terminology_override_catalog(standards)

    with st.expander('Standards mapping overrides'):
        st.caption('Confirm or override suggested CDISC, FHIR, and terminology mappings when you want a more directed readiness review. Auto-detected values remain the default until you change them.')
        if st.button('Reset standards mapping overrides', key=f'{key_prefix}_reset_standards_overrides'):
            for key in list(st.session_state.keys()):
                if key.startswith('standards_override_') or key.startswith('terminology_override_'):
                    del st.session_state[key]
            st.rerun()

        if not override_catalog.empty:
            st.markdown('#### Standards mapping review')
            for group_name, group in override_catalog.groupby('mapping_group'):
                st.write(f'**{group_name}**')
                for row in group.itertuples(index=False):
                    widget_key = _override_key(f'{key_prefix}_standards_override', row.target_field)
                    default_choice = row.default_source if row.default_source in columns else 'Auto-detect'
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = default_choice
                    st.selectbox(
                        row.target_field,
                        options,
                        key=widget_key,
                        help=row.help_text,
                    )

        if not terminology_catalog.empty:
            st.markdown('#### Terminology confirmation')
            for row in terminology_catalog.itertuples(index=False):
                widget_key = _override_key(f'{key_prefix}_terminology_override', row.terminology_type)
                default_choice = row.default_source if row.default_source in columns else 'Auto-detect'
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = default_choice
                st.selectbox(
                    row.terminology_type,
                    options,
                    key=widget_key,
                    help=row.help_text,
                )

    overrides = {
        row.target_field: _normalize_choice(st.session_state.get(_override_key(f'{key_prefix}_standards_override', row.target_field)))
        for row in override_catalog.itertuples(index=False)
    }
    terminology_overrides = {
        row.terminology_type: _normalize_choice(st.session_state.get(_override_key(f'{key_prefix}_terminology_override', row.terminology_type)))
        for row in terminology_catalog.itertuples(index=False)
    }
    pipeline['standards'] = apply_standards_mapping_overrides(standards, overrides, terminology_overrides)

def _build_standards_prefill_actions(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    standards = pipeline['standards']
    if not standards.get('available'):
        return []
    actions: list[dict[str, str]] = []
    cdisc = standards.get('cdisc_report', {})
    interop = standards.get('interoperability_report', {})
    cdisc_map = safe_df(cdisc.get('mapping_suggestions'))
    interop_map = safe_df(interop.get('mapping_suggestions'))
    terminology = safe_df(interop.get('terminology_validation'))
    if not cdisc_map.empty:
        actions.append({'label': 'Prefill CDISC mappings', 'action': 'apply_detected_cdisc_mappings'})
        core_targets = {'STUDYID', 'USUBJID', 'SUBJID', 'VISIT', 'VISITNUM', 'DOMAIN'}
        if set(cdisc_map.get('cdisc_field', pd.Series(dtype=str)).astype(str)).intersection(core_targets):
            actions.append({'label': 'Prefill CDISC core fields', 'action': 'apply_detected_cdisc_core_mappings'})
        actions.append({'label': 'Prepare CDISC readiness review', 'action': 'prepare_cdisc_readiness_review'})
    if not interop_map.empty:
        actions.append({'label': 'Prefill interoperability mappings', 'action': 'apply_detected_interop_mappings'})
        fhir_prefixes = ('Patient.', 'Encounter.', 'Condition.', 'Procedure.', 'Observation.', 'Medication')
        targets = interop_map.get('reference_model_target', pd.Series(dtype=str)).astype(str).tolist()
        if any(target.startswith(fhir_prefixes) for target in targets):
            actions.append({'label': 'Prefill FHIR mappings', 'action': 'apply_detected_fhir_mappings'})
    if not terminology.empty:
        actions.append({'label': 'Prefill terminology confirmations', 'action': 'apply_detected_terminology'})
    return actions

def _run_standards_prefill_action(pipeline: dict[str, Any], action_name: str) -> str:
    standards = pipeline['standards']
    cdisc_map = safe_df(standards.get('cdisc_report', {}).get('mapping_suggestions'))
    interop_map = safe_df(standards.get('interoperability_report', {}).get('mapping_suggestions'))
    terminology = safe_df(standards.get('interoperability_report', {}).get('terminology_validation'))

    if action_name == 'apply_detected_cdisc_mappings':
        applied = _apply_mapping_suggestions(cdisc_map, 'cdisc_field', 'suggested_source_column', 'standards_override')
        return f'Prefilled {applied} CDISC mapping suggestions for review.'
    if action_name == 'apply_detected_cdisc_core_mappings':
        core_targets = {'STUDYID', 'USUBJID', 'SUBJID', 'VISIT', 'VISITNUM', 'DOMAIN'}
        applied = _apply_mapping_suggestions(cdisc_map, 'cdisc_field', 'suggested_source_column', 'standards_override', lambda target: target in core_targets)
        return f'Prefilled {applied} core CDISC fields for review.'
    if action_name == 'prepare_cdisc_readiness_review':
        st.session_state['analysis_template'] = 'General Review'
        st.session_state['report_mode'] = 'Analyst Report'
        st.session_state['export_policy_name'] = 'Internal Review'
        return 'Prepared the workspace for a CDISC-focused readiness review.'
    if action_name == 'apply_detected_interop_mappings':
        applied = _apply_mapping_suggestions(interop_map, 'reference_model_target', 'suggested_source_column', 'standards_override')
        return f'Prefilled {applied} interoperability mappings for review.'
    if action_name == 'apply_detected_fhir_mappings':
        prefixes = ('Patient.', 'Encounter.', 'Condition.', 'Procedure.', 'Observation.', 'Medication')
        applied = _apply_mapping_suggestions(interop_map, 'reference_model_target', 'suggested_source_column', 'standards_override', lambda target: str(target).startswith(prefixes))
        return f'Prefilled {applied} FHIR-style mappings for review.'
    if action_name == 'apply_detected_terminology':
        applied = 0
        if not terminology.empty:
            for row in terminology.itertuples(index=False):
                term_type = str(getattr(row, 'terminology_type', ''))
                column_name = str(getattr(row, 'column_name', ''))
                if term_type and column_name:
                    st.session_state[_override_key('terminology_override', term_type)] = column_name
                    applied += 1
        return f'Prefilled {applied} terminology confirmations for review.'
    return 'No standards prefill action was applied.'

def _build_ehr_onboarding_actions(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    interop = pipeline['standards'].get('interoperability_report', {})
    ehr_patterns = safe_df(interop.get('ehr_export_patterns'))
    hl7_patterns = safe_df(interop.get('hl7_patterns'))
    actions: list[dict[str, str]] = []
    if not ehr_patterns.empty and 'export_pattern' in ehr_patterns.columns:
        detected = set(ehr_patterns['export_pattern'].astype(str))
        if 'Epic-like export' in detected:
            actions.append({'label': 'Prepare Epic-style onboarding', 'action': 'prepare_epic_onboarding'})
        if 'Cerner-like export' in detected:
            actions.append({'label': 'Prepare Cerner-style onboarding', 'action': 'prepare_cerner_onboarding'})
    if not hl7_patterns.empty:
        actions.append({'label': 'Prepare HL7-style onboarding', 'action': 'prepare_hl7_onboarding'})
    return actions

def _run_ehr_onboarding_action(action_name: str) -> str:
    st.session_state['analysis_template'] = 'General Review'
    st.session_state['export_policy_name'] = 'Internal Review'
    if action_name == 'prepare_epic_onboarding':
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Prepared an Epic-style onboarding workflow with analyst-oriented review settings.'
    if action_name == 'prepare_cerner_onboarding':
        st.session_state['report_mode'] = 'Operational Report'
        return 'Prepared a Cerner-style onboarding workflow with operational review settings.'
    if action_name == 'prepare_hl7_onboarding':
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Prepared an HL7-style onboarding workflow for structure and standards review.'
    return 'No onboarding preset was applied.'

def _build_ehr_onboarding_explanation(pipeline: dict[str, Any]) -> pd.DataFrame:
    interop = pipeline['standards'].get('interoperability_report', {})
    rows: list[dict[str, object]] = []
    ehr_patterns = safe_df(interop.get('ehr_export_patterns'))
    if not ehr_patterns.empty:
        for row in ehr_patterns.itertuples(index=False):
            rows.append({
                'detected_signal': getattr(row, 'export_pattern', 'EHR export pattern'),
                'status': getattr(row, 'status', 'Detected'),
                'why_it_appeared': getattr(row, 'matched_signals', 'Column naming pattern and export structure heuristics'),
            })
    hl7_patterns = safe_df(interop.get('hl7_patterns'))
    if not hl7_patterns.empty:
        for row in hl7_patterns.itertuples(index=False):
            rows.append({
                'detected_signal': f"HL7 segment {getattr(row, 'segment', '')}",
                'status': getattr(row, 'status', 'Detected'),
                'why_it_appeared': getattr(row, 'matched_signals', 'Segment-like naming and field pattern signals'),
            })
    return pd.DataFrame(rows)

def _build_compliance_quick_actions(pipeline: dict[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    standards = pipeline['standards']
    privacy = pipeline['privacy_review']
    if standards.get('available') and standards.get('cdisc_report', {}).get('available'):
        actions.append({'label': 'Prepare Clinical Outcomes workflow', 'action': 'prepare_clinical_outcomes'})
    if standards.get('interoperability_report', {}).get('available'):
        actions.append({'label': 'Prepare Analyst handoff', 'action': 'prepare_analyst_handoff'})
    hipaa_risk = privacy.get('hipaa', {}).get('risk_level', 'Low')
    if hipaa_risk in {'Moderate', 'High'}:
        actions.append({'label': 'Apply HIPAA-style export posture', 'action': 'apply_hipaa_export'})
        actions.append({'label': 'Apply research-safe export posture', 'action': 'apply_research_export'})
    return actions

def _run_compliance_quick_action(action_name: str) -> str:
    if action_name == 'prepare_clinical_outcomes':
        st.session_state['analysis_template'] = 'Clinical Outcomes Analysis'
        st.session_state['report_mode'] = 'Clinical Report'
        return 'Prepared the workspace for a clinical outcomes review.'
    if action_name == 'prepare_analyst_handoff':
        st.session_state['analysis_template'] = 'General Review'
        st.session_state['report_mode'] = 'Analyst Report'
        return 'Prepared the workspace for an analyst-oriented handoff review.'
    if action_name == 'apply_hipaa_export':
        st.session_state['export_policy_name'] = 'HIPAA-style Limited Dataset'
        return 'Applied the HIPAA-style limited dataset export posture.'
    if action_name == 'apply_research_export':
        st.session_state['export_policy_name'] = 'Research-safe Extract'
        return 'Applied the research-safe export posture.'
    return 'No compliance action was applied.'

def _build_next_compliance_action(pipeline: dict[str, Any]) -> dict[str, str] | None:
    standards = pipeline['standards']
    privacy = pipeline['privacy_review']
    if privacy.get('hipaa', {}).get('risk_level') in {'Moderate', 'High'}:
        return {
            'title': 'Reduce privacy exposure before sharing',
            'detail': 'Sensitive columns were detected. Use a stricter export policy or review the de-identification preview before handing the dataset to a wider audience.',
        }
    if standards.get('available') and standards.get('combined_readiness_score', 0) < 45:
        return {
            'title': 'Confirm standards mappings',
            'detail': 'Use the standards mapping overrides to confirm trial or interoperability fields so downstream healthcare reviews can rely on clearer structure.',
        }
    if not standards.get('available'):
        return {
            'title': 'Start with schema and remediation review',
            'detail': 'The dataset does not yet resemble a strong standards-aligned structure. Use the remediation assistant to fill the highest-impact gaps first.',
        }
    return {
        'title': 'Prepare a governed handoff bundle',
        'detail': 'The dataset is in a reasonable position for standards and privacy review. Generate a compliance handoff or governance packet before sharing results.',
    }

def _build_standards_override_summary(standards: dict[str, Any]) -> pd.DataFrame:
    effective = safe_df(standards.get('effective_mappings'))
    interop = standards.get('interoperability_report', {}) if isinstance(standards, dict) else {}
    terminology = safe_df(interop.get('effective_terminology', interop.get('terminology_validation', pd.DataFrame())))
    rows: list[dict[str, object]] = []
    if not effective.empty:
        mapped_count = int((effective['source_column'].astype(str) != '').sum()) if 'source_column' in effective.columns else len(effective)
        manual_count = int((effective.get('mapping_source', pd.Series(dtype=str)).astype(str) == 'Manual override').sum()) if 'mapping_source' in effective.columns else 0
        rows.append({
            'focus_area': 'Standards mappings',
            'status': f'{mapped_count} effective mappings',
            'detail': f'{manual_count} manually confirmed' if manual_count else 'Using auto-detected mappings',
        })
    if not terminology.empty:
        confirmed = int((terminology.get('status', pd.Series(dtype=str)).astype(str) == 'Manually confirmed').sum()) if 'status' in terminology.columns else 0
        rows.append({
            'focus_area': 'Terminology confirmation',
            'status': f'{len(terminology)} terminology fields',
            'detail': f'{confirmed} manually confirmed' if confirmed else 'Using detected coded-field signals',
        })
    return pd.DataFrame(rows)

def render_standards(pipeline: dict[str, Any], key_prefix: str = 'standards') -> None:
    standards = pipeline['standards']
    st.markdown('### Standards Compliance')
    st.caption('Use this validator to confirm whether the dataset is closer to a research, interoperability, or operational healthcare structure and what to map next for stronger downstream analysis.')
    if not standards.get('available'):
        st.info(standards.get('reason', 'Standards readiness is not available for the current dataset.'))
        info_or_table(safe_df(standards.get('summary_table')), 'No standards summary is available yet.')
        return
    _render_standards_mapping_overrides(pipeline, key_prefix)
    standards = pipeline['standards']
    metric_row([
        ('Combined Readiness', fmt(standards.get('combined_readiness_score', 0.0), 'float')),
        ('Detected Standard', standards.get('detected_standard', 'Not detected')),
        ('Confidence', fmt(standards.get('compliance_confidence', 0.0), 'pct')),
        ('Missing Required Fields', fmt(standards.get('missing_required_fields', 0))),
    ])
    st.caption(
        f"{standards.get('confidence_label', 'Low')} confidence. "
        + str(standards.get('note', 'Readiness-oriented validator only; not a formal certification engine.'))
    )
    info_or_table(safe_df(standards.get('summary_table')), 'No standards summary is available yet.')
    profiles = safe_df(standards.get('standards_profiles'))
    if not profiles.empty:
        st.markdown('#### Standards detection detail')
        info_or_table(
            profiles[['standard_type', 'status', 'confidence_score', 'standards_readiness_score', 'required_fields_present', 'required_fields_total', 'missing_required_fields', 'suggested_mappings']],
            'Standards detection detail will appear here when the validator finds healthcare-specific structure signals.',
        )
    required_field_review = safe_df(standards.get('required_field_review'))
    if not required_field_review.empty:
        st.markdown('#### Required vs missing fields')
        focus_standard = str(standards.get('detected_standard', ''))
        focus_rows = required_field_review[required_field_review['standard_type'] == focus_standard]
        info_or_table(
            focus_rows if not focus_rows.empty else required_field_review,
            'Required-field guidance will appear here when a standards profile is in scope.',
        )
    standards_mappings = safe_df(standards.get('mapping_suggestions'))
    if not standards_mappings.empty:
        st.markdown('#### Suggested standards mappings')
        info_or_table(standards_mappings, 'Suggested standards mappings will appear here when the validator finds strong mapping signals.')
    summary_table = _build_standards_override_summary(standards)
    if not summary_table.empty:
        st.markdown('#### Mapping confirmation snapshot')
        info_or_table(summary_table, 'No mapping confirmation snapshot is available yet.')
    effective = safe_df(standards.get('effective_mappings'))
    if not effective.empty:
        st.markdown('#### Effective standards mappings')
        info_or_table(effective, 'No effective mappings are available yet.')
    cols = st.columns(2)
    cdisc = standards.get('cdisc_report', {})
    interop = standards.get('interoperability_report', {})
    with cols[0]:
        st.markdown('#### CDISC validation')
        if cdisc.get('available'):
            info_or_table(safe_df(cdisc.get('validation_report')), 'No CDISC validation details are available.')
            info_or_table(safe_df(cdisc.get('effective_mappings', cdisc.get('mapping_suggestions'))), 'No CDISC mapping suggestions were detected.')
            info_or_table(safe_df(cdisc.get('domain_templates')), 'No SDTM domain-template guidance is available yet.')
        else:
            st.info('The current dataset does not look trial-oriented enough for a useful CDISC review yet.')
    with cols[1]:
        st.markdown('#### FHIR / HL7 interoperability')
        if interop.get('available'):
            info_or_table(safe_df(interop.get('validation_report')), 'No interoperability validation details are available.')
            info_or_table(safe_df(interop.get('effective_mappings', interop.get('mapping_suggestions'))), 'No interoperability mapping suggestions were detected.')
            info_or_table(safe_df(interop.get('fhir_resources')), 'No FHIR resource signals were detected.')
            info_or_table(safe_df(interop.get('hl7_patterns')), 'No HL7-like patterns were detected.')
            info_or_table(safe_df(interop.get('ehr_export_patterns')), 'No likely Epic-like or Cerner-like export pattern was detected.')
            info_or_table(safe_df(interop.get('effective_terminology', interop.get('terminology_validation', pd.DataFrame()))), 'No terminology-like fields were detected.')
        else:
            st.info('The current dataset does not yet look interoperable enough for a useful FHIR or HL7 readiness review.')

def render_privacy(pipeline: dict[str, Any]) -> None:
    role = st.session_state.get('active_role', 'Analyst')
    st.markdown('### Privacy & Security Review')
    if not can_access(role, 'sensitive_review'):
        st.info(f'The active {role} role can view standards guidance, but privacy review details are limited for this role.')
        return
    privacy = pipeline['privacy_review']
    sensitive = safe_df(privacy.get('sensitive_fields'))
    hipaa = privacy.get('hipaa', {})
    metric_row([
        ('Sensitive Columns', fmt(len(sensitive))),
        ('HIPAA-style Risk', hipaa.get('risk_level', 'Low')),
        ('Direct Identifiers', fmt(hipaa.get('direct_identifier_count', 0))),
        ('Safe Harbor', 'Ready' if hipaa.get('safe_harbor_ready') else 'Needs review'),
    ])
    info_or_table(sensitive, 'No likely PHI or PII columns were detected in the current dataset sample.')
    info_or_table(safe_df(privacy.get('deidentification_preview')), 'No de-identification preview is available.')
    info_or_table(safe_df(privacy.get('gdpr_impact')), 'No GDPR-style impact notes are available.')
    info_or_table(safe_df(privacy.get('privacy_rule_pack')), 'No privacy rule pack details are available.')

