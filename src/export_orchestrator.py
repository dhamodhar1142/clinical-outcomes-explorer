from __future__ import annotations

from typing import Any

from src.export_utils import recommended_report_mode_for_role
from src.validation_orchestrator import _is_cloud_runtime


REPORT_MODE_TO_LABEL = {
    'Executive Summary': 'Executive Report',
    'Analyst Report': 'Analyst Report',
    'Data Readiness Review': 'Data Readiness Report',
    'Clinical Report': 'Clinical Summary',
    'Operational Report': 'Claims Validation Report',
    'Population Health Summary': 'Executive Report',
}


def recommended_report_label(role: str, pipeline: dict[str, Any] | None = None) -> str:
    pipeline = pipeline or {}
    mode = str(recommended_report_mode_for_role(role))
    label = REPORT_MODE_TO_LABEL.get(mode, 'Analyst Report')
    claims_available = bool((pipeline.get('healthcare') or {}).get('claims_validation_utilization', {}).get('available'))
    if label == 'Claims Validation Report' and not claims_available:
        return 'Analyst Report'
    return label


def build_export_runtime_profile(pipeline: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline = pipeline or {}
    sample_info = dict(pipeline.get('sample_info', {}))
    row_count = int((pipeline.get('overview') or {}).get('rows', 0) or 0)
    sampling_applied = bool(sample_info.get('sampling_applied'))
    large_dataset_mode = bool(sampling_applied or row_count > 100000)
    cloud_runtime = _is_cloud_runtime()
    supports_governed_packaging = not cloud_runtime
    return {
        'cloud_runtime': cloud_runtime,
        'runtime_label': 'Streamlit Cloud' if cloud_runtime else 'Local / workstation',
        'row_count': row_count,
        'sampling_applied': sampling_applied,
        'large_dataset_mode': large_dataset_mode,
        'supports_governed_packaging': supports_governed_packaging,
        'detail': (
            'This runtime is suitable for interactive report generation and lightweight exports. '
            'Heavier governed packaging should run locally or in staging.'
            if cloud_runtime
            else 'This runtime can generate both lightweight exports and heavier governed packaging actions.'
        ),
    }


def build_export_execution_plan(
    pipeline: dict[str, Any],
    *,
    role: str,
    export_allowed: bool,
    advanced_exports_allowed: bool,
    governance_exports_allowed: bool,
    stakeholder_bundle_allowed: bool,
) -> list[dict[str, Any]]:
    profile = build_export_runtime_profile(pipeline)
    recommended_label = recommended_report_label(role, pipeline)
    rows = [
        {
            'action': f'Generate {recommended_label}',
            'task': 'report_generation',
            'priority': 'High',
            'allowed': bool(export_allowed),
            'execution_posture': 'Run here' if export_allowed else 'Blocked by export policy',
            'why': 'A stakeholder-ready report is the safest first export for the active audience and dataset context.',
            'gating_reason': (
                'This report can be generated in the current runtime.'
                if export_allowed
                else 'The active role or workspace export policy currently blocks export generation.'
            ),
        },
        {
            'action': 'Prepare ZIP export bundle',
            'task': 'governed_bundle',
            'priority': 'Medium',
            'allowed': bool(export_allowed and profile.get('supports_governed_packaging')),
            'execution_posture': 'Run here' if export_allowed and profile.get('supports_governed_packaging') else 'Run locally or in staging',
            'why': 'ZIP bundles combine multiple governed artifacts and are the heaviest export action in the current workflow.',
            'gating_reason': (
                'This runtime can build governed ZIP bundles directly.'
                if export_allowed and profile.get('supports_governed_packaging')
                else 'Governed bundle packaging is intentionally gated away from lighter cloud runtimes.'
            ),
        },
        {
            'action': 'Publish stakeholder/shared bundle manifests',
            'task': 'bundle_manifest',
            'priority': 'Medium',
            'allowed': bool(advanced_exports_allowed or stakeholder_bundle_allowed),
            'execution_posture': 'Run here' if advanced_exports_allowed or stakeholder_bundle_allowed else 'Blocked by plan or policy',
            'why': 'Manifest downloads are lightweight and help teams coordinate handoffs before packaging a heavier export bundle.',
            'gating_reason': (
                'Manifest exports are safe for the current plan and workspace policy.'
                if advanced_exports_allowed or stakeholder_bundle_allowed
                else 'Bundle manifests are limited by the active plan or workspace export controls.'
            ),
        },
        {
            'action': 'Generate compliance handoff package',
            'task': 'compliance_handoff',
            'priority': 'Medium',
            'allowed': bool(advanced_exports_allowed and profile.get('supports_governed_packaging')),
            'execution_posture': 'Run here' if advanced_exports_allowed and profile.get('supports_governed_packaging') else 'Run locally or in staging',
            'why': 'Compliance handoff exports package structured governance material that is better suited to a heavier operator runtime.',
            'gating_reason': (
                'Compliance handoff packaging is available in the current runtime.'
                if advanced_exports_allowed and profile.get('supports_governed_packaging')
                else 'Compliance handoff packaging is intentionally gated to local or staging environments with fuller export support.'
            ),
        },
        {
            'action': 'Generate governance audit packet',
            'task': 'governance_packet',
            'priority': 'Medium',
            'allowed': bool(governance_exports_allowed and profile.get('supports_governed_packaging')),
            'execution_posture': 'Run here' if governance_exports_allowed and profile.get('supports_governed_packaging') else 'Run locally or in staging',
            'why': 'Governance packets are useful for release and audit workflows, but they are not the first export to prioritize in constrained runtimes.',
            'gating_reason': (
                'Governance packet generation is available in the current runtime.'
                if governance_exports_allowed and profile.get('supports_governed_packaging')
                else 'Governance packet generation is intentionally gated by runtime capacity or export policy.'
            ),
        },
    ]
    return rows


__all__ = [
    'build_export_execution_plan',
    'build_export_runtime_profile',
    'recommended_report_label',
]
