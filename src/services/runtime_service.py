from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Any

import src.auth as auth_module
from src.jobs import build_job_runtime
from src.persistence import build_persistence_service
from src.profiler import default_profile_cache_metrics
from src.product_settings import DEFAULT_PRODUCT_SETTINGS
from src.services.admin_ops_service import build_admin_ops_service
from src.services.application_service import build_workspace_application_service
from src.storage import build_storage_service


def _get_auth_module():
    global auth_module
    required_attrs = (
        'build_auth_service',
        'build_guest_auth_session',
    )
    if all(hasattr(auth_module, name) for name in required_attrs):
        return auth_module
    sys.modules.pop('src.auth', None)
    auth_module = importlib.import_module('src.auth')
    return auth_module


@dataclass(frozen=True)
class RuntimeServices:
    persistence_service: Any
    auth_service: Any
    storage_service: Any
    job_runtime: dict[str, Any]
    application_service: Any
    admin_ops_service: Any


def ensure_runtime_services(session_state: dict[str, Any]) -> RuntimeServices:
    persistence_service = session_state.get('persistence_service')
    if persistence_service is None:
        persistence_service = build_persistence_service()
        session_state['persistence_service'] = persistence_service

    auth_service = session_state.get('auth_service')
    if auth_service is None:
        auth_service = _get_auth_module().build_auth_service()
        session_state['auth_service'] = auth_service

    storage_service = session_state.get('storage_service')
    if storage_service is None:
        storage_service = build_storage_service()
        session_state['storage_service'] = storage_service

    job_runtime = session_state.get('job_runtime')
    if not isinstance(job_runtime, dict):
        job_runtime = build_job_runtime()
        session_state['job_runtime'] = job_runtime
    application_service = session_state.get('application_service')
    if application_service is None:
        application_service = build_workspace_application_service(persistence_service)
        session_state['application_service'] = application_service
    admin_ops_service = session_state.get('admin_ops_service')
    if admin_ops_service is None:
        admin_ops_service = build_admin_ops_service(application_service, persistence_service)
        session_state['admin_ops_service'] = admin_ops_service

    return RuntimeServices(
        persistence_service=persistence_service,
        auth_service=auth_service,
        storage_service=storage_service,
        job_runtime=job_runtime,
        application_service=application_service,
        admin_ops_service=admin_ops_service,
    )


def initialize_app_session_state(session_state: dict[str, Any]) -> RuntimeServices:
    services = ensure_runtime_services(session_state)
    guest_session = _get_auth_module().build_guest_auth_session()
    defaults: dict[str, Any] = {
        'analysis_log': [],
        'saved_snapshots': {},
        'workflow_packs': {},
        'workspace_saved_snapshots': {},
        'workspace_workflow_packs': {},
        'collaboration_notes': [],
        'workspace_collaboration_notes': {},
        'beta_interest_submissions': [],
        'workspace_beta_interest_submissions': {},
        'beta_interest_storage_mode': 'LOCAL',
        'beta_interest_api_endpoint': '',
        'beta_interest_feedback': None,
        'workspace_analysis_logs': {},
        'workspace_run_history': {},
        'analysis_template': 'General Review',
        'report_mode': 'Executive Summary',
        'export_policy_name': 'Internal Review',
        'active_role': 'Analyst',
        'accuracy_benchmark_profile': 'Auto',
        'active_benchmark_pack_name': 'None',
        'accuracy_reporting_threshold_profile': 'Role default',
        'accuracy_reporting_min_trust_score': 0.76,
        'accuracy_allow_directional_external_reporting': False,
        'organization_benchmark_packs': {},
        'semantic_mapping_profiles': {},
        'dataset_review_approvals': {},
        'active_plan': 'Pro',
        'plan_enforcement_mode': 'Demo-safe',
        'run_history': [],
        'demo_synthetic_helper_mode': 'Auto',
        'demo_bmi_remediation_mode': 'median',
        'demo_synthetic_cost_mode': 'Auto',
        'demo_synthetic_readmission_mode': 'Auto',
        'demo_executive_summary_verbosity': 'Concise',
        'demo_scenario_simulation_mode': 'Basic',
        'workspace_user_name': 'Guest User',
        'workspace_user_email': '',
        'workspace_name': 'Guest Demo Workspace',
        'workspace_role': 'Viewer',
        'workspace_governance_redaction_level': 'Medium',
        'workspace_governance_export_access': 'Editors and owners',
        'workspace_governance_watermark_sensitive_exports': True,
        'auth_session': guest_session,
        'workspace_identity': services.auth_service.build_workspace_identity(guest_session, 'Guest Demo Workspace'),
        'visited_sections': [],
        'demo_usage_seeded_keys': [],
        'generated_report_outputs': {},
        'job_runs': [],
        'profile_cache_metrics': default_profile_cache_metrics(),
        'active_dataset_bundle': None,
        'semantic_mapping_overrides_by_dataset': {},
        'latest_dataset_artifact': None,
        'pending_uploaded_dataset_state': None,
        'active_dataset_diagnostics': {},
    }
    defaults.update(DEFAULT_PRODUCT_SETTINGS)
    for key, value in defaults.items():
        session_state.setdefault(key, value)
    services.application_service.hydrate_workspace_state(session_state)
    return services


__all__ = ['RuntimeServices', 'ensure_runtime_services', 'initialize_app_session_state']
