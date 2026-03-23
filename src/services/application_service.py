from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data_loader import estimate_memory_mb
from src.jobs import build_heavy_task_catalog, build_job_runtime
from src.ops_hardening import (
    build_column_validation_report,
    build_deployment_health_checks,
    build_long_task_notice,
    build_performance_diagnostics,
    build_preflight_guardrails,
)
from src.pipeline import AnalysisCancelledError, build_updated_run_history, finalize_runtime_pipeline, run_analysis_pipeline
from src.plan_awareness import build_plan_awareness
from src.profiler import default_profile_cache_metrics
from src.product_settings import get_large_dataset_profile
from src.deployment_readiness import build_environment_checks, build_startup_readiness_summary
from src.usage_analytics import build_demo_usage_seed_events
from src.workspace import persist_active_workspace_state, sync_workspace_views

ACTIVE_CONTROL_KEYS = [
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
    'demo_dataset_name',
]

PERSISTED_USER_SETTING_KEYS = [
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
    'product_large_dataset_profile',
    'product_copilot_response_style',
    'product_copilot_show_workflow_preview',
    'workspace_governance_redaction_level',
    'workspace_governance_export_access',
    'workspace_governance_watermark_sensitive_exports',
]


@dataclass(frozen=True)
class AnalysisExecutionResult:
    dataset_name: str
    source_meta: dict[str, Any]
    preflight: dict[str, Any]
    column_validation: dict[str, Any]
    job_runtime: dict[str, Any]
    large_dataset_profile: dict[str, Any]
    long_task_notice: str
    empty_column_warning: str | None
    other_column_warnings: list[str]
    demo_config: dict[str, str]
    blocked: bool
    pipeline: dict[str, Any] | None = None
    demo_usage_seed: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkspaceApplicationService:
    persistence_service: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.persistence_service is not None)

    def hydrate_workspace_state(self, session_state: dict[str, Any]) -> None:
        sync_workspace_views(session_state)
        self.hydrate_user_settings(session_state)

    def persist_workspace_state(self, session_state: dict[str, Any]) -> None:
        persist_active_workspace_state(session_state)

    def record_usage_event(self, identity: dict[str, Any], event: dict[str, Any]) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.record_usage_event(identity, event)

    def record_dataset_metadata(self, identity: dict[str, Any], metadata: dict[str, Any]) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.save_dataset_metadata(identity, metadata)
        if hasattr(self.persistence_service, 'save_dataset_version'):
            self.persistence_service.save_dataset_version(
                identity,
                {
                    'dataset_name': metadata.get('dataset_name', 'Current dataset'),
                    'version_hash': metadata.get('dataset_version_hash', ''),
                    'version_label': metadata.get('version_label', metadata.get('dataset_name', 'Current dataset')),
                    'source_mode': metadata.get('source_mode', 'unknown'),
                    'row_count': metadata.get('row_count', 0),
                    'column_count': metadata.get('column_count', 0),
                    'file_size_mb': metadata.get('file_size_mb', 0.0),
                    'metadata_json': metadata,
                    'is_active': True,
                },
            )

    def list_dataset_metadata(self, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_dataset_metadata(identity)

    def record_report_metadata(self, identity: dict[str, Any], metadata: dict[str, Any]) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.save_report_metadata(identity, metadata)

    def list_report_metadata(self, identity: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_report_metadata(identity, limit=limit)

    def save_beta_interest_submission(self, identity: dict[str, Any], submission: dict[str, Any]) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.save_beta_interest_submission(identity, submission)

    def list_beta_interest_submissions(self, identity: dict[str, Any], limit: int = 250) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_beta_interest_submissions(identity, limit=limit)

    def update_beta_interest_submission_status(
        self,
        identity: dict[str, Any],
        submission_id: str,
        *,
        follow_up_status: str,
        contacted_at: str = '',
        completed_at: str = '',
    ) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.update_beta_interest_submission_status(
            identity,
            submission_id,
            follow_up_status=follow_up_status,
            contacted_at=contacted_at,
            completed_at=completed_at,
        )

    def list_usage_events(self, identity: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_usage_events(identity, limit=limit)

    def load_workspace_summary(self, identity: dict[str, Any]) -> dict[str, Any]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return {
                'workspace': {},
                'user_count': 0,
                'dataset_count': 0,
                'report_count': 0,
                'usage_event_count': 0,
            }
        return self.persistence_service.load_workspace_summary(identity)

    def save_user_settings(self, identity: dict[str, Any], settings: dict[str, Any]) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.save_user_settings(identity, settings)

    def load_user_settings(self, identity: dict[str, Any]) -> dict[str, Any]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return {}
        return self.persistence_service.load_user_settings(identity)

    def hydrate_user_settings(self, session_state: dict[str, Any]) -> None:
        identity = session_state.get('workspace_identity', {})
        settings = self.load_user_settings(identity)
        if not settings:
            return
        for key in PERSISTED_USER_SETTING_KEYS:
            if key in settings and key not in session_state:
                session_state[key] = settings[key]

    def persist_user_settings(self, session_state: dict[str, Any]) -> None:
        identity = session_state.get('workspace_identity', {})
        settings = {
            key: session_state.get(key)
            for key in PERSISTED_USER_SETTING_KEYS
            if key in session_state
        }
        self.save_user_settings(identity, settings)

    def list_dataset_versions(self, identity: dict[str, Any], dataset_name: str | None = None) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_dataset_versions(identity, dataset_name=dataset_name)

    def save_workspace_snapshot_record(
        self,
        identity: dict[str, Any],
        *,
        snapshot_name: str,
        dataset_name: str,
        dataset_version_id: str = '',
        snapshot_payload: dict[str, Any],
    ) -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.save_workspace_snapshot(
            identity,
            {
                'snapshot_name': snapshot_name,
                'dataset_name': dataset_name,
                'dataset_version_id': dataset_version_id,
                'snapshot_payload': snapshot_payload,
                'created_by_user_id': str(identity.get('user_id', 'guest-user')),
            },
        )

    def list_workspace_snapshot_records(self, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_workspace_snapshots(identity)

    def upsert_collaboration_presence(self, identity: dict[str, Any], *, session_id: str, active_section: str, presence_state: str = 'active') -> None:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return
        self.persistence_service.upsert_collaboration_session(
            identity,
            {
                'session_id': session_id,
                'active_section': active_section,
                'presence_state': presence_state,
            },
        )

    def list_collaboration_presence(self, identity: dict[str, Any]) -> list[dict[str, Any]]:
        if self.persistence_service is None or not bool(getattr(self.persistence_service, 'enabled', False)):
            return []
        return self.persistence_service.list_collaboration_sessions(identity)

    def update_run_history(
        self,
        session_state: dict[str, Any],
        *,
        dataset_name: str,
        pipeline: dict[str, Any],
        demo_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        updated = build_updated_run_history(
            session_state.get('run_history', []),
            dataset_name,
            pipeline,
            demo_config,
        )
        session_state['run_history'] = updated
        return updated

    def build_active_controls(self, session_state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: session_state.get(key)
            for key in ACTIVE_CONTROL_KEYS
            if key in session_state
        }

    def build_execution_context(self, session_state: dict[str, Any]) -> dict[str, Any]:
        context = {
            'active_plan': session_state.get('active_plan', 'Pro'),
            'plan_enforcement_mode': session_state.get('plan_enforcement_mode', 'Demo-safe'),
            'workflow_packs': dict(session_state.get('workflow_packs', {})),
            'saved_snapshots': dict(session_state.get('saved_snapshots', {})),
            'analysis_log': list(session_state.get('analysis_log', [])),
            'run_history': list(session_state.get('run_history', [])),
            'product_demo_mode_enabled': bool(session_state.get('product_demo_mode_enabled', True)),
            'demo_usage_seeded_keys': list(session_state.get('demo_usage_seeded_keys', [])),
            'profile_cache_metrics': dict(session_state.get('profile_cache_metrics', default_profile_cache_metrics())),
        }
        context.update(self.build_active_controls(session_state))
        context.update(self.build_demo_config(session_state))
        for key in PERSISTED_USER_SETTING_KEYS:
            if key in session_state:
                context[key] = session_state.get(key)
        return context

    def build_demo_config(self, session_state: dict[str, Any]) -> dict[str, str]:
        return {
            'synthetic_helper_mode': str(session_state.get('demo_synthetic_helper_mode', 'Auto')),
            'bmi_remediation_mode': str(session_state.get('demo_bmi_remediation_mode', 'median')),
            'synthetic_cost_mode': str(session_state.get('demo_synthetic_cost_mode', 'Auto')),
            'synthetic_readmission_mode': str(session_state.get('demo_synthetic_readmission_mode', 'Auto')),
            'executive_summary_verbosity': str(session_state.get('demo_executive_summary_verbosity', 'Concise')),
            'scenario_simulation_mode': str(session_state.get('demo_scenario_simulation_mode', 'Basic')),
        }

    def split_column_validation_warnings(self, column_validation: dict[str, Any]) -> tuple[str | None, list[str]]:
        warnings = [str(item) for item in column_validation.get('warnings', []) if str(item).strip()]
        empty_column_warning = None
        remaining: list[str] = []
        for warning in warnings:
            if empty_column_warning is None and 'completely empty' in warning.lower():
                empty_column_warning = warning
            else:
                remaining.append(warning)
        return empty_column_warning, remaining

    def execute_analysis_run(
        self,
        session_state: dict[str, Any],
        *,
        data: Any,
        dataset_name: str,
        source_meta: dict[str, Any],
        progress_callback: Any | None = None,
        cancel_check: Any | None = None,
        persist_runtime_state: bool = True,
    ) -> AnalysisExecutionResult:
        if progress_callback:
            progress_callback(
                0.05,
                '1. Loading data...',
                step_index=1,
                total_steps=5,
                current_operation='Loading data and validating the dataset shape',
            )
        large_dataset_profile = get_large_dataset_profile(
            session_state.get('product_large_dataset_profile', 'Standard')
        )
        large_dataset_profile['profile_name'] = str(
            session_state.get('product_large_dataset_profile', 'Standard')
        )
        preflight = build_preflight_guardrails(
            source_meta,
            estimate_memory_mb(data),
            len(data),
            len(data.columns),
            profile_config=large_dataset_profile,
        )
        column_validation = build_column_validation_report(data)
        empty_column_warning, other_column_warnings = self.split_column_validation_warnings(column_validation)
        job_runtime = session_state.get('job_runtime') or build_job_runtime()
        session_state['job_runtime'] = job_runtime
        long_task_notice = build_long_task_notice(
            source_meta,
            len(data),
            estimate_memory_mb(data),
            job_runtime=job_runtime,
            profile_config=large_dataset_profile,
        )
        demo_config = self.build_demo_config(session_state)
        if preflight.get('blocked'):
            return AnalysisExecutionResult(
                dataset_name=dataset_name,
                source_meta=source_meta,
                preflight=preflight,
                column_validation=column_validation,
                job_runtime=job_runtime,
                large_dataset_profile=large_dataset_profile,
                long_task_notice=long_task_notice,
                empty_column_warning=empty_column_warning,
                other_column_warnings=other_column_warnings,
                demo_config=demo_config,
                blocked=True,
            )
        pipeline = run_analysis_pipeline(
            data,
            dataset_name,
            source_meta,
            demo_config=demo_config,
            active_control_values=self.build_active_controls(session_state),
            large_dataset_profile=large_dataset_profile,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            cache_metrics=session_state.setdefault('profile_cache_metrics', default_profile_cache_metrics()),
        )
        if progress_callback:
            progress_callback(
                0.98,
                '5. Finalizing insights...',
                step_index=5,
                total_steps=5,
                current_operation='Finalizing runtime views and packaging the analysis results',
            )
        startup_readiness = build_startup_readiness_summary(
            preflight,
            column_validation,
            source_meta,
            pipeline['sample_info'],
        )
        plan_awareness = build_plan_awareness(
            session_state.get('active_plan', 'Pro'),
            session_state.get('plan_enforcement_mode', 'Demo-safe'),
            source_meta,
            workflow_pack_count=len(session_state.get('workflow_packs', {})),
            snapshot_count=len(session_state.get('saved_snapshots', {})),
        )
        deployment_health_checks = build_deployment_health_checks(pipeline, source_meta)
        performance_diagnostics = build_performance_diagnostics(
            pipeline['overview'],
            pipeline['sample_info'],
            source_meta,
            profile_config=large_dataset_profile,
        )
        if persist_runtime_state:
            self.finalize_analysis_run(
                session_state,
                dataset_name=dataset_name,
                pipeline=pipeline,
                demo_config=demo_config,
            )
        demo_usage_seed = build_demo_usage_seed_events(
            dataset_name,
            source_meta,
            demo_mode_enabled=bool(session_state.get('product_demo_mode_enabled', True)),
            seeded_keys=session_state.get('demo_usage_seeded_keys', []),
        )
        pipeline = finalize_runtime_pipeline(
            pipeline,
            dataset_name=dataset_name,
            source_meta=source_meta,
            preflight=preflight,
            column_validation=column_validation,
            job_runtime=job_runtime,
            heavy_task_catalog=build_heavy_task_catalog(),
            environment_checks=build_environment_checks(),
            startup_readiness=startup_readiness,
            plan_awareness=plan_awareness,
            deployment_health_checks=deployment_health_checks,
            performance_diagnostics=performance_diagnostics,
            run_history=session_state.get('run_history', []),
            analysis_log=session_state.get('analysis_log', []),
            demo_config=demo_config,
        )
        if progress_callback:
            progress_callback(
                1.0,
                'Analysis ready.',
                step_index=5,
                total_steps=5,
                current_operation='Analysis ready',
            )
        return AnalysisExecutionResult(
            dataset_name=dataset_name,
            source_meta=source_meta,
            preflight=preflight,
            column_validation=column_validation,
            job_runtime=job_runtime,
            large_dataset_profile=large_dataset_profile,
            long_task_notice=long_task_notice,
            empty_column_warning=empty_column_warning,
            other_column_warnings=other_column_warnings,
            demo_config=demo_config,
            blocked=False,
            pipeline=pipeline,
            demo_usage_seed=demo_usage_seed,
        )

    def apply_completed_analysis_result(
        self,
        session_state: dict[str, Any],
        *,
        analysis_result: AnalysisExecutionResult,
    ) -> list[dict[str, Any]]:
        if analysis_result.blocked or analysis_result.pipeline is None:
            return list(session_state.get('run_history', []))
        return self.finalize_analysis_run(
            session_state,
            dataset_name=analysis_result.dataset_name,
            pipeline=analysis_result.pipeline,
            demo_config=analysis_result.demo_config,
        )

    def finalize_analysis_run(
        self,
        session_state: dict[str, Any],
        *,
        dataset_name: str,
        pipeline: dict[str, Any],
        demo_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        updated = self.update_run_history(
            session_state,
            dataset_name=dataset_name,
            pipeline=pipeline,
            demo_config=demo_config,
        )
        self.persist_user_settings(session_state)
        self.persist_workspace_state(session_state)
        return updated


def build_workspace_application_service(persistence_service: Any | None) -> WorkspaceApplicationService:
    return WorkspaceApplicationService(persistence_service=persistence_service)


__all__ = [
    'AnalysisExecutionResult',
    'WorkspaceApplicationService',
    'build_workspace_application_service',
]
