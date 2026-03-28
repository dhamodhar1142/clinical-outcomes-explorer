from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.services.copilot_service import execute_copilot_prompt, plan_copilot_workflow
from src.services.admin_ops_service import build_admin_ops_service
from src.services.application_service import build_workspace_application_service
from src.services.dataset_service import build_uploaded_dataset_bundle, load_primary_dataset_from_ui
from src.services.export_service import generate_export_report_output, prepare_policy_aware_export_bundle
from src.services.job_service import read_background_task, submit_background_task
from src.services.report_service import generate_report_deliverable
from src.services.runtime_service import initialize_app_session_state
from src.services.workspace_service import (
    add_collaboration_note_to_workspace,
    load_snapshot_into_session,
    load_workflow_pack_into_session,
    save_snapshot_to_workspace,
    save_workflow_pack_to_workspace,
)


class ServiceLayerTests(unittest.TestCase):
    def test_application_service_coordinates_persistence_metadata_and_history(self) -> None:
        class _PersistenceService:
            enabled = True

            def __init__(self) -> None:
                self.dataset_calls = []
                self.report_calls = []
                self.usage_calls = []

            def save_dataset_metadata(self, identity, metadata):
                self.dataset_calls.append((identity, metadata))

            def save_report_metadata(self, identity, metadata):
                self.report_calls.append((identity, metadata))

            def record_usage_event(self, identity, event):
                self.usage_calls.append((identity, event))

        persistence_service = _PersistenceService()
        application_service = build_workspace_application_service(persistence_service)
        identity = {'workspace_id': 'workspace-a', 'user_id': 'user-a'}
        session_state = {'run_history': []}

        application_service.record_dataset_metadata(identity, {'dataset_name': 'demo.csv'})
        application_service.record_report_metadata(identity, {'report_type': 'Executive Report'})
        application_service.record_usage_event(identity, {'event_type': 'Dataset Selected'})
        updated = application_service.update_run_history(
            session_state,
            dataset_name='demo.csv',
            pipeline={
                'overview': {'rows': 1},
                'readiness': {'readiness_score': 1.0, 'major_blockers': []},
                'quality': {'quality_score': 1.0},
                'healthcare': {'healthcare_readiness_score': 1.0},
            },
            demo_config={'synthetic_helper_mode': 'Auto'},
        )

        self.assertEqual(persistence_service.dataset_calls[0][1]['dataset_name'], 'demo.csv')
        self.assertEqual(persistence_service.report_calls[0][1]['report_type'], 'Executive Report')
        self.assertEqual(persistence_service.usage_calls[0][1]['event_type'], 'Dataset Selected')
        self.assertEqual(session_state['run_history'], updated)

    def test_application_service_builds_control_payloads_and_finalizes_analysis(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        session_state = {
            'analysis_template': 'General Review',
            'report_mode': 'Executive Summary',
            'active_role': 'Analyst',
            'demo_synthetic_helper_mode': 'Auto',
            'demo_bmi_remediation_mode': 'median',
            'demo_synthetic_cost_mode': 'Auto',
            'demo_synthetic_readmission_mode': 'Auto',
            'demo_executive_summary_verbosity': 'Concise',
            'demo_scenario_simulation_mode': 'Basic',
            'run_history': [],
        }

        controls = application_service.build_active_controls(session_state)
        demo_config = application_service.build_demo_config(session_state)
        updated = application_service.finalize_analysis_run(
            session_state,
            dataset_name='demo.csv',
            pipeline={
                'overview': {'rows': 1},
                'readiness': {'readiness_score': 1.0, 'major_blockers': []},
                'quality': {'quality_score': 1.0},
                'healthcare': {'healthcare_readiness_score': 1.0},
            },
            demo_config=demo_config,
        )

        self.assertEqual(controls['analysis_template'], 'General Review')
        self.assertEqual(demo_config['executive_summary_verbosity'], 'Concise')
        self.assertEqual(session_state['run_history'], updated)

    def test_application_service_hydrates_persisted_settings_over_defaults(self) -> None:
        class _PersistenceService:
            enabled = True

            def load_user_settings(self, identity):
                return {
                    'accuracy_benchmark_profile': 'Hospital Encounters',
                    'evolution_execution_queue': [{'proposal_title': 'Queued item'}],
                    'governance_default_owner': 'Data Steward',
                    'validation_runtime_preference': 'Prefer local/staging for heavy actions',
                    'approval_routing_rules': {'release_signoff': 'Admin'},
                    'governance_policy_packs': {'Enterprise Release Controls': {'export_policy_name': 'Internal Review'}},
                }

            def save_user_settings(self, identity, settings):
                return None

        application_service = build_workspace_application_service(_PersistenceService())
        session_state = {
            'workspace_identity': {'workspace_id': 'workspace-a', 'user_id': 'user-a'},
            'accuracy_benchmark_profile': 'Auto',
            'evolution_execution_queue': [],
        }

        application_service.hydrate_user_settings(session_state)

        self.assertEqual(session_state['accuracy_benchmark_profile'], 'Hospital Encounters')
        self.assertEqual(session_state['evolution_execution_queue'][0]['proposal_title'], 'Queued item')
        self.assertEqual(session_state['governance_default_owner'], 'Data Steward')
        self.assertEqual(session_state['validation_runtime_preference'], 'Prefer local/staging for heavy actions')
        self.assertEqual(session_state['approval_routing_rules']['release_signoff'], 'Admin')
        self.assertIn('Enterprise Release Controls', session_state['governance_policy_packs'])

    def test_application_service_build_active_controls_includes_policy_center_fields(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        session_state = {
            'analysis_template': 'General Review',
            'report_mode': 'Executive Summary',
            'governance_default_owner': 'Admin',
            'governance_default_reviewer_role': 'Data Steward',
            'governance_release_gate_mode': 'Strict signoff',
            'validation_runtime_preference': 'Auto',
            'export_runtime_preference': 'Allow lightweight cloud actions only',
            'governance_policy_packs': {'Pack A': {'export_policy_name': 'Internal Review'}},
            'approval_routing_rules': {'mapping_approval': 'Data Steward'},
        }

        controls = application_service.build_active_controls(session_state)

        self.assertEqual(controls['governance_default_owner'], 'Admin')
        self.assertEqual(controls['governance_default_reviewer_role'], 'Data Steward')
        self.assertEqual(controls['governance_release_gate_mode'], 'Strict signoff')
        self.assertEqual(controls['validation_runtime_preference'], 'Auto')
        self.assertEqual(controls['export_runtime_preference'], 'Allow lightweight cloud actions only')
        self.assertIn('Pack A', controls['governance_policy_packs'])
        self.assertEqual(controls['approval_routing_rules']['mapping_approval'], 'Data Steward')

    def test_application_service_executes_analysis_and_runtime_finalization(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        session_state = {
            'active_plan': 'Pro',
            'plan_enforcement_mode': 'Demo-safe',
            'workflow_packs': {},
            'saved_snapshots': {},
            'analysis_log': [],
            'run_history': [],
            'product_demo_mode_enabled': True,
            'demo_usage_seeded_keys': [],
        }
        data = pd.DataFrame([{'patient_id': 1, 'cost': 10.0}])
        pipeline = {
            'sample_info': {'sampling_applied': False},
            'overview': {'rows': 1, 'columns': 2},
            'quality': {'quality_score': 0.9},
            'readiness': {'readiness_score': 0.8, 'major_blockers': []},
            'healthcare': {'healthcare_readiness_score': 0.7},
        }

        with patch('src.services.application_service.build_preflight_guardrails', return_value={'blocked': False, 'warnings': []}), \
             patch('src.services.application_service.build_column_validation_report', return_value={'warnings': [], 'summary': 'ok'}), \
             patch('src.services.application_service.build_job_runtime', return_value={'mode': 'sync'}), \
             patch('src.services.application_service.build_long_task_notice', return_value=''), \
             patch('src.services.application_service.run_analysis_pipeline', return_value=pipeline), \
             patch('src.services.application_service.build_startup_readiness_summary', return_value={'status': 'ready'}), \
             patch('src.services.application_service.build_plan_awareness', return_value={'plan': 'Pro'}), \
             patch('src.services.application_service.build_deployment_health_checks', return_value={'status': 'ok'}), \
             patch('src.services.application_service.build_performance_diagnostics', return_value={'status': 'ok'}), \
             patch('src.services.application_service.build_demo_usage_seed_events', return_value={'seeded': True, 'seeded_keys': ['demo.csv'], 'events': []}), \
             patch('src.services.application_service.build_heavy_task_catalog', return_value={'report_generation': {}}), \
             patch('src.services.application_service.build_environment_checks', return_value={'python': 'ok'}), \
             patch('src.services.application_service.finalize_runtime_pipeline', return_value={**pipeline, 'landing_summary': {}}):
            result = application_service.execute_analysis_run(
                session_state,
                data=data,
                dataset_name='demo.csv',
                source_meta={'source_mode': 'Demo dataset'},
            )

        self.assertFalse(result.blocked)
        self.assertEqual(result.job_runtime['mode'], 'sync')
        self.assertTrue(result.demo_usage_seed['seeded'])
        self.assertIn('landing_summary', result.pipeline)
        self.assertEqual(session_state['run_history'][0]['dataset_name'], 'demo.csv')

    def test_application_service_can_execute_with_detached_context_and_apply_result_later(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        session_state = {
            'active_plan': 'Pro',
            'plan_enforcement_mode': 'Demo-safe',
            'workflow_packs': {},
            'saved_snapshots': {},
            'analysis_log': [],
            'run_history': [],
            'product_demo_mode_enabled': True,
            'demo_usage_seeded_keys': [],
            'analysis_template': 'General Review',
            'report_mode': 'Executive Summary',
        }
        data = pd.DataFrame([{'patient_id': 1, 'cost': 10.0}])
        pipeline = {
            'sample_info': {'sampling_applied': False},
            'overview': {'rows': 1, 'columns': 2},
            'quality': {'quality_score': 0.9},
            'readiness': {'readiness_score': 0.8, 'major_blockers': []},
            'healthcare': {'healthcare_readiness_score': 0.7},
        }
        execution_context = application_service.build_execution_context(session_state)

        with patch('src.services.application_service.build_preflight_guardrails', return_value={'blocked': False, 'warnings': []}), \
             patch('src.services.application_service.build_column_validation_report', return_value={'warnings': [], 'summary': 'ok'}), \
             patch('src.services.application_service.build_job_runtime', return_value={'mode': 'sync'}), \
             patch('src.services.application_service.build_long_task_notice', return_value=''), \
             patch('src.services.application_service.run_analysis_pipeline', return_value=pipeline), \
             patch('src.services.application_service.build_startup_readiness_summary', return_value={'status': 'ready'}), \
             patch('src.services.application_service.build_plan_awareness', return_value={'plan': 'Pro'}), \
             patch('src.services.application_service.build_deployment_health_checks', return_value={'status': 'ok'}), \
             patch('src.services.application_service.build_performance_diagnostics', return_value={'status': 'ok'}), \
             patch('src.services.application_service.build_demo_usage_seed_events', return_value={'seeded': False, 'seeded_keys': [], 'events': []}), \
             patch('src.services.application_service.build_heavy_task_catalog', return_value={'report_generation': {}}), \
             patch('src.services.application_service.build_environment_checks', return_value={'python': 'ok'}), \
             patch('src.services.application_service.finalize_runtime_pipeline', return_value={**pipeline, 'landing_summary': {}}):
            result = application_service.execute_analysis_run(
                execution_context,
                data=data,
                dataset_name='demo.csv',
                source_meta={'source_mode': 'Demo dataset'},
                persist_runtime_state=False,
            )

        self.assertEqual(session_state['run_history'], [])
        application_service.apply_completed_analysis_result(
            session_state,
            analysis_result=result,
        )
        self.assertEqual(session_state['run_history'][0]['dataset_name'], 'demo.csv')

    def test_application_service_emits_structured_progress_updates(self) -> None:
        application_service = build_workspace_application_service(persistence_service=None)
        session_state = {
            'active_plan': 'Pro',
            'plan_enforcement_mode': 'Demo-safe',
            'workflow_packs': {},
            'saved_snapshots': {},
            'analysis_log': [],
            'run_history': [],
            'product_demo_mode_enabled': True,
            'demo_usage_seeded_keys': [],
        }
        data = pd.DataFrame([{'patient_id': 1, 'cost': 10.0}])
        events: list[tuple[float, str, dict[str, object]]] = []

        def _progress(value, message, **metadata):
            events.append((value, message, metadata))

        with patch('src.services.application_service.build_preflight_guardrails', return_value={'blocked': False, 'warnings': []}), \
             patch('src.services.application_service.build_column_validation_report', return_value={'warnings': [], 'summary': 'ok'}), \
             patch('src.services.application_service.build_job_runtime', return_value={'mode': 'sync'}), \
             patch('src.services.application_service.build_long_task_notice', return_value=''), \
             patch('src.services.application_service.run_analysis_pipeline', return_value={'sample_info': {}, 'overview': {}, 'quality': {}, 'readiness': {}, 'healthcare': {}}), \
             patch('src.services.application_service.build_startup_readiness_summary', return_value={'status': 'ready'}), \
             patch('src.services.application_service.build_plan_awareness', return_value={'plan': 'Pro'}), \
             patch('src.services.application_service.build_deployment_health_checks', return_value={'status': 'ok'}), \
             patch('src.services.application_service.build_performance_diagnostics', return_value={'status': 'ok'}), \
             patch('src.services.application_service.build_demo_usage_seed_events', return_value={'seeded': False, 'seeded_keys': [], 'events': []}), \
             patch('src.services.application_service.build_heavy_task_catalog', return_value={'report_generation': {}}), \
             patch('src.services.application_service.build_environment_checks', return_value={'python': 'ok'}), \
             patch('src.services.application_service.finalize_runtime_pipeline', return_value={'landing_summary': {}}):
            application_service.execute_analysis_run(
                session_state,
                data=data,
                dataset_name='demo.csv',
                source_meta={'source_mode': 'Demo dataset'},
                progress_callback=_progress,
            )

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0][1], '1. Loading data...')
        self.assertEqual(events[0][2]['step_index'], 1)
        self.assertEqual(events[-1][2]['total_steps'], 5)

    def test_runtime_service_initializes_backend_adapters_and_defaults(self) -> None:
        session_state: dict[str, object] = {}
        services = initialize_app_session_state(session_state)
        self.assertIn('persistence_service', session_state)
        self.assertIn('auth_service', session_state)
        self.assertIn('storage_service', session_state)
        self.assertIn('job_runtime', session_state)
        self.assertIn('application_service', session_state)
        self.assertIn('admin_ops_service', session_state)
        self.assertEqual(session_state['analysis_template'], 'General Review')
        self.assertEqual(session_state['workspace_name'], 'Guest Demo Workspace')
        self.assertIn('semantic_mapping_profiles', session_state)
        self.assertIn('organization_benchmark_packs', session_state)
        self.assertIn('dataset_review_approvals', session_state)
        self.assertIn('evolution_memory', session_state)
        self.assertIn('evolution_execution_queue', session_state)
        self.assertIs(session_state['persistence_service'], services.persistence_service)
        admin_view = services.admin_ops_service.build_admin_ops_view(
            workspace_identity=session_state['workspace_identity'],
            plan_awareness={},
            analysis_log=[],
            run_history=[],
        )
        self.assertIn('support_diagnostics', admin_view)
        self.assertIn('support_diagnostics_table', admin_view['product_admin'])

    def test_uploaded_dataset_bundle_returns_metadata_and_artifact(self) -> None:
        df = pd.DataFrame([{'patient_id': 1, 'cost': 12.5}])

        class _StorageService:
            enabled = True

            def save_dataset_upload(self, identity, **kwargs):
                return {'artifact_path': 'artifacts/demo.csv'}

        class _PersistenceService:
            enabled = True

            def __init__(self) -> None:
                self.saved_metadata = None

            def save_dataset_metadata(self, identity, metadata):
                self.saved_metadata = metadata

        persistence_service = _PersistenceService()
        with patch(
            'src.services.dataset_service.load_uploaded_file_bundle',
            return_value=(
                df,
                {'patient_id': 'patient_id'},
                {
                    'ingestion_strategy': 'standard_csv',
                    'sampling_mode': 'full',
                    'source_row_count': 1,
                    'analyzed_row_count': 1,
                    'chunk_count': 1,
                    'dataset_cache_key': 'cache-key',
                },
            ),
        ):
            data, original_lookup, source_meta, artifact = build_uploaded_dataset_bundle(
                'demo.csv',
                b'patient_id,cost\n1,12.5\n',
                storage_service=_StorageService(),
                persistence_service=persistence_service,
                workspace_identity={'workspace_id': 'workspace-a'},
            )
        self.assertEqual(len(data), 1)
        self.assertEqual(original_lookup['patient_id'], 'patient_id')
        self.assertEqual(source_meta['source_mode'], 'Uploaded dataset')
        self.assertEqual(artifact['artifact_path'], 'artifacts/demo.csv')
        self.assertEqual(persistence_service.saved_metadata['dataset_name'], 'demo.csv')

    def test_load_primary_dataset_from_ui_returns_demo_bundle(self) -> None:
        class _Sidebar:
            def radio(self, *args, **kwargs):
                return 'Built-in example dataset'

            def selectbox(self, *args, **kwargs):
                return 'Healthcare Operations Demo'

        class _Ui:
            def info(self, *args, **kwargs):
                return None

            def error(self, *args, **kwargs):
                return None

            def caption(self, *args, **kwargs):
                return None

            def warning(self, *args, **kwargs):
                return None

        with patch('src.services.dataset_service.build_demo_dataset_bundle', return_value=(pd.DataFrame([{'a': 1}]), {'a': 'a'}, {'source_mode': 'Demo dataset'})):
            selection = load_primary_dataset_from_ui(
                sidebar=_Sidebar(),
                ui=_Ui(),
                session_state={'application_service': None, 'persistence_service': None, 'workspace_identity': {}},
            )

        self.assertEqual(selection.source_mode, 'Built-in example dataset')
        self.assertEqual(selection.dataset_name, 'Healthcare Operations Demo')
        self.assertEqual(len(selection.data), 1)

    def test_load_primary_dataset_from_ui_reuses_active_uploaded_bundle_when_no_replacement_is_selected(self) -> None:
        active_frame = pd.DataFrame([{'patient_id': 'P-1', 'value': 10}])

        class _Sidebar:
            def radio(self, *args, **kwargs):
                return 'Uploaded dataset'

            def markdown(self, *args, **kwargs):
                return None

            def file_uploader(self, *args, **kwargs):
                return None

            def caption(self, *args, **kwargs):
                return None

        class _Ui:
            def info(self, *args, **kwargs):
                return None

        selection = load_primary_dataset_from_ui(
            sidebar=_Sidebar(),
            ui=_Ui(),
            session_state={
                'application_service': None,
                'persistence_service': None,
                'workspace_identity': {},
                'active_dataset_bundle': {
                    'source_mode': 'Uploaded dataset',
                    'data': active_frame,
                    'original_lookup': {'patient_id': 'patient_id', 'value': 'value'},
                    'dataset_name': 'uploaded.csv',
                    'source_meta': {'source_mode': 'Uploaded dataset', 'dataset_cache_key': 'upload-key'},
                },
            },
        )

        self.assertEqual(selection.source_mode, 'Uploaded dataset')
        self.assertEqual(selection.dataset_name, 'uploaded.csv')
        self.assertEqual(selection.source_meta['dataset_cache_key'], 'upload-key')
        self.assertEqual(len(selection.data), 1)

    def test_uploaded_dataset_bundle_blocks_signed_in_viewer_uploads(self) -> None:
        with self.assertRaises(PermissionError):
            build_uploaded_dataset_bundle(
                'demo.csv',
                b'patient_id,cost\n1,12.5\n',
                workspace_identity={'workspace_id': 'workspace-a', 'auth_mode': 'local', 'role': 'viewer', 'workspace_name': 'Pilot Workspace'},
            )

    def test_uploaded_dataset_bundle_blocks_invalid_signed_in_membership(self) -> None:
        with self.assertRaises(PermissionError):
            build_uploaded_dataset_bundle(
                'demo.csv',
                b'patient_id,cost\n1,12.5\n',
                workspace_identity={
                    'workspace_id': 'workspace-a',
                    'auth_mode': 'local',
                    'role': 'admin',
                    'workspace_name': 'Pilot Workspace',
                    'membership_validated': False,
                },
            )

    def test_workspace_service_saves_and_loads_snapshot_and_pack(self) -> None:
        session_state = {}
        workspace_identity = {'workspace_id': 'workspace-a', 'user_id': 'user-a', 'display_name': 'Analyst', 'role': 'admin'}
        save_snapshot_to_workspace(
            session_state,
            snapshot_name='baseline',
            dataset_name='demo.csv',
            controls={'active_role': 'Analyst'},
            workspace_identity=workspace_identity,
        )
        session_state['active_role'] = 'Viewer'
        snapshot = load_snapshot_into_session(session_state, 'baseline')
        self.assertEqual(snapshot['dataset_name'], 'demo.csv')
        self.assertEqual(session_state['active_role'], 'Analyst')

        save_workflow_pack_to_workspace(
            session_state,
            workflow_name='ops-pack',
            details={'summary': 'Ops review'},
            summary={'status': 'Ready'},
            controls={'report_mode': 'Operational Report'},
            workspace_identity=workspace_identity,
        )
        session_state['report_mode'] = 'Analyst Report'
        workflow = load_workflow_pack_into_session(session_state, 'ops-pack')
        self.assertEqual(workflow['details']['summary'], 'Ops review')
        self.assertEqual(session_state['report_mode'], 'Operational Report')

    def test_workspace_service_blocks_signed_in_viewer_writes(self) -> None:
        with self.assertRaises(PermissionError):
            save_snapshot_to_workspace(
                {},
                snapshot_name='baseline',
                dataset_name='demo.csv',
                controls={'active_role': 'Analyst'},
                workspace_identity={'workspace_id': 'workspace-a', 'user_id': 'user-a', 'display_name': 'Viewer', 'role': 'viewer', 'auth_mode': 'local', 'workspace_name': 'Pilot Workspace'},
            )

    def test_workspace_service_blocks_invalid_signed_in_membership_for_writes(self) -> None:
        with self.assertRaises(PermissionError):
            save_workflow_pack_to_workspace(
                {},
                workflow_name='ops-pack',
                details={'summary': 'Ops review'},
                summary={'status': 'Ready'},
                controls={'report_mode': 'Operational Report'},
                workspace_identity={
                    'workspace_id': 'workspace-a',
                    'user_id': 'user-a',
                    'display_name': 'Admin',
                    'role': 'admin',
                    'auth_mode': 'local',
                    'workspace_name': 'Pilot Workspace',
                    'membership_validated': False,
                },
            )

    def test_workspace_service_blocks_cross_workspace_snapshot_loads(self) -> None:
        session_state = {
            'saved_snapshots': {
                'foreign-snapshot': {
                    'dataset_name': 'demo.csv',
                    'workspace_id': 'workspace-b',
                    'controls': {'active_role': 'Analyst'},
                }
            }
        }
        with self.assertRaises(PermissionError):
            load_snapshot_into_session(
                session_state,
                'foreign-snapshot',
                workspace_identity={'workspace_id': 'workspace-a', 'auth_mode': 'local', 'role': 'admin', 'workspace_name': 'Pilot Workspace', 'membership_validated': True},
            )

    def test_workspace_service_adds_collaboration_note(self) -> None:
        session_state = {}
        note = add_collaboration_note_to_workspace(
            session_state,
            target_type='Dataset',
            target_name='demo.csv',
            note_text='Review cost completeness.',
            workspace_identity={'display_name': 'Analyst', 'workspace_name': 'Pilot Workspace', 'workspace_id': 'workspace-a', 'user_id': 'user-a', 'role': 'admin'},
        )
        self.assertEqual(note['target_name'], 'demo.csv')
        self.assertEqual(note['author'], 'Analyst')

    def test_copilot_service_wraps_prompt_and_workflow_planning(self) -> None:
        class _FakeCopilot:
            @staticmethod
            def append_copilot_message(*args, **kwargs):
                return None

            @staticmethod
            def run_copilot_question(*args, **kwargs):
                return {'answer': 'Summary ready.'}

            @staticmethod
            def initialize_copilot_memory():
                return [{'role': 'assistant', 'content': 'Summary ready.'}]

            @staticmethod
            def plan_workflow_action(*args, **kwargs):
                return {'planned_action': 'open_export_center'}

        with patch('src.services.copilot_service._load_ai_copilot', return_value=_FakeCopilot()):
            result = execute_copilot_prompt(
                'Summarize the dataset',
                data=pd.DataFrame([{'a': 1}]),
                schema_context={'matched_schema': {}, 'dataset_name': 'demo.csv'},
            )
        self.assertEqual(result['answer'], 'Summary ready.')
        self.assertTrue(result['messages'])

        with patch('src.services.copilot_service._load_ai_copilot', return_value=_FakeCopilot()):
            workflow = plan_copilot_workflow(
                'Generate a report',
                data=pd.DataFrame([{'a': 1}]),
                canonical_map={},
                readiness={},
                healthcare={},
                remediation=pd.DataFrame(),
            )
        self.assertEqual(workflow['planned_action'], 'open_export_center')

    def test_report_service_returns_job_payload(self) -> None:
        session_state = {'job_runs': []}
        pipeline = {
            'overview': {'rows': 1},
            'quality': {'quality_score': 1.0},
            'readiness': {'readiness_score': 1.0},
            'healthcare': {'healthcare_readiness_score': 1.0},
            'insights': {'summary_lines': []},
            'action_recommendations': pd.DataFrame(),
        }
        with patch('src.services.report_service.submit_background_task', return_value={'job_id': 'job-1'}), \
             patch('src.services.report_service.read_background_task', return_value={'status': {'status': 'completed'}, 'result': b'report body'}):
            result = generate_report_deliverable(
                session_state,
                job_runtime={'status_label': 'Synchronous fallback'},
                report_label='Executive Report',
                dataset_name='demo.csv',
                pipeline=pipeline,
                workspace_identity={'workspace_id': 'guest-demo-workspace', 'auth_mode': 'guest', 'role': 'viewer'},
            )
        self.assertEqual(result['job']['job_id'], 'job-1')
        self.assertEqual(result['status']['status'], 'completed')
        self.assertEqual(result['result'], b'report body')

    def test_job_service_wraps_background_job_contract(self) -> None:
        session_state: dict[str, object] = {'job_runs': []}
        submission = submit_background_task(
            session_state,
            job_runtime={'backend_configured': False, 'mode': 'sync'},
            task_key='report_generation',
            task_label='Executive Report',
            detail='Generated report.',
            runner=lambda: b'report',
        )
        job_state = read_background_task(session_state, submission['job_id'])
        self.assertEqual(job_state['status']['status'], 'completed')
        self.assertEqual(job_state['result'], b'report')

    def test_export_service_prepares_policy_aware_bundle(self) -> None:
        export_bundle = prepare_policy_aware_export_bundle(
            role='Analyst',
            report_mode='Executive Summary',
            policy_name='Internal Review',
            export_allowed=True,
            privacy_review={},
        )
        self.assertFalse(export_bundle['bundle_manifest'].empty)
        self.assertTrue(export_bundle['bundle_title'])
        self.assertFalse(export_bundle['bundle_table'].empty)

    def test_export_service_coordinates_report_metadata_and_artifacts(self) -> None:
        class _ApplicationService:
            def __init__(self) -> None:
                self.report_calls = []

            def record_report_metadata(self, identity, metadata):
                self.report_calls.append((identity, metadata))

        class _StorageService:
            enabled = True

            def save_report_artifact(self, identity, **kwargs):
                return {'stored': True, 'artifact_path': 'artifacts/executive_report.txt'}

        app_service = _ApplicationService()
        storage_service = _StorageService()
        with patch('src.services.export_service.generate_report_deliverable', return_value={'job': {'job_id': 'job-1'}, 'status': {'status': 'completed'}, 'result': b'report body'}):
            result = generate_export_report_output(
                {'job_runs': []},
                job_runtime={'status_label': 'Synchronous fallback'},
                report_label='Executive Report',
                dataset_name='demo.csv',
                pipeline={},
                workspace_identity={'workspace_id': 'workspace-a'},
                role='Executive',
                application_service=app_service,
                storage_service=storage_service,
            )
        self.assertEqual(result['artifact']['artifact_path'], 'artifacts/executive_report.txt')
        self.assertEqual(app_service.report_calls[0][1]['report_type'], 'Executive Report')

    def test_admin_ops_service_consolidates_workspace_activity_views(self) -> None:
        class _ApplicationService:
            enabled = True

            def load_workspace_summary(self, identity):
                return {'dataset_count': 2, 'report_count': 1, 'usage_event_count': 3}

            def list_usage_events(self, identity, limit=25):
                return [{'event_type': 'Dataset Selected'}]

            def list_report_metadata(self, identity, limit=25):
                return [{'dataset_name': 'demo.csv', 'report_type': 'Executive Report', 'status': 'generated', 'generated_at': '2026-03-14'}]

            def list_dataset_metadata(self, identity):
                return [{'dataset_name': 'demo.csv', 'row_count': 12}]

        service = build_admin_ops_service(_ApplicationService(), persistence_service=None)
        result = service.build_admin_ops_view(
            workspace_identity={'workspace_id': 'workspace-a', 'workspace_name': 'Pilot Workspace', 'display_name': 'Analyst', 'role_label': 'Admin'},
            plan_awareness={'active_plan': 'Pro'},
            analysis_log=[{'event_type': 'Dataset Selected'}],
            run_history=[{'dataset_name': 'demo.csv', 'major_blockers_detected': '', 'synthetic_helper_fields_added': 0}],
            visited_sections=['Data Intake'],
            generated_report_outputs={'Executive Report': 'report body'},
            saved_snapshots={'baseline': {}},
            workflow_packs={'ops-pack': {}},
        )
        self.assertIn('usage', result)
        self.assertIn('customer_success', result)
        self.assertIn('product_admin', result)
        self.assertEqual(result['persisted_workspace_summary']['dataset_count'], 2)
        self.assertEqual(result['product_admin']['report_action_count'], 1)

    def test_report_service_blocks_signed_in_viewer_exports(self) -> None:
        session_state = {'job_runs': []}
        pipeline = {
            'overview': {'rows': 1},
            'quality': {'quality_score': 1.0},
            'readiness': {'readiness_score': 1.0},
            'healthcare': {'healthcare_readiness_score': 1.0},
            'insights': {'summary_lines': []},
            'action_recommendations': pd.DataFrame(),
        }
        with self.assertRaises(PermissionError):
            generate_report_deliverable(
                session_state,
                job_runtime={'status_label': 'Synchronous fallback'},
                report_label='Executive Report',
                dataset_name='demo.csv',
                pipeline=pipeline,
                workspace_identity={'workspace_id': 'workspace-a', 'auth_mode': 'local', 'role': 'viewer', 'workspace_name': 'Pilot Workspace'},
                role='Executive',
            )

    def test_report_service_blocks_invalid_signed_in_workspace_scope(self) -> None:
        session_state = {'job_runs': []}
        pipeline = {
            'overview': {'rows': 1},
            'quality': {'quality_score': 1.0},
            'readiness': {'readiness_score': 1.0},
            'healthcare': {'healthcare_readiness_score': 1.0},
            'insights': {'summary_lines': []},
            'action_recommendations': pd.DataFrame(),
        }
        with self.assertRaises(PermissionError):
            generate_report_deliverable(
                session_state,
                job_runtime={'status_label': 'Synchronous fallback'},
                report_label='Executive Report',
                dataset_name='demo.csv',
                pipeline=pipeline,
                workspace_identity={'workspace_id': '', 'auth_mode': 'local', 'role': 'admin', 'workspace_name': 'Pilot Workspace', 'membership_validated': False},
                role='Executive',
            )


if __name__ == '__main__':
    unittest.main()
