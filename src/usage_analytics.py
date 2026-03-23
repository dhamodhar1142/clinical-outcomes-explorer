from __future__ import annotations

from typing import Any

import pandas as pd


def build_product_admin_summary(
    *,
    workspace_identity: dict[str, Any] | None,
    persistence_service: Any | None,
    application_service: Any | None = None,
    plan_awareness: dict[str, Any] | None,
    analysis_log: list[dict[str, Any]] | None,
    run_history: list[dict[str, Any]] | None,
    generated_report_outputs: dict[str, Any] | None = None,
    saved_snapshots: dict[str, Any] | None = None,
    workflow_packs: dict[str, Any] | None = None,
    persisted_workspace_summary: dict[str, Any] | None = None,
    persisted_usage_events: list[dict[str, Any]] | None = None,
    persisted_reports: list[dict[str, Any]] | None = None,
    persisted_datasets: list[dict[str, Any]] | None = None,
    support_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace_identity = workspace_identity or {}
    plan_awareness = plan_awareness or {}
    analysis_log = analysis_log or []
    run_history = run_history or []
    generated_report_outputs = generated_report_outputs or {}
    saved_snapshots = saved_snapshots or {}
    workflow_packs = workflow_packs or {}

    workspace_name = str(workspace_identity.get('workspace_name', 'Guest Demo Workspace'))
    workspace_id = str(workspace_identity.get('workspace_id', 'guest-demo-workspace'))
    role_label = str(workspace_identity.get('role_label', workspace_identity.get('role', 'Viewer')))
    plan_name = str(plan_awareness.get('active_plan', workspace_identity.get('plan_name', 'Pro')))
    strict_enforcement = bool(plan_awareness.get('strict_enforcement'))
    persistence_enabled = bool(getattr(persistence_service, 'enabled', False)) if persistence_service is not None else False
    persistence_status = getattr(persistence_service, 'status', None)
    persistence_mode = str(getattr(persistence_status, 'mode', 'session'))
    storage_target = str(getattr(persistence_status, 'storage_target', 'session-only'))

    persisted_workspace_summary = persisted_workspace_summary or {}
    persisted_usage_events = persisted_usage_events or []
    persisted_reports = persisted_reports or []
    persisted_datasets = persisted_datasets or []
    support_diagnostics = support_diagnostics or {}
    if not persisted_workspace_summary and not persisted_usage_events and not persisted_reports and not persisted_datasets:
        if application_service is not None and bool(getattr(application_service, 'enabled', False)):
            persisted_workspace_summary = application_service.load_workspace_summary(workspace_identity)
            persisted_usage_events = application_service.list_usage_events(workspace_identity, limit=25)
            persisted_reports = application_service.list_report_metadata(workspace_identity, limit=25)
            persisted_datasets = application_service.list_dataset_metadata(workspace_identity)
        elif persistence_enabled and persistence_service is not None:
            persisted_workspace_summary = persistence_service.load_workspace_summary(workspace_identity)
            persisted_usage_events = persistence_service.list_usage_events(workspace_identity, limit=25)
            persisted_reports = persistence_service.list_report_metadata(workspace_identity, limit=25)
            persisted_datasets = persistence_service.list_dataset_metadata(workspace_identity)

    customer_success = build_customer_success_summary(analysis_log, run_history)
    usage_summary = build_usage_analytics_view(analysis_log, run_history, [])

    report_action_count = len(persisted_reports) if persisted_reports else (
        len(generated_report_outputs)
        or sum(
            1
            for event in analysis_log
            if str(event.get('event_type', '')).strip() in {'Export Generated', 'Export Downloaded', 'Export Preset Applied'}
        )
    )

    summary_cards = [
        {'label': 'Active Workspace', 'value': workspace_name},
        {'label': 'Workspace Role', 'value': role_label},
        {'label': 'Active Plan', 'value': plan_name},
        {'label': 'Persistence Mode', 'value': 'SQLite-backed' if persistence_mode == 'sqlite' else 'Session-only'},
    ]

    workspace_rows = [
        {'metric': 'Workspace name', 'value': workspace_name},
        {'metric': 'Workspace ID', 'value': workspace_id},
        {'metric': 'Owner', 'value': str(workspace_identity.get('owner_label', 'Guest session'))},
        {'metric': 'Signed-in user', 'value': str(workspace_identity.get('display_name', 'Guest User'))},
        {'metric': 'Workspace role', 'value': role_label},
        {'metric': 'Persistence target', 'value': storage_target},
        {'metric': 'Saved snapshots', 'value': f"{len(saved_snapshots):,}"},
        {'metric': 'Workflow packs', 'value': f"{len(workflow_packs):,}"},
    ]
    if persisted_workspace_summary:
        workspace_rows.extend(
            [
                {'metric': 'Persisted datasets', 'value': f"{int(persisted_workspace_summary.get('dataset_count', 0)):,}"},
                {'metric': 'Persisted reports', 'value': f"{int(persisted_workspace_summary.get('report_count', 0)):,}"},
                {'metric': 'Persisted usage events', 'value': f"{int(persisted_workspace_summary.get('usage_event_count', 0)):,}"},
            ]
        )

    plan_rows = [
        {'plan_area': 'Current plan', 'status': plan_name, 'detail': str(plan_awareness.get('description', 'Plan packaging guidance is loaded for the active workspace.'))},
        {'plan_area': 'Enforcement mode', 'status': 'Strict' if strict_enforcement else 'Demo-safe', 'detail': 'Strict mode pauses selected premium saves and exports. Demo-safe mode keeps the product walkthrough visible.'},
        {'plan_area': 'Workflow pack limit', 'status': 'At limit' if bool(plan_awareness.get('workflow_pack_limit_reached')) else 'Available', 'detail': 'Soft limits pause additional saves only when strict enforcement is enabled.'},
        {'plan_area': 'Premium exports', 'status': 'Included' if bool(plan_awareness.get('advanced_exports_available', True)) else 'Upgrade prompt', 'detail': 'Advanced export bundles stay visible and explain upgrade paths in demo-safe mode.'},
    ]

    usage_rows = [
        {'ops_signal': card['label'], 'value': card['value'], 'detail': 'Observed in the active workspace session.'}
        for card in usage_summary.get('summary_cards', [])
    ]
    if persisted_usage_events:
        usage_rows.append(
            {
                'ops_signal': 'Persisted usage events',
                'value': f"{len(persisted_usage_events):,}",
                'detail': 'Recent activity stored in the configured persistence layer for pilot operations follow-up.',
            }
        )

    reports_rows = []
    if persisted_reports:
        for report in persisted_reports[:10]:
            reports_rows.append(
                {
                    'dataset_name': str(report.get('dataset_name', 'Current dataset')),
                    'report_type': str(report.get('report_type', 'Generated Report')),
                    'status': str(report.get('status', 'generated')),
                    'generated_at': str(report.get('generated_at', '')),
                }
            )
    elif generated_report_outputs:
        for report_name in sorted(generated_report_outputs.keys())[:10]:
            reports_rows.append(
                {
                    'dataset_name': workspace_name,
                    'report_type': str(report_name),
                    'status': 'Generated in session',
                    'generated_at': 'Current session',
                }
            )
    reports_table = pd.DataFrame(reports_rows, columns=['dataset_name', 'report_type', 'status', 'generated_at'])

    customer_success_table = customer_success.get('value_table', pd.DataFrame())
    if customer_success_table.empty:
        customer_success_table = pd.DataFrame(
            [
                {
                    'success_area': 'Customer value tracking',
                    'status': 'Not enough activity yet',
                    'detail': 'Datasets, workflows, and reports will populate this summary as the pilot workspace is used.',
                }
            ]
        )

    dataset_ops_table = pd.DataFrame(persisted_datasets)
    notes = [
        'This admin-facing summary is designed for early pilot operations, internal reviews, and lightweight customer success check-ins.',
        'When persistence is not configured, the summary falls back to session-safe workspace signals so demo mode remains fully usable.',
    ]
    if not persistence_enabled:
        notes.append('Enable SQLite persistence to keep workspace, report, and usage summaries available across sessions.')
    if report_action_count == 0:
        notes.append('Generate a report or export bundle to make reporting activity visible in the pilot operations summary.')

    return {
        'summary_cards': summary_cards,
        'workspace_table': pd.DataFrame(workspace_rows),
        'plan_table': pd.DataFrame(plan_rows),
        'usage_table': pd.DataFrame(usage_rows),
        'reports_table': reports_table,
        'dataset_ops_table': dataset_ops_table,
        'customer_success_table': customer_success_table,
        'support_diagnostics_table': pd.DataFrame(support_diagnostics.get('diagnostics_table', pd.DataFrame())),
        'support_diagnostics_summary_cards': support_diagnostics.get('summary_cards', []),
        'support_error_frequency_table': pd.DataFrame(support_diagnostics.get('error_frequency_table', pd.DataFrame())),
        'support_recurring_error_table': pd.DataFrame(support_diagnostics.get('recurring_error_table', pd.DataFrame())),
        'support_recent_error_table': pd.DataFrame(support_diagnostics.get('recent_error_table', pd.DataFrame())),
        'notes': notes,
        'report_action_count': report_action_count,
    }


def build_demo_usage_seed_events(
    dataset_name: str,
    source_meta: dict[str, Any] | None,
    *,
    demo_mode_enabled: bool,
    seeded_keys: list[str] | None = None,
) -> dict[str, Any]:
    source_meta = source_meta or {}
    seeded_keys = [str(item) for item in (seeded_keys or []) if str(item).strip()]
    source_mode = str(source_meta.get('source_mode', '')).strip()
    seed_key = dataset_name.strip()

    if (
        not demo_mode_enabled
        or source_mode != 'Demo dataset'
        or not seed_key
        or seed_key in seeded_keys
    ):
        return {'seeded': False, 'seed_key': seed_key, 'seeded_keys': seeded_keys, 'events': []}

    events = [
        {
            'event_type': 'Workflow Action',
            'details': f"Seeded demo workflow started for '{seed_key}'.",
            'user_interaction': 'Demo mode seed',
            'analysis_step': 'Demo onboarding',
        },
        {
            'event_type': 'AI Copilot Question',
            'details': f"Seeded demo copilot prompt for '{seed_key}'.",
            'user_interaction': 'Demo mode seed',
            'analysis_step': 'Demo onboarding',
        },
        {
            'event_type': 'Export Downloaded',
            'details': f"Seeded demo export generated for '{seed_key}'.",
            'user_interaction': 'Demo mode seed',
            'analysis_step': 'Demo onboarding',
        },
    ]
    return {
        'seeded': True,
        'seed_key': seed_key,
        'seeded_keys': seeded_keys + [seed_key],
        'events': events,
    }


def build_usage_analytics_view(
    analysis_log: list[dict[str, Any]],
    run_history: list[dict[str, Any]],
    visited_sections: list[str] | None = None,
) -> dict[str, Any]:
    log_df = pd.DataFrame(analysis_log)
    history_df = pd.DataFrame(run_history)
    visited_sections = visited_sections or []

    def _count(event_type: str) -> int:
        if log_df.empty or 'event_type' not in log_df.columns:
            return 0
        return int((log_df['event_type'].astype(str) == event_type).sum())

    datasets_loaded = _count('Dataset Selected')
    workflows_started = _count('Workflow Action') + _count('Workflow Pack Loaded') + _count('Session Bundle Restored')
    reports_generated = _count('Export Downloaded') + _count('Export Preset Applied')
    copilot_prompts = _count('AI Copilot Question')
    exports_downloaded = _count('Export Downloaded')

    summary_cards = [
        {'label': 'Datasets Loaded', 'value': f'{datasets_loaded:,}'},
        {'label': 'Workflows Started', 'value': f'{workflows_started:,}'},
        {'label': 'AI Copilot Prompts', 'value': f'{copilot_prompts:,}'},
        {'label': 'Exports Downloaded', 'value': f'{exports_downloaded:,}'},
    ]

    module_table = pd.DataFrame([{'module_name': name, 'status': 'Rendered in session'} for name in visited_sections])
    if log_df.empty:
        activity_table = pd.DataFrame(columns=['event_type', 'events'])
    else:
        activity_table = (
            log_df.groupby('event_type')
            .size()
            .reset_index(name='events')
            .sort_values('events', ascending=False)
            .reset_index(drop=True)
        )

    if history_df.empty:
        dataset_runs = pd.DataFrame(columns=['dataset_name', 'runs'])
    else:
        dataset_runs = (
            history_df.groupby('dataset_name')
            .size()
            .reset_index(name='runs')
            .sort_values('runs', ascending=False)
            .reset_index(drop=True)
        )

    notes = [
        'Usage analytics are session-safe and local to this demo workspace.',
        'Rendered sections are used as a lightweight proxy for module visits in this Streamlit interface.',
    ]
    if exports_downloaded == 0:
        notes.append('Export activity will appear here after a report or bundle is downloaded in this session.')

    return {
        'summary_cards': summary_cards,
        'module_table': module_table,
        'activity_table': activity_table,
        'dataset_runs': dataset_runs,
        'notes': notes,
    }


def build_customer_success_summary(
    analysis_log: list[dict[str, Any]],
    run_history: list[dict[str, Any]],
) -> dict[str, Any]:
    log_df = pd.DataFrame(analysis_log)
    history_df = pd.DataFrame(run_history)

    def _count(event_type: str) -> int:
        if log_df.empty or 'event_type' not in log_df.columns:
            return 0
        return int((log_df['event_type'].astype(str) == event_type).sum())

    datasets_analyzed = int(history_df['dataset_name'].nunique()) if not history_df.empty and 'dataset_name' in history_df.columns else _count('Dataset Selected')
    issues_detected = 0
    readiness_lift_candidates = 0
    if not history_df.empty:
        if 'major_blockers_detected' in history_df.columns:
            issues_detected = int(
                history_df['major_blockers_detected']
                .astype(str)
                .str.strip()
                .replace('', pd.NA)
                .dropna()
                .shape[0]
            )
        if 'synthetic_helper_fields_added' in history_df.columns:
            readiness_lift_candidates = int((pd.to_numeric(history_df['synthetic_helper_fields_added'], errors='coerce').fillna(0) > 0).sum())

    reports_generated = _count('Export Generated') + _count('Export Downloaded') + _count('Export Preset Applied')
    workflows_completed = _count('Workflow Pack Saved') + _count('Workflow Pack Loaded') + _count('Session Bundle Restored')

    summary_cards = [
        {'label': 'Datasets Analyzed', 'value': f'{datasets_analyzed:,}'},
        {'label': 'Issues Detected', 'value': f'{issues_detected:,}'},
        {'label': 'Reports Generated', 'value': f'{reports_generated:,}'},
        {'label': 'Workflows Completed', 'value': f'{workflows_completed:,}'},
    ]

    value_rows = [
        {
            'success_area': 'Datasets analyzed',
            'status': f'{datasets_analyzed:,} datasets reviewed',
            'detail': 'Shows how many unique datasets have moved through the platform in this workspace.',
        },
        {
            'success_area': 'Issues detected',
            'status': f'{issues_detected:,} runs with blockers',
            'detail': 'Tracks runs where major blockers or readiness issues were surfaced for follow-up.',
        },
        {
            'success_area': 'Reports generated',
            'status': f'{reports_generated:,} report actions',
            'detail': 'Counts export generation and download actions that turned findings into deliverables.',
        },
        {
            'success_area': 'Workflow completion',
            'status': f'{workflows_completed:,} saved or restored workflows',
            'detail': 'Shows how often teams captured or reused repeatable review workflows.',
        },
    ]
    if readiness_lift_candidates:
        value_rows.append(
            {
                'success_area': 'Readiness lift candidates',
                'status': f'{readiness_lift_candidates:,} runs with helper-backed uplift',
                'detail': 'Highlights runs where helper or remediation support helped unlock more of the workflow.',
            }
        )

    notes = [
        'Customer success summary is local to this workspace and is meant to show value delivered during pilots or early beta reviews.',
        'Counts reflect observed workflow activity and generated deliverables, not external adoption analytics.',
    ]
    if reports_generated == 0:
        notes.append('Generate or download a report to show stakeholder-ready value in this summary.')

    return {
        'summary_cards': summary_cards,
        'value_table': pd.DataFrame(value_rows),
        'notes': notes,
    }
