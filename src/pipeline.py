from __future__ import annotations

import concurrent.futures
import os
from typing import Any

import pandas as pd

from src.analysis_trust import build_analysis_trust_summary
from src.data_loader import estimate_memory_mb
from src.dataset_intelligence import build_dataset_intelligence_report
from src.decision_support import (
    build_executive_summary,
    build_intervention_recommendations,
    build_kpi_benchmarking_layer,
    build_prioritized_insights,
    build_scenario_simulation_studio,
)
from src.enterprise_features import build_data_lineage_view, build_quality_rule_engine
from src.evolution_engine import build_evolution_summary
from src.healthcare_analysis import run_healthcare_analysis
from src.insights_engine import build_action_recommendations, build_automated_insight_board, build_key_insights
from src.ops_hardening import build_performance_diagnostics
from src.portfolio_support import (
    build_app_metadata,
    build_dataset_onboarding_summary,
    build_demo_mode_content,
    build_documentation_support,
    build_screenshot_support,
)
from src.presentation_support import (
    build_audit_summary,
    build_compliance_governance_summary,
    build_executive_report_pack,
    build_landing_summary,
    build_printable_reports,
    build_run_history_entry,
    build_stakeholder_export_bundle,
    update_run_history,
)
from src.profiler import (
    analysis_sample_info,
    build_dataset_overview,
    build_profile_cache_summary,
    build_quality_checks,
    build_structure_profile_bundle,
)
from src.readiness_engine import evaluate_analysis_readiness
from src.remediation_engine import apply_remediation_augmentations
from src.result_accuracy import build_result_accuracy_summary
from src.schema_detection import detect_structure
from src.semantic_mapper import build_data_dictionary, build_data_remediation_assistant, build_dataset_improvement_plan, infer_semantic_mapping
import src.logger as logger_module
from src.solution_layers import (
    build_demo_guidance_system,
    build_market_specific_solution_views,
    build_solution_layer_guidance,
    build_solution_packages,
    build_use_case_detection,
)
from src.standards_validator import validate_healthcare_standards
from src.temporal_detection import augment_temporal_fields


class AnalysisCancelledError(RuntimeError):
    pass


def _dataset_identifier(dataset_name: str, source_meta: dict[str, Any], data: pd.DataFrame) -> str:
    cache_key = str(source_meta.get('dataset_cache_key', '') or data.attrs.get('dataset_cache_key', '')).strip()
    if cache_key:
        return cache_key
    return f"{dataset_name}:{len(data)}x{len(data.columns)}"


def _healthcare_analysis_timeout_seconds() -> float:
    try:
        return max(5.0, float(os.getenv('SMART_DATASET_ANALYZER_HEALTHCARE_TIMEOUT_SECONDS', '20')))
    except (TypeError, ValueError):
        return 20.0


def _healthcare_timeout_fallback(
    semantic: dict[str, object],
    synthetic_fields: set[str],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    try:
        from src.healthcare_analysis import assess_healthcare_dataset

        readiness = assess_healthcare_dataset(semantic['canonical_map'], synthetic_fields=synthetic_fields)
    except Exception:
        readiness = {
            'healthcare_readiness_score': 0.0,
            'likely_dataset_type': 'Healthcare analysis timeout fallback',
            'matched_healthcare_fields': [],
        }
    reason = f'Healthcare analytics timed out after {timeout_seconds:.0f} seconds. A lightweight fallback was used so the app stays responsive.'
    unavailable = {'available': False, 'reason': reason}
    return {
        **readiness,
        'utilization': unavailable,
        'cost': unavailable,
        'provider': unavailable,
        'diagnosis': unavailable,
        'readmission': unavailable,
        'length_of_stay_prediction': unavailable,
        'mortality_adverse_events': unavailable,
        'population_health': unavailable,
        'clinical_outcome_benchmarks': unavailable,
        'risk_segmentation': unavailable,
        'ai_insight_summary': [],
        'anomaly_detection': unavailable,
        'default_cohort_summary': unavailable,
        'scenario': unavailable,
        'survival_outcomes': unavailable,
        'driver_analysis': unavailable,
        'segment_discovery': unavailable,
        'care_pathway': unavailable,
        'explainability_fairness': unavailable,
        'timeout_fallback': True,
        'timeout_reason': reason,
    }


def _safe_df(table: Any) -> pd.DataFrame:
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _emit_progress(
    progress_callback,
    value: float,
    message: str,
    *,
    step_index: int,
    total_steps: int,
    current_operation: str | None = None,
) -> None:
    if progress_callback:
        progress_callback(
            value,
            message,
            step_index=step_index,
            total_steps=total_steps,
            current_operation=current_operation or message,
        )


def _check_cancel(cancel_check) -> None:
    if cancel_check and bool(cancel_check()):
        raise AnalysisCancelledError('Analysis was cancelled before the next pipeline stage completed.')


def run_analysis_pipeline(
    data: pd.DataFrame,
    dataset_name: str,
    source_meta: dict[str, str],
    demo_config: dict[str, str] | None = None,
    active_control_values: dict[str, object] | None = None,
    large_dataset_profile: dict[str, Any] | None = None,
    progress_callback=None,
    cancel_check=None,
    cache_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    demo_config = demo_config or {}
    active_control_values = active_control_values or {}
    large_dataset_profile = large_dataset_profile or {}
    source_column_count = int(len(data.columns))
    dataset_identifier = _dataset_identifier(dataset_name, source_meta, data)
    logger_module.log_platform_event(
        'pipeline_started',
        logger_name='pipeline',
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        row_count=int(len(data)),
        source_columns=source_column_count,
        source_mode=source_meta.get('source_mode', 'unknown'),
    )

    analysis_data, temporal_context = augment_temporal_fields(data)
    analysis_data, remediation_context = apply_remediation_augmentations(
        analysis_data,
        bmi_mode=str(demo_config.get('bmi_remediation_mode', 'median')).lower(),
        helper_mode=str(demo_config.get('synthetic_helper_mode', 'Auto')),
        synthetic_cost_mode=str(demo_config.get('synthetic_cost_mode', 'Auto')),
        synthetic_readmission_mode=str(demo_config.get('synthetic_readmission_mode', 'Auto')),
    )
    analysis_data.attrs['dataset_cache_key'] = str(
        analysis_data.attrs.get('dataset_cache_key', source_meta.get('dataset_cache_key', dataset_identifier))
    )
    analysis_data.attrs['dataset_identifier'] = dataset_identifier
    analysis_data.attrs['dataset_name'] = dataset_name
    analysis_data.attrs['source_mode'] = str(source_meta.get('source_mode', 'unknown'))

    source_meta = dict(source_meta)
    source_meta['dataset_identifier'] = dataset_identifier
    if temporal_context.get('synthetic_date_created'):
        source_meta['temporal_note'] = str(
            temporal_context.get('note', 'Synthetic event_date generated for temporal analysis.')
        )
        source_meta['best_for'] = (
            f"{source_meta.get('best_for', 'General analysis')} "
            'Temporal trend modules are enabled with a synthetic event_date derived from existing date parts.'
        )

    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.12,
        '2. Profiling columns...',
        step_index=2,
        total_steps=5,
        current_operation='Detecting structure and preparing profile inputs',
    )
    profile_bundle = build_structure_profile_bundle(
        analysis_data,
        sampling_plan=large_dataset_profile,
        cache_metrics=cache_metrics,
    )
    structure = profile_bundle['structure']

    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.32,
        '2. Profiling columns...',
        step_index=2,
        total_steps=5,
        current_operation='Building field profiling diagnostics',
    )
    field_profile = profile_bundle['field_profile']
    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.48,
        '3. Analyzing quality...',
        step_index=3,
        total_steps=5,
        current_operation='Running data quality checks',
    )
    quality = build_quality_checks(analysis_data, structure, field_profile, sampling_plan=large_dataset_profile)

    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.60,
        '3. Analyzing quality...',
        step_index=3,
        total_steps=5,
        current_operation='Summarizing dataset overview and semantic mappings',
    )
    overview = build_dataset_overview(analysis_data, estimate_memory_mb(analysis_data))
    overview['source_columns'] = source_column_count
    overview['analyzed_columns'] = int(len(analysis_data.columns))
    overview['helper_columns_added'] = max(int(len(analysis_data.columns)) - source_column_count, 0)
    semantic = infer_semantic_mapping(
        analysis_data,
        structure,
        manual_overrides=source_meta.get('manual_semantic_overrides'),
    )
    helper_fields_table = _safe_df(remediation_context.get('helper_fields'))
    analysis_data.attrs['helper_field_names'] = (
        helper_fields_table['helper_field'].astype(str).tolist()
        if 'helper_field' in helper_fields_table.columns
        else []
    )
    helper_fields = (
        set(
            helper_fields_table.loc[
                lambda df: df.get('helper_type', pd.Series(dtype=str)).isin(['synthetic', 'derived']),
                'helper_field',
            ]
            .astype(str)
            .tolist()
        )
        if 'helper_field' in helper_fields_table.columns
        else set()
    )

    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.72,
        '4. Computing healthcare metrics...',
        step_index=4,
        total_steps=5,
        current_operation='Evaluating readiness and healthcare analytics',
    )
    readiness = evaluate_analysis_readiness(semantic['canonical_map'], synthetic_fields=helper_fields)
    logger_module.log_platform_event(
        'readiness_scored',
        logger_name='pipeline',
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        readiness_score=float(readiness.get('readiness_score', 0.0)),
        available_modules=int(readiness.get('available_count', 0)),
    )
    healthcare_timeout_seconds = _healthcare_analysis_timeout_seconds()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        run_healthcare_analysis,
        analysis_data,
        semantic['canonical_map'],
        synthetic_fields=helper_fields,
    )
    try:
        healthcare = future.result(timeout=healthcare_timeout_seconds)
    except concurrent.futures.TimeoutError:
        future.cancel()
        healthcare = _healthcare_timeout_fallback(
            semantic,
            helper_fields,
            timeout_seconds=healthcare_timeout_seconds,
        )
        logger_module.log_platform_event(
            'healthcare_analysis_timeout',
            logger_name='pipeline',
            dataset_name=dataset_name,
            timeout_seconds=float(healthcare_timeout_seconds),
            row_count=int(len(analysis_data)),
            column_count=int(len(analysis_data.columns)),
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    logger_module.log_platform_event(
        'healthcare_analysis_completed',
        logger_name='pipeline',
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        healthcare_readiness_score=float(healthcare.get('healthcare_readiness_score', 0.0)),
        likely_dataset_type=healthcare.get('likely_dataset_type', ''),
        readmission_available=bool((healthcare.get('readmission') or {}).get('available')),
    )

    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.84,
        '5. Finalizing insights...',
        step_index=5,
        total_steps=5,
        current_operation='Running standards, privacy, and governance checks',
    )
    standards = validate_healthcare_standards(analysis_data, structure, semantic)
    from src.modules.privacy_security import run_privacy_security_review

    privacy_review = run_privacy_security_review(analysis_data)
    data_dictionary = build_data_dictionary(structure, semantic)
    remediation = build_data_remediation_assistant(structure, semantic, readiness)
    improvement_plan = build_dataset_improvement_plan(structure, semantic, readiness)

    _check_cancel(cancel_check)
    _emit_progress(
        progress_callback,
        0.94,
        '5. Finalizing insights...',
        step_index=5,
        total_steps=5,
        current_operation='Preparing rule engine, insights, and action guidance',
    )
    rule_engine = build_quality_rule_engine(analysis_data, semantic['canonical_map'])
    insights = build_key_insights(overview, field_profile, quality, readiness, semantic, healthcare, structure)
    action_recommendations = build_action_recommendations(quality, readiness, semantic, healthcare)
    insight_board = build_automated_insight_board(overview, readiness, healthcare, insights, action_recommendations)
    intervention_recommendations = build_intervention_recommendations(
        healthcare,
        quality,
        readiness,
        remediation_context,
        model_comparison=None,
    )
    sample_info = analysis_sample_info(analysis_data, sampling_plan=large_dataset_profile)
    analysis_trust_summary = build_analysis_trust_summary(
        dataset_name=dataset_name,
        source_meta=source_meta,
        overview=overview,
        readiness=readiness,
        healthcare=healthcare,
        remediation_context=remediation_context,
        sample_info=sample_info,
    )
    kpi_benchmarking = build_kpi_benchmarking_layer(healthcare, quality, remediation_context)
    scenario_studio = build_scenario_simulation_studio(
        healthcare,
        quality,
        remediation_context,
        scenario_mode=str(demo_config.get('scenario_simulation_mode', 'Basic')).lower(),
    )
    prioritized_insights = build_prioritized_insights(
        overview,
        quality,
        readiness,
        healthcare,
        action_recommendations,
        intervention_recommendations,
        model_comparison=None,
    )
    compliance_governance_summary = build_compliance_governance_summary(
        standards,
        privacy_review,
        {},
        remediation_context,
        readiness,
    )
    dataset_intelligence = build_dataset_intelligence_report(
        analysis_data,
        structure,
        semantic,
        readiness,
        healthcare,
        quality,
        remediation,
        remediation_context,
        standards,
        privacy_review,
        compliance_governance_summary,
    )
    result_accuracy_summary = build_result_accuracy_summary(
        dataset_name=dataset_name,
        source_meta=source_meta,
        overview=overview,
        semantic=semantic,
        readiness=readiness,
        healthcare=healthcare,
        remediation_context=remediation_context,
        dataset_intelligence=dataset_intelligence,
        trust_summary=analysis_trust_summary,
        active_control_values=active_control_values,
    )
    evolution_summary = build_evolution_summary(
        dataset_name=dataset_name,
        source_meta=source_meta,
        structure=structure,
        semantic=semantic,
        readiness=readiness,
        healthcare=healthcare,
        trust_summary=analysis_trust_summary,
        accuracy_summary=result_accuracy_summary,
        overview=overview,
        sample_info=sample_info,
        active_control_values=active_control_values,
    )
    lineage = build_data_lineage_view(
        dataset_name,
        source_meta,
        semantic,
        readiness,
        active_control_values,
        accuracy_summary=result_accuracy_summary,
    )
    if not _safe_df(remediation_context.get('helper_fields')).empty:
        helper_rows = remediation_context['helper_fields'].rename(
            columns={'helper_field': 'source_column', 'helper_type': 'derived_role', 'note': 'business_meaning'}
        )
        derived_fields = _safe_df(lineage.get('derived_fields_table'))
        lineage['derived_fields_table'] = pd.concat([derived_fields, helper_rows], ignore_index=True, sort=False)
        steps = list(lineage.get('transformation_steps', []))
        for key in ['bmi_remediation', 'synthetic_cost', 'synthetic_clinical', 'synthetic_readmission']:
            note = remediation_context.get(key, {}).get('lineage_note')
            if note:
                steps.append(note)
        lineage['transformation_steps'] = steps
    compliance_governance_summary = build_compliance_governance_summary(
        standards,
        privacy_review,
        lineage,
        remediation_context,
        readiness,
    )
    executive_summary = build_executive_summary(
        dataset_name,
        overview,
        readiness,
        healthcare,
        action_recommendations,
        intervention_recommendations,
        remediation_context,
        trust_summary=analysis_trust_summary,
        accuracy_summary=result_accuracy_summary,
        verbosity=str(demo_config.get('executive_summary_verbosity', 'Concise')).lower(),
    )
    executive_report_pack = build_executive_report_pack(
        dataset_name,
        overview,
        quality,
        readiness,
        healthcare,
        dataset_intelligence,
        executive_summary,
        action_recommendations,
        intervention_recommendations,
        kpi_benchmarking,
        scenario_studio,
        prioritized_insights,
        remediation_context,
        demo_config,
    )
    printable_reports = build_printable_reports(executive_report_pack, compliance_governance_summary)
    stakeholder_bundle = build_stakeholder_export_bundle(
        executive_report_pack,
        dataset_intelligence,
        kpi_benchmarking,
        intervention_recommendations,
        healthcare.get('explainability_fairness', {}),
        healthcare.get('readmission', {}),
        quality,
        compliance_governance_summary,
    )
    dataset_onboarding = build_dataset_onboarding_summary(
        dataset_name,
        source_meta,
        {
            'readiness': readiness,
            'remediation': remediation,
            'remediation_context': remediation_context,
            'dataset_intelligence': dataset_intelligence,
        },
    )
    demo_mode_content = build_demo_mode_content(
        dataset_name,
        source_meta,
        {
            'readiness': readiness,
            'remediation_context': remediation_context,
            'dataset_intelligence': dataset_intelligence,
        },
        demo_config,
    )
    documentation_support = build_documentation_support(
        dataset_name,
        {'readiness': readiness, 'healthcare': healthcare, 'remediation_context': remediation_context},
    )
    screenshot_support = build_screenshot_support(
        dataset_name,
        {'readiness': readiness, 'remediation_context': remediation_context},
    )
    app_metadata = build_app_metadata({'readiness': readiness, 'remediation_context': remediation_context})
    solution_layers = build_solution_layer_guidance(dataset_intelligence, healthcare, readiness)
    use_case_detection = build_use_case_detection(dataset_intelligence, readiness, healthcare)
    solution_packages = build_solution_packages(dataset_intelligence, use_case_detection)
    market_solution_views = build_market_specific_solution_views(
        dataset_intelligence,
        use_case_detection,
        solution_packages,
        solution_layers,
    )
    demo_guidance = build_demo_guidance_system(dataset_intelligence, use_case_detection, solution_packages)

    if progress_callback:
        progress_callback(1.0, 'Analysis preparation complete.')
    logger_module.log_platform_event(
        'pipeline_completed',
        logger_name='pipeline',
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        analyzed_columns=int(overview.get('analyzed_columns', overview.get('columns', 0))),
        helper_columns_added=int(overview.get('helper_columns_added', 0)),
    )

    return {
        'data': analysis_data,
        'dataset_runtime_diagnostics': {
            'dataset_name': dataset_name,
            'source_mode': str(source_meta.get('source_mode', 'unknown')),
            'dataset_identifier': dataset_identifier,
            'dataset_cache_key': str(analysis_data.attrs.get('dataset_cache_key', '')),
            'row_count': int(len(analysis_data)),
            'column_count': int(len(analysis_data.columns)),
            'cache_identity_preserved': bool(str(analysis_data.attrs.get('dataset_cache_key', '')).strip()),
            'trust_level': str(analysis_trust_summary.get('trust_level', 'Unknown')),
            'trust_score': float(analysis_trust_summary.get('trust_score', 0.0)),
            'sampling_active': bool(analysis_trust_summary.get('sampling_active', False)),
            'timeout_fallback': bool(analysis_trust_summary.get('timeout_fallback', False)),
            'external_reporting_ready_modules': list(result_accuracy_summary.get('safe_for_external_reporting_modules', [])),
            'benchmark_profile': str(result_accuracy_summary.get('benchmark_profile', {}).get('profile_name', '')),
            'reporting_threshold_profile': str(result_accuracy_summary.get('reporting_policy', {}).get('profile_name', '')),
        },
        'overview': overview,
        'structure': structure,
        'field_profile': field_profile,
        'quality': quality,
        'semantic': semantic,
        'readiness': readiness,
        'healthcare': healthcare,
        'standards': standards,
        'privacy_review': privacy_review,
        'data_dictionary': data_dictionary,
        'remediation': remediation,
        'improvement_plan': improvement_plan,
        'rule_engine': rule_engine,
        'insights': insights,
        'action_recommendations': action_recommendations,
        'insight_board': insight_board,
        'intervention_recommendations': intervention_recommendations,
        'executive_summary': executive_summary,
        'kpi_benchmarking': kpi_benchmarking,
        'scenario_studio': scenario_studio,
        'prioritized_insights': prioritized_insights,
        'compliance_governance_summary': compliance_governance_summary,
        'dataset_intelligence': dataset_intelligence,
        'result_accuracy_summary': result_accuracy_summary,
        'evolution_summary': evolution_summary,
        'executive_report_pack': executive_report_pack,
        'printable_reports': printable_reports,
        'stakeholder_export_bundle': stakeholder_bundle,
        'dataset_onboarding': dataset_onboarding,
        'demo_mode_content': demo_mode_content,
        'documentation_support': documentation_support,
        'screenshot_support': screenshot_support,
        'app_metadata': app_metadata,
        'solution_layers': solution_layers,
        'use_case_detection': use_case_detection,
        'solution_packages': solution_packages,
        'market_solution_views': market_solution_views,
        'demo_guidance': demo_guidance,
        'lineage': lineage,
        'sample_info': sample_info,
        'analysis_trust_summary': analysis_trust_summary,
        'temporal_context': temporal_context,
        'remediation_context': remediation_context,
        'demo_config': demo_config,
        'large_dataset_profile': large_dataset_profile,
        'profile_cache_summary': build_profile_cache_summary(cache_metrics),
        'profile_cache_bundle': profile_bundle,
    }


def finalize_runtime_pipeline(
    pipeline: dict[str, Any],
    *,
    dataset_name: str,
    source_meta: dict[str, str],
    preflight: dict[str, Any],
    column_validation: dict[str, Any],
    job_runtime: dict[str, Any],
    heavy_task_catalog: dict[str, Any],
    environment_checks: dict[str, Any],
    startup_readiness: dict[str, Any],
    plan_awareness: dict[str, Any],
    deployment_health_checks: dict[str, Any],
    performance_diagnostics: dict[str, Any],
    run_history: list[dict[str, object]],
    analysis_log: list[dict[str, object]],
    demo_config: dict[str, str],
) -> dict[str, Any]:
    pipeline = dict(pipeline)
    pipeline['source_meta'] = dict(source_meta)
    pipeline['preflight'] = preflight
    pipeline['column_validation'] = column_validation
    pipeline['job_runtime'] = job_runtime
    pipeline['heavy_task_catalog'] = heavy_task_catalog
    pipeline['environment_checks'] = environment_checks
    pipeline['startup_readiness'] = startup_readiness
    pipeline['plan_awareness'] = plan_awareness
    pipeline['deployment_health_checks'] = deployment_health_checks
    pipeline['performance_diagnostics'] = performance_diagnostics
    pipeline['audit_summary_bundle'] = build_audit_summary(run_history, analysis_log)
    pipeline['landing_summary'] = build_landing_summary(pipeline, demo_config, dataset_name)
    return pipeline


def build_updated_run_history(
    existing_history: list[dict[str, object]],
    dataset_name: str,
    pipeline: dict[str, object],
    demo_config: dict[str, object],
) -> list[dict[str, object]]:
    run_entry = build_run_history_entry(dataset_name, pipeline, demo_config)
    return update_run_history(existing_history, run_entry)
