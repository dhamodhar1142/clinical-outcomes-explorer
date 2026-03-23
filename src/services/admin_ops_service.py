from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import src.logger as logger_module
from src.usage_analytics import (
    build_customer_success_summary,
    build_product_admin_summary,
    build_usage_analytics_view,
)


@dataclass(frozen=True)
class AdminOpsService:
    application_service: Any | None = None
    persistence_service: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.application_service is not None or self.persistence_service is not None

    def build_admin_ops_view(
        self,
        *,
        workspace_identity: dict[str, Any] | None,
        plan_awareness: dict[str, Any] | None,
        analysis_log: list[dict[str, Any]] | None,
        run_history: list[dict[str, Any]] | None,
        visited_sections: list[str] | None = None,
        generated_report_outputs: dict[str, Any] | None = None,
        saved_snapshots: dict[str, Any] | None = None,
        workflow_packs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace_identity = workspace_identity or {}
        analysis_log = analysis_log or []
        run_history = run_history or []

        persisted_workspace_summary: dict[str, Any] = {}
        persisted_usage_events: list[dict[str, Any]] = []
        persisted_reports: list[dict[str, Any]] = []
        persisted_datasets: list[dict[str, Any]] = []
        if self.application_service is not None and bool(getattr(self.application_service, 'enabled', False)):
            persisted_workspace_summary = self.application_service.load_workspace_summary(workspace_identity)
            persisted_usage_events = self.application_service.list_usage_events(workspace_identity, limit=25)
            persisted_reports = self.application_service.list_report_metadata(workspace_identity, limit=25)
            persisted_datasets = self.application_service.list_dataset_metadata(workspace_identity)

        usage = build_usage_analytics_view(
            analysis_log,
            run_history,
            visited_sections or [],
        )
        customer_success = build_customer_success_summary(analysis_log, run_history)
        support_diagnostics = logger_module.build_support_diagnostics()
        product_admin = build_product_admin_summary(
            workspace_identity=workspace_identity,
            persistence_service=self.persistence_service,
            application_service=self.application_service,
            plan_awareness=plan_awareness,
            analysis_log=analysis_log,
            run_history=run_history,
            generated_report_outputs=generated_report_outputs,
            saved_snapshots=saved_snapshots,
            workflow_packs=workflow_packs,
            persisted_workspace_summary=persisted_workspace_summary,
            persisted_usage_events=persisted_usage_events,
            persisted_reports=persisted_reports,
            persisted_datasets=persisted_datasets,
            support_diagnostics=support_diagnostics,
        )
        return {
            'usage': usage,
            'customer_success': customer_success,
            'product_admin': product_admin,
            'support_diagnostics': support_diagnostics,
            'persisted_workspace_summary': persisted_workspace_summary,
            'persisted_usage_events': persisted_usage_events,
            'persisted_reports': persisted_reports,
            'persisted_datasets': persisted_datasets,
        }


def build_admin_ops_service(
    application_service: Any | None,
    persistence_service: Any | None,
) -> AdminOpsService:
    return AdminOpsService(
        application_service=application_service,
        persistence_service=persistence_service,
    )


__all__ = ['AdminOpsService', 'build_admin_ops_service']
