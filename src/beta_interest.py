from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd


BETA_INTEREST_STORAGE_OPTIONS = ['LOCAL', 'DATABASE', 'API']
BETA_INTEREST_STATUS_OPTIONS = ['New', 'Contacted', 'Completed']
BETA_INTEREST_API_URL_ENV = 'SMART_DATASET_ANALYZER_BETA_INTEREST_API_URL'
BETA_INTEREST_API_TOKEN_ENV = 'SMART_DATASET_ANALYZER_BETA_INTEREST_API_TOKEN'
BETA_INTEREST_SUPPORT_EMAIL_ENV = 'SMART_DATASET_ANALYZER_SUPPORT_EMAIL'
DEFAULT_SUPPORT_EMAIL = 'support@example.com'
NAME_PATTERN = re.compile(r"^[A-Za-z0-9 .,'-]{2,100}$")
EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _clean(value: str | None) -> str:
    return (value or '').strip()


def support_email() -> str:
    return _clean(os.getenv(BETA_INTEREST_SUPPORT_EMAIL_ENV, DEFAULT_SUPPORT_EMAIL)) or DEFAULT_SUPPORT_EMAIL


def validate_beta_interest(name: str, email: str, organization: str, use_case: str) -> list[str]:
    errors: list[str] = []
    clean_name = _clean(name)
    clean_email = _clean(email)
    clean_organization = _clean(organization)
    clean_use_case = _clean(use_case)

    if not clean_name or len(clean_name) < 2 or len(clean_name) > 100 or not NAME_PATTERN.fullmatch(clean_name):
        errors.append('Name must be 2-100 characters and use letters, numbers, spaces, or basic punctuation only.')
    if not clean_email or len(clean_email) > 255 or not EMAIL_PATTERN.fullmatch(clean_email):
        errors.append('Invalid email address. Use a standard work email such as name@organization.com.')
    if clean_organization and len(clean_organization) > 200:
        errors.append('Organization must be 200 characters or fewer.')
    if not clean_use_case or len(clean_use_case) < 10 or len(clean_use_case) > 500:
        errors.append('Use case must be 10-500 characters.')
    return errors


def build_beta_interest_record(
    *,
    name: str,
    email: str,
    organization: str,
    use_case: str,
    workspace_name: str,
    workspace_id: str,
    submitted_by: str,
    dataset_name: str,
    dataset_source_mode: str,
    capture_mode: str,
    submission_id: str | None = None,
    follow_up_status: str = 'New',
    submitted_at: str | None = None,
    contacted_at: str = '',
    completed_at: str = '',
) -> dict[str, Any]:
    clean_email = _clean(email).lower()
    submission_timestamp = submitted_at or datetime.now(UTC).isoformat()
    submission_key = submission_id or f"{workspace_id}:{clean_email}:{uuid.uuid4().hex}"
    return {
        'submission_id': submission_key,
        'name': _clean(name),
        'email': clean_email,
        'organization': _clean(organization),
        'use_case': _clean(use_case),
        'workspace_id': _clean(workspace_id) or 'guest-demo-workspace',
        'workspace_name': _clean(workspace_name) or 'Guest Demo Workspace',
        'submitted_by': _clean(submitted_by) or 'Guest User',
        'dataset_name': _clean(dataset_name) or 'Current dataset',
        'dataset_source_mode': _clean(dataset_source_mode) or 'Unknown',
        'submitted_at': submission_timestamp,
        'capture_mode': _clean(capture_mode) or 'Local demo',
        'follow_up_status': follow_up_status if follow_up_status in BETA_INTEREST_STATUS_OPTIONS else 'New',
        'contacted_at': _clean(contacted_at),
        'completed_at': _clean(completed_at),
    }


def append_beta_interest_submission(
    submissions: list[dict[str, Any]],
    *,
    name: str,
    email: str,
    organization: str,
    use_case: str,
    workspace_name: str,
    submitted_by: str,
    workspace_id: str = 'guest-demo-workspace',
    dataset_name: str = 'Current dataset',
    dataset_source_mode: str = 'Unknown',
    capture_mode: str = 'Local demo',
) -> dict[str, Any]:
    record = build_beta_interest_record(
        name=name,
        email=email,
        organization=organization,
        use_case=use_case,
        workspace_name=workspace_name,
        workspace_id=workspace_id,
        submitted_by=submitted_by,
        dataset_name=dataset_name,
        dataset_source_mode=dataset_source_mode,
        capture_mode=capture_mode,
    )
    submissions.append(record)
    return record


def _dedupe_submissions(submissions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for submission in submissions:
        key = str(submission.get('submission_id', '') or '')
        if key:
            deduped[key] = dict(submission)
        else:
            fallback = f"{submission.get('email', '')}:{submission.get('submitted_at', '')}:{submission.get('workspace_id', '')}"
            deduped[fallback] = dict(submission)
    return list(deduped.values())


def merge_beta_interest_submissions(*submission_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for submission_list in submission_lists:
        merged.extend(dict(item) for item in submission_list or [])
    records = _dedupe_submissions(merged)
    records.sort(key=lambda item: str(item.get('submitted_at', '')), reverse=True)
    return records


def save_beta_interest_submission(
    *,
    local_submissions: list[dict[str, Any]],
    name: str,
    email: str,
    organization: str,
    use_case: str,
    workspace_identity: dict[str, Any],
    dataset_name: str,
    dataset_source_mode: str,
    storage_mode: str,
    application_service: Any | None = None,
    api_url: str = '',
    api_token: str = '',
) -> dict[str, Any]:
    errors = validate_beta_interest(name, email, organization, use_case)
    if errors:
        raise ValueError('\n'.join(errors))

    normalized_mode = str(storage_mode or 'LOCAL').strip().upper()
    if normalized_mode not in BETA_INTEREST_STORAGE_OPTIONS:
        normalized_mode = 'LOCAL'
    capture_mode = {
        'LOCAL': 'Local demo',
        'DATABASE': 'Database',
        'API': 'External API',
    }[normalized_mode]
    record = build_beta_interest_record(
        name=name,
        email=email,
        organization=organization,
        use_case=use_case,
        workspace_name=str(workspace_identity.get('workspace_name', 'Guest Demo Workspace')),
        workspace_id=str(workspace_identity.get('workspace_id', 'guest-demo-workspace')),
        submitted_by=str(workspace_identity.get('display_name', 'Guest User')),
        dataset_name=dataset_name,
        dataset_source_mode=dataset_source_mode,
        capture_mode=capture_mode,
    )

    if normalized_mode == 'DATABASE':
        if application_service is None or not bool(getattr(application_service, 'enabled', False)):
            raise RuntimeError('Database beta-interest capture requires an enabled persistence backend.')
        application_service.save_beta_interest_submission(workspace_identity, record)
    elif normalized_mode == 'API':
        endpoint = _clean(api_url or os.getenv(BETA_INTEREST_API_URL_ENV, ''))
        if not endpoint:
            raise RuntimeError('API beta-interest capture requires a configured endpoint.')
        payload = json.dumps(record).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        token = _clean(api_token or os.getenv(BETA_INTEREST_API_TOKEN_ENV, ''))
        if token:
            headers['Authorization'] = f'Bearer {token}'
        request = urllib_request.Request(endpoint, data=payload, headers=headers, method='POST')
        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                status_code = getattr(response, 'status', 200)
        except urllib_error.URLError as exc:
            raise RuntimeError(f'Unable to reach the configured beta-interest API endpoint ({exc}).') from exc
        if int(status_code) >= 400:
            raise RuntimeError(f'The configured beta-interest API endpoint returned HTTP {status_code}.')
    local_submissions.append(record)
    return record


def update_beta_interest_status(
    submissions: list[dict[str, Any]],
    submission_id: str,
    *,
    follow_up_status: str,
) -> dict[str, Any] | None:
    target_status = follow_up_status if follow_up_status in BETA_INTEREST_STATUS_OPTIONS else 'New'
    for submission in submissions:
        if str(submission.get('submission_id', '')) != str(submission_id):
            continue
        submission['follow_up_status'] = target_status
        if target_status == 'Contacted' and not submission.get('contacted_at'):
            submission['contacted_at'] = datetime.now(UTC).isoformat()
        if target_status == 'Completed' and not submission.get('completed_at'):
            submission['completed_at'] = datetime.now(UTC).isoformat()
            if not submission.get('contacted_at'):
                submission['contacted_at'] = submission['completed_at']
        return submission
    return None


def beta_interest_csv_bytes(submissions: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            'submission_id',
            'submitted_at',
            'name',
            'email',
            'organization',
            'use_case',
            'workspace_name',
            'workspace_id',
            'submitted_by',
            'dataset_name',
            'dataset_source_mode',
            'capture_mode',
            'follow_up_status',
            'contacted_at',
            'completed_at',
        ],
    )
    writer.writeheader()
    for submission in merge_beta_interest_submissions(submissions):
        writer.writerow({key: submission.get(key, '') for key in writer.fieldnames})
    return output.getvalue().encode('utf-8')


def build_beta_interest_summary(submissions: list[dict[str, Any]]) -> dict[str, Any]:
    records = merge_beta_interest_submissions(submissions)
    if not records:
        empty = pd.DataFrame(
            columns=[
                'submitted_at',
                'name',
                'email',
                'organization',
                'use_case',
                'workspace_name',
                'submitted_by',
                'capture_mode',
                'follow_up_status',
            ]
        )
        return {
            'summary_cards': [
                {'label': 'Captured Leads', 'value': '0'},
                {'label': 'Organizations', 'value': '0'},
                {'label': 'Use Cases', 'value': '0'},
                {'label': 'Contacted', 'value': '0'},
            ],
            'history': empty,
            'notes': [
                'Beta interest capture is enabled for this workspace.',
                'Choose LOCAL, DATABASE, or API storage mode depending on how you want to route follow-up leads.',
            ],
        }

    history = pd.DataFrame(records).sort_values('submitted_at', ascending=False).reset_index(drop=True)
    if 'follow_up_status' not in history.columns:
        history['follow_up_status'] = 'New'
    if 'contacted_at' not in history.columns:
        history['contacted_at'] = ''
    if 'completed_at' not in history.columns:
        history['completed_at'] = ''
    org_count = int(history['organization'].astype(str).str.strip().replace('', pd.NA).dropna().nunique()) if 'organization' in history.columns else 0
    use_case_count = int(history['use_case'].astype(str).str.strip().replace('', pd.NA).dropna().nunique()) if 'use_case' in history.columns else 0
    contacted_count = int((history['follow_up_status'] == 'Contacted').sum() + (history['follow_up_status'] == 'Completed').sum())
    return {
        'summary_cards': [
            {'label': 'Captured Leads', 'value': f'{len(history):,}'},
            {'label': 'Organizations', 'value': f'{org_count:,}'},
            {'label': 'Use Cases', 'value': f'{use_case_count:,}'},
            {'label': 'Contacted', 'value': f'{contacted_count:,}'},
        ],
        'history': history,
        'notes': [
            'Multiple submissions from the same email are allowed so teams can capture new use cases over time.',
            f"For save failures, use the fallback support address: {support_email()}",
        ],
    }


def build_beta_conversion_panel(beta_enabled: bool, submissions: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_beta_interest_summary(submissions)
    ctas = [
        {
            'label': 'Request Early Access',
            'preset_use_case': 'Healthcare data readiness assessment',
            'note': 'Best for analysts and small teams evaluating the platform for active healthcare datasets.',
            'cta_summary': 'Start an early-access conversation for a real dataset review.',
        },
        {
            'label': 'Join Beta',
            'preset_use_case': 'Healthcare analytics beta program',
            'note': 'Best for users who want guided access to dataset readiness, analytics, and reporting workflows.',
            'cta_summary': 'Join the guided beta for readiness, analytics, Copilot, and reporting workflows.',
        },
        {
            'label': 'Contact for Pilot',
            'preset_use_case': 'Design-partner pilot conversation',
            'note': 'Best for consulting groups, research teams, and organizations planning a real pilot dataset review.',
            'cta_summary': 'Open a pilot conversation for a scoped healthcare dataset engagement.',
        },
    ]
    status_note = (
        'Beta capture is live in this workspace and can save locally, to the database, or to an external CRM/API.'
        if beta_enabled
        else 'Beta capture is currently muted in Product Settings. You can still use the CTAs as startup positioning without enabling follow-up collection.'
    )
    return {
        'headline': 'Request early access, join the beta, or start a pilot conversation',
        'subheadline': 'Clinverity is positioned as a clinical data quality, remediation, and audit platform for pilot teams who need faster onboarding, clearer blockers, and stakeholder-ready outputs.',
        'summary_cards': summary.get('summary_cards', []),
        'cta_cards': ctas,
        'status_note': status_note,
        'conversion_steps': [
            'Choose the CTA that matches the conversation you want to start.',
            'Capture the contact, organization, and use case in the in-app beta form.',
            'Route the submission to local storage, the workspace database, or an external CRM/API endpoint.',
        ],
        'reassurance_note': 'This flow stays demo-safe by default and can graduate to database or API capture when a real follow-up backend is ready.',
    }


__all__ = [
    'BETA_INTEREST_API_TOKEN_ENV',
    'BETA_INTEREST_API_URL_ENV',
    'BETA_INTEREST_STATUS_OPTIONS',
    'BETA_INTEREST_STORAGE_OPTIONS',
    'append_beta_interest_submission',
    'beta_interest_csv_bytes',
    'build_beta_conversion_panel',
    'build_beta_interest_record',
    'build_beta_interest_summary',
    'merge_beta_interest_submissions',
    'save_beta_interest_submission',
    'support_email',
    'update_beta_interest_status',
    'validate_beta_interest',
]
