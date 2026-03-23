from __future__ import annotations

ROLE_PERMISSIONS = {
    'Admin': {'sensitive_review', 'standards_validation', 'exports', 'audit_review'},
    'Analyst': {'sensitive_review', 'standards_validation', 'exports'},
    'Executive': {'standards_validation', 'exports'},
    'Clinician': {'sensitive_review', 'standards_validation', 'exports'},
    'Researcher': {'sensitive_review', 'standards_validation'},
    'Data Steward': {'sensitive_review', 'standards_validation', 'exports', 'audit_review'},
    'Viewer': {'standards_validation'},
}


def get_role_permissions(role: str) -> set[str]:
    return set(ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS['Analyst']))


def can_access(role: str, feature: str) -> bool:
    return feature in get_role_permissions(role)
