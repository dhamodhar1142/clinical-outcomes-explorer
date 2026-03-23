from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.auth import (
    build_auth_service,
    build_guest_auth_session,
    build_hosted_auth_session,
    build_oidc_authorization_url,
    build_workspace_security_summary,
    enforce_workspace_boundary,
    enforce_workspace_minimum_role,
    enforce_workspace_permission,
    generate_totp_code,
    get_workspace_role_rank,
    get_workspace_role_permissions,
    verify_two_factor_code,
    workspace_can_access,
    workspace_identity_has_minimum_role,
    workspace_identity_matches_workspace,
    workspace_identity_can_access,
    workspace_role_at_least,
)


class AuthFoundationTests(unittest.TestCase):
    def test_auth_service_defaults_to_guest_mode_without_database(self) -> None:
        service = build_auth_service(database_url='', sqlite_path='')
        guest = build_guest_auth_session()
        identity = service.build_workspace_identity(guest, 'Guest Demo Workspace')
        self.assertFalse(service.status.enabled)
        self.assertEqual(guest['auth_mode'], 'guest')
        self.assertEqual(guest['role'], 'viewer')
        self.assertEqual(identity['workspace_id'], 'guest-demo-workspace')
        self.assertEqual(identity['owner_label'], 'Guest session')
        self.assertEqual(identity['role_label'], 'Viewer')

    def test_local_sign_in_returns_owned_workspace_identity(self) -> None:
        service = build_auth_service(database_url='', sqlite_path='')
        auth_session, identity = service.sign_in_local('Dana Analyst', 'dana@example.com', 'Care Ops')
        self.assertTrue(auth_session['signed_in'])
        self.assertEqual(auth_session['email'], 'dana@example.com')
        self.assertEqual(auth_session['role'], 'owner')
        self.assertEqual(identity['workspace_name'], 'Care Ops')
        self.assertEqual(identity['owner_label'], 'Dana Analyst')
        self.assertEqual(identity['auth_mode'], 'local')
        self.assertEqual(identity['role'], 'owner')
        self.assertTrue(identity['is_workspace_owner'])
        self.assertTrue(str(identity['workspace_id']).startswith('care-ops::'))

    def test_sqlite_auth_service_persists_local_users_and_workspace_owners(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'auth.sqlite3'
            service = build_auth_service(sqlite_path=sqlite_path)
            auth_session, identity = service.sign_in_local('Dana Analyst', 'dana@example.com', 'Care Ops', 'Admin')
            self.assertTrue(service.status.enabled)
            self.assertEqual(auth_session['provider'], 'local-demo')
            self.assertEqual(auth_session['role'], 'admin')
            self.assertEqual(identity['owner_user_id'], auth_session['user_id'])
            self.assertEqual(identity['role'], 'admin')
            self.assertTrue(sqlite_path.exists())

    def test_sign_out_returns_guest_session(self) -> None:
        service = build_auth_service(database_url='', sqlite_path='')
        auth_session, identity = service.sign_out()
        self.assertFalse(auth_session['signed_in'])
        self.assertEqual(identity['workspace_name'], 'Guest Demo Workspace')
        self.assertEqual(identity['status_label'], 'Guest session')

    def test_workspace_role_permissions_are_available(self) -> None:
        self.assertIn('workspace_admin', get_workspace_role_permissions('owner'))
        self.assertIn('export_download', get_workspace_role_permissions('analyst'))
        self.assertTrue(workspace_can_access('admin', 'workspace_write'))
        self.assertFalse(workspace_can_access('viewer', 'workspace_write'))
        self.assertTrue(workspace_role_at_least('admin', 'analyst'))
        self.assertFalse(workspace_role_at_least('viewer', 'analyst'))
        self.assertGreater(get_workspace_role_rank('owner'), get_workspace_role_rank('viewer'))

    def test_guest_mode_keeps_demo_permissions_available(self) -> None:
        guest = build_guest_auth_session()
        identity = build_auth_service(database_url='', sqlite_path='').build_workspace_identity(guest, 'Guest Demo Workspace')
        self.assertTrue(workspace_identity_can_access(identity, 'dataset_upload'))
        enforce_workspace_permission(identity, 'export_generate', resource_label='report generation')
        self.assertTrue(workspace_identity_matches_workspace(identity, 'any-workspace'))
        enforce_workspace_boundary(identity, workspace_id='another-workspace', resource_label='tenant-scoped asset')

    def test_signed_in_identity_respects_workspace_boundary(self) -> None:
        identity = {
            'workspace_id': 'workspace-a',
            'workspace_name': 'Pilot Workspace',
            'auth_mode': 'local',
            'role': 'admin',
            'membership_validated': True,
        }
        self.assertTrue(workspace_identity_matches_workspace(identity, 'workspace-a'))
        self.assertFalse(workspace_identity_matches_workspace(identity, 'workspace-b'))
        with self.assertRaises(PermissionError):
            enforce_workspace_boundary(identity, workspace_id='workspace-b', resource_label='snapshot load')

    def test_signed_in_identity_requires_verified_membership_for_permissions(self) -> None:
        identity = {
            'workspace_id': 'workspace-a',
            'workspace_name': 'Pilot Workspace',
            'auth_mode': 'local',
            'role': 'admin',
            'membership_validated': False,
        }
        self.assertFalse(workspace_identity_can_access(identity, 'workspace_write'))
        self.assertFalse(workspace_identity_has_minimum_role(identity, 'analyst'))

    def test_workspace_minimum_role_enforcement_stays_demo_safe_for_guest(self) -> None:
        guest = build_guest_auth_session()
        identity = build_auth_service(database_url='', sqlite_path='').build_workspace_identity(guest, 'Guest Demo Workspace')
        self.assertTrue(workspace_identity_has_minimum_role(identity, 'admin'))
        enforce_workspace_minimum_role(identity, 'admin', resource_label='workspace settings')

    def test_workspace_minimum_role_enforcement_blocks_viewer_for_analyst_actions(self) -> None:
        viewer_identity = {
            'workspace_id': 'workspace-a',
            'workspace_name': 'Pilot Workspace',
            'auth_mode': 'local',
            'role': 'viewer',
            'membership_validated': True,
        }
        self.assertFalse(workspace_identity_has_minimum_role(viewer_identity, 'analyst'))
        with self.assertRaises(PermissionError):
            enforce_workspace_minimum_role(viewer_identity, 'analyst', resource_label='dataset uploads')

    def test_workspace_security_summary_is_available(self) -> None:
        service = build_auth_service(database_url='', sqlite_path='')
        auth_session, identity = service.sign_in_local('Dana Analyst', 'dana@example.com', 'Care Ops', 'Admin')
        summary = build_workspace_security_summary(identity, service.status)
        self.assertTrue(summary['summary_cards'])
        self.assertTrue(summary['boundaries_table'])
        self.assertTrue(any(row['control_area'] == 'Tenant boundary' for row in summary['boundaries_table']))
        self.assertTrue(any(row['control_area'] == 'Workspace role enforcement' for row in summary['boundaries_table']))
        self.assertTrue(any(row['control_area'] == 'Export permissions' for row in summary['boundaries_table']))

    def test_sqlite_auth_service_marks_membership_validation_on_workspace_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'auth.sqlite3'
            service = build_auth_service(sqlite_path=sqlite_path)
            auth_session, identity = service.sign_in_local('Dana Analyst', 'dana@example.com', 'Care Ops', 'Owner')
            rebuilt = service.build_workspace_identity(auth_session, 'Care Ops')
            self.assertTrue(rebuilt['membership_validated'])
            self.assertTrue(rebuilt['ownership_validated'])

    def test_hosted_auth_session_builder_preserves_provider_subject_and_tenant(self) -> None:
        session = build_hosted_auth_session(
            display_name='Dana Analyst',
            email='dana@example.com',
            provider_subject='auth0|123',
            provider='auth0',
            tenant_id='tenant-1',
            role='Admin',
        )
        self.assertEqual(session['auth_mode'], 'hosted')
        self.assertEqual(session['provider'], 'auth0')
        self.assertEqual(session['provider_subject'], 'auth0|123')
        self.assertEqual(session['tenant_id'], 'tenant-1')
        self.assertEqual(session['role'], 'admin')

    def test_hosted_auth_preparation_mode_remains_fallback_safe_without_sqlite(self) -> None:
        with patch.dict(
            'os.environ',
            {
                'SMART_DATASET_ANALYZER_AUTH_PROVIDER': 'auth0',
                'SMART_DATASET_ANALYZER_AUTH_ISSUER': 'https://tenant.example.com/',
                'SMART_DATASET_ANALYZER_AUTH_AUDIENCE': 'smart-dataset-analyzer',
            },
            clear=False,
        ):
            service = build_auth_service(database_url='', sqlite_path='')
        self.assertFalse(service.status.enabled)
        self.assertEqual(service.status.mode, 'hosted-prepared')
        self.assertEqual(service.status.provider_mode, 'hosted-prepared')
        self.assertTrue(service.status.hosted_configured)

    def test_sqlite_auth_service_supports_hosted_membership_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'auth.sqlite3'
            with patch.dict(
                'os.environ',
                {
                    'SMART_DATASET_ANALYZER_AUTH_PROVIDER': 'auth0',
                    'SMART_DATASET_ANALYZER_AUTH_ISSUER': 'https://tenant.example.com/',
                    'SMART_DATASET_ANALYZER_AUTH_AUDIENCE': 'smart-dataset-analyzer',
                    'SMART_DATASET_ANALYZER_AUTH_TENANT_ID': 'tenant-1',
                },
                clear=False,
            ):
                service = build_auth_service(sqlite_path=sqlite_path)
                auth_session, identity = service.sign_in_hosted(
                    display_name='Dana Analyst',
                    email='dana@example.com',
                    provider_subject='auth0|abc',
                    workspace_name='Care Ops',
                    workspace_role='Admin',
                )
                rebuilt = service.build_workspace_identity(auth_session, 'Care Ops')
        self.assertTrue(service.status.enabled)
        self.assertEqual(service.status.mode, 'hybrid')
        self.assertEqual(identity['auth_mode'], 'hosted')
        self.assertEqual(rebuilt['membership_validated'], True)
        self.assertEqual(rebuilt['membership_source'], 'hosted')
        self.assertEqual(rebuilt['tenant_id'], 'tenant-1')
        self.assertEqual(rebuilt['provider_subject'], 'auth0|abc')
        self.assertEqual(rebuilt['role'], 'admin')

    def test_totp_verification_supports_second_factor(self) -> None:
        secret = 'JBSWY3DPEHPK3PXP'
        code = generate_totp_code(secret, timestamp=1_700_000_000)
        self.assertTrue(verify_two_factor_code(secret, code, timestamp=1_700_000_000, window=0))
        self.assertFalse(verify_two_factor_code(secret, '000000', timestamp=1_700_000_000, window=0))

    def test_hosted_auth_service_builds_oidc_start_configuration(self) -> None:
        with patch.dict(
            'os.environ',
            {
                'SMART_DATASET_ANALYZER_AUTH_PROVIDER': 'okta',
                'SMART_DATASET_ANALYZER_AUTH_ISSUER': 'https://tenant.okta.com/oauth2/default',
                'SMART_DATASET_ANALYZER_AUTH_CLIENT_ID': 'client-123',
                'SMART_DATASET_ANALYZER_AUTH_REDIRECT_URI': 'https://app.example.com/callback',
            },
            clear=False,
        ):
            service = build_auth_service(database_url='', sqlite_path='')
        start = service.build_oidc_start()
        self.assertIsNotNone(start)
        self.assertIn('authorization_url', start)
        self.assertIn('client_id=client-123', start['authorization_url'])

    def test_sqlite_auth_service_tracks_and_revokes_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'auth.sqlite3'
            with patch.dict('os.environ', {'SMART_DATASET_ANALYZER_2FA_SECRET': 'JBSWY3DPEHPK3PXP'}, clear=False):
                service = build_auth_service(sqlite_path=sqlite_path)
                code = generate_totp_code('JBSWY3DPEHPK3PXP')
                auth_session, identity = service.sign_in_local('Dana Analyst', 'dana@example.com', 'Care Ops', 'Admin', two_factor_code=code)
                sessions = service.list_active_sessions(identity)
                self.assertTrue(sessions)
                self.assertTrue(auth_session['mfa_verified'])
                service.revoke_session(str(sessions[0]['session_id']))
                self.assertEqual(service.list_active_sessions(identity), [])

    def test_sqlite_auth_service_supports_workspace_invitations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / 'auth.sqlite3'
            service = build_auth_service(sqlite_path=sqlite_path)
            inviter_session, inviter_identity = service.sign_in_local('Dana Analyst', 'dana@example.com', 'Care Ops', 'Owner')
            invitation = service.create_workspace_invitation(inviter_identity, email='teammate@example.com', role='Analyst')
            self.assertIsNotNone(invitation)
            invited_session, _ = service.sign_in_local('Teammate User', 'teammate@example.com', 'Care Ops', 'Viewer')
            accepted = service.accept_workspace_invitation(invited_session, str(invitation['invitation_id']))
            self.assertIsNotNone(accepted)
            rebuilt = service.build_workspace_identity(invited_session, 'Care Ops')
            self.assertEqual(rebuilt['membership_source'], 'invitation')
            self.assertEqual(rebuilt['role'], 'analyst')


if __name__ == '__main__':
    unittest.main()
