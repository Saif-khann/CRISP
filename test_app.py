"""
CRISP - Construction Recognition & Intelligence for Stage Progress
Unit and integration test suite.
"""

import os
import tempfile
import uuid

import pytest

# Point the app at a scratch database before importing it, so tests never
# touch the real data/crisp.db.
_TEST_DB = os.path.join(tempfile.gettempdir(), f'crisp-test-{uuid.uuid4().hex}.db')
os.environ['CRISP_DB_PATH'] = _TEST_DB
os.environ['FLASK_SECRET_KEY'] = 'test-secret-key'
os.environ['ALLOW_DEMO_LOGIN'] = 'true'

import database
from app import (
    app, allowed_file, calculate_progress, get_progress_message,
    stages, stage_weights, sub_stage_weights
)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope='session', autouse=True)
def _cleanup_test_db():
    yield
    for suffix in ('', '-wal', '-shm'):
        try:
            os.remove(_TEST_DB + suffix)
        except OSError:
            pass


@pytest.fixture
def account():
    """Factory creating throwaway accounts."""
    def _make(role='worker'):
        email = f'{uuid.uuid4().hex[:10]}@test.local'
        password = 'test-password-123'
        user_id = database.create_user(email, password, role)
        return {'email': email, 'password': password, 'role': role, 'id': user_id}
    return _make


def _csrf(client):
    """Establish a session and return its CSRF token.

    Every state-changing request needs one; enforcement is global via a
    before_request hook rather than per-route opt-in.
    """
    client.get('/login')
    with client.session_transaction() as sess:
        return sess.get('_csrf_token')


def _post(client, path, data=None, **kwargs):
    """POST with a valid CSRF token attached."""
    payload = dict(data or {})
    payload['csrf_token'] = _csrf(client)
    return client.post(path, data=payload, follow_redirects=False, **kwargs)


def _login(client, acct):
    return _post(client, '/login',
                 {'email': acct['email'], 'password': acct['password']})


# ---------------------------------------------------------------------------
# Route Tests
# ---------------------------------------------------------------------------

def test_unauthenticated_redirect_to_login(client):
    """Visiting root redirects to login when unauthenticated."""
    rv = client.get('/', follow_redirects=False)
    assert rv.status_code == 302
    assert '/login' in rv.headers['Location']


def test_login_page_renders(client):
    rv = client.get('/login')
    assert rv.status_code == 200
    assert b'CRISP' in rv.data


def test_signup_page_renders(client):
    rv = client.get('/signup')
    assert rv.status_code == 200
    assert b'CRISP' in rv.data


def test_dashboard_requires_login(client):
    rv = client.get('/dashboard', follow_redirects=False)
    assert rv.status_code == 302
    assert '/login' in rv.headers['Location']


def test_geo_map_requires_login(client):
    rv = client.get('/geo-map', follow_redirects=False)
    assert rv.status_code == 302
    assert '/login' in rv.headers['Location']


def test_visual_comparison_requires_login(client):
    rv = client.get('/visual_comparison', follow_redirects=False)
    assert rv.status_code == 302
    assert '/login' in rv.headers['Location']


# ---------------------------------------------------------------------------
# Authentication - regression guards
#
# The login form previously accepted ANY email/password whenever Firebase
# was unconfigured, and inferred the role from whether "expert" appeared
# in the email string. These tests exist so that cannot come back.
# ---------------------------------------------------------------------------

def test_arbitrary_credentials_are_rejected(client):
    """An unregistered email must never be granted a session."""
    rv = _post(client, '/login',
               {'email': 'expert@anything.com', 'password': 'whatever'})
    assert rv.status_code == 401
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_wrong_password_is_rejected(client, account):
    acct = account('worker')
    rv = _post(client, '/login',
               {'email': acct['email'], 'password': 'not-the-password'})
    assert rv.status_code == 401
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_correct_password_is_accepted(client, account):
    acct = account('expert')
    rv = _login(client, acct)
    assert rv.status_code == 302
    assert 'expert_dashboard' in rv.headers['Location']


def test_role_comes_from_the_account_not_the_email(client):
    """Role must be read from the stored account, never inferred from the
    email address."""
    email = f'expert-{uuid.uuid4().hex[:8]}@test.local'
    database.create_user(email, 'test-password-123', 'worker')
    rv = _post(client, '/login', {'email': email, 'password': 'test-password-123'})
    assert rv.status_code == 302
    # Despite "expert" appearing in the address, this account is a worker.
    assert 'worker_dashboard' in rv.headers['Location']


def test_passwords_are_not_stored_in_plaintext(account):
    acct = account('worker')
    with database.get_connection() as conn:
        row = conn.execute(
            'SELECT password_hash FROM users WHERE id = ?', (acct['id'],)
        ).fetchone()
    assert acct['password'] not in row['password_hash']
    assert len(row['password_hash']) > 40


def test_short_passwords_are_refused():
    with pytest.raises(ValueError):
        database.create_user(f'{uuid.uuid4().hex}@test.local', 'short', 'worker')


def test_duplicate_email_is_refused(account):
    acct = account('worker')
    with pytest.raises(ValueError):
        database.create_user(acct['email'], 'another-password-1', 'expert')


def test_invalid_role_is_refused():
    with pytest.raises(ValueError):
        database.create_user(
            f'{uuid.uuid4().hex}@test.local', 'test-password-123', 'admin'
        )


# ---------------------------------------------------------------------------
# Role gating
# ---------------------------------------------------------------------------

def test_worker_cannot_reach_expert_dashboard(client, account):
    _login(client, account('worker'))
    rv = client.get('/expert_dashboard', follow_redirects=False)
    assert rv.status_code == 302
    assert 'expert_dashboard' not in rv.headers['Location']


def test_visual_analyzer_is_expert_only(client, account):
    """The Visual Analyzer is a review tool for experts."""
    _login(client, account('worker'))
    assert client.get('/visual_comparison', follow_redirects=False).status_code == 302
    assert _post(client, '/visual_comparison').status_code in (302, 403)


def test_visual_analyzer_allows_experts(client, account):
    _login(client, account('expert'))
    assert client.get('/visual_comparison').status_code == 200


# ---------------------------------------------------------------------------
# Per-account data isolation
# ---------------------------------------------------------------------------

def test_projects_are_scoped_to_their_owner(account):
    owner = account('expert')
    other = account('expert')
    project_id = database.create_project(
        owner['id'], 'Owner Site', 'desc', 'loc', '2024-01-01', '2025-12-31', 1.0, 2.0
    )
    assert database.get_project(owner['id'], project_id) is not None
    # Guessing the id as a different user must not reveal it.
    assert database.get_project(other['id'], project_id) is None
    assert database.list_projects(other['id']) == []


def test_cannot_open_another_users_project_over_http(client, account):
    owner = account('expert')
    project_id = database.create_project(
        owner['id'], 'Private Site', 'desc', 'loc', '2024-01-01', '2025-12-31', 1.0, 2.0
    )
    _login(client, account('expert'))
    rv = client.get(f'/home?project_id={project_id}', follow_redirects=False)
    assert rv.status_code == 302


def test_validations_are_scoped_to_their_owner(account):
    owner = account('worker')
    other = account('worker')
    project_id = database.create_project(
        owner['id'], 'S', 'd', 'l', '2024-01-01', '2025-12-31', 1.0, 2.0
    )
    validation_id = database.create_validation(
        owner['id'], project_id, 'foundation', 'Excavation', 90.0, 80.0, None
    )
    assert database.get_validation(owner['id'], validation_id) is not None
    assert database.get_validation(other['id'], validation_id) is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_progress_is_derived_from_validation_history(account):
    acct = account('worker')
    project_id = database.create_project(
        acct['id'], 'S', 'd', 'l', '2024-01-01', '2025-12-31', 1.0, 2.0
    )
    assert database.get_latest_validation(acct['id'], project_id) is None

    database.create_validation(
        acct['id'], project_id, 'superstructure',
        'Structural_Frame_Erection_(framing)', 95.0, 85.0, None
    )
    latest = database.get_latest_validation(acct['id'], project_id)
    assert latest['primary_stage'] == 'superstructure'

    _, overall, completed = calculate_progress(
        latest['primary_stage'], latest['specific_classification']
    )
    assert overall > 0
    assert 'foundation' in completed


# ---------------------------------------------------------------------------
# Helper / Business Logic Tests
# ---------------------------------------------------------------------------

def test_allowed_file():
    """Test allowed file extensions."""
    assert allowed_file('site_photo.jpg') is True
    assert allowed_file('site_photo.jpeg') is True
    assert allowed_file('site_photo.png') is True
    assert allowed_file('site_photo.PNG') is True
    assert allowed_file('document.pdf') is False
    assert allowed_file('script.py') is False
    assert allowed_file('executable.exe') is False
    assert allowed_file('no_extension') is False


def test_calculate_progress_foundation():
    """Test progress calculation for foundation sub-stages."""
    stage_prog, overall_prog, completed = calculate_progress('foundation', 'Excavation')
    assert stage_prog == 25
    assert overall_prog == (20 * 0.25)
    assert completed == []

    stage_prog, overall_prog, completed = calculate_progress('foundation', 'concrete_pouring')
    # Excavation (25) + Reinforcement (25) + concrete curing (25) + concrete_pouring (25) = 100
    assert stage_prog == 100
    assert 'foundation' in completed


def test_calculate_progress_invalid_stage():
    """Test progress calculation with non-existent stage."""
    stage_prog, overall_prog, completed = calculate_progress('invalid_stage', 'sub')
    assert stage_prog == 0
    assert overall_prog == 0
    assert completed == []


def test_progress_message_advancement():
    """Test progress message generation when advancing stages."""
    status, msg = get_progress_message(
        'foundation', 'Excavation',
        'superstructure', 'Structural_Frame_Erection_(framing)'
    )
    assert status == 'advanced'
    assert 'Progress has advanced' in msg


def test_progress_message_invalid_regression():
    """Test progress message when attempting an invalid backward stage jump."""
    status, msg = get_progress_message(
        'superstructure', 'Roof_Decking',
        'foundation', 'Excavation'
    )
    assert status == 'invalid'
    assert 'Invalid progress' in msg


def test_progress_message_same_stage():
    """Test progress message when stage has not changed."""
    status, msg = get_progress_message(
        'foundation', 'Excavation',
        'foundation', 'Excavation'
    )
    assert status == 'same'
    assert 'No progress detected' in msg


def test_rounding_is_consistent_for_exact_ties():
    """Weighted stage math lands on exact .x5 ties constantly; the value
    must be rounded once at the source so every consumer agrees on the
    digit shown (Python and JavaScript round such ties differently)."""
    _, overall, _ = calculate_progress('finishing works', 'Painting')
    assert overall == 90.3
    _, msg = get_progress_message(
        'foundation', 'Excavation', 'finishing works', 'Painting'
    )
    assert '90.3% complete' in msg


def test_stages_and_weights_consistency():
    """Test that all stages have defined weights totaling 100%."""
    total_weight = sum(stage_weights.values())
    assert total_weight == 100

    for stage, sub_dict in sub_stage_weights.items():
        assert sum(sub_dict.values()) == 100, f"Sub-stage weights for {stage} must sum to 100"
        assert set(sub_dict.keys()) == set(stages[stage]), f"Sub-stages for {stage} must match stage definition"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_healthz(client):
    rv = client.get('/healthz')
    assert rv.status_code == 200
    assert rv.get_json()['status'] == 'ok'


# ---------------------------------------------------------------------------
# CSRF protection
# ---------------------------------------------------------------------------

def test_post_without_csrf_token_is_rejected(client, account):
    """A state-changing request with no token must not authenticate."""
    acct = account('worker')
    client.post(
        '/login',
        data={'email': acct['email'], 'password': acct['password']},
        follow_redirects=False
    )
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_post_with_wrong_csrf_token_is_rejected(client, account):
    acct = account('worker')
    client.get('/login')
    client.post(
        '/login',
        data={'email': acct['email'], 'password': acct['password'],
              'csrf_token': 'not-the-real-token'},
        follow_redirects=False
    )
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_csrf_token_accepted_via_header(client, account):
    """AJAX callers send the token as a header instead of a form field."""
    acct = account('expert')
    token = _csrf(client)
    rv = client.post(
        '/login',
        data={'email': acct['email'], 'password': acct['password']},
        headers={'X-CSRFToken': token},
        follow_redirects=False
    )
    assert rv.status_code == 302
    assert 'expert_dashboard' in rv.headers['Location']


def test_get_requests_need_no_csrf_token(client):
    assert client.get('/login').status_code == 200


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------

def test_password_change_requires_correct_current_password(account):
    acct = account('worker')
    with pytest.raises(ValueError):
        database.change_password(acct['id'], 'wrong-current', 'brand-new-password')


def test_password_change_rejects_short_new_password(account):
    acct = account('worker')
    with pytest.raises(ValueError):
        database.change_password(acct['id'], acct['password'], 'short')


def test_password_change_rejects_reusing_the_same_password(account):
    acct = account('worker')
    with pytest.raises(ValueError):
        database.change_password(acct['id'], acct['password'], acct['password'])


def test_password_change_succeeds_and_old_password_stops_working(account):
    acct = account('worker')
    database.change_password(acct['id'], acct['password'], 'a-brand-new-password')
    assert database.verify_user(acct['email'], acct['password']) is None
    assert database.verify_user(acct['email'], 'a-brand-new-password') is not None


def test_account_page_requires_login(client):
    rv = client.get('/account', follow_redirects=False)
    assert rv.status_code == 302


def test_account_page_renders_for_signed_in_user(client, account):
    _login(client, account('worker'))
    assert client.get('/account').status_code == 200


# ---------------------------------------------------------------------------
# Login throttling and audit log
# ---------------------------------------------------------------------------

def test_failed_logins_are_recorded_in_the_audit_log(client, account):
    acct = account('worker')
    _post(client, '/login', {'email': acct['email'], 'password': 'wrong'})
    events = database.list_auth_events(limit=20, email=acct['email'])
    assert any(e['event'] == 'login_failed' for e in events)


def test_successful_login_is_recorded_in_the_audit_log(client, account):
    acct = account('expert')
    _login(client, acct)
    events = database.list_auth_events(limit=20, email=acct['email'])
    assert any(e['event'] == 'login_success' for e in events)


def test_throttle_counter_persists_in_the_database(client, account):
    """The lockout must survive a restart, so it cannot live in a dict."""
    acct = account('worker')
    for _ in range(3):
        _post(client, '/login', {'email': acct['email'], 'password': 'wrong'})

    with database.get_connection() as conn:
        row = conn.execute('SELECT SUM(attempts) AS total FROM login_attempts').fetchone()
    assert row['total'] >= 3


def test_security_headers_are_set(client):
    rv = client.get('/login')
    assert rv.headers.get('X-Content-Type-Options') == 'nosniff'
    assert rv.headers.get('X-Frame-Options') == 'DENY'
    assert 'Referrer-Policy' in rv.headers
