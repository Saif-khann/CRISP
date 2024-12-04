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
