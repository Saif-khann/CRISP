"""
CRISP - Construction Recognition & Intelligence for Stage Progress
Authentication: credential checks, session guards, role gating, CSRF.

Credentials are verified against scrypt-hashed passwords in the SQLite
store (see database.py). Failed attempts are throttled, and every
authentication event is written to an append-only audit table.
"""

import hmac
import logging
import secrets
from functools import wraps

from flask import session, redirect, url_for, flash, request, jsonify

import database

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 8
LOCKOUT_SECONDS = 300

# Methods that can change state and therefore require a CSRF token.
UNSAFE_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}

CSRF_SESSION_KEY = '_csrf_token'
CSRF_FORM_FIELD = 'csrf_token'
CSRF_HEADER = 'X-CSRFToken'


# ---------------------------------------------------------------------------
# CSRF protection
#
# SameSite=Lax cookies already block the cross-site form POST that classic
# CSRF depends on. These tokens are defence in depth: they also cover
# same-site injection and browsers that mishandle SameSite.
# ---------------------------------------------------------------------------

def get_csrf_token():
    """Return this session's CSRF token, creating one on first use."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def _submitted_csrf_token():
    return (request.form.get(CSRF_FORM_FIELD)
            or request.headers.get(CSRF_HEADER)
            or '')


def csrf_token_is_valid():
    expected = session.get(CSRF_SESSION_KEY)
    submitted = _submitted_csrf_token()
    if not expected or not submitted:
        return False
    # Constant-time comparison so a token cannot be recovered by timing.
    return hmac.compare_digest(expected, submitted)


def wants_json():
    """True when the caller expects JSON rather than a rendered page."""
    if request.headers.get(CSRF_HEADER) or request.is_json:
        return True
    accept = request.headers.get('Accept', '')
    return 'application/json' in accept and 'text/html' not in accept


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------

def _throttle_key(email):
    return f"{request.remote_addr or 'unknown'}|{(email or '').strip().lower()}"


def is_locked_out(email):
    """True if this IP and email pair has failed too many times recently."""
    attempts, first_seen = database.get_login_attempts(_throttle_key(email))
    if attempts < MAX_ATTEMPTS or first_seen is None:
        return False

    from datetime import datetime, timezone
    elapsed = (datetime.now(timezone.utc) - first_seen).total_seconds()
    if elapsed > LOCKOUT_SECONDS:
        database.clear_login_attempts(_throttle_key(email))
        return False
    return True


def record_failed_attempt(email):
    return database.record_login_failure(_throttle_key(email), LOCKOUT_SECONDS)


def clear_failed_attempts(email):
    database.clear_login_attempts(_throttle_key(email))


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def audit(event, email=None, user_id=None, detail=None):
    database.record_auth_event(
        event=event,
        email=email,
        user_id=user_id,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent'),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def create_user(email, password, role='worker', display_name=None):
    """Register a new account. Raises ValueError with a user-safe message."""
    return database.create_user(email, password, role, display_name)


def verify_user(email, password):
    """Return the user dict on valid credentials, else None."""
    return database.verify_user(email, password)


def establish_session(user):
    """Populate the session for an authenticated user.

    The session is cleared first so a pre-authentication session identifier
    can never be carried over, and a fresh CSRF token is issued.
    """
    session.clear()
    session['user_id'] = user['id']
    session['email'] = user['email']
    session['role'] = user['role']
    session['display_name'] = user['display_name']
    session['is_demo'] = user.get('is_demo', False)
    session[CSRF_SESSION_KEY] = secrets.token_urlsafe(32)
    session.permanent = True


# ---------------------------------------------------------------------------
# Route guards
# ---------------------------------------------------------------------------

def login_required(f):
    """Require an authenticated session.

    Redirects silently. This fires on an ordinary first visit to a
    protected URL as often as on an expired session, and greeting a new
    visitor with a red error banner reads as broken rather than helpful.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if wants_json():
                return jsonify({'success': False,
                                'error': 'Please sign in first.'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def role_required(allowed_roles):
    """Require one of the given roles.

    Accepts a single role string ('expert') or a list. A bare string is
    treated as one exact role, never as a substring-match target.
    """
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]

    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            user_role = session.get('role')
            if not user_role or user_role not in allowed_roles:
                if wants_json():
                    return jsonify({
                        'success': False,
                        'error': 'You do not have permission to do that.'
                    }), 403
                flash('You do not have permission to access that page.', 'error')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator
