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
