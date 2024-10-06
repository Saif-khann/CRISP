"""
CRISP - SQLite persistence layer.

This is the application's single source of truth for users, projects and
validations. It replaces the previous in-memory dictionaries (which were
global, shared between every visitor, and wiped on restart) and the
half-configured Firebase path.

Design notes:
  * One connection per operation, WAL journal mode - plenty for the
    concurrency a Gunicorn worker pool generates here, and avoids the
    "SQLite objects created in a thread" pitfall entirely.
  * Every project/validation query is scoped by user_id so one account
    can never read or mutate another account's records by guessing an id.
  * Timestamps are stored as ISO-8601 UTC strings and handed back to
    callers as real datetime objects, because the Jinja templates call
    .strftime() on them.
"""

import os
import sqlite3
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

DB_PATH = os.getenv('CRISP_DB_PATH', os.path.join('data', 'crisp.db'))

VALID_ROLES = ('expert', 'worker')

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('expert', 'worker')),
    display_name  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    is_demo       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name              TEXT NOT NULL,
    description       TEXT,
    location          TEXT,
    start_date        TEXT,
    end_date          TEXT,
    latitude          REAL,
    longitude         REAL,
    status            TEXT NOT NULL DEFAULT 'active',
    current_stage     TEXT,
    current_sub_stage TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);

CREATE TABLE IF NOT EXISTS validations (
    id                      TEXT PRIMARY KEY,
    project_id              TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id                 TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    primary_stage           TEXT NOT NULL,
    specific_classification TEXT NOT NULL,
    stage_confidence        REAL NOT NULL,
    global_confidence       REAL NOT NULL,
    image_path              TEXT,
    ai_description          TEXT,
    timestamp               TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_validations_project ON validations(project_id, timestamp DESC);

-- Failed sign-in attempts. Persisted rather than held in process memory so
-- the lockout survives a restart and is shared by every worker, instead of
-- resetting the moment an attacker triggers one.
CREATE TABLE IF NOT EXISTS login_attempts (
    throttle_key TEXT PRIMARY KEY,
    attempts     INTEGER NOT NULL DEFAULT 0,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL
);

-- Append-only record of authentication events, queryable rather than only
-- present in the application log.
CREATE TABLE IF NOT EXISTS auth_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    event      TEXT NOT NULL,
    email      TEXT,
    user_id    TEXT,
    ip_address TEXT,
    user_agent TEXT,
    detail     TEXT
);

CREATE INDEX IF NOT EXISTS idx_auth_events_time ON auth_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_auth_events_email ON auth_events(email, timestamp DESC);
"""


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value):
    """Parse a stored ISO timestamp back into a datetime (templates call
    .strftime on these). Returns None rather than raising on bad data."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


@contextmanager
def get_connection():
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Safe to call on every boot."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        try:
            conn.execute('PRAGMA journal_mode = WAL')
        except sqlite3.Error:
            # WAL is unavailable on some network/container filesystems;
            # the default rollback journal still works correctly.
            logger.debug("Could not enable WAL journal mode; using default.")
    logger.info("SQLite database ready at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def _row_to_user(row):
    if row is None:
        return None
    return {
        'id': row['id'],
        'user_id': row['id'],
        'email': row['email'],
        'role': row['role'],
        'display_name': row['display_name'],
        'status': row['status'],
        'is_demo': bool(row['is_demo']),
        'created_at': _parse_ts(row['created_at']),
    }


def create_user(email, password, role='worker', display_name=None, is_demo=False):
    """Create a user with a hashed password. Raises ValueError on invalid
    input and on duplicate email."""
    email = (email or '').strip()
    if not email:
        raise ValueError("Email is required")
    if not password:
        raise ValueError("Password is required")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if role not in VALID_ROLES:
        raise ValueError("Role must be 'expert' or 'worker'")

    user_id = f'usr-{uuid.uuid4().hex[:12]}'
    record = (
        user_id,
        email,
        generate_password_hash(password),
        role,
        display_name or email.split('@')[0],
        'active',
        1 if is_demo else 0,
        _utcnow_iso(),
    )

    try:
        with get_connection() as conn:
            conn.execute(
                'INSERT INTO users (id, email, password_hash, role, display_name,'
                ' status, is_demo, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                record
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with that email already exists") from exc

    return user_id


def get_user_by_email(email):
    with get_connection() as conn:
        row = conn.execute(
            'SELECT * FROM users WHERE email = ? COLLATE NOCASE', ((email or '').strip(),)
        ).fetchone()
    return _row_to_user(row)


def get_user(user_id):
    with get_connection() as conn:
        row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    return _row_to_user(row)


def verify_user(email, password):
    """Return the user dict if the credentials are valid, else None.

    Always runs a hash comparison even when the account doesn't exist, so
    a missing account and a wrong password take the same amount of time
    and can't be told apart by timing.
    """
    with get_connection() as conn:
        row = conn.execute(
            'SELECT * FROM users WHERE email = ? COLLATE NOCASE', ((email or '').strip(),)
        ).fetchone()

    if row is None:
        # Dummy comparison to equalise response time with the found case.
        check_password_hash(
            'pbkdf2:sha256:600000$dummysaltvalue$'
            '0000000000000000000000000000000000000000000000000000000000000000',
            password or ''
        )
        return None

    if not check_password_hash(row['password_hash'], password or ''):
        return None

    if row['status'] != 'active':
        return None

    return _row_to_user(row)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def _row_to_project(row):
    if row is None:
        return None
    return {
        'id': row['id'],
        'user_id': row['user_id'],
        'name': row['name'],
        'description': row['description'],
        'location': row['location'],
        'start_date': row['start_date'],
        'end_date': row['end_date'],
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'status': row['status'],
        'current_stage': row['current_stage'],
        'current_sub_stage': row['current_sub_stage'],
        'created_at': _parse_ts(row['created_at']),
    }


def create_project(user_id, name, description, location, start_date, end_date,
                   latitude, longitude):
    project_id = f'proj-{uuid.uuid4().hex[:10]}'
    with get_connection() as conn:
        conn.execute(
            'INSERT INTO projects (id, user_id, name, description, location,'
            ' start_date, end_date, latitude, longitude, status, current_stage,'
            ' current_sub_stage, created_at)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (project_id, user_id, name, description, location, start_date,
             end_date, latitude, longitude, 'active', None, None, _utcnow_iso())
        )
    return project_id


def list_projects(user_id):
    with get_connection() as conn:
        rows = conn.execute(
            'SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
    return [_row_to_project(r) for r in rows]
