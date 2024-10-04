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
