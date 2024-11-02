"""
CRISP - Construction Recognition & Intelligence for Stage Progress
Main Flask application for construction progress monitoring using CNN models.
"""

import os
import math
import logging
import uuid
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, jsonify, send_file,
    session, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import tempfile
import secrets
import requests

import database

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

IS_PRODUCTION = os.getenv('FLASK_ENV', '').lower() == 'production' or \
    os.getenv('CRISP_ENV', '').lower() == 'production'

app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    if IS_PRODUCTION:
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. Without it, session "
            "cookies are signed with a publicly-known key and can be forged."
        )
    # Ephemeral per-process key for local development: still unguessable,
    # and it invalidates sessions on restart rather than sharing a
    # hardcoded secret that could reach production by accident.
    app.secret_key = secrets.token_hex(32)
    logger.warning(
        "FLASK_SECRET_KEY is not set - generated a temporary key for this "
        "process. Sessions will not survive a restart. Set FLASK_SECRET_KEY "
        "before deploying."
    )

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,       # not readable from JavaScript
    SESSION_COOKIE_SAMESITE='Lax',      # blocks cross-site form CSRF
    SESSION_COOKIE_SECURE=IS_PRODUCTION,  # HTTPS-only once deployed
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB upload ceiling
)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'demo_samples'), exist_ok=True)

# Create tables on boot (idempotent).
database.init_db()

# Seed the one-click demo accounts, unless demo access is switched off.
ALLOW_DEMO_LOGIN = os.getenv('ALLOW_DEMO_LOGIN', 'true').lower() == 'true'
if ALLOW_DEMO_LOGIN:
    from seed_demo import seed_demo_data, DEMO_ACCOUNTS
    seed_demo_data()
else:
    DEMO_ACCOUNTS = {}

# Timezone from env (default UTC)
APP_TIMEZONE = os.getenv('APP_TIMEZONE', 'UTC')

# AI gateway configuration (any OpenAI-compatible endpoint)
AI_GATEWAY_URL = os.getenv('AI_GATEWAY_URL') or os.getenv('OPENAI_BASE_URL')
AI_GATEWAY_KEY = os.getenv('AI_GATEWAY_KEY') or os.getenv('OPENAI_API_KEY') or 'dummy-key'
AI_GATEWAY_MODEL = os.getenv('AI_GATEWAY_MODEL', 'gpt-4o-mini')
AI_GATEWAY_TIMEOUT = int(os.getenv('AI_GATEWAY_TIMEOUT', '30'))

# ---------------------------------------------------------------------------
# Gemini API - optional, graceful if missing
# ---------------------------------------------------------------------------
API_KEY = os.getenv('GOOGLE_API_KEY')
_genai = None

if API_KEY and API_KEY != '..':
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        _genai = genai
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.warning("Gemini API initialization failed: %s", e)
else:
    logger.info("Gemini API not configured")

if AI_GATEWAY_URL:
    logger.info("Photo descriptions: AI gateway at %s (model: %s)",
                AI_GATEWAY_URL, AI_GATEWAY_MODEL)
elif API_KEY and API_KEY != '..':
    logger.info("Photo descriptions: Google Gemini")
else:
    logger.info("Photo descriptions: built-in offline generator (no API key needed)")

# Import auth functions
from auth import (
    create_user, verify_user, login_required, role_required,
    establish_session, is_locked_out, record_failed_attempt,
    clear_failed_attempts, audit, get_csrf_token, csrf_token_is_valid,
    wants_json, UNSAFE_METHODS, LOCKOUT_SECONDS,
)

# ---------------------------------------------------------------------------
# ML Models - deferred loading (not at import time)
# ---------------------------------------------------------------------------
global_mobilenet = None
global_inception = None
global_vgg = None
stage_specific_models = {}
MODELS_LOADED = False


MODEL_FILES = (
    'mobilenet.keras', 'inception.keras', 'vgg16.keras',
    'Foundation_mobile.keras', 'Superstructure_mobile.keras',
    'Facade_inception.keras', 'Interior_mobile.keras', 'finishing_mobile.keras',
)


def missing_model_files():
    """Return the model weights that are not on disk.

    The weights ship as GitHub Release assets rather than in the repo, so
    a fresh clone legitimately starts without them. Naming the missing
    files beats a generic TensorFlow load error.
    """
    return [name for name in MODEL_FILES
            if not os.path.exists(os.path.join('models', name))]


def load_models():
    """Load all ML models. Called on first request, not at import time."""
    global global_mobilenet, global_inception, global_vgg
    global stage_specific_models, MODELS_LOADED

    if MODELS_LOADED:
        return

    absent = missing_model_files()
    if absent:
        logger.error(
            "Cannot load models - %d weight file(s) missing from models/: %s. "
            "Run 'python download_models.py' to fetch them.",
            len(absent), ', '.join(absent)
        )
        MODELS_LOADED = False
        return

    try:
        import tensorflow as tf

        global_mobilenet = tf.keras.models.load_model(
            "models/mobilenet.keras", compile=False
        )
        global_inception = tf.keras.models.load_model(
            "models/inception.keras", compile=False
        )
        global_vgg = tf.keras.models.load_model(
            "models/vgg16.keras", compile=False
        )

        stage_specific_models.update({
            "foundation": tf.keras.models.load_model(
                "models/Foundation_mobile.keras", compile=False
            ),
            "superstructure": tf.keras.models.load_model(
                "models/Superstructure_mobile.keras", compile=False
            ),
            "facade": tf.keras.models.load_model(
                "models/Facade_inception.keras", compile=False
            ),
            "Interior": tf.keras.models.load_model(
                "models/Interior_mobile.keras", compile=False
            ),
            "finishing works": tf.keras.models.load_model(
                "models/finishing_mobile.keras", compile=False
            ),
        })

        MODELS_LOADED = True
        logger.info("All ML models loaded successfully")
    except Exception as e:
        logger.error("Error loading models: %s", e)
        MODELS_LOADED = False


@app.before_request
def enforce_csrf():
    """Reject state-changing requests without a valid CSRF token.

    Runs before every view, so a route added later is protected by default
    rather than only when someone remembers to opt in.
    """
    if request.method not in UNSAFE_METHODS:
        return None
    if getattr(app.view_functions.get(request.endpoint), '_csrf_exempt', False):
        return None
    if csrf_token_is_valid():
        return None

    logger.warning("Rejected %s %s: missing or invalid CSRF token",
                   request.method, request.path)
    if wants_json():
        return jsonify({
            'success': False,
            'error': 'Your session expired. Reload the page and try again.'
        }), 400
    flash('Your session expired. Please try again.', 'error')
    return redirect(request.referrer or url_for('login'))


@app.after_request
def set_security_headers(response):
    """Baseline hardening headers.

    No Content-Security-Policy here: the templates rely on inline scripts
    and CDN-hosted Bootstrap/Leaflet, so a meaningful policy would need
    those refactored first. Claiming one that had to be loosened into
    uselessness would be worse than not setting it.
    """
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    if IS_PRODUCTION:
        response.headers.setdefault(
            'Strict-Transport-Security', 'max-age=31536000; includeSubDomains'
        )
    return response


@app.before_request
def ensure_models_loaded():
    """Load models on the first request that needs them.

    'compare_progress' is intentionally excluded - it only reads
    previously-stored validation results and never runs inference,
    so it shouldn't trigger a multi-second TensorFlow model load.
    """
    if not MODELS_LOADED and request.endpoint in (
        'validate_image', 'validate_project_images'
    ):
        load_models()


# ---------------------------------------------------------------------------
# Construction stage definitions & weights
# ---------------------------------------------------------------------------
stages = {
    "facade": [
        "Exterior_Cladding_and_Finishes",
        "Window_and_Door_Installation",
        "exterior_wall_construction",
    ],
    "finishing works": [
        "Painting",
        "fixture installation",
        "Millwork and carpentry"
    ],
    "foundation": [
        "Excavation",
        "Reinforcement Placement",
        "concrete curing",
        "concrete_pouring"
    ],
    "Interior": [
        "Ceiling Installation",
        "Flooring Installation",
        "Staircase Finishing"
    ],
    "superstructure": [
        "Roof_Decking",
        "Stair Case",
        "Structural_Frame_Erection_(framing)",
        "Structural_Wall_Construction",
    ],
}


stage_weights = {
    "foundation": 20,
    "superstructure": 30,
    "facade": 20,
    "Interior": 15,
    "finishing works": 15
}


sub_stage_weights = {
    "foundation": {
        "Excavation": 25,
        "Reinforcement Placement": 25,
        "concrete curing": 25,
        "concrete_pouring": 25
    },
    "superstructure": {
        "Roof_Decking": 15,
        "Stair Case": 20,
        "Structural_Frame_Erection_(framing)": 40,
        "Structural_Wall_Construction": 25
    },
    "facade": {
        "Exterior_Cladding_and_Finishes": 25,
        "Window_and_Door_Installation": 35,
        "exterior_wall_construction": 40
    },
    "Interior": {
        "Ceiling Installation": 35,
        "Flooring Installation": 35,
        "Staircase Finishing": 30
    },
    "finishing works": {
        "Painting": 35,
        "fixture installation": 35,
        "Millwork and carpentry": 30
    }
}


STAGE_ORDER = [
    "foundation", "superstructure", "facade", "Interior", "finishing works"
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_timezone():
    """Get the configured timezone object."""
    import pytz
    return pytz.timezone(APP_TIMEZONE)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _round1(value):
    """Round to 1 decimal place using round-half-up, to match how
    JavaScript's toFixed(1) rounds an exact .x5 tie - Python's own
    round()/"%.1f" use banker's rounding and can land on a different
    digit for the same number. See calculate_progress() for why this
    matters here specifically."""
    return math.floor(value * 10 + 0.5) / 10


def _normalize_image_url(path):
    """Convert a stored server-side image path into a static/ URL suffix."""
    if not path:
        return ''
    if 'uploads' in path:
        return f"uploads/{os.path.basename(path)}"
    if 'demo_samples' in path:
        return f"demo_samples/{os.path.basename(path)}"
    return path


def _decorate_progress(project, user_id):
    """Attach live progress fields derived from the project's most recent
    validation. Progress is always computed from validation history rather
    than stored denormalised, so it can never drift out of sync."""
    latest = database.get_latest_validation(user_id, project['id'])

    if latest:
        stage = latest['primary_stage']
        sub_stage = latest['specific_classification']
        stage_progress, overall, completed = calculate_progress(stage, sub_stage)
        project['current_stage'] = stage
        project['current_sub_stage'] = sub_stage
        project['stage_progress'] = stage_progress
        project['progress_percentage'] = overall
        project['completed_stages'] = completed
        project['last_updated'] = latest['timestamp']
    else:
        project['current_stage'] = project.get('current_stage') or 'Not started'
        project['current_sub_stage'] = project.get('current_sub_stage') or ''
        project['stage_progress'] = 0
        project['progress_percentage'] = 0
        project['completed_stages'] = []
        project['last_updated'] = None

    return project


def _save_upload(file_storage):
    """Persist an upload under a collision-proof name and return its path.

    secure_filename alone is not enough: two users uploading 'site.jpg'
    would overwrite each other's evidence photos, and a validation record
    written yesterday would silently start pointing at someone else's
    image. Prefixing a random token keeps every record's image immutable.
    """
    safe_name = secure_filename(file_storage.filename) or 'upload.jpg'
    unique_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file_storage.save(path)
    return path


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    # PORT is what most container platforms inject; HOST defaults to
    # loopback locally but must be 0.0.0.0 to be reachable in a container.
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0' if IS_PRODUCTION else '127.0.0.1')
    app.run(host=host, port=port, debug=debug_mode)
