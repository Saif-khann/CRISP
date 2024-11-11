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


def _discard_upload(path):
    """Delete an upload that isn't going to be referenced by any record."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not remove temporary upload %s: %s", path, exc)


# ---------------------------------------------------------------------------
# ML prediction functions
# ---------------------------------------------------------------------------


def ensemble_predict(image):
    """Run ensemble prediction across MobileNet, Inception, and VGG16."""
    import tensorflow as tf

    if not MODELS_LOADED:
        raise ValueError("Models not loaded")

    inception_preprocessed = tf.image.resize(image, (299, 299))
    inception_preprocessed = tf.cast(inception_preprocessed, tf.float32) / 255.0
    inception_preprocessed = tf.reshape(inception_preprocessed, (1, 299, 299, 3))

    mobilenet_preprocessed = tf.image.resize(image, (224, 224))
    mobilenet_preprocessed = tf.cast(mobilenet_preprocessed, tf.float32) / 255.0
    mobilenet_preprocessed = tf.reshape(mobilenet_preprocessed, (1, 224, 224, 3))

    vgg_preprocessed = tf.image.resize(image, (224, 224))
    vgg_preprocessed = tf.cast(vgg_preprocessed, tf.float32) / 255.0
    vgg_preprocessed = tf.reshape(vgg_preprocessed, (1, 224, 224, 3))

    mobilenet_output = global_mobilenet.predict(mobilenet_preprocessed, verbose=0)[0]
    inception_output = global_inception.predict(inception_preprocessed, verbose=0)[0]
    vgg_output = global_vgg.predict(vgg_preprocessed, verbose=0)[0]

    ensemble_output = (
        0.3 * mobilenet_output +
        0.4 * inception_output +
        0.3 * vgg_output
    )

    predicted_stage_index = np.argmax(ensemble_output)
    confidence_score = float(ensemble_output[predicted_stage_index] * 100)

    stage_list = list(stages.keys())
    predicted_stage = stage_list[predicted_stage_index]

    return predicted_stage, confidence_score


def classify_stage(image, selected_stage):
    """Classify the sub-stage within a given stage."""
    import tensorflow as tf

    if not MODELS_LOADED or selected_stage not in stage_specific_models:
        raise ValueError("Models not loaded or invalid stage")

    if selected_stage == "facade":
        preprocessed = tf.image.resize(image, (299, 299))
    else:
        preprocessed = tf.image.resize(image, (224, 224))

    preprocessed = tf.cast(preprocessed, tf.float32) / 255.0
    preprocessed = tf.expand_dims(preprocessed, axis=0)

    model = stage_specific_models[selected_stage]
    predictions = model.predict(preprocessed, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index] * 100)

    sub_stages = stages[selected_stage]
    predicted_sub_stage = sub_stages[predicted_index]

    return predicted_sub_stage, confidence


def _predict_and_classify(image_array, selected_stage):
    """Run the ensemble global-stage predictor and the stage-specific
    sub-stage classifier on one preprocessed image array.

    Shared by validate_image and validate_project_images (demo
    direct-upload path) - both ran this exact pair of calls
    independently before.
    """
    predicted_stage, global_confidence = ensemble_predict(image_array)
    predicted_sub_stage, sub_stage_confidence = classify_stage(image_array, selected_stage)
    return predicted_stage, global_confidence, predicted_sub_stage, sub_stage_confidence


# ---------------------------------------------------------------------------
# AI image description: OpenAI-compatible gateway -> Gemini -> offline fallback
# ---------------------------------------------------------------------------


def _generate_local_description(stage, sub_stage, confidence):
    """
    100% Free, offline local intelligent construction narrative.
    Generates domain-accurate engineering notes with zero API calls.
    """
    sub_clean = sub_stage.replace('_', ' ').title() if sub_stage else 'General Stage Work'
    stage_clean = stage.title() if stage else 'Active Construction'
    
    narratives = {
        'foundation': {
            'Excavation': f"Site visual analysis shows active earthmoving and excavation operations. Ground preparation is underway with trenching to target structural founding depth (AI Confidence: {confidence:.1f}%).",
            'Reinforcement Placement': f"Rebar cage fabrication and footing reinforcement placement are actively positioned in excavated footing zones prior to pour scheduling (AI Confidence: {confidence:.1f}%).",
            'concrete_pouring': f"Active concrete pouring and placement operations observed with pumping equipment positioned on site (AI Confidence: {confidence:.1f}%).",
            'concrete curing': f"Foundation slab/footing curing phase in progress with moisture retention measures visible across completed structural foundation elements (AI Confidence: {confidence:.1f}%)."
        },
        'superstructure': {
            'Structural_Frame_Erection_(framing)': f"Primary structural frame erection underway. Vertical load-bearing columns, steel beams, and structural framing elements visible (AI Confidence: {confidence:.1f}%).",
            'Structural_Wall_Construction': f"Load-bearing wall construction and shear wall reinforcement formwork active across the structural floor plate (AI Confidence: {confidence:.1f}%).",
            'Stair Case': f"Cast-in-place staircase formwork and vertical transit core structural elements under construction (AI Confidence: {confidence:.1f}%).",
            'Roof_Decking': f"Upper roof decking, slab formwork, and top structural diaphragm assembly in progress (AI Confidence: {confidence:.1f}%)."
        },
        'facade': {
            'exterior_wall_construction': f"Exterior perimeter wall assembly, masonry infill, and envelope weatherproofing layers being erected (AI Confidence: {confidence:.1f}%).",
            'Window_and_Door_Installation': f"Fenestration package active: Exterior window frame anchoring and glazing installation visible across building facade (AI Confidence: {confidence:.1f}%).",
            'Exterior_Cladding_and_Finishes': f"Architectural cladding panels, insulation backing, and external facade finishing materials in progress (AI Confidence: {confidence:.1f}%)."
        },
        'Interior': {
            'Ceiling Installation': f"Interior MEP overhead ducting and drop ceiling suspension framing active across internal zones (AI Confidence: {confidence:.1f}%).",
            'Flooring Installation': f"Interior floor substrate preparation, screeding, and architectural flooring installation in progress (AI Confidence: {confidence:.1f}%).",
            'Staircase Finishing': f"Internal staircase architectural finishes, tread cladding, and safety balustrade installation observed (AI Confidence: {confidence:.1f}%)."
        },
        'finishing works': {
            'Painting': f"Internal/external primer and architectural surface painting coats actively applied across finished partitions (AI Confidence: {confidence:.1f}%).",
            'fixture installation': f"Final trade electrical fixtures, plumbing trim, and lighting device installations in progress (AI Confidence: {confidence:.1f}%).",
            'Millwork and carpentry': f"Final joinery, interior door hanging, trim, and architectural millwork packages nearing completion (AI Confidence: {confidence:.1f}%)."
        }
    }

    stage_dict = narratives.get(stage.lower() if stage else '', {})
    if sub_stage in stage_dict:
        return stage_dict[sub_stage]
    
    return f"CRISP Intelligent Analysis: Active {stage_clean} phase detected, specifically {sub_clean} with a model confidence of {confidence:.1f}%. Site progress aligns with scheduled stage milestones."


def describe_image_with_ai(image_path, stage="foundation", sub_stage="Excavation", confidence=95.0):
    """
    Multi-tier description generator:
    1. OpenAI-compatible gateway (if AI_GATEWAY_URL is set)
    2. Google Gemini API (if GOOGLE_API_KEY set)
    3. 100% Free Offline Local Narrative Generator (Default)
    """
    # Tier 1: any OpenAI-compatible gateway
    if AI_GATEWAY_URL:
        try:
            import base64
            import mimetypes

            with open(image_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')

            # Send the real content type. Labelling a PNG as JPEG makes some
            # vision endpoints reject the request outright.
            mime = mimetypes.guess_type(image_path)[0] or 'image/jpeg'

            headers = {
                'Authorization': f'Bearer {AI_GATEWAY_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': AI_GATEWAY_MODEL,
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': f'Describe this construction site photo briefly (1-2 sentences), focusing on the visible stage ({stage} - {sub_stage}).'
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:{mime};base64,{img_b64}'
                                }
                            }
                        ]
                    }
                ],
                'max_tokens': 150
            }
            # Vision inference is slower than text completion, and a local
            # gateway may be loading a model on first use; 8s was too tight.
            resp = requests.post(
                f"{AI_GATEWAY_URL.rstrip('/')}/chat/completions",
                headers=headers, json=payload, timeout=AI_GATEWAY_TIMEOUT
            )
            if resp.ok:
                content = resp.json()['choices'][0]['message']['content']
                if content:
                    logger.info("Photo description generated via AI gateway")
                    return content.strip()
            logger.warning(
                "AI gateway returned HTTP %s - falling back. Body: %s",
                resp.status_code, resp.text[:200]
            )
        except requests.Timeout:
            logger.warning(
                "AI gateway timed out after %ss - falling back", AI_GATEWAY_TIMEOUT
            )
        except Exception as e:
            logger.warning("AI gateway call failed: %s - falling back", e)

    # Tier 2: Google Gemini API (if configured)
    if _genai and API_KEY:
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            model = _genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=[
                    f"Describe this construction site image briefly (1-2 sentences), focusing on the visible stage ({stage} - {sub_stage}).",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ],
                generation_config={"temperature": 0.3, "max_output_tokens": 150}
            )
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning("Gemini API call failed: %s - using local descriptor", e)

    # Tier 3: 100% Free Offline Local Narrative Generator
    return _generate_local_description(stage, sub_stage, confidence)


# ---------------------------------------------------------------------------
# Progress calculation functions
# ---------------------------------------------------------------------------


def calculate_progress(stage, sub_stage):
    """Calculate stage-level and overall project progress."""
    if stage not in stages or sub_stage not in stages[stage]:
        return 0, 0, []

    stage_sub_stages = stages[stage]
    current_index = stage_sub_stages.index(sub_stage)

    stage_progress = 0
    for i in range(current_index + 1):
        sstg = stage_sub_stages[i]
        stage_progress += sub_stage_weights[stage][sstg]

    current_stage_index = STAGE_ORDER.index(stage) + 1

    overall_progress = 0
    completed_stages = []

    for s in STAGE_ORDER:
        if STAGE_ORDER.index(s) + 1 < current_stage_index:
            overall_progress += stage_weights[s]
            completed_stages.append(s)

    stage_contribution = (stage_weights[stage] * (stage_progress / 100.0))
    overall_progress += stage_contribution

    if stage_progress == 100:
        completed_stages.append(stage)

    completed_stages.sort(key=lambda x: STAGE_ORDER.index(x) + 1)

    # Round once, here, at the source. overall_progress is a weighted sum
    # of integer percentages (e.g. 15 * 0.35 = 5.25) and lands on an exact
    # .x5 tie constantly. Left unrounded, Python's "%.1f" (banker's
    # rounding) and JS's toFixed(1) (round-half-away-from-zero) disagree
    # on which way to round that tie - e.g. one showing 90.2%, the other
    # 90.3% for the same underlying number. Rounding once here means
    # every caller (the JSON payload, the text summary, the dashboards)
    # displays the same already-decided digit instead of re-rounding the
    # raw float differently.
    overall_progress = _round1(overall_progress)

    return stage_progress, overall_progress, completed_stages


def get_progress_message(prev_stage, prev_sub_stage, curr_stage, curr_sub_stage):
    """Generate a human-readable progress comparison message."""
    prev_stage_progress, prev_overall_progress, prev_completed = \
        calculate_progress(prev_stage, prev_sub_stage)
    curr_stage_progress, curr_overall_progress, curr_completed = \
        calculate_progress(curr_stage, curr_sub_stage)

    prev_stage_idx = STAGE_ORDER.index(prev_stage) + 1
    curr_stage_idx = STAGE_ORDER.index(curr_stage) + 1

    overall_progress_diff = curr_overall_progress - prev_overall_progress

    if curr_stage_idx < prev_stage_idx:
        status = 'invalid'
        message = [
            f"Invalid progress: Cannot move from {prev_stage} "
            f"(Stage {prev_stage_idx}) to {curr_stage} (Stage {curr_stage_idx})",
            "Construction stages must proceed in order."
        ]
        return status, "\n".join(message)

    if curr_overall_progress > prev_overall_progress:
        status = 'advanced'
        message = [
            f"Progress has advanced from Stage {prev_stage_idx}: "
            f"{prev_stage} ({prev_sub_stage}) to Stage {curr_stage_idx}: "
            f"{curr_stage} ({curr_sub_stage})"
        ]
        newly_completed = set(curr_completed) - set(prev_completed)
        if newly_completed:
            completed_info = [
                f"Stage {STAGE_ORDER.index(s)+1}: {s}"
                for s in sorted(newly_completed, key=lambda x: STAGE_ORDER.index(x))
            ]
            message.append(f"Completed stages: {', '.join(completed_info)} (100%)")

        message.append(
            f"Current Stage {curr_stage_idx} ({curr_stage}) is "
            f"{curr_stage_progress:.1f}% complete "
            f"(contributing {((stage_weights[curr_stage] * curr_stage_progress) / 100.0):.1f}% "
            f"to overall progress)"
        )

        message.append("Project Progress Breakdown:")
        for completed_stage in curr_completed:
            if completed_stage == curr_stage and curr_stage_progress < 100:
                partial = (stage_weights[curr_stage] * (curr_stage_progress / 100.0))
                message.append(
                    f"- Stage {curr_stage_idx}: {curr_stage}: "
                    f"{curr_stage_progress:.1f}% "
                    f"(contributing {partial:.1f}% out of possible "
                    f"{stage_weights[curr_stage]}%)"
                )
            else:
                idx = STAGE_ORDER.index(completed_stage) + 1
                message.append(
                    f"- Stage {idx}: {completed_stage}: 100% "
                    f"(contributing {stage_weights[completed_stage]}%)"
                )

        message.append(
            f"Overall project is {curr_overall_progress:.1f}% complete "
            f"(+{overall_progress_diff:.1f}%)"
        )

    elif curr_overall_progress == prev_overall_progress:
        status = 'same'
        message = [
            f"No progress detected. Staying at Stage {curr_stage_idx}: "
            f"{curr_stage} ({curr_sub_stage})",
            f"Current stage is {curr_stage_progress:.1f}% complete",
            f"Overall project is {curr_overall_progress:.1f}% complete"
        ]
    else:
        status = 'regressed'
        message = [
            f"Progress has regressed from Stage {prev_stage_idx}: "
            f"{prev_stage} ({prev_sub_stage}) to Stage {curr_stage_idx}: "
            f"{curr_stage} ({curr_sub_stage})",
            f"Current stage ({curr_stage}) is {curr_stage_progress:.1f}% complete",
            f"Overall project is {curr_overall_progress:.1f}% complete "
            f"({overall_progress_diff:.1f}% change)"
        ]

    return status, "\n".join(message)

# ---------------------------------------------------------------------------
# Routes: Authentication
# ---------------------------------------------------------------------------


@app.route('/demo-login')
def demo_login():
    """Sign in to a real, pre-seeded demo account.

    This is a genuine login against a real hashed-password account in the
    database - not an auth bypass. The demo accounts own their own seeded
    projects, so exploring the demo can never touch another user's data.
    Disable entirely with ALLOW_DEMO_LOGIN=false.
    """
    if os.getenv('ALLOW_DEMO_LOGIN', 'true').lower() != 'true':
        flash('Demo access is disabled on this deployment.', 'error')
        return redirect(url_for('login'))

    role = request.args.get('role', 'expert')
    if role not in ('expert', 'worker'):
        role = 'expert'

    user = database.get_user_by_email(DEMO_ACCOUNTS[role]['email'])
    if not user:
        flash('Demo account is unavailable. Please sign up instead.', 'error')
        return redirect(url_for('login'))

    establish_session(user)
    flash(f'Signed in to the {role} demo account.', 'success')
    return redirect(url_for(f'{role}_dashboard'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Someone already signed in has no business on the sign-in form.
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please enter both your email and password.', 'error')
            return render_template('login.html'), 400

        if is_locked_out(email):
            audit('login_blocked', email=email, detail='rate limited')
            flash(
                'Too many failed sign-in attempts. Please wait a few minutes '
                'and try again.', 'error'
            )
            return render_template('login.html'), 429

        user = verify_user(email, password)
        if not user:
            attempts = record_failed_attempt(email)
            audit('login_failed', email=email, detail=f'attempt {attempts}')
            # Deliberately vague: naming which half was wrong would let an
            # attacker enumerate which emails have accounts.
            flash('Incorrect email or password.', 'error')
            return render_template('login.html'), 401

        clear_failed_attempts(email)
        establish_session(user)
        audit('login_success', email=user['email'], user_id=user['id'])
        logger.info("User signed in: %s (%s)", user['email'], user['role'])
        return redirect(url_for(f"{user['role']}_dashboard"))

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', '')

        try:
            user_id = create_user(email, password, role)
        except ValueError as exc:
            flash(str(exc), 'error')
            return render_template('signup.html'), 400
        except Exception as exc:
            logger.error("Signup failed for %s: %s", email, exc)
            flash('Could not create the account. Please try again.', 'error')
            return render_template('signup.html'), 500

        user = database.get_user(user_id)
        establish_session(user)
        audit('signup', email=email, user_id=user_id, detail=f'role {role}')
        logger.info("Account created: %s (%s)", email, role)
        flash('Welcome to CRISP! Your account is ready.', 'success')
        return redirect(url_for(f"{role}_dashboard"))

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    audit('logout', email=session.get('email'), user_id=session.get('user_id'))
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('login'))


@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    """Account settings: change password, review recent sign-in activity."""
    user_id = session['user_id']

    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if new_password != confirm_password:
            flash('The new passwords do not match.', 'error')
        else:
            try:
                database.change_password(user_id, current_password, new_password)
            except ValueError as exc:
                audit('password_change_failed', email=session.get('email'),
                      user_id=user_id, detail=str(exc))
                flash(str(exc), 'error')
            else:
                audit('password_changed', email=session.get('email'),
                      user_id=user_id)
                flash('Your password has been updated.', 'success')
                return redirect(url_for('account'))

    return render_template(
        'account.html',
        events=database.list_auth_events(limit=15, email=session.get('email')),
    )


# ---------------------------------------------------------------------------
# Routes: Dashboards
# ---------------------------------------------------------------------------


@app.route('/dashboard')
@login_required
def dashboard():
    return redirect(
        url_for('expert_dashboard' if session.get('role') == 'expert'
                else 'worker_dashboard')
    )


@app.route('/expert_dashboard')
@role_required('expert')
def expert_dashboard():
    user_id = session['user_id']
    projects = [_decorate_progress(p, user_id) for p in database.list_projects(user_id)]
    return render_template(
        'expert_dashboard.html',
        projects=projects,
        user_id=user_id,
        user_role='expert',
        total_validations=database.count_validations(user_id),
    )


@app.route('/worker_dashboard')
@role_required('worker')
def worker_dashboard():
    user_id = session['user_id']
    projects = [_decorate_progress(p, user_id) for p in database.list_projects(user_id)]
    return render_template(
        'worker_dashboard.html',
        projects=projects,
        user_id=user_id,
        user_role='worker',
        total_validations=database.count_validations(user_id),
    )


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    # PORT is what most container platforms inject; HOST defaults to
    # loopback locally but must be 0.0.0.0 to be reachable in a container.
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0' if IS_PRODUCTION else '127.0.0.1')
    app.run(host=host, port=port, debug=debug_mode)
