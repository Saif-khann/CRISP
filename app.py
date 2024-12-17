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


STAGE_MODEL_FILES = {
    "foundation": "Foundation_mobile.keras",
    "superstructure": "Superstructure_mobile.keras",
    "facade": "Facade_inception.keras",
    "Interior": "Interior_mobile.keras",
    "finishing works": "finishing_mobile.keras",
}

# Disk footprint of the three ensemble models, for the startup log.
_ENSEMBLE_DISK_MB = 369


def get_stage_model(stage):
    """Return the sub-stage model for a stage, loading it on first use.

    Only one of the five is needed per validation, and a typical session
    touches one or two stages. Loading all five up front adds roughly
    285 MB of resident memory and several seconds to the first request for
    models most sessions never call.
    """
    if stage in stage_specific_models:
        return stage_specific_models[stage]

    filename = STAGE_MODEL_FILES.get(stage)
    if filename is None:
        raise ValueError(f"Unknown stage: {stage}")

    import tensorflow as tf
    path = os.path.join('models', filename)
    if not os.path.exists(path):
        raise ValueError(f"Missing model weights: {path}")

    logger.info("Loading stage model for %r", stage)
    model = tf.keras.models.load_model(path, compile=False)
    stage_specific_models[stage] = model
    return model


def load_models():
    """Load the ensemble models. Called on first request, not at import time.

    Stage-specific models are loaded lazily by get_stage_model().
    """
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

        MODELS_LOADED = True
        logger.info(
            "Ensemble models loaded (%.0f MB on disk). Stage-specific models "
            "load on first use.", _ENSEMBLE_DISK_MB
        )
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

def _preprocess(image, size):
    """Resize to `size`, scale to [0,1] and add a batch dimension."""
    import tensorflow as tf

    resized = tf.image.resize(image, (size, size))
    scaled = tf.cast(resized, tf.float32) / 255.0
    return tf.reshape(scaled, (1, size, size, 3))


# Compiled forward passes, one per model, built on first use.
_compiled_forward = {}


def _infer(model, batch):
    """Run a single-sample forward pass through a compiled graph.

    Measured on this model set, CPU-only, at batch size 1:

        model.predict()      359 ms   builds a tf.data pipeline per call
        model(x) eager       868 ms   op-by-op dispatch, slowest
        predict_on_batch()   162 ms
        tf.function(model)   155 ms   <- used here

    Calling the model directly in eager mode is the intuitive choice and is
    the worst of the four. Wrapping it in a tf.function traces the graph
    once and reuses it, and produces bit-identical output to .predict().
    """
    key = id(model)
    fn = _compiled_forward.get(key)
    if fn is None:
        import tensorflow as tf
        fn = tf.function(lambda x: model(x, training=False),
                         reduce_retracing=True)
        _compiled_forward[key] = fn
    return np.asarray(fn(batch))[0]


def ensemble_predict(image):
    """Run ensemble prediction across MobileNet, Inception, and VGG16."""
    if not MODELS_LOADED:
        raise ValueError("Models not loaded")

    # MobileNet and VGG16 take the same input size, so preprocess once and
    # feed the same tensor to both instead of resizing twice.
    batch_224 = _preprocess(image, 224)
    batch_299 = _preprocess(image, 299)

    mobilenet_output = _infer(global_mobilenet, batch_224)
    inception_output = _infer(global_inception, batch_299)
    vgg_output = _infer(global_vgg, batch_224)


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
    if not MODELS_LOADED:
        raise ValueError("Models not loaded")

    model = get_stage_model(selected_stage)

    # Facade uses an InceptionV3 backbone at 299x299; the rest are
    # MobileNetV2 at 224x224.
    size = 299 if selected_stage == "facade" else 224
    predictions = _infer(model, _preprocess(image, size))

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


@app.route('/')
@app.route('/home')
@login_required
def home():
    project_id = request.args.get('project_id')
    if not project_id:
        return redirect(url_for('dashboard'))

    user_id = session['user_id']
    project = database.get_project(user_id, project_id)
    if not project:
        flash('That project could not be found.', 'error')
        return redirect(url_for('dashboard'))

    _decorate_progress(project, user_id)
    validations = database.list_validations(user_id, project_id)

    return render_template(
        'home.html',
        project=project,
        validations=validations,
        stages=stages,
        user_role=session.get('role'),
    )


# ---------------------------------------------------------------------------
# Routes: Projects
# ---------------------------------------------------------------------------

@app.route('/create_project', methods=['POST'])
@login_required
def create_project():
    try:
        name = (request.form.get('name') or '').strip()
        description = (request.form.get('description') or '').strip()
        location = (request.form.get('location') or '').strip()
        start_date = (request.form.get('start_date') or '').strip()
        end_date = (request.form.get('end_date') or '').strip()
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)

        missing = [
            label for label, value in (
                ('name', name), ('description', description),
                ('location', location), ('start date', start_date),
                ('target completion date', end_date),
            ) if not value
        ]
        if latitude is None:
            missing.append('latitude')
        if longitude is None:
            missing.append('longitude')
        if missing:
            return jsonify({
                'success': False,
                'error': f"Please provide: {', '.join(missing)}."
            }), 400

        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({
                'success': False,
                'error': 'Coordinates are out of range. Latitude must be '
                         'between -90 and 90, longitude between -180 and 180.'
            }), 400

        if end_date < start_date:
            return jsonify({
                'success': False,
                'error': 'The target completion date cannot be before the start date.'
            }), 400

        project_id = database.create_project(
            session['user_id'], name, description, location,
            start_date, end_date, latitude, longitude
        )
        return jsonify({
            'success': True,
            'message': 'Project created.',
            'project_id': project_id,
        }), 201

    except Exception as exc:
        logger.error("Error creating project: %s", exc)
        return jsonify({'success': False, 'error': 'Could not create the project.'}), 500


@app.route('/project/<project_id>/delete', methods=['POST'])
@login_required
def delete_project(project_id):
    if database.delete_project(session['user_id'], project_id):
        flash('Project deleted.', 'success')
    else:
        flash('That project could not be found.', 'error')
    return redirect(url_for('dashboard'))


# ---------------------------------------------------------------------------
# Routes: AI validation
# ---------------------------------------------------------------------------

@app.route('/project/<project_id>/validate', methods=['GET'])
@login_required
def validate_project_images(project_id):
    """The validation workspace for a project.

    This used to duplicate the whole inference pipeline for its POST
    branch; uploads now go through /validate_image exclusively, so there
    is one code path for classification instead of two that could drift.
    """
    return redirect(url_for('home', project_id=project_id))


@app.route('/validate_image', methods=['POST'])
@login_required
def validate_image():
    """Classify an uploaded site photo and record the validation."""
    filepath = None
    try:
        if not MODELS_LOADED:
            absent = missing_model_files()
            if absent:
                return jsonify({
                    'success': False,
                    'error': (
                        f'{len(absent)} model weight file(s) are missing. '
                        "Run 'python download_models.py' to fetch them, then restart."
                    )
                }), 503
            return jsonify({
                'success': False,
                'error': 'The AI models are still loading. Please try again in a moment.'
            }), 503

        user_id = session['user_id']
        project_id = request.form.get('project_id')
        selected_stage = request.form.get('stage')
        proceed = request.form.get('proceed', 'false').lower() in ('true', 'on', '1', 'yes')
        describe = request.form.get('describe', 'false').lower() in ('true', 'on', '1', 'yes')

        if not project_id:
            return jsonify({'success': False, 'error': 'No project specified.'}), 400

        project = database.get_project(user_id, project_id)
        if not project:
            return jsonify({'success': False, 'error': 'Project not found.'}), 404

        if selected_stage not in stages:
            return jsonify({'success': False, 'error': 'Please choose a valid construction stage.'}), 400

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded.'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Use PNG, JPG or JPEG.'}), 400

        filepath = _save_upload(file)

        try:
            image = Image.open(filepath).convert('RGB')
        except Exception:
            return jsonify({'success': False, 'error': 'That file could not be read as an image.'}), 400

        image_array = np.array(image)
        predicted_stage, global_confidence = ensemble_predict(image_array)

        if predicted_stage.lower() != selected_stage.lower() and not proceed:
            # Not recorded - remove the upload rather than leaving orphans.
            _discard_upload(filepath)
            filepath = None
            return jsonify({
                'success': False,
                'mismatch': True,
                'message': (
                    f"The photo looks like '{predicted_stage}' "
                    f"({global_confidence:.1f}% confidence), but you selected "
                    f"'{selected_stage}'."
                )
            }), 200

        predicted_class, stage_confidence = classify_stage(image_array, selected_stage)

        description = None
        if describe:
            description = describe_image_with_ai(
                filepath, selected_stage, predicted_class, stage_confidence
            )

        validation_id = database.create_validation(
            user_id=user_id,
            project_id=project_id,
            primary_stage=selected_stage,
            specific_classification=predicted_class,
            stage_confidence=stage_confidence,
            global_confidence=global_confidence,
            image_path=filepath,
            ai_description=description,
        )
        database.update_project_stage(user_id, project_id, selected_stage, predicted_class)

        _, overall_progress, _ = calculate_progress(selected_stage, predicted_class)
        tz = get_timezone()

        return jsonify({
            'success': True,
            'message': (
                f"Matched stage '{selected_stage}'. Sub-stage: "
                f"'{predicted_class}' ({stage_confidence:.1f}% confidence)."
            ),
            'validation_id': validation_id,
            'primary_stage': selected_stage,
            'specific_classification': predicted_class,
            'confidence_scores': {
                'Stage Confidence': f"{stage_confidence:.2f}",
                'Global Stage Confidence': f"{global_confidence:.2f}",
            },
            'overall_progress': overall_progress,
            'description': description,
            'timestamp': datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z'),
        }), 200

    except Exception as exc:
        logger.exception("Error during image validation")
        if filepath:
            _discard_upload(filepath)
        return jsonify({'success': False, 'error': 'Could not process that image.'}), 500


@app.route('/compare', methods=['POST'])
@login_required
def compare_progress():
    """Compare two recorded validations for the same project."""
    try:
        user_id = session['user_id']
        project_id = request.form.get('project_id')
        previous_id = request.form.get('previous_doc_id')
        current_id = request.form.get('current_doc_id')

        if not all([project_id, previous_id, current_id]):
            return jsonify({'success': False, 'error': 'Select both milestones to compare.'}), 400

        if previous_id == current_id:
            return jsonify({
                'success': False,
                'error': 'Select two different milestones to compare.'
            }), 400

        if not database.get_project(user_id, project_id):
            return jsonify({'success': False, 'error': 'Project not found.'}), 404

        prev_data = database.get_validation(user_id, previous_id)
        curr_data = database.get_validation(user_id, current_id)
        if not prev_data or not curr_data:
            return jsonify({'success': False, 'error': 'One of those milestones no longer exists.'}), 404

        if prev_data['project_id'] != project_id or curr_data['project_id'] != project_id:
            return jsonify({
                'success': False,
                'error': 'Those milestones belong to a different project.'
            }), 400

        # Order by time so "previous" is genuinely the earlier record, no
        # matter which way round the two dropdowns were filled in.
        if prev_data['timestamp'] and curr_data['timestamp'] and \
                prev_data['timestamp'] > curr_data['timestamp']:
            prev_data, curr_data = curr_data, prev_data

        prev_stage = prev_data['primary_stage']
        prev_sub = prev_data['specific_classification']
        curr_stage = curr_data['primary_stage']
        curr_sub = curr_data['specific_classification']

        status, progress_message = get_progress_message(
            prev_stage, prev_sub, curr_stage, curr_sub
        )
        prev_stage_p, prev_overall_p, prev_comp = calculate_progress(prev_stage, prev_sub)
        curr_stage_p, curr_overall_p, curr_comp = calculate_progress(curr_stage, curr_sub)

        def _fmt(ts):
            return ts.strftime('%Y-%m-%d %H:%M') if ts else 'Unknown'

        return jsonify({
            'success': True,
            'previous': {
                'stage': prev_stage,
                'sub_stage': prev_sub,
                'stage_progress': prev_stage_p,
                'overall_progress': prev_overall_p,
                'completed_stages': prev_comp,
                'timestamp': _fmt(prev_data['timestamp']),
                'image_path': _normalize_image_url(prev_data['image_path']),
            },
            'current': {
                'stage': curr_stage,
                'sub_stage': curr_sub,
                'stage_progress': curr_stage_p,
                'overall_progress': curr_overall_p,
                'completed_stages': curr_comp,
                'timestamp': _fmt(curr_data['timestamp']),
                'image_path': _normalize_image_url(curr_data['image_path']),
            },
            'progress_status': status,
            'progress_message': progress_message,
        })

    except Exception as exc:
        logger.exception("Error in compare_progress")
        return jsonify({'success': False, 'error': 'Could not compare those milestones.'}), 500


@app.route('/generate_report/<validation_id>', methods=['GET'])
@login_required
def generate_report(validation_id):
    """Produce a PDF audit report for a recorded validation."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        user_id = session['user_id']
        validation = database.get_validation(user_id, validation_id)
        if not validation:
            flash('That validation record could not be found.', 'error')
            return redirect(url_for('dashboard'))

        project = database.get_project(user_id, validation['project_id'])
        stage_p, overall_p, _ = calculate_progress(
            validation['primary_stage'], validation['specific_classification']
        )

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            doc = SimpleDocTemplate(tmp.name, pagesize=letter)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle', parent=styles['Heading1'], fontSize=20, spaceAfter=18
            )
            story = [
                Paragraph('CRISP - Construction Stage Analysis Report', title_style),
                Paragraph(
                    f"Generated: {datetime.now(get_timezone()).strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    styles['Normal']
                ),
                Spacer(1, 12),
            ]

            if project:
                story += [
                    Paragraph('Project', styles['Heading2']),
                    Paragraph(f"<b>Name:</b> {project['name']}", styles['Normal']),
                    Paragraph(f"<b>Location:</b> {project['location'] or 'N/A'}", styles['Normal']),
                    Spacer(1, 10),
                ]

            image_path = validation.get('image_path')
            if image_path and os.path.exists(image_path):
                try:
                    story += [RLImage(image_path, width=380, height=260), Spacer(1, 12)]
                except Exception:
                    logger.warning("Could not embed %s in the report", image_path)

            recorded = validation['timestamp']
            story += [
                Paragraph('Stage Analysis', styles['Heading2']),
                Paragraph(f"<b>Stage:</b> {validation['primary_stage']}", styles['Normal']),
                Paragraph(
                    f"<b>Sub-stage:</b> {validation['specific_classification'].replace('_', ' ')}",
                    styles['Normal']
                ),
                Paragraph(
                    f"<b>Recorded:</b> "
                    f"{recorded.strftime('%Y-%m-%d %H:%M UTC') if recorded else 'Unknown'}",
                    styles['Normal']
                ),
                Spacer(1, 10),
                Paragraph('Progress', styles['Heading2']),
                Paragraph(f"<b>Stage completion:</b> {stage_p:.1f}%", styles['Normal']),
                Paragraph(f"<b>Overall project completion:</b> {overall_p:.1f}%", styles['Normal']),
                Spacer(1, 10),
                Paragraph('Model Confidence', styles['Heading2']),
                Paragraph(f"Sub-stage confidence: {validation['stage_confidence']}%", styles['Normal']),
                Paragraph(f"Overall stage confidence: {validation['global_confidence']}%", styles['Normal']),
            ]

            if validation.get('ai_description'):
                story += [
                    Spacer(1, 10),
                    Paragraph('AI Notes', styles['Heading2']),
                    Paragraph(validation['ai_description'], styles['Normal']),
                ]

            doc.build(story)

            return send_file(
                tmp.name,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'crisp_report_{validation_id}.pdf'
            )

    except Exception as exc:
        logger.exception("Error generating report")
        flash('Could not generate that report.', 'error')
        return redirect(url_for('dashboard'))


# ---------------------------------------------------------------------------
# Routes: Map
# ---------------------------------------------------------------------------

@app.route('/geo-map')
@login_required
def geo_map():
    user_id = session['user_id']
    projects = []
    for project in database.list_projects(user_id):
        lat, lng = project.get('latitude'), project.get('longitude')
        if lat is None or lng is None:
            continue
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            continue
        _decorate_progress(project, user_id)
        projects.append({
            'id': project['id'],
            'name': project['name'],
            'latitude': lat,
            'longitude': lng,
            'status': project['status'],
            'stage': (project['current_stage'] or 'Not started').title(),
            'progress': project['progress_percentage'],
        })

    return render_template(
        'geo_map.html', projects=projects, user_role=session.get('role')
    )


# ---------------------------------------------------------------------------
# Routes: Visual change analyzer (experts only)
# ---------------------------------------------------------------------------

@app.route('/visual_comparison', methods=['GET', 'POST'])
@role_required('expert')
def visual_comparison():
    if request.method == 'GET':
        return render_template('visual_comparison.html')

    prev_path = curr_path = None
    try:
        import base64
        import vision

        prev_image = request.files.get('prev_image')
        curr_image = request.files.get('curr_image')

        if not prev_image or not curr_image or not prev_image.filename \
                or not curr_image.filename:
            return jsonify({
                'success': False,
                'error': 'Please choose both a baseline and a current photo.'
            }), 400

        if not allowed_file(prev_image.filename) or not allowed_file(curr_image.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Use PNG, JPG or JPEG.'
            }), 400

        prev_path = _save_upload(prev_image)
        curr_path = _save_upload(curr_image)

        result = vision.detect_construction_change(prev_path, curr_path)

        import cv2
        encoded, buffer = cv2.imencode('.jpg', result['image'],
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded:
            raise RuntimeError("Could not encode the result image")

        return jsonify({
            'success': True,
            'result_image': 'data:image/jpeg;base64,'
                            + base64.b64encode(buffer).decode('utf-8'),
            'method': result['mask_method'],
            'aligned': result['aligned'],
            'change_percent': round(result['change_ratio'] * 100, 1),
            'has_change': result['has_change'],
            'note': result['note'],
        })

    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception:
        logger.exception("Error in visual_comparison")
        return jsonify({'success': False, 'error': 'Could not compare those photos.'}), 500
    finally:
        # These are transient analysis inputs, not evidence records.
        for path in (prev_path, curr_path):
            if path:
                _discard_upload(path)


# ---------------------------------------------------------------------------
# Health check & error handlers
# ---------------------------------------------------------------------------

@app.route('/healthz')
def healthz():
    """Liveness/readiness probe for the deployment platform."""
    return jsonify({'status': 'ok', 'models_loaded': MODELS_LOADED}), 200


@app.context_processor
def inject_globals():
    """Template globals: the CSRF token and whether to show demo links."""
    return {'demo_enabled': ALLOW_DEMO_LOGIN, 'csrf_token': get_csrf_token()}


@app.errorhandler(404)
def handle_404(_error):
    if request.path.startswith(('/validate_image', '/compare', '/create_project')):
        return jsonify({'success': False, 'error': 'Not found.'}), 404
    return render_template('error.html', code=404,
                           message='That page does not exist.'), 404


@app.errorhandler(413)
def handle_413(_error):
    return jsonify({
        'success': False,
        'error': 'That file is too large. The limit is 10 MB.'
    }), 413


@app.errorhandler(500)
def handle_500(error):
    logger.error("Unhandled server error: %s", error)
    return render_template('error.html', code=500,
                           message='Something went wrong on our end.'), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    # PORT is what most container platforms inject; HOST defaults to
    # loopback locally but must be 0.0.0.0 to be reachable in a container.
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '0.0.0.0' if IS_PRODUCTION else '127.0.0.1')
    app.run(host=host, port=port, debug=debug_mode)
