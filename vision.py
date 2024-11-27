"""
CRISP - construction change detection between two site photos.

The previous implementation subtracted raw grayscale pixels and painted
every differing pixel green. On two drone shots taken on different days
that lights up the entire frame: sky, road, trees and all, because a
sunny-vs-overcast exposure shift changes almost every pixel. It also
never ran its "SegFormer" branch, because that branch imported
transformers' SegformerImageProcessor, which requires PyTorch.

What this module does instead:

  1. Aligns the two photos (ORB + RANSAC homography), so a drone that
     drifted between visits doesn't register as site-wide change.
  2. Normalises exposure between them, so weather/time-of-day doesn't.
  3. Compares *structure* (local SSIM) rather than raw intensity.
  4. Restricts results to construction regions - sky, vegetation and
     smooth ground are excluded, and the area must actually carry
     built structure in at least one of the two photos.
  5. Cleans up the mask morphologically and drops specks, so what's
     highlighted is contiguous, real, and outlined.

If models/segformer.onnx is present it is used for step 4 (a proper
semantic building mask); the preprocessing here is pure NumPy, so that
path needs onnxruntime only - no torch, no transformers.
"""

import os
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'segformer.onnx')

# ADE20K class indices that represent built structure. The previous code
# used id 2, which is 'sky' in ADE20K - 'building' is 1.
ADE20K_STRUCTURE_CLASSES = (
    0,   # wall
    1,   # building / edifice
    25,  # house
    48,  # skyscraper
    84,  # tower
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_WORK_WIDTH = 1400          # cap working resolution for predictable runtime
MIN_COMPONENT_AREA_RATIO = 5e-4  # drop change blobs smaller than 0.05% of frame

_onnx_session = None
_onnx_checked = False


# ---------------------------------------------------------------------------
# Optional ONNX SegFormer building segmentation (torch-free)
# ---------------------------------------------------------------------------


def _get_onnx_session():
    """Load models/segformer.onnx once, if it exists. Returns None when
    unavailable - callers fall back to the heuristic structure mask."""
    global _onnx_session, _onnx_checked

    if _onnx_checked:
        return _onnx_session
    _onnx_checked = True

    if not os.path.exists(MODEL_PATH):
        logger.info(
            "models/segformer.onnx not present - using heuristic construction "
            "masking for change detection."
        )
        return None

    try:
        import onnxruntime as ort
        _onnx_session = ort.InferenceSession(
            MODEL_PATH, providers=['CPUExecutionProvider']
        )
        logger.info("SegFormer ONNX session initialised.")
    except Exception as exc:
        logger.warning("Could not initialise SegFormer ONNX session: %s", exc)
        _onnx_session = None

    return _onnx_session


def _preprocess_for_segformer(bgr_image, size=640):
    """Resize + ImageNet-normalise into NCHW float32. Pure NumPy so this
    works with onnxruntime alone."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def _segformer_structure_mask(bgr_image):
    """Semantic building mask via ONNX, or None if unavailable/failed."""
    session = _get_onnx_session()
    if session is None:
        return None

    try:
        h, w = bgr_image.shape[:2]
        inputs = {session.get_inputs()[0].name: _preprocess_for_segformer(bgr_image)}
        logits = session.run(None, inputs)[0]
        classes = np.argmax(logits[0], axis=0).astype(np.int32)
        mask = np.isin(classes, ADE20K_STRUCTURE_CLASSES).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception as exc:
        logger.warning("SegFormer inference failed, falling back: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Heuristic region masks
# ---------------------------------------------------------------------------
