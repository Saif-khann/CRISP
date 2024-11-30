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


def _sky_mask(bgr_image, structure_density=None):
    """Sky = bright/blue AND featureless AND anchored to the top of frame.

    Colour alone is not enough. Pale concrete and white cladding are also
    bright and desaturated, so a brightness-only rule swallows the very
    buildings we care about - and a morphological close can bridge a pale
    facade up to the sky, dragging the whole structure into the mask.

    Sky is separated here by being *textureless*: buildings carry edges,
    openings and panel lines even when flat-toned, while sky does not. A
    vertical prior and a top-anchoring requirement complete the split.
    """
    img_h, img_w = bgr_image.shape[:2]
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    overcast = (v > 150) & (s < 45)                       # white/grey sky
    blue = (h >= 95) & (h <= 135) & (s > 35) & (v > 110)  # clear blue sky

    if structure_density is None:
        structure_density = _structure_density(bgr_image)
    featureless = structure_density < 0.06

    candidate = ((overcast | blue) & featureless)

    # Sky does not occupy the bottom of an aerial construction shot.
    vertical_cutoff = int(img_h * 0.65)
    candidate[vertical_cutoff:, :] = False
    candidate = candidate.astype(np.uint8) * 255

    # Gentle cleanup only - a large kernel here is what bridges facades
    # into the sky region.
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate)
    if num <= 1:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    band = max(1, img_h // 25)
    touching_top = set(np.unique(labels[0:band, :]))
    touching_top.discard(0)

    keep = [
        label for label in touching_top
        # Centroid must sit in the upper third: a component that merely
        # grazes the top edge but extends far down is a building, not sky.
        if centroids[label][1] < img_h * 0.33
        and stats[label, cv2.CC_STAT_AREA] > (img_h * img_w) * 0.002
    ]
    if not keep:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    mask = (np.isin(labels, keep).astype(np.uint8)) * 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)


def _vegetation_mask(bgr_image):
    """Green vegetation - trees, grass verges, landscaping."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    veg = ((h >= 32) & (h <= 92) & (s > 45) & (v > 25)).astype(np.uint8) * 255
    return cv2.morphologyEx(veg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _fill_holes(mask):
    """Fill regions fully enclosed by the mask.

    A large flat wall or slab only produces edges at its boundary, so an
    edge-density mask outlines it but leaves the middle empty. Filling
    enclosed holes recovers the whole structure instead of a ring around
    it - without this, change inside a big new facade goes unreported.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _structure_density(bgr_image):
    """Local edge density - high on framing, scaffolding, formwork and
    facades; low on road surface, water and open ground."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    density = cv2.boxFilter(
        edges.astype(np.float32), -1, (51, 51), normalize=True
    )
    peak = float(density.max())
    if peak <= 0:
        return np.zeros_like(density)
    return density / peak


def _construction_mask(img_a, img_b):
    """Where change is allowed to count: not sky, not vegetation, and
    carrying built structure in at least one of the two photos."""
    seg_a = _segformer_structure_mask(img_a)
    seg_b = _segformer_structure_mask(img_b)

    if seg_a is not None and seg_b is not None:
        combined = cv2.bitwise_or(seg_a, seg_b)
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
        )
        combined = cv2.dilate(combined, np.ones((9, 9), np.uint8), iterations=1)
        return combined, 'segformer'

    density_a = _structure_density(img_a)
    density_b = _structure_density(img_b)

    sky = cv2.bitwise_or(_sky_mask(img_a, density_a), _sky_mask(img_b, density_b))
    veg = cv2.bitwise_or(_vegetation_mask(img_a), _vegetation_mask(img_b))

    density = np.maximum(density_a, density_b)
    structural = (density > 0.14).astype(np.uint8) * 255

    # Consolidate, then fill enclosed interiors so solid walls and slabs
    # count as construction and not just their outlines.
    structural = cv2.morphologyEx(structural, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    structural = _fill_holes(structural)

    mask = cv2.bitwise_and(structural, cv2.bitwise_not(sky))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(veg))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return mask, 'heuristic'


# ---------------------------------------------------------------------------
# Alignment, exposure normalisation, structural comparison
# ---------------------------------------------------------------------------


def _align(src, dst):
    """Warp `src` onto `dst` using ORB features + RANSAC homography.

    Returns (warped_src, validity_mask). Falls back to a plain resize when
    there aren't enough reliable matches - better a slightly misaligned
    comparison than a wildly warped one.
    """
    h, w = dst.shape[:2]
    fallback = (cv2.resize(src, (w, h)), np.full((h, w), 255, dtype=np.uint8))

    try:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(4000)
        kp1, des1 = orb.detectAndCompute(gray_src, None)
        kp2, des2 = orb.detectAndCompute(gray_dst, None)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return fallback

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 15:
            return fallback

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or inliers is None or int(inliers.sum()) < 12:
            return fallback

        # Reject degenerate/extreme warps (mirror flips, huge scale jumps).
        det = float(np.linalg.det(H[:2, :2]))
        if not np.isfinite(det) or det < 0.2 or det > 5.0:
            return fallback

        warped = cv2.warpPerspective(src, H, (w, h))
        valid = cv2.warpPerspective(
            np.full(src.shape[:2], 255, dtype=np.uint8), H, (w, h)
        )
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8), iterations=1)
        return warped, valid
    except cv2.error as exc:
        logger.warning("Alignment failed, comparing unaligned: %s", exc)
        return fallback
