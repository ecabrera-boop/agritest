"""
Shared leaf preprocessing + feature extraction for train and inference.

Both ``ml.train_models`` and ``ml.pipeline`` call the same path:
    preprocess_leaf_image()  →  extract_features_from_pil()

Steps (identical at train and serve time):
  1. RGB load
  2. Very light luminance normalization
  3. Letterbox resize to 128×128

  Background removal is off by default (``REMOVE_BG_DEFAULT = False``); enable only via
  ``preprocess_leaf(..., remove_bg=True)`` for experiments (see ``ml/validate_preprocessing.py``).

Feature vector (~427 dims):
    color_moments (RGB + HSV + LAB) : 27
    LBP histogram                   : 18
    GLCM texture (5 props × 2d × 4a): 40
    HOG descriptor (32×32 patch)    : 324
    Edge energy histogram           : 18
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .preprocessor_shared import (
    REMOVE_BG_DEFAULT,
    _BG_COLOR,
    _letterbox_resize,
    plant_mask_from_rgb,
    preprocess_leaf,
    remove_background,
)

try:
    from skimage.color import rgb2lab
    from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
    from skimage.filters import sobel

    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

_IMG_SIZE = 128
_HOG_SIZE = 32
EXPECTED_FEATURE_DIM = 427


def preprocess_leaf_image(img: Image.Image, size: int = _IMG_SIZE) -> Image.Image:
    """
    Canonical train/serve preprocessing: optional background removal + light normalize + resize.

    Call this (via ``prepare_image``) before augmentation during training and before
    feature extraction at inference so distributions stay matched.
    """
    return preprocess_leaf(img, size)


def prepare_image(img: Image.Image, size: int = _IMG_SIZE) -> Image.Image:
    """Alias used by training — same as ``preprocess_leaf_image``."""
    return preprocess_leaf_image(img, size)


def prepare_image_raw(img: Image.Image, size: int = _IMG_SIZE) -> Image.Image:
    """
    Like :func:`prepare_image` but WITHOUT any image enhancement.

    Only does:
      1. RGB conversion
      2. Letterbox resize to ``size`` × ``size`` (so the feature vector keeps
         a fixed dimension)

    No luminance normalization, no background removal — the pixels you get
    back are the same pixels the user uploaded (just resized). This is what
    Layer 0 (rice-leaf gate) uses so the gate cannot artificially benefit
    from inference-time enhancement.
    """
    rgb = img.convert("RGB")
    if rgb.size != (size, size):
        rgb = _letterbox_resize(rgb, size, _BG_COLOR)
    return rgb


def load_preprocessed_image(image_path: str, size: int = _IMG_SIZE) -> Image.Image:
    """Load from disk and run the shared preprocessing pipeline."""
    with Image.open(image_path) as opened:
        return preprocess_leaf_image(opened, size)


def _to_array(
    img: Image.Image,
    size: int = _IMG_SIZE,
    *,
    preprocess: bool = True,
) -> np.ndarray:
    if preprocess:
        img = preprocess_leaf_image(img, size)
    else:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = _letterbox_resize(img, size, _BG_COLOR)
    return np.array(img, dtype=np.uint8)


def _color_moments(arr: np.ndarray) -> np.ndarray:
    """27-dim: mean/std/skew for each channel in RGB, HSV, LAB."""
    feats = []
    spaces = [arr.astype(np.float32)]

    r, g, b = arr[..., 0] / 255.0, arr[..., 1] / 255.0, arr[..., 2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-8
    h = np.zeros_like(r)
    mr, mg, mb = cmax == r, cmax == g, cmax == b
    h[mr] = 60 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    h[mg] = 60 * (((b[mg] - r[mg]) / delta[mg]) + 2)
    h[mb] = 60 * (((r[mb] - g[mb]) / delta[mb]) + 4)
    s = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
    v = cmax
    spaces.append((np.stack([h / 360.0, s, v], axis=-1) * 255).astype(np.float32))

    if _SKIMAGE:
        lab = rgb2lab(arr.astype(np.float32) / 255.0)
        lab_norm = (lab - lab.min()) / (lab.max() - lab.min() + 1e-8) * 255
        spaces.append(lab_norm.astype(np.float32))
    else:
        spaces.append(arr.astype(np.float32))

    for space in spaces:
        for c in range(space.shape[2]):
            ch = space[..., c]
            mean = float(ch.mean())
            std = float(ch.std())
            skew = float(np.mean(((ch - mean) / (std + 1e-8)) ** 3))
            feats.extend([mean, std, skew])

    return np.array(feats, dtype=np.float32)


def _lbp_features(gray: np.ndarray, P: int = 16, R: int = 2) -> np.ndarray:
    if _SKIMAGE:
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    else:
        hist, _ = np.histogram(gray.ravel(), bins=18, range=(0, 256))
    return (hist / (hist.sum() + 1e-8)).astype(np.float32)


def _glcm_features(gray: np.ndarray) -> np.ndarray:
    if not _SKIMAGE:
        return np.zeros(40, dtype=np.float32)
    g = (gray // 32).astype(np.uint8)
    gcm = graycomatrix(
        g,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = []
    for prop in props:
        feats.extend(graycoprops(gcm, prop).ravel().tolist())
    return np.array(feats, dtype=np.float32)


def _hog_features(arr: np.ndarray) -> np.ndarray:
    if not _SKIMAGE:
        return np.zeros(324, dtype=np.float32)
    small = np.array(
        Image.fromarray(arr).resize((_HOG_SIZE, _HOG_SIZE), Image.LANCZOS)
    )
    gray = small.mean(axis=2).astype(np.float32)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    if len(feat) != 324:
        raise ValueError(
            f"HOG produced {len(feat)} features; expected 324. "
            "Retrain models after changing HOG parameters."
        )
    return feat.astype(np.float32)


def _edge_features(gray: np.ndarray) -> np.ndarray:
    if _SKIMAGE:
        edges = sobel(gray.astype(np.float64))
    else:
        edges = np.array(Image.fromarray(gray).filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    e_min = float(edges.min())
    e_max = float(edges.max())
    if e_max <= e_min:
        e_max = e_min + 1.0
    hist, _ = np.histogram(edges.ravel(), bins=16, range=(e_min, e_max))
    hist = (hist / (hist.sum() + 1e-8)).astype(np.float32)
    return np.concatenate([[edges.mean(), edges.std()], hist]).astype(np.float32)


def extract_features_from_pil(img: Image.Image, *, preprocess: bool = True) -> np.ndarray:
    """
    Extract rich feature vector from a PIL Image (1-D float32, ~427 dims).

    Set ``preprocess=False`` when the image was already passed through
    ``prepare_image`` / ``preprocess_leaf_image`` (training augmentation path).
    """
    arr = _to_array(img, preprocess=preprocess)
    gray = arr.mean(axis=2).astype(np.uint8)
    parts = [
        _color_moments(arr),
        _lbp_features(gray),
        _glcm_features(gray),
        _hog_features(arr),
        _edge_features(gray),
    ]
    vec = np.concatenate(parts).astype(np.float32)
    if vec.size != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Feature vector has {vec.size} dims; expected {EXPECTED_FEATURE_DIM}. "
            "Retrain models if the extractor changed."
        )
    return vec


def extract_features(image_path: str, *, preprocess: bool = True) -> np.ndarray:
    """
    Load image from disk; return shape (1, n_features) for sklearn predict.

    Set ``preprocess=False`` to skip ALL image enhancement (no luminance
    normalization, no background removal) — only RGB convert + letterbox
    resize happens. Layer 0 (rice-leaf gate) uses this so the gate cannot
    benefit from inference-time enhancement that wouldn't apply to real
    uploads in production.
    """
    with Image.open(image_path) as opened:
        return extract_features_from_pil(opened, preprocess=preprocess).reshape(1, -1)


def extract_features_raw(image_path: str) -> np.ndarray:
    """
    Like :func:`extract_features` but with ``preprocess=False`` baked in.

    Use for Layer 0 (rice-leaf gate) and any other place where the model
    must judge the user's photo as-uploaded, not a normalized variant.
    """
    return extract_features(image_path, preprocess=False)


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA)
# ---------------------------------------------------------------------------

_N_TTA = 5  # number of deterministic variants


def _tta_variants(base: Image.Image) -> list[Image.Image]:
    """Five deterministic TTA variants of an already-preprocessed 128×128 image."""
    return [
        base,
        ImageOps.mirror(base),
        ImageOps.flip(base),
        base.rotate(12, resample=Image.BILINEAR, fillcolor=_BG_COLOR),
        base.rotate(-12, resample=Image.BILINEAR, fillcolor=_BG_COLOR),
    ]


def extract_features_tta(image_path: str) -> np.ndarray:
    """
    Load image, preprocess once, then return shape (_N_TTA, n_features).

    Each row is a feature vector from a different deterministic augmentation.
    Pass to layer1/layer2 TTA functions to average probabilities across rows —
    this smooths out sensitivity to exact orientation and lighting.
    """
    with Image.open(image_path) as opened:
        base = preprocess_leaf_image(opened)
    variants = _tta_variants(base)
    rows = [extract_features_from_pil(v, preprocess=False) for v in variants]
    return np.stack(rows)  # (_N_TTA, EXPECTED_FEATURE_DIM)
