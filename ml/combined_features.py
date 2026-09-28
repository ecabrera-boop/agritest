"""
Tag-aware feature extraction for the ``_01`` RiceGuard model set.

The ``train_model_01.py`` trainer feeds its models a 456-dim feature row =
the baseline 427-dim vector from ``ml.preprocessor`` concatenated with the
29-dim lesion-aware vector from ``ml.lesion_features``:

      baseline (full leaf):   color moments + LBP + GLCM + HOG + edge histogram
      lesion-aware (spot):    GLCM Haralick + HSV moments + LBP + shape descriptors

This module owns that concatenation so that the trainer and the inference
pipeline cannot drift apart. It also exposes three small dispatchers that
return the **right** feature shape for whichever model set is currently
active (legacy 427 vs ``_01`` 456), so the rest of the inference code
doesn't have to know about model tags at all.

Public surface
--------------

Combined extractors (always return the 456-dim row that ``_01`` models expect):

    extract_combined_features_from_pil(img, *, preprocess) -> (456,)  float32
    extract_combined_features(image_path, *, preprocess)   -> (1, 456) float32
    extract_combined_features_tta(image_path)              -> (N, 456) float32

Active-tag dispatchers (legacy 427-dim when tag is empty, 456 when tag=``01``):

    extract_features_tta_for_active_model(image_path)
    extract_features_for_active_model(image_path, *, preprocess=False)
    extract_features_from_pil_for_active_model(img, *, preprocess=False)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .lesion_features import LESION_FEATURE_DIM, extract_lesion_features_from_pil
from .preprocessor import (
    EXPECTED_FEATURE_DIM,
    _BG_COLOR,
    _IMG_SIZE,
    extract_features,
    extract_features_from_pil,
    extract_features_tta,
)
from .preprocessor_shared import _letterbox_resize, preprocess_leaf

COMBINED_FEATURE_DIM = EXPECTED_FEATURE_DIM + LESION_FEATURE_DIM  # 427 + 29 = 456


# ---------------------------------------------------------------------------
# Canonical 128x128 RGB conversion (must match training so GLCM / shape stats
# are scale-comparable to the rows the *_01 models were trained on)
# ---------------------------------------------------------------------------

def _canonicalize(img: Image.Image, *, preprocess: bool) -> Image.Image:
    """
    Return a 128x128 PIL image suitable for both the baseline and lesion
    extractors. With ``preprocess=True`` this is the full ``preprocess_leaf``
    pipeline (matches Layers 1 / 2 training); with ``preprocess=False`` it is
    only an RGB-cast plus letterbox to 128x128 (matches the Layer 0 path that
    must judge the raw upload).
    """
    if preprocess:
        return preprocess_leaf(img, _IMG_SIZE)
    rgb = img.convert("RGB")
    if rgb.size == (_IMG_SIZE, _IMG_SIZE):
        return rgb
    return _letterbox_resize(rgb, _IMG_SIZE, _BG_COLOR)


# ---------------------------------------------------------------------------
# Combined (baseline + lesion) feature extractors
# ---------------------------------------------------------------------------

def extract_combined_features_from_pil(
    img: Image.Image, *, preprocess: bool = True
) -> np.ndarray:
    """Return a 1-D ``(COMBINED_FEATURE_DIM,)`` float32 vector for ``img``."""
    canonical = _canonicalize(img, preprocess=preprocess)
    base = extract_features_from_pil(canonical, preprocess=False).ravel().astype(np.float32)
    lesion = extract_lesion_features_from_pil(canonical).ravel().astype(np.float32)
    vec = np.concatenate([base, lesion]).astype(np.float32)
    if vec.size != COMBINED_FEATURE_DIM:
        raise ValueError(
            f"Combined feature vector has {vec.size} dims; expected {COMBINED_FEATURE_DIM}."
        )
    return vec


def extract_combined_features(image_path: str, *, preprocess: bool = True) -> np.ndarray:
    """Like :func:`extract_combined_features_from_pil` but takes a file path; returns ``(1, 456)``."""
    with Image.open(image_path) as opened:
        return extract_combined_features_from_pil(opened, preprocess=preprocess).reshape(1, -1)


def extract_combined_features_tta(image_path: str) -> np.ndarray:
    """
    Five-row TTA matrix of combined features for one image.

    Mirrors ``ml.preprocessor.extract_features_tta`` but with the 29-dim lesion
    vector appended to each row, so the result is ``(_N_TTA, 456)``.
    """
    # Re-use the existing TTA variant generator inside ``preprocessor`` so the
    # variants here are exactly the variants used to compute the legacy 427-dim
    # TTA rows. This guarantees train / serve TTA parity.
    from .preprocessor import _tta_variants  # noqa: WPS433 (private but stable here)

    with Image.open(image_path) as opened:
        base = preprocess_leaf(opened, _IMG_SIZE)
    rows = [
        extract_combined_features_from_pil(v, preprocess=False)
        for v in _tta_variants(base)
    ]
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Active-model dispatchers
# ---------------------------------------------------------------------------
#
# Routing rule: read the **active training report**'s ``feature_version`` and
# only emit combined (456-dim) features when that field carries the v2 tag
# emitted by ``train_model_01.py``. Routing on the model tag alone would feed
# 456-dim rows into a stale ``_01.pkl`` that was trained on the old 427-dim
# vector and crash at predict time; the report tells us exactly what shape the
# current pickle expects regardless of which trainer wrote it.

_V2_FEATURE_TAG = "lesion"  # substring present in all combined-features versions


def _active_feature_version() -> str:
    """Return the currently active model set's ``feature_version`` string (or '')."""
    try:
        from .model_paths import report_path  # local import keeps the cycle break clean
    except ImportError:
        return ""
    try:
        path = report_path()
    except Exception:
        return ""
    if not path.is_file():
        return ""
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("feature_version", ""))


def _active_tag_uses_combined() -> bool:
    return _V2_FEATURE_TAG in _active_feature_version().lower()


def extract_features_tta_for_active_model(image_path: str) -> np.ndarray:
    """Five-row TTA matrix sized for whichever model set is active."""
    if _active_tag_uses_combined():
        return extract_combined_features_tta(image_path)
    return extract_features_tta(image_path)


def extract_features_for_active_model(
    image_path: str, *, preprocess: bool = False
) -> np.ndarray:
    """Single feature row ``(1, n)`` sized for whichever model set is active."""
    if _active_tag_uses_combined():
        return extract_combined_features(image_path, preprocess=preprocess)
    return extract_features(image_path, preprocess=preprocess)


def extract_features_from_pil_for_active_model(
    img: Image.Image, *, preprocess: bool = False
) -> np.ndarray:
    """Single feature row ``(1, n)`` from an in-memory PIL image."""
    if _active_tag_uses_combined():
        return extract_combined_features_from_pil(img, preprocess=preprocess).reshape(1, -1)
    return extract_features_from_pil(img, preprocess=preprocess).reshape(1, -1)
