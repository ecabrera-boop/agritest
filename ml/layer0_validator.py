"""
Layer 0 — Rice-leaf validation gate.

Runs before Layers 1–2 to reject out-of-distribution inputs (faces, objects,
non-rice plants). When ``rf_rice_gate.pkl`` is present (trained on all images
under ``data/sources`` via ``train_models.py``), the gate trusts that classifier
first.

IMPORTANT — no inference-time enhancement
-----------------------------------------
The gate sees the **raw** uploaded photo (just RGB-converted + letterbox-resized
to the model's input size). It does NOT run the gentle luminance normalization
that Layers 1–2 use, so the gate cannot artificially benefit from preprocessing
that brightens / equalizes the upload. The rice gate is trained with the same
raw features (see ``train_models._train_rice_gate``) so train and serve match.

Without a trained gate, only light colour heuristics are used (no strict blade
shape rules that reject full-frame dataset photos).

Returns rejection reasons that map to layer3_recommendation keys:
  "not_rice"       — clearly not a rice leaf
  "wrong_shape"    — reserved; kept for API compatibility
  "low_confidence" — classifier uncertain and image lacks plant-like colour
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from PIL import Image

from .combined_features import (
    extract_features_for_active_model,
    extract_features_from_pil_for_active_model,
)
from .preprocessor import plant_mask_from_rgb

try:
    from skimage.measure import label as sk_label

    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

from .model_paths import model_path

_model = None

# Hard reject only for obvious non-plants (faces, indoor scenes).
MIN_PLANT_PIXEL_FRACTION_HARD = 0.05

# Soft plant signal used when the classifier is uncertain.
MIN_PLANT_PIXEL_FRACTION = 0.12

# Classifier thresholds (rf_rice_gate.pkl trained on data/sources rice vs data/not_rice)
RICE_GATE_ACCEPT_CONFIDENCE = 0.50
RICE_GATE_REJECT_CONFIDENCE = 0.60

# Legacy heuristic fallback (no trained gate) — relaxed for full-frame leaf photos.
_FALLBACK_MIN_PLANT = 0.10
_FALLBACK_MIN_ASPECT = 1.15


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str
    confidence: float
    plant_fraction: float
    aspect_ratio: float
    classifier_used: bool = False


def _load_model():
    global _model
    path = model_path("rf_rice_gate")
    if _model is None and path.is_file():
        _model = joblib.load(path)
    return _model


def _open_rgb(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def plant_pixel_fraction(image_path: str) -> float:
    """
    Fraction of pixels in plant-like HSV ranges (green + yellow-brown leaf tones).

    Faces and indoor scenes typically score near zero; rice leaves score high.
    """
    arr = np.array(_open_rgb(image_path))
    return _plant_fraction_from_array(arr)


def _plant_fraction_from_array(arr: np.ndarray) -> float:
    return float(plant_mask_from_rgb(arr).mean())


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    if not _SKIMAGE:
        return mask
    labeled = sk_label(mask)
    if labeled.max() == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def _elongation_from_mask(mask: np.ndarray) -> float:
    """
    Rotation-invariant aspect ratio (major / minor axis) from mask pixel coordinates.
    """
    if not mask.any():
        return 1.0
    ys, xs = np.where(mask)
    if len(ys) < 2:
        return 1.0
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    coords -= coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    if cov.ndim < 2:
        return 1.0
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if eigvals[-1] <= 1e-9:
        return 1.0
    return float(np.sqrt(eigvals[0] / eigvals[-1]))


def _dominant_plant_shape_metrics(arr: np.ndarray) -> tuple[float, float]:
    """Aspect ratio and fill ratio of the largest plant-mask connected component."""
    component = _largest_component(plant_mask_from_rgb(arr))
    if not component.any():
        return 1.0, 1.0

    ys, xs = np.where(component)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    aspect = _elongation_from_mask(component)
    fill = float(component.sum()) / float(bbox_w * bbox_h)
    return aspect, fill


def _leaf_aspect_ratio_from_array(arr: np.ndarray) -> float:
    """Rotation-invariant elongation of the green-dominant region, >= 1.0."""
    g = arr[:, :, 1].astype(float)
    r, b = arr[:, :, 0].astype(float), arr[:, :, 2].astype(float)
    mask = (g > 80) & (g > r * 1.05) & (g > b * 1.05)

    if not mask.any():
        h, w = arr.shape[:2]
        return max(w, h) / max(min(w, h), 1)

    return _elongation_from_mask(mask)


def _heuristic_fallback(arr: np.ndarray) -> tuple[bool, str]:
    """Light colour/aspect checks when rf_rice_gate.pkl is missing."""
    plant_frac = _plant_fraction_from_array(arr)
    if plant_frac < _FALLBACK_MIN_PLANT:
        return False, "not_rice"

    aspect = _leaf_aspect_ratio_from_array(arr)
    if aspect < _FALLBACK_MIN_ASPECT:
        return False, "not_rice"

    return True, ""


def _gate_from_array(
    arr: np.ndarray,
    features: np.ndarray | None,
) -> GateResult:
    plant_frac = _plant_fraction_from_array(arr)
    aspect = _leaf_aspect_ratio_from_array(arr)

    if plant_frac < MIN_PLANT_PIXEL_FRACTION_HARD:
        return GateResult(
            accepted=False,
            reason="not_rice",
            confidence=plant_frac,
            plant_fraction=plant_frac,
            aspect_ratio=aspect,
        )

    model = _load_model()
    if model is not None and features is not None:
        is_rice, clf_conf = predict_rice_gate(features)

        if is_rice and clf_conf >= RICE_GATE_ACCEPT_CONFIDENCE:
            return GateResult(
                accepted=True,
                reason="",
                confidence=clf_conf,
                plant_fraction=plant_frac,
                aspect_ratio=aspect,
                classifier_used=True,
            )

        if not is_rice and clf_conf >= RICE_GATE_REJECT_CONFIDENCE:
            return GateResult(
                accepted=False,
                reason="not_rice",
                confidence=clf_conf,
                plant_fraction=plant_frac,
                aspect_ratio=aspect,
                classifier_used=True,
            )

        # Uncertain classifier: accept typical rice-leaf colour coverage from training set.
        if plant_frac >= MIN_PLANT_PIXEL_FRACTION:
            return GateResult(
                accepted=True,
                reason="",
                confidence=clf_conf,
                plant_fraction=plant_frac,
                aspect_ratio=aspect,
                classifier_used=True,
            )

        return GateResult(
            accepted=False,
            reason="low_confidence",
            confidence=clf_conf,
            plant_fraction=plant_frac,
            aspect_ratio=aspect,
            classifier_used=True,
        )

    accepted, reason = _heuristic_fallback(arr)
    return GateResult(
        accepted=accepted,
        reason=reason,
        confidence=plant_frac if accepted else plant_frac / max(_FALLBACK_MIN_PLANT, 1e-6),
        plant_fraction=plant_frac,
        aspect_ratio=aspect,
        classifier_used=False,
    )


def is_rice_leaf(image: str | Image.Image) -> tuple[bool, str]:
    """
    Rice-leaf gate used by the web app and pipeline.

    Uses the trained ``rf_rice_gate.pkl`` when available. Falls back to light
    heuristics otherwise.

    The gate uses **raw** features (``preprocess=False``) — no luminance
    normalization, no background removal — so the verdict reflects the actual
    uploaded photo, not a beautified version of it.

    Returns (accepted, reason) where reason is ``""``, ``"not_rice"``, or
    ``"low_confidence"``.
    """
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
        arr = np.array(pil)
        # preprocess=False: the gate must judge the raw upload, not an enhanced variant.
        features = extract_features_from_pil_for_active_model(pil, preprocess=False)
    else:
        arr = np.array(_open_rgb(image))
        features = (
            extract_features_for_active_model(image, preprocess=False)
            if _load_model() is not None else None
        )

    result = _gate_from_array(arr, features)
    return result.accepted, result.reason


def leaf_aspect_ratio(image_path: str) -> float:
    """Aspect ratio of the green-dominant region (max/min side), >= 1.0."""
    arr = np.array(_open_rgb(image_path))
    return _leaf_aspect_ratio_from_array(arr)


def predict_rice_gate(features: np.ndarray) -> tuple[bool, float]:
    """
    Binary rice-leaf classifier.

    Returns (is_rice, confidence) where confidence is the winning class probability.
    """
    model = _load_model()
    if model is None:
        return True, 1.0

    if features.ndim == 1:
        features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    label = str(prediction).strip().lower()
    is_rice = label in ("rice", "rice_leaf", "1", "true", "yes")

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        classes = [str(c).strip().lower() for c in model.classes_]
        proba = model.predict_proba(features)[0]
        if is_rice:
            rice_idx = next(
                (i for i, c in enumerate(classes) if c in ("rice", "rice_leaf", "1", "true", "yes")),
                int(np.argmax(proba)),
            )
            confidence = float(proba[rice_idx])
        else:
            not_idx = next(
                (i for i, c in enumerate(classes) if c in ("not_rice", "0", "false", "no")),
                int(np.argmax(proba)),
            )
            confidence = float(proba[not_idx])

    return is_rice, confidence


def validate_rice_leaf(image_path: str) -> GateResult:
    """
    Run Layer 0 checks and return structured metrics.

    Features are extracted with ``preprocess=False`` so the gate scores the
    raw uploaded photo, not a luminance-normalized variant.

    Returns GateResult with reason in ("", "not_rice", "wrong_shape", "low_confidence").
    """
    arr = np.array(_open_rgb(image_path))
    features = (
        extract_features_for_active_model(image_path, preprocess=False)
        if _load_model() is not None else None
    )
    return _gate_from_array(arr, features)


def is_likely_rice_leaf(image_path: str) -> bool:
    """Backward-compatible boolean wrapper used by older imports."""
    return validate_rice_leaf(image_path).accepted
