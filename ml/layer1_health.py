"""
Layer 1 — Health classifier.

Loads rf_health.pkl (sklearn Pipeline: scaler → PCA → voting ensemble)
and predicts whether the leaf is healthy or diseased.
"""

from dataclasses import dataclass

import joblib
import numpy as np

from .model_paths import model_path

_model = None


def _load_model():
    global _model
    if _model is None:
        path = model_path("rf_health")
        if not path.is_file():
            raise FileNotFoundError(
                f"Health model not found at {path}. "
                "Train with: python ml/train_model_01.py --data data/sources"
            )
        _model = joblib.load(path)
    return _model


def predict_health(features: np.ndarray) -> str:
    """Predict whether the leaf is healthy or diseased."""
    label, _ = predict_health_with_confidence(features)
    return label


def predict_health_with_confidence(features: np.ndarray) -> tuple[str, float]:
    """
    Predict health label and model confidence (max class probability).

    Returns
    -------
    tuple[str, float]
        ("healthy" or "diseased", confidence in [0.0, 1.0])
    """
    model = _load_model()
    prediction = model.predict(features)[0]

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    label = str(prediction).lower()
    if label in ("healthy", "0", "0.0"):
        health = "healthy"
    else:
        health = "diseased"

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(np.max(proba))

    return health, confidence


def _normalize_health_label(raw) -> str:
    label = str(raw).lower()
    if label in ("healthy", "0", "0.0"):
        return "healthy"
    return "diseased"


def predict_health_proba_tta(features_matrix: np.ndarray) -> dict[str, float]:
    """
    TTA-averaged P(healthy) / P(diseased).

    Returns
    -------
    dict[str, float]
        Keys ``healthy`` and ``diseased`` (whichever classes the estimator has),
        values in [0.0, 1.0] summing to ~1.
    """
    model = _load_model()
    avg_proba = model.predict_proba(features_matrix).mean(axis=0)
    out: dict[str, float] = {}
    for cls, p in zip(model.classes_, avg_proba):
        key = _normalize_health_label(cls)
        out[key] = out.get(key, 0.0) + float(p)
    return out


def predict_health_tta(features_matrix: np.ndarray) -> tuple[str, float]:
    """
    TTA health prediction: average predict_proba across N augmentation rows.

    Parameters
    ----------
    features_matrix : np.ndarray
        Shape (N, n_features) — one row per TTA variant from extract_features_tta().

    Returns
    -------
    tuple[str, float]
        ("healthy" or "diseased", averaged confidence)
    """
    probs = predict_health_proba_tta(features_matrix)
    health = max(probs, key=probs.get)
    return health, float(probs[health])


@dataclass(frozen=True)
class HealthResult:
    passed: bool    # confidence >= HEALTH_MIN_CONFIDENCE
    health: str     # "healthy" | "diseased"
    confidence: float
    tta_used: bool = True


def validate_health(image_path: str) -> HealthResult:
    """
    Run Layer 1 health checks on *image_path* and return structured metrics.

    Uses TTA (5 augmentations) for confidence.  ``passed`` is True when
    confidence meets the HEALTH_MIN_CONFIDENCE threshold (0.60).
    """
    from .combined_features import extract_features_tta_for_active_model
    from .validation import HEALTH_MIN_CONFIDENCE

    features_tta = extract_features_tta_for_active_model(image_path)
    health, confidence = predict_health_tta(features_tta)
    return HealthResult(
        passed=confidence >= HEALTH_MIN_CONFIDENCE,
        health=health,
        confidence=confidence,
    )
