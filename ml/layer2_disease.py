"""
Layer 2 — Disease classifier.

Loads rf_disease.pkl (sklearn Pipeline: scaler → PCA → voting ensemble)
and predicts the specific disease label and confidence score.
Only called when Layer 1 returns "diseased".

Predictions are restricted to disease classes that currently have training
data on disk under data/sources (see `_allowed_disease_classes`). This is
a safety net: if a folder is later removed (or never added back) the live
pickle's stale classes — e.g. an out-of-date "tungro" — can no longer win
a prediction.
"""

from __future__ import annotations

import joblib
import numpy as np

from .dataset_io import SOURCES_DIR, STRESS_CLASS, available_disease_classes
from .model_paths import model_path

_model = None
_allowed_classes: frozenset[str] | None = None


def _get_model_classes(model) -> list[str]:
    """
    Return class labels aligned with model.predict_proba() output columns.

    For sklearn Pipeline models, prefer pipeline-level `classes_` because it is
    guaranteed to match predict/predict_proba output ordering.
    """
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        classes = getattr(clf, "classes_", None)
    if classes is None:
        raise AttributeError(
            "Disease model has no classes_. Re-train and export rf_disease.pkl again."
        )
    return [str(c) for c in classes]


def _load_model():
    global _model
    if _model is None:
        path = model_path("rf_disease")
        if not path.is_file():
            raise FileNotFoundError(
                f"Disease model not found at {path}. "
                "Train with: python ml/train_model_01.py --data data/sources"
            )
        _model = joblib.load(path)
    return _model


def _allowed_disease_classes() -> frozenset[str]:
    """
    Disease labels that currently have training images under data/sources.

    Computed once per process and cached. If the scan finds nothing (e.g. the
    source folder is missing in a deployed container), we fall back to "no
    restriction" so predictions still work — better than refusing to answer.
    """
    global _allowed_classes
    if _allowed_classes is None:
        try:
            found = set(available_disease_classes(SOURCES_DIR))
        except Exception:
            found = set()
        if found:
            # "physiological stress" is trained from a synthetic train-only
            # folder (not a class folder on disk), so the folder scan never
            # finds it — always keep it out of the stale-class mask.
            found.add(STRESS_CLASS)
        _allowed_classes = frozenset(found)
    return _allowed_classes


def _mask_disallowed(
    classes: list[str], proba: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """
    Zero out any class whose folder is no longer present on disk and renormalize
    the survivors so the probabilities still sum to 1. Returns the inputs
    untouched if the allowlist is empty or no class survives the mask
    (defensive: never produce all-zero output).
    """
    allowed = _allowed_disease_classes()
    if not allowed:
        return classes, proba
    mask = np.array([c in allowed for c in classes], dtype=bool)
    if not mask.any():
        return classes, proba
    filtered = proba.astype(float, copy=True)
    filtered[~mask] = 0.0
    total = float(filtered.sum())
    if total > 0.0:
        filtered = filtered / total
    return classes, filtered


def predict_disease(features: np.ndarray) -> tuple[str, float]:
    """
    Predict the specific rice disease and the model's confidence score.

    Parameters
    ----------
    features : np.ndarray
        Feature vector produced by preprocessor.extract_features().

    Returns
    -------
    tuple[str, float]
        (disease_label, confidence) where confidence is in [0.0, 1.0].
    """
    model = _load_model()
    classes = _get_model_classes(model)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes, proba = _mask_disallowed(classes, proba)
        top = int(np.argmax(proba))
        return str(classes[top]), float(proba[top])

    prediction = model.predict(features)[0]
    return str(prediction), 0.0


def predict_disease_tta(
    features_matrix: np.ndarray,
) -> tuple[str, float, float, dict[str, float]]:
    """
    TTA disease prediction: average predict_proba across N augmentation rows.

    Parameters
    ----------
    features_matrix : np.ndarray
        Shape (N, n_features) — one row per TTA variant from extract_features_tta().

    Returns
    -------
    tuple[str, float, float, dict[str, float]]
        (disease_label, confidence, margin_vs_2nd, all_class_probs)

        - confidence   : averaged top-class probability
        - margin_vs_2nd: top-class prob minus second-class prob (0–1)
        - all_class_probs: {class_name: averaged_probability, ...}
          (classes masked out by `_allowed_disease_classes` are omitted)
    """
    model = _load_model()
    avg_proba = model.predict_proba(features_matrix).mean(axis=0)
    classes = _get_model_classes(model)
    if len(classes) != len(avg_proba):
        raise RuntimeError(
            "Class/probability mismatch in disease model. "
            f"classes={len(classes)} probs={len(avg_proba)}"
        )

    classes, avg_proba = _mask_disallowed(classes, avg_proba)

    sorted_idx = np.argsort(avg_proba)[::-1]
    top_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx

    disease_label = str(classes[top_idx])
    confidence = float(avg_proba[top_idx])
    margin = float(abs(avg_proba[top_idx] - avg_proba[second_idx])) if len(sorted_idx) > 1 else 1.0
    all_probs = dict(sorted(
        {
            str(c): round(float(p) * 100, 1)
            for c, p in zip(classes, avg_proba)
            if p > 0.0
        }.items(),
        key=lambda x: -x[1],
    ))

    return disease_label, confidence, margin, all_probs
