"""
ML Pipeline orchestrator.

Calls four stages in order:
  0. layer0_validator — rice-leaf gate (heuristic is_rice_leaf before preprocessing)
  1. layer1_health    — healthy vs. diseased (with confidence gate)
  2. layer2_disease   — specific disease label + confidence (only if diseased)
  3. layer3_recommendation — treatment text

Each diseased prediction uses Test-Time Augmentation (TTA): probabilities are
averaged over 5 deterministic augmentations of the preprocessed leaf for the
label. Acceptance uses max(TTA, original-only) confidence and margin so gates
match held-out evaluation (single-pass). If the top two diseases are nearly tied,
the pipeline reports "unknown" instead of guessing.
"""
from __future__ import annotations

import logging
import os

from .combined_features import extract_features_tta_for_active_model
from .layer0_validator import is_rice_leaf
from .layer1_health import predict_health_proba_tta
from .layer2_disease import predict_disease_tta
from .layer3_recommendation import get_recommendation
from .validation import (
    DISEASE_MARGIN_MIN,
    DISEASE_MIN_CONFIDENCE,
    HEALTH_MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# Attached to the result when Layer 0 flagged the upload but Layer 1 was
# confident enough to proceed anyway (lenient gate).
GATE_WARNING_TEXT = (
    "The rice-leaf check was not fully confident about this photo "
    "(reason: {reason}). The analysis below still ran, but treat the result "
    "with extra care — retake the photo closer, in good light, with one leaf "
    "filling the frame if it looks wrong."
)

# Engine selection: "cnn" (default when trained + torch present), "rf" (force the
# hand-crafted pipeline), or "auto" (CNN if available, else rf). Override with the
# ML_ENGINE environment variable.
_ENGINE = os.environ.get("ML_ENGINE", "auto").strip().lower()


def _layer0_recommendation_key(reason: str) -> str:
    """Map Layer 0 heuristic reasons to layer3 recommendation keys."""
    if reason in ("not_rice", "wrong_shape"):
        return "not_rice"
    return "low_confidence"


def _unknown_result(
    reason: str,
    confidence: float = 0.0,
    *,
    all_disease_probs: dict | None = None,
    health_probs: dict | None = None,
    gate_warning: str = "",
) -> dict:
    return {
        "health": "unknown",
        "disease_name": "",
        "confidence": confidence,
        "all_disease_probs": all_disease_probs or {},
        "health_probs": health_probs or {},
        "recommendation": get_recommendation(reason),
        "rejection_reason": reason,
        "gate_warning": gate_warning,
        "engine": "rf",
    }


def run_pipeline(image_path: str) -> dict:
    """
    Run the full detection pipeline on a leaf image.

    Returns
    -------
    dict with keys:
        health           : "healthy" | "diseased" | "unknown"
        disease_name     : specific disease or ""
        confidence       : TTA-averaged top-class probability
        all_disease_probs: {disease: probability} for all classes (empty when healthy/unknown)
        health_probs     : {healthy/diseased: probability in [0, 1]} from Layer 1
        recommendation   : actionable text
        rejection_reason : non-empty when result is unknown
        gate_warning     : non-empty when Layer 0 flagged the image but Layer 1
                           was confident enough to analyse it anyway

    Uses the CNN (transfer-learning) pipeline when its heads are trained and torch
    is installed; otherwise falls back to the hand-crafted rf_* pipeline. Force one
    engine with the ML_ENGINE env var ("cnn" | "rf" | "auto").
    """
    if _ENGINE != "rf":
        try:
            from .cnn_pipeline import cnn_available, run_cnn_pipeline

            if cnn_available():
                return run_cnn_pipeline(image_path)
            if _ENGINE == "cnn":
                logger.warning("ML_ENGINE=cnn but CNN artifacts/torch missing; using rf pipeline.")
        except Exception:  # torch import or inference error → fall back to rf
            logger.exception("CNN pipeline failed; falling back to hand-crafted rf pipeline.")

    accepted, layer0_reason = is_rice_leaf(image_path)

    # Extract features for all TTA variants at once. Width is 427 for the legacy
    # model set and 456 for the ``_01`` set (baseline + lesion features); the
    # dispatcher picks the right one based on which pkl is currently active.
    features_tta = extract_features_tta_for_active_model(image_path)

    health_probs = predict_health_proba_tta(features_tta)
    health = max(health_probs, key=health_probs.get)
    health_conf = float(health_probs[health])

    # Strict Layer 0: a "not_rice"/"wrong_shape" flag ends the pipeline with a
    # "not a rice leaf" result instead of a disease analysis + warning banner.
    # Only a soft "low_confidence" gate reason continues with a warning.
    gate_warning = ""
    if not accepted:
        if layer0_reason in ("not_rice", "wrong_shape"):
            return _unknown_result(
                _layer0_recommendation_key(layer0_reason),
                confidence=health_conf,
                health_probs=health_probs,
            )
        gate_warning = GATE_WARNING_TEXT.format(reason=layer0_reason)

    if health_conf < HEALTH_MIN_CONFIDENCE:
        return _unknown_result(
            "low_confidence",
            confidence=health_conf,
            health_probs=health_probs,
            gate_warning=gate_warning,
        )

    if health == "healthy":
        return {
            "health": "healthy",
            "disease_name": "",
            "confidence": health_conf,
            "all_disease_probs": {},
            "health_probs": health_probs,
            "recommendation": get_recommendation("healthy"),
            "rejection_reason": "",
            "gate_warning": gate_warning,
            "engine": "rf",
        }

    disease_name, disease_conf, margin, all_probs = predict_disease_tta(features_tta)

    # TTA averages probabilities across flips/rotations, which can dilute the top
    # score vs. single-pass inference (how models are evaluated in train_models).
    # Use the stronger of TTA-averaged vs. original-only scores for acceptance.
    _, single_conf, single_margin, _ = predict_disease_tta(features_tta[0:1])
    gate_conf = max(disease_conf, single_conf)
    gate_margin = max(margin, single_margin)

    if gate_conf < DISEASE_MIN_CONFIDENCE or gate_margin < DISEASE_MARGIN_MIN:
        return _unknown_result(
            "low_confidence",
            confidence=gate_conf,
            all_disease_probs=all_probs,
            health_probs=health_probs,
            gate_warning=gate_warning,
        )

    return {
        "health": "diseased",
        "disease_name": disease_name,
        "confidence": gate_conf,
        "all_disease_probs": all_probs,
        "health_probs": health_probs,
        "recommendation": get_recommendation(disease_name),
        "rejection_reason": "",
        "gate_warning": gate_warning,
        "engine": "rf",
    }
