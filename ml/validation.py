"""
Confidence thresholds and re-exports for the Layer 0 rice-leaf gate.

Layer 0 heuristics and classifier live in layer0_validator.py.
"""

from __future__ import annotations

from .layer0_validator import (
    GateResult,
    is_likely_rice_leaf,
    is_rice_leaf,
    leaf_aspect_ratio,
    plant_pixel_fraction,
    validate_rice_leaf,
)

# Layer 1 / Layer 2 confidence floors (max predict_proba).
# Disease gating uses max(TTA-averaged, original-only) so thresholds align with
# held-out evaluation (single-pass); TTA is still used to pick the label.
HEALTH_MIN_CONFIDENCE = 0.60
# Layer 2 is a 3-class problem (chance = 0.33). The soft-voting ensemble spreads
# probability across classes, so genuinely-diseased real-world uploads often land
# around 0.40-0.50 on the top class — especially field photos with busy
# backgrounds. A higher floor rejected those as "unknown" (shown as
# "unclassified") even though Layer 1 was confident the leaf is sick and the top
# class had a clear lead. 0.40 still sits comfortably above chance; the top-2
# margin guard below still blocks genuine near-ties.
DISEASE_MIN_CONFIDENCE = 0.40

# Minimum gap between top-1 and top-2 disease probabilities (after TTA).
# If the model is nearly tied between two diseases, report "unknown" rather
# than committing to a wrong label with false confidence. Lowered 0.15 -> 0.10:
# a 10-point lead over the runner-up is enough to name the disease; anything
# tighter (a near coin-flip between two diseases) still returns "unknown".
DISEASE_MARGIN_MIN = 0.10

# Lenient Layer 0 (see run_pipeline / run_cnn_pipeline): when the gate flags an
# upload as "not_rice"/"wrong_shape", the pipeline no longer rejects it outright.
# Layer 1 still runs and the result carries a gate warning. The image is only
# rejected completely when the gate said "not_rice" AND Layer 1's confidence is
# below this floor.
GATE_OVERRIDE_MIN_HEALTH_CONFIDENCE = 0.40

__all__ = [
    "DISEASE_MIN_CONFIDENCE",
    "GATE_OVERRIDE_MIN_HEALTH_CONFIDENCE",
    "GateResult",
    "HEALTH_MIN_CONFIDENCE",
    "is_likely_rice_leaf",
    "is_rice_leaf",
    "leaf_aspect_ratio",
    "plant_pixel_fraction",
    "validate_rice_leaf",
]
