"""
Standalone Layer 0 gate diagnostic.

Usage:
    python -m ml.debug_gate <image_path>

Prints the four heuristic metrics against their thresholds and PASS/FAIL
for each check. Reuses layer0_validator helpers — no duplicated logic.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from ml.layer0_validator import (
    MIN_PLANT_PIXEL_FRACTION,
    MIN_PLANT_PIXEL_FRACTION_HARD,
    _dominant_plant_shape_metrics,
    _gate_from_array,
    _leaf_aspect_ratio_from_array,
    _load_model,
    _plant_fraction_from_array,
    predict_rice_gate,
)
from ml.preprocessor import extract_features


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def diagnose(image_path: str) -> int:
    arr = np.array(Image.open(image_path).convert("RGB"))

    plant_fraction = _plant_fraction_from_array(arr)
    leaf_aspect_ratio = _leaf_aspect_ratio_from_array(arr)
    comp_aspect, comp_fill = _dominant_plant_shape_metrics(arr)

    hard_plant_ok = plant_fraction >= MIN_PLANT_PIXEL_FRACTION_HARD
    soft_plant_ok = plant_fraction >= MIN_PLANT_PIXEL_FRACTION

    # preprocess=False matches the gate's production path — no luma normalization,
    # no background removal. Diagnostic numbers must reflect the raw upload.
    features = (
        extract_features(image_path, preprocess=False) if _load_model() is not None else None
    )
    gate = _gate_from_array(arr, features)

    print(f"Image: {image_path}")
    print()
    print(
        f"  plant_fraction      = {plant_fraction:.4f}  "
        f"(hard reject < {MIN_PLANT_PIXEL_FRACTION_HARD})  {_status(hard_plant_ok)}"
    )
    print(
        f"  soft_plant_ok       = {soft_plant_ok}  "
        f"(uncertain accept if >= {MIN_PLANT_PIXEL_FRACTION})"
    )
    print(f"  leaf_aspect_ratio   = {leaf_aspect_ratio:.4f}  (info)")
    print(f"  comp_aspect         = {comp_aspect:.4f}  (info)")
    print(f"  comp_fill           = {comp_fill:.4f}  (info)")

    if features is not None:
        is_rice, clf_conf = predict_rice_gate(features)
        print(
            f"  rice_gate_clf       = {'rice' if is_rice else 'not_rice'}  "
            f"conf={clf_conf:.4f}  model=rf_rice_gate.pkl"
        )
    else:
        print("  rice_gate_clf       = (no model — heuristic fallback only)")

    print()
    if gate.accepted:
        print(f"Overall: PASS — {gate.reason or 'accepted'} (conf={gate.confidence:.4f})")
    else:
        print(f"Overall: FAIL — reason={gate.reason} (conf={gate.confidence:.4f})")

    return 0 if gate.accepted else 1


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m ml.debug_gate <image_path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(diagnose(sys.argv[1]))


if __name__ == "__main__":
    main()
