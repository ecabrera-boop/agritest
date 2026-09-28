"""
Read-only: disease-model confidence on the held-out diseased TEST split.

Prints max predict_proba distribution and accuracy above/below a threshold
to justify DISEASE_MIN_CONFIDENCE in validation.py.

Usage:
    python ml/analyze_disease_confidence.py
    python ml/analyze_disease_confidence.py --data data/sources --threshold 0.55
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import SOURCES_DIR, collect_samples, default_training_root  # noqa: E402
from ml.train_models import (  # noqa: E402
    DISEASE_MODEL,
    _features_original_only,
    _split_train_val_test,
)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _accuracy(y_true: list[str], y_pred: np.ndarray) -> float:
    if not y_true:
        return 0.0
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def analyze(data_root: Path, threshold: float) -> None:
    paths, health_labels, disease_labels = collect_samples(data_root)
    d_paths_all = [paths[i] for i, h in enumerate(health_labels) if h == "diseased"]
    d_labels_all = [disease_labels[i] for i, h in enumerate(health_labels) if h == "diseased"]
    if not d_paths_all:
        raise RuntimeError(f"No diseased images under {data_root}")

    if not DISEASE_MODEL.exists():
        raise FileNotFoundError(f"Disease model not found: {DISEASE_MODEL}")

    _, _, d_test_idx = _split_train_val_test(d_paths_all, d_labels_all, 43)
    d_test_paths = [d_paths_all[i] for i in d_test_idx]
    d_test_y = [d_labels_all[i] for i in d_test_idx]

    disease_clf = joblib.load(DISEASE_MODEL)
    X_test = _features_original_only(d_test_paths)
    y_pred = disease_clf.predict(X_test)
    proba = disease_clf.predict_proba(X_test)
    max_conf = np.max(proba, axis=1)

    dist = _percentiles(max_conf)
    above = max_conf >= threshold
    below = ~above

    acc_all = _accuracy(d_test_y, y_pred)
    acc_above = _accuracy(
        [d_test_y[i] for i in range(len(d_test_y)) if above[i]],
        y_pred[above],
    )
    acc_below = _accuracy(
        [d_test_y[i] for i in range(len(d_test_y)) if below[i]],
        y_pred[below],
    )

    n_above = int(np.sum(above))
    n_below = int(np.sum(below))

    print(f"Dataset root: {data_root}")
    print(f"Diseased TEST split: {len(d_test_y)} images (seed=43, same as evaluate_models)")
    print(f"Threshold (DISEASE_MIN_CONFIDENCE): {threshold:.2f}")
    print()
    print("Max predict_proba distribution:")
    for key, val in dist.items():
        print(f"  {key:8s}: {val:.4f}")
    print()
    print(f"Overall test accuracy (no threshold): {acc_all:.2%} ({len(d_test_y)} samples)")
    print()
    print(f"At or above threshold ({n_above} samples):")
    print(f"  accuracy: {acc_above:.2%}")
    print()
    print(f"Below threshold ({n_below} samples):")
    print(f"  accuracy: {acc_below:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disease-model confidence distribution on held-out diseased TEST set"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SOURCES_DIR,
        help="Data root (default: data/sources)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Confidence floor to compare (default: 0.55)",
    )
    args = parser.parse_args()
    data_root = args.data if args.data.exists() else default_training_root()
    analyze(data_root, args.threshold)


if __name__ == "__main__":
    main()
