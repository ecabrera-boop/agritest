"""
Compare held-out test accuracy with background removal ON vs OFF.

Uses the same stratified 70/10/20 split as ml.train_models and the saved
rf_health.pkl / rf_disease.pkl pipelines. Does not retrain.

Run:
    python ml/validate_preprocessing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.preprocessor import extract_features_from_pil  # noqa: E402
from ml.preprocessor_shared import preprocess_leaf  # noqa: E402
from ml.dataset_io import SOURCES_DIR, collect_samples, default_training_root  # noqa: E402
from ml.train_models import (  # noqa: E402
    HEALTH_MODEL,
    DISEASE_MODEL,
    TEST_SIZE,
    VAL_SIZE_OF_REMAINDER,
    _split_train_val_test,
)

# Current production baseline — background removal ON must meet both to ship.
BASELINE_HEALTH_ACCURACY = 0.990
BASELINE_DISEASE_ACCURACY = 0.966


def _features_for_paths(paths: list[str], *, remove_bg: bool) -> np.ndarray:
    rows: list[np.ndarray] = []
    n = len(paths)
    label = "ON" if remove_bg else "OFF"
    for i, path in enumerate(paths):
        with Image.open(path) as opened:
            preprocessed = preprocess_leaf(opened, remove_bg=remove_bg)
        rows.append(extract_features_from_pil(preprocessed, preprocess=False).ravel())
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  features (remove_bg={label}): {i + 1}/{n}")
    return np.vstack(rows)


def _accuracy_for_setting(
    h_test_paths: list[str],
    h_test_y: list[str],
    d_test_paths: list[str],
    d_test_y: list[str],
    health_clf,
    disease_clf,
    *,
    remove_bg: bool,
) -> tuple[float, float]:
    label = "ON" if remove_bg else "OFF"
    print(f"\nExtracting features (remove_bg={label}) ...")
    X_h = _features_for_paths(h_test_paths, remove_bg=remove_bg)
    X_d = _features_for_paths(d_test_paths, remove_bg=remove_bg)

    h_pred = health_clf.predict(X_h)
    d_pred = disease_clf.predict(X_d)
    h_acc = float(accuracy_score(h_test_y, h_pred))
    d_acc = float(accuracy_score(d_test_y, d_pred))
    return h_acc, d_acc


def validate_preprocessing(data_root: Path | None = None) -> None:
    data_root = data_root or default_training_root()
    paths, health_labels, disease_labels = collect_samples(data_root)
    if len(paths) < 10:
        raise RuntimeError(f"Need at least 10 images under {data_root}")

    if not HEALTH_MODEL.exists() or not DISEASE_MODEL.exists():
        raise FileNotFoundError(
            f"Trained models not found. Run: python ml/train_models.py --quick\n"
            f"  Expected: {HEALTH_MODEL}\n"
            f"           {DISEASE_MODEL}"
        )

    health_clf = joblib.load(HEALTH_MODEL)
    disease_clf = joblib.load(DISEASE_MODEL)

    _, _, h_test_idx = _split_train_val_test(paths, health_labels, 42)
    h_test_paths = [paths[i] for i in h_test_idx]
    h_test_y = [health_labels[i] for i in h_test_idx]

    d_paths_all = [paths[i] for i, h in enumerate(health_labels) if h == "diseased"]
    d_labels_all = [disease_labels[i] for i, h in enumerate(health_labels) if h == "diseased"]
    _, _, d_test_idx = _split_train_val_test(d_paths_all, d_labels_all, 43)
    d_test_paths = [d_paths_all[i] for i in d_test_idx]
    d_test_y = [d_labels_all[i] for i in d_test_idx]

    train_pct = int((1 - TEST_SIZE) * (1 - VAL_SIZE_OF_REMAINDER) * 100)
    val_pct = int((1 - TEST_SIZE) * VAL_SIZE_OF_REMAINDER * 100)
    test_pct = int(TEST_SIZE * 100)

    print(f"Dataset: {len(paths)} images from {data_root}")
    print(f"Split: {train_pct}% train / {val_pct}% val / {test_pct}% test (stratified)")
    print(f"Test samples: health={len(h_test_paths)}, disease={len(d_test_paths)}")
    print(f"Models: {HEALTH_MODEL.name}, {DISEASE_MODEL.name}")

    h_off, d_off = _accuracy_for_setting(
        h_test_paths,
        h_test_y,
        d_test_paths,
        d_test_y,
        health_clf,
        disease_clf,
        remove_bg=False,
    )
    h_on, d_on = _accuracy_for_setting(
        h_test_paths,
        h_test_y,
        d_test_paths,
        d_test_y,
        health_clf,
        disease_clf,
        remove_bg=True,
    )

    print("\n" + "=" * 62)
    print("BACKGROUND REMOVAL A/B (held-out test)")
    print("=" * 62)
    print(f"{'Setting':<22} {'Health acc':>12} {'Disease acc':>12}")
    print("-" * 62)
    print(f"{'remove_bg=False (OFF)':<22} {h_off:>11.2%} {d_off:>12.2%}")
    print(f"{'remove_bg=True  (ON)':<22} {h_on:>11.2%} {d_on:>12.2%}")
    print("-" * 62)
    print(
        f"{'Baseline (ship bar)':<22} "
        f"{BASELINE_HEALTH_ACCURACY:>11.2%} "
        f"{BASELINE_DISEASE_ACCURACY:>12.2%}"
    )
    print(
        f"{'Delta ON vs OFF':<22} "
        f"{h_on - h_off:>+11.2%} "
        f"{d_on - d_off:>+12.2%}"
    )

    meets_baseline = h_on >= BASELINE_HEALTH_ACCURACY and d_on >= BASELINE_DISEASE_ACCURACY
    print()
    if meets_baseline:
        print(
            "VERDICT: Keep background removal ON - health and disease accuracy "
            f"both meet baseline (health >= {BASELINE_HEALTH_ACCURACY:.1%}, "
            f"disease >= {BASELINE_DISEASE_ACCURACY:.1%})."
        )
    else:
        failing = []
        if h_on < BASELINE_HEALTH_ACCURACY:
            failing.append(f"health {h_on:.2%} < {BASELINE_HEALTH_ACCURACY:.1%}")
        if d_on < BASELINE_DISEASE_ACCURACY:
            failing.append(f"disease {d_on:.2%} < {BASELINE_DISEASE_ACCURACY:.1%}")
        print(
            "VERDICT: Recommend background removal OFF - "
            + "; ".join(failing)
            + "."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare test accuracy with background removal ON vs OFF"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SOURCES_DIR,
        help="Data root with class folders (default: data/sources)",
    )
    args = parser.parse_args()
    validate_preprocessing(args.data)


if __name__ == "__main__":
    main()
