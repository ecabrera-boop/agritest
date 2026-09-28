"""
Retrain disease model using audit results to exclude mislabeled/ambiguous images.

Reads ml/saved_models/audit_results.csv (produced by audit_dataset.py) and
optionally ml/saved_models/manual_exclusions.txt (one image path per line),
then retrains only rf_disease.pkl on the filtered dataset.

Usage
-----
  python ml/retrain_clean.py
  python ml/retrain_clean.py --mislabel-threshold 0.80 --ambiguous-threshold 0.50
  python ml/retrain_clean.py --dry-run   # show stats without retraining
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_AUDIT_CSV = _ML_DIR / "saved_models" / "audit_results.csv"
_MANUAL_EXCLUSIONS = _ML_DIR / "saved_models" / "manual_exclusions.txt"
_DISEASE_MODEL = _ML_DIR / "saved_models" / "rf_disease.pkl"
_DISEASE_MODEL_CLEAN = _ML_DIR / "saved_models" / "rf_disease_clean.pkl"


def _load_exclusions(
    audit_csv: Path,
    mislabel_threshold: float,
    ambiguous_threshold: float,
) -> set[str]:
    """
    Return a set of image paths to exclude based on audit results.
    Excludes:
      - images where model >= mislabel_threshold confident of a DIFFERENT label
      - images where model confidence < ambiguous_threshold
    """
    excluded: set[str] = set()
    if not audit_csv.exists():
        print(f"  Audit CSV not found: {audit_csv}")
        return excluded

    with open(audit_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["confidence"])
            correct = row["correct"].lower() in ("true", "1", "yes")
            if not correct and conf >= mislabel_threshold:
                excluded.add(row["path"])
            elif conf < ambiguous_threshold:
                excluded.add(row["path"])

    if _MANUAL_EXCLUSIONS.exists():
        for line in _MANUAL_EXCLUSIONS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                excluded.add(line)

    return excluded


def retrain(
    mislabel_threshold: float = 0.80,
    ambiguous_threshold: float = 0.50,
    dry_run: bool = False,
) -> None:
    from ml.dataset_io import collect_samples, default_training_root
    from ml.train_models import (
        MODEL_DIR,
        _apply_smote,
        _aug_per_image_for_train_size,
        _build_ensemble,
        _features_original_only,
        _features_for_paths,
        _fit_and_score,
        _select_hyperparameters,
        _split_train_val_test,
    )

    print(f"\n=== Retrain with cleaned dataset ===")
    print(f"Mislabel threshold : {mislabel_threshold:.0%}  (confidence for WRONG label)")
    print(f"Ambiguous threshold: {ambiguous_threshold:.0%}  (minimum confidence to keep)")

    # Load exclusion list
    excluded = _load_exclusions(_AUDIT_CSV, mislabel_threshold, ambiguous_threshold)
    print(f"\nImages to exclude: {len(excluded)}")

    # Collect full dataset
    data_root = default_training_root()
    paths, health_labels, disease_labels = collect_samples(data_root)
    diseased_idx = [i for i, h in enumerate(health_labels) if h == "diseased"]
    d_paths = [paths[i] for i in diseased_idx]
    d_labels = [disease_labels[i] for i in diseased_idx]

    before_count = Counter(d_labels)
    # Filter
    keep = [(p, l) for p, l in zip(d_paths, d_labels) if p not in excluded]
    d_paths_clean = [p for p, _ in keep]
    d_labels_clean = [l for _, l in keep]
    after_count = Counter(d_labels_clean)

    print(f"\nDataset after filtering:")
    print(f"  {'Class':<22} {'Before':>8} {'After':>7} {'Removed':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8}")
    all_classes = sorted(set(list(before_count) + list(after_count)))
    for cls in all_classes:
        b, a = before_count.get(cls, 0), after_count.get(cls, 0)
        print(f"  {cls:<22} {b:>8} {a:>7} {b-a:>8}")
    print(f"  {'TOTAL':<22} {sum(before_count.values()):>8} {sum(after_count.values()):>7} "
          f"{sum(before_count.values())-sum(after_count.values()):>8}")

    if dry_run:
        print("\n--dry-run: stopping here (no model saved).")
        return

    # Split, augment, train
    d_train_idx, d_val_idx, d_test_idx = _split_train_val_test(d_paths_clean, d_labels_clean, 43)

    def _subset(idxs, paths, labels):
        return [paths[i] for i in idxs], [labels[i] for i in idxs]

    d_train_p, d_train_y = _subset(d_train_idx, d_paths_clean, d_labels_clean)
    d_val_p, d_val_y = _subset(d_val_idx, d_paths_clean, d_labels_clean)
    d_test_p, d_test_y = _subset(d_test_idx, d_paths_clean, d_labels_clean)

    print(f"\nSplit: train {len(d_train_p)} | val {len(d_val_p)} | test {len(d_test_p)}")

    aug = _aug_per_image_for_train_size(len(d_train_p), quick=False)
    print(f"Augmentation: {aug}/image")

    print("\nCV features (path-level, no aug) ...")
    X_cv = _features_original_only(d_train_p)
    d_params = _select_hyperparameters(X_cv, d_train_y, "Disease ensemble (clean)",
                                        groups=np.arange(len(d_train_p)))

    X_val = _features_original_only(d_val_p)
    X_test = _features_original_only(d_test_p)

    print("\nExtracting augmented train features ...")
    X_train, y_train = _features_for_paths(d_train_p, d_train_y, aug, 43)
    X_train, y_train = _apply_smote(X_train, y_train, 43)

    print(f"\nFitting on {len(y_train):,} samples ...")
    clf, tr, va, te = _fit_and_score(d_params, X_train, y_train, X_val, d_val_y, X_test, d_test_y)

    print(f"\nClean disease model: train {tr:.2%} | val {va:.2%} | test {te:.2%}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, _DISEASE_MODEL_CLEAN)
    print(f"Saved -> {_DISEASE_MODEL_CLEAN}")
    print("\nTo deploy: rename rf_disease_clean.pkl -> rf_disease.pkl")
    print("           (keep a backup of the original rf_disease.pkl first)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain disease model with cleaned dataset")
    parser.add_argument("--mislabel-threshold", type=float, default=0.80)
    parser.add_argument("--ambiguous-threshold", type=float, default=0.50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    retrain(args.mislabel_threshold, args.ambiguous_threshold, args.dry_run)


if __name__ == "__main__":
    main()
