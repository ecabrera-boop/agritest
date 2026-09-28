"""
Evaluate saved Random Forest ensemble models on the held-out test split.

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import SOURCES_DIR, collect_samples, default_training_root  # noqa: E402
from ml.model_paths import get_model_tag, model_path  # noqa: E402
from ml.train_models import (  # noqa: E402
    MODEL_DIR,
    TEST_SIZE,
    VAL_SIZE_OF_REMAINDER,
    _features_original_only,
    _split_train_val_test,
)

VALIDATION_SUMMARY_PATH = MODEL_DIR / "validation_summary.json"


def _active_validation_summary_path() -> Path:
    tag = get_model_tag()
    return MODEL_DIR / (f"validation_summary_{tag}.json" if tag else "validation_summary.json")


def _active_health_model() -> Path:
    return model_path("rf_health")


def _active_disease_model() -> Path:
    return model_path("rf_disease")

# Metrics before shared background-removal preprocessing (retrain required after change).
BASELINE_BEFORE_PREPROCESS = {
    "health_accuracy": 0.9429695181907571,
    "disease_accuracy": 0.8629893238434164,
}
SHIP_THRESHOLDS = {"health_accuracy": 0.99, "disease_accuracy": 0.96}


def _roc_curve_points(
    y_true: list[str],
    y_proba: np.ndarray,
    labels: list[str],
) -> dict:
    """One-vs-rest ROC curve points (fpr/tpr) per class, for plotting.

    Binary tasks collapse to the positive class (index 1). Points are
    down-sampled to at most 200 per curve so the JSON stays small.
    """
    y_bin = label_binarize(y_true, classes=labels)

    def _thin(fpr: np.ndarray, tpr: np.ndarray, cap: int = 200) -> tuple[list, list]:
        if len(fpr) <= cap:
            return fpr.tolist(), tpr.tolist()
        idx = np.linspace(0, len(fpr) - 1, cap).round().astype(int)
        return fpr[idx].tolist(), tpr[idx].tolist()

    curves: dict[str, dict] = {}
    if len(labels) == 2:
        if y_bin[:, 0].sum() == 0:
            return curves
        fpr, tpr, _ = roc_curve(y_bin[:, 0], y_proba[:, 1])
        f, t = _thin(fpr, tpr)
        curves[labels[1]] = {"fpr": f, "tpr": t}
    else:
        for i, label in enumerate(labels):
            # Skip OvR curves when the test split has no positives for this class
            # (e.g. physiological stress held out of test on purpose).
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            f, t = _thin(fpr, tpr)
            curves[label] = {"fpr": f, "tpr": t}
    return curves


def _print_metrics_block(
    task_name: str,
    y_true: list[str],
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
) -> dict:
    """Compute and print classification metrics; return values for JSON export."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    n_classes = len(labels)
    if n_classes == 2:
        # Binary: use positive-class column (index 1)
        y_bin = label_binarize(y_true, classes=labels)
        roc_auc = roc_auc_score(y_bin, y_proba[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=labels)
        roc_auc = roc_auc_score(
            y_bin,
            y_proba,
            multi_class="ovr",
            average="weighted",
        )

    print(f"\n{'=' * 60}")
    print(f"{task_name} — Test Set Evaluation")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}  ({acc:.2%})")
    print(f"  Precision : {precision:.4f}  (weighted)")
    print(f"  Recall    : {recall:.4f}  (weighted)")
    print(f"  F1-Score  : {f1:.4f}  (weighted)")
    print(f"  ROC-AUC   : {roc_auc:.4f}  (weighted OvR)")
    print()
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    return {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "roc_auc_weighted": float(roc_auc),
        "roc_curve": _roc_curve_points(y_true, y_proba, labels),
        "n_test_samples": len(y_true),
        "classes": labels,
    }


def _compute_metrics(
    data_root: Path,
    *,
    verbose: bool = True,
) -> dict:
    """Run test-set evaluation; return metrics dict (no file I/O)."""
    paths, health_labels, disease_labels = collect_samples(data_root)
    if len(paths) < 10:
        raise RuntimeError(f"Need at least 10 images under {data_root}")

    if not _active_health_model().exists() or not _active_disease_model().exists():
        raise FileNotFoundError(
            "Trained models not found. Run: python ml/train_model_01.py --data data/sources\n"
            f"  Expected: {_active_health_model()}\n"
            f"           {_active_disease_model()}"
        )

    if verbose:
        print(f"Dataset: {len(paths)} images from {data_root}")
        print(
            f"Split: {int((1 - TEST_SIZE) * (1 - VAL_SIZE_OF_REMAINDER) * 100)}% train / "
            f"{int((1 - TEST_SIZE) * VAL_SIZE_OF_REMAINDER * 100)}% val / "
            f"{int(TEST_SIZE * 100)}% test (stratified)"
        )

    health_clf = joblib.load(_active_health_model())
    disease_clf = joblib.load(_active_disease_model())

    _, _, h_test_idx = _split_train_val_test(paths, health_labels, 42)
    h_test_paths = [paths[i] for i in h_test_idx]
    h_test_y = [health_labels[i] for i in h_test_idx]
    X_h_test = _features_original_only(h_test_paths)

    h_labels = list(health_clf.classes_)
    h_pred = health_clf.predict(X_h_test)
    h_proba = health_clf.predict_proba(X_h_test)

    if verbose:
        health_metrics = _print_metrics_block(
            "Layer 1 — Health (Random Forest Ensemble)",
            h_test_y,
            h_pred,
            h_proba,
            h_labels,
        )
    else:
        health_metrics = _metrics_dict(h_test_y, h_pred, h_proba, h_labels)

    d_paths_all = [paths[i] for i, h in enumerate(health_labels) if h == "diseased"]
    d_labels_all = [disease_labels[i] for i, h in enumerate(health_labels) if h == "diseased"]
    _, _, d_test_idx = _split_train_val_test(d_paths_all, d_labels_all, 43)
    d_test_paths = [d_paths_all[i] for i in d_test_idx]
    d_test_y = [d_labels_all[i] for i in d_test_idx]
    X_d_test = _features_original_only(d_test_paths)

    d_labels = list(disease_clf.classes_)
    d_pred = disease_clf.predict(X_d_test)
    d_proba = disease_clf.predict_proba(X_d_test)

    if verbose:
        disease_metrics = _print_metrics_block(
            "Layer 2 — Disease (Random Forest Ensemble)",
            d_test_y,
            d_pred,
            d_proba,
            d_labels,
        )
    else:
        disease_metrics = _metrics_dict(d_test_y, d_pred, d_proba, d_labels)

    return {
        "dataset_images": len(paths),
        "health": health_metrics,
        "disease": disease_metrics,
    }


def _metrics_dict(
    y_true: list[str],
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
) -> dict:
    """Same metrics as _print_metrics_block without printing."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    y_bin = label_binarize(y_true, classes=labels)
    if len(labels) == 2:
        roc_auc = roc_auc_score(y_bin, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
    return {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "roc_auc_weighted": float(roc_auc),
        "roc_curve": _roc_curve_points(y_true, y_proba, labels),
        "n_test_samples": len(y_true),
        "classes": labels,
    }


def save_validation_summary(data_root: Path | None = None) -> dict:
    """Evaluate on test split and write validation_summary.json."""
    data_root = data_root or default_training_root()
    summary = _compute_metrics(data_root, verbose=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _active_validation_summary_path()
    out_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate(data_root: Path | None = None) -> None:
    data_root = data_root or default_training_root()
    summary = _compute_metrics(data_root, verbose=True)
    health_metrics = summary["health"]
    disease_metrics = summary["disease"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _active_validation_summary_path()
    out_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<12} {'Health':>10} {'Disease':>10}")
    print("-" * 34)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision"),
        ("recall_weighted", "Recall"),
        ("f1_weighted", "F1-Score"),
        ("roc_auc_weighted", "ROC-AUC"),
    ]:
        print(
            f"{label:<12} {health_metrics[key]:>10.4f} {disease_metrics[key]:>10.4f}"
        )
    print()
    print(f"Active model tag: {get_model_tag() or 'default'}")
    print(f"Health model : {_active_health_model().resolve()}")
    print(f"Disease model: {_active_disease_model().resolve()}")
    print(f"Test samples : health={health_metrics['n_test_samples']}, "
          f"disease={disease_metrics['n_test_samples']}")
    print()
    _print_preprocess_gate(health_metrics["accuracy"], disease_metrics["accuracy"])
    print()


def _print_preprocess_gate(health_acc: float, disease_acc: float) -> None:
    """Compare held-out accuracy to pre-preprocess baseline and ship thresholds."""
    base_h = BASELINE_BEFORE_PREPROCESS["health_accuracy"]
    base_d = BASELINE_BEFORE_PREPROCESS["disease_accuracy"]
    delta_h = health_acc - base_h
    delta_d = disease_acc - base_d
    print("=" * 60)
    print("PREPROCESSING VALIDATION (held-out test)")
    print("=" * 60)
    print(f"{'':12} {'Health':>10} {'Disease':>10}")
    print(f"{'Baseline':12} {base_h:>10.2%} {base_d:>10.2%}")
    print(f"{'Current':12} {health_acc:>10.2%} {disease_acc:>10.2%}")
    print(f"{'Delta':12} {delta_h:>+10.2%} {delta_d:>+10.2%}")
    print(f"{'Target':12} {SHIP_THRESHOLDS['health_accuracy']:>10.2%} "
          f"{SHIP_THRESHOLDS['disease_accuracy']:>10.2%}")
    holds = health_acc >= base_h and disease_acc >= base_d
    meets_target = (
        health_acc >= SHIP_THRESHOLDS["health_accuracy"]
        and disease_acc >= SHIP_THRESHOLDS["disease_accuracy"]
    )
    if meets_target:
        print("\nResult: SHIP — meets target thresholds.")
    elif holds:
        print("\nResult: SHIP — accuracy held vs baseline (below 99%/96% targets but not worse).")
    else:
        print("\nResult: HOLD — accuracy dropped vs baseline; revisit preprocessing.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RF ensemble models (accuracy, precision, recall, F1, ROC-AUC)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SOURCES_DIR,
        help="Data root with class folders (default: data/sources)",
    )
    args = parser.parse_args()
    evaluate(args.data)


if __name__ == "__main__":
    main()
