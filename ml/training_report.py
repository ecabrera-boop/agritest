"""Load and print the latest ML training report (shared by CLI and Django)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_paths import (
    get_model_tag,
    model_path,
    report_path,
    validation_summary_path,
)

_ML_DIR = Path(__file__).resolve().parent
MODEL_DIR = _ML_DIR / "saved_models"

METRIC_ROWS = (
    ("accuracy", "Accuracy"),
    ("precision_weighted", "Precision"),
    ("recall_weighted", "Recall"),
    ("f1_weighted", "F1-Score"),
    ("roc_auc_weighted", "ROC-AUC"),
)


def get_validation_summary() -> dict[str, Any] | None:
    """Metrics for templates/API: validation_summary.json, else training_report fallbacks."""
    summary_path = validation_summary_path()
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["model_tag"] = get_model_tag() or "default"
        return data

    report_file = report_path()
    if not report_file.is_file():
        return None

    report = json.loads(report_file.read_text(encoding="utf-8"))
    if "models" in report:
        h = report["models"]["health"]
        d = report["models"]["disease"]
        h_test, d_test = h["test"], d["test"]

        def _from_layer(test: dict, n_test: int | None) -> dict:
            return {
                "accuracy": float(test["accuracy"]),
                "precision_weighted": float(test.get("macro_f1", test["accuracy"])),
                "recall_weighted": float(test.get("balanced_accuracy", test["accuracy"])),
                "f1_weighted": float(test.get("macro_f1", test["accuracy"])),
                "roc_auc_weighted": None,
                "n_test_samples": n_test,
                "classes": [],
            }

        return {
            "dataset_images": report.get("total_images"),
            "health": _from_layer(h_test, h.get("n_test")),
            "disease": _from_layer(d_test, d.get("n_test")),
            "from_training_report_only": True,
            "model_tag": get_model_tag() or "default",
        }


def print_validation_summary() -> None:
    """Print FINAL RESULTS plus full validation metrics (terminal / Django startup)."""
    print_final_results()

    summary = get_validation_summary()
    if summary is None:
        print("No validation summary. Run: python ml/evaluate_models.py\n")
        return

    if summary.get("from_training_report_only"):
        print(
            "Full metrics (Precision, Recall, F1, ROC-AUC) not cached yet.\n"
            "Run: python ml/evaluate_models.py\n"
        )
        return

    health = summary["health"]
    disease = summary["disease"]
    print("=" * 60)
    print("VALIDATION SUMMARY (held-out test set)")
    print("=" * 60)
    print(f"{'Metric':<12} {'Health':>10} {'Disease':>10}")
    print("-" * 34)
    for key, label in METRIC_ROWS:
        h_val = health.get(key)
        d_val = disease.get(key)
        h_str = "—" if h_val is None else f"{float(h_val):>10.4f}"
        d_str = "—" if d_val is None else f"{float(d_val):>10.4f}"
        print(f"{label:<12} {h_str:>10} {d_str:>10}")
    print()
    print(
        f"Test samples: health={health['n_test_samples']}, "
        f"disease={disease['n_test_samples']}"
    )
    if summary.get("dataset_images"):
        print(f"Dataset images: {summary['dataset_images']}")
    print(f"Cached: {validation_summary_path().resolve()}\n")


def print_final_results() -> None:
    """Print the same FINAL RESULTS block as ml/train_models.py."""
    report_file = report_path()
    if not report_file.is_file():
        print(
            "\nNo training report found. Train models:\n"
            "  python ml/train_model_01.py --data data/sources\n"
        )
        return

    report = json.loads(report_file.read_text(encoding="utf-8"))
    tag = get_model_tag() or "default"
    if "models" in report:
        health_acc = float(report["models"]["health"]["test"]["accuracy"])
        disease_acc = float(report["models"]["disease"]["test"]["accuracy"])
    else:
        health_acc = float(report["health_model"]["test_accuracy"])
        disease_acc = float(report["disease_model"]["test_accuracy"])

    health_model = model_path("rf_health")
    disease_model = model_path("rf_disease")

    print("\n" + "=" * 60)
    print("FINAL RESULTS (held-out test split)")
    print("=" * 60)
    print(f"Active model set: {tag}")
    print(f"Health  test accuracy: {health_acc:.2%}")
    print(f"Disease test accuracy: {disease_acc:.2%}")
    print(f"Report: {report_file.resolve()}")
    print(f"Models: {health_model.resolve()}, {disease_model.resolve()}")
    print()


def print_full_report() -> None:
    """Extended summary (CLI helper)."""
    report_file = report_path()
    if not report_file.is_file():
        print("No training report found. Run: python ml/train_model_01.py --data data/sources")
        return

    r = json.loads(report_file.read_text(encoding="utf-8"))
    if "models" in r:
        print()
        print("=" * 60)
        print("LATEST TRAINING RESULTS (saved report)")
        print("=" * 60)
        print(f"Model tag: {r.get('model_tag', get_model_tag() or 'default')}")
        if r.get("total_images"):
            print(f"Dataset: {r['total_images']} images")
        print(
            f"Split: {int(r['split']['train'] * 100)}% train / "
            f"{int(r['split']['validation'] * 100)}% val / "
            f"{int(r['split']['test'] * 100)}% test  |  CV folds: {r['cv_folds']}"
        )
        print()
        for layer, title in (("health", "Health"), ("disease", "Disease")):
            m = r["models"][layer]
            test = m["test"]
            print(f"--- {title} model ---")
            print(f"  n_estimators: {m.get('n_estimators')}  |  aug_per_image: {m.get('aug_per_image')}")
            print(
                f"  Train: {m['train']['accuracy']:.2%}  |  Val: {m['val']['accuracy']:.2%}  |  "
                f"Test: {test['accuracy']:.2%}"
            )
            print()
        print("Disease classes:", ", ".join(r["disease_classes"]))
        print_final_results()
        return

    h = r["health_model"]
    d = r["disease_model"]

    print()
    print("=" * 60)
    print("LATEST TRAINING RESULTS (saved report)")
    print("=" * 60)
    if r.get("total_images"):
        print(f"Dataset: {r['total_images']} images")
    elif r.get("training_hours_budget"):
        print(f"Legacy run: {r['training_hours_budget']} hour budget, {r.get('rounds_completed', '?')} rounds")
    print(
        f"Split: {int(r['split']['train'] * 100)}% train / "
        f"{int(r['split']['validation'] * 100)}% val / "
        f"{int(r['split']['test'] * 100)}% test  |  CV folds: {r['cv_folds']}"
    )
    if r.get("feature_extractor"):
        print(f"Features: {r['feature_extractor']}")
    if r.get("ensemble"):
        print(f"Model: {r['ensemble']}")
    if r.get("pipeline"):
        print(f"Pipeline: {r['pipeline']}")
    print()
    print("--- Health model (rf_health.pkl) ---")
    print(f"  Best round: {h['round']}  |  aug_per_image: {h['aug_per_image']}")
    print(f"  n_estimators: {h['n_estimators']}  |  max_depth: {h['max_depth']}")
    print(f"  CV accuracy:  {h['cv_accuracy']:.2%}")
    print(
        f"  Train: {h['train_accuracy']:.2%}  |  Val: {h['val_accuracy']:.2%}  |  "
        f"Test: {h['test_accuracy']:.2%}"
    )
    print(f"  Overfit gap (train-test): {h['train_accuracy'] - h['test_accuracy']:.2%}")
    print()
    print("--- Disease model (rf_disease.pkl) ---")
    print(f"  Best round: {d['round']}  |  aug_per_image: {d['aug_per_image']}")
    print(f"  n_estimators: {d['n_estimators']}  |  max_depth: {d['max_depth']}")
    print(f"  CV accuracy:  {d['cv_accuracy']:.2%}")
    print(
        f"  Train: {d['train_accuracy']:.2%}  |  Val: {d['val_accuracy']:.2%}  |  "
        f"Test: {d['test_accuracy']:.2%}"
    )
    print(f"  Overfit gap (train-test): {d['train_accuracy'] - d['test_accuracy']:.2%}")
    print()
    print("Disease classes:", ", ".join(r["disease_classes"]))
    print_final_results()
