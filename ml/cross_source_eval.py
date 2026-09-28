"""
Cross-source (leave-one-dataset-out) generalization report.

Why this exists
---------------
The in-distribution test accuracy reported by ``train_models.py`` is ~99% with
ROC-AUC ~1.0. That number is REAL but MISLEADING: after de-duplication the three
disease classes come almost entirely from a single source folder (leaf/data_1),
and *blast* exists in that one source only. When train and test images share the
same source, the model memorises that source's lighting/background/camera and
scores near-100% — while generalising poorly to leaves from any other source
(a real phone upload gets the wrong disease).

This script measures the honest number a reviewer will ask for:
  - IN-SOURCE   : train on data_1, test on a held-out slice of data_1.
  - CROSS-SOURCE: train on data_1, test on data_3 (an independent source the
                  model never saw).

It writes ml/saved_models/cross_source_report.json and prints a summary you can
cite directly in the study.

Run:  python ml/cross_source_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import collect_samples, default_training_root  # noqa: E402
from ml.preprocessor import extract_features  # noqa: E402
from ml.train_models import _build_rf_pipeline  # noqa: E402

REPORT_PATH = _ML_DIR / "saved_models" / "cross_source_report.json"
SOURCES = ("data_1", "data_2", "data_3", "data_4", "data_5", "data_6")
TRAIN_CAP_PER_CLASS = 400  # keep the report fast; representative of full training


def _source_of(path: str) -> str:
    for s in SOURCES:
        if s in path:
            return s
    return "other"


def _features(items: list[tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    X = np.vstack([extract_features(p).ravel() for p, _ in items])
    y = [label for _, label in items]
    return X, y


def _cap_per_class(items: list[tuple[str, str]], cap: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[tuple[str, str]]] = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    out: list[tuple[str, str]] = []
    for label, group in by.items():
        if len(group) <= cap:
            out += group
        else:
            idx = rng.choice(len(group), size=cap, replace=False)
            out += [group[i] for i in idx]
    return out


def evaluate() -> dict:
    data_root = default_training_root()
    paths, health, disease = collect_samples(data_root)

    # Diseased-only rows, tagged with their source folder.
    rows = [
        (p, dl, _source_of(p))
        for p, hl, dl in zip(paths, health, disease)
        if hl == "diseased"
    ]
    by_source_class = Counter((s, dl) for _, dl, s in rows)

    # Hold out data_4 (the independent second source, all 3 disease classes) as
    # the unseen test set; train on every other source (data_1..3). Falls back to
    # data_3 if data_4 has not been added yet.
    test_source = "data_4" if any(s == "data_4" for _, _, s in rows) else "data_3"
    train_rows = [(p, dl) for p, dl, s in rows if s != test_source]
    test_rows = [(p, dl) for p, dl, s in rows if s == test_source]

    print("Disease uniques by (source, class):")
    for (s, c) in sorted(by_source_class):
        print(f"  {s:7s} {c:18s} {by_source_class[(s, c)]}")

    if not test_rows:
        raise SystemExit(
            f"No independent {test_source} diseased images found — cannot run a "
            "cross-source test. Add a second source to measure generalization."
        )

    train = _cap_per_class(train_rows, TRAIN_CAP_PER_CLASS, seed=0)
    data3 = test_rows  # kept name for the report fields below
    print(f"\nTrain (all except {test_source}) per class: {Counter(dl for _, dl in train)}")
    print(f"Test  ({test_source}, unseen) per class: {Counter(dl for _, dl in test_rows)}")

    print("\nExtracting features ...")
    X_tr, y_tr = _features(train)
    X_te, y_te = _features(data3)

    # Same preprocessing/feature pipeline as the deployed model (scaler->PCA->RF).
    # RF is the dominant member (2x weight) of the shipped voting ensemble, so it
    # is a faithful, fast proxy for the cross-source number.
    clf = _build_rf_pipeline(400, None, min_samples_leaf=3)
    clf.fit(X_tr, y_tr)

    # In-source control: hold out 20% of data_1 (same-source test).
    from sklearn.model_selection import train_test_split

    Xa, Xb, ya, yb = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=1, stratify=y_tr
    )
    ctrl = _build_rf_pipeline(400, None, min_samples_leaf=3).fit(Xa, ya)
    in_source_acc = float(accuracy_score(yb, ctrl.predict(Xb)))

    pred = clf.predict(X_te)
    cross_acc = float(accuracy_score(y_te, pred))
    cross_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    test_classes = sorted(set(y_te))

    print(f"\nIN-SOURCE    (same-source held-out slice)          : acc = {in_source_acc:.3f}")
    print(f"CROSS-SOURCE (test {test_source}, unseen source)          : acc = {cross_acc:.3f}  macro-F1 = {cross_f1:.3f}")
    print("\nCross-source per-class report:")
    print(classification_report(y_te, pred, labels=test_classes, zero_division=0))

    single_source_classes = sorted(
        {c for (_, c) in by_source_class}
        - {c for (s, c) in by_source_class if s != "data_1"}
    )

    report = {
        "note": (
            "In-source accuracy is optimistic because disease classes are "
            "single-source (see single_source_classes). Cross-source accuracy is "
            "the honest generalization estimate."
        ),
        "disease_uniques_by_source_class": {
            f"{s}/{c}": n for (s, c), n in sorted(by_source_class.items())
        },
        "single_source_classes": single_source_classes,
        "train_source": f"all except {test_source}",
        "test_source": test_source,
        "train_class_counts": dict(Counter(dl for _, dl in train)),
        "test_class_counts": dict(Counter(dl for _, dl in data3)),
        "in_source_accuracy": round(in_source_acc, 4),
        "cross_source_accuracy": round(cross_acc, 4),
        "cross_source_macro_f1": round(cross_f1, 4),
        "cross_source_classes": test_classes,
        "limitation": (
            "Blast has no independent source, so it cannot be cross-validated; "
            "the cross-source figure covers only the classes present in data_3. "
            "Collect disease images (especially blast) from additional sources to "
            "obtain a full generalization estimate."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    evaluate()
