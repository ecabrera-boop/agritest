"""
Audit every image in a dataset folder against the trained disease model.

For each image:
  - extracts features (single-pass, no TTA — fast)
  - predicts disease via rf_disease.pkl
  - compares the prediction against the folder label

Outputs
-------
  saved_models/audit_results.csv   — one row per image
  saved_models/audit_report.txt    — summary report
  (prints confusion matrix and mislabel candidates to stdout)

Usage
-----
  python ml/audit_dataset.py
  python ml/audit_dataset.py --root data/sources/hf_amdmqd/rice_leaf_disease_dataset
  python ml/audit_dataset.py --max-per-class 200   # quick run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import IMAGE_SUFFIXES, normalize_label  # noqa: E402
from ml.preprocessor import extract_features_from_pil  # noqa: E402
from ml.preprocessor_shared import preprocess_leaf  # noqa: E402

_DISEASE_MODEL = _ML_DIR / "saved_models" / "rf_disease.pkl"
_OUT_CSV = _ML_DIR / "saved_models" / "audit_results.csv"
_OUT_REPORT = _ML_DIR / "saved_models" / "audit_report.txt"

# Confidence thresholds for flagging
_MISLABEL_CONF_THRESHOLD = 0.80   # model is this sure of a DIFFERENT label -> suspect mislabel
_AMBIGUOUS_CONF_THRESHOLD = 0.55  # model confidence below this -> image is ambiguous


def _resolve_folder_label(path: Path, root: Path) -> str | None:
    """Walk parents of path (relative to root) for the first known class folder."""
    root = root.resolve()
    for parent in path.parents:
        if parent.resolve() == root:
            break
        label = normalize_label(parent.name.strip())
        if label:
            return label
    return None


def _collect_images(root: Path, max_per_class: int | None) -> list[tuple[Path, str]]:
    """Return [(image_path, folder_label)] sorted by class then path."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = _resolve_folder_label(p, root)
        if label and label != "healthy":
            by_class[label].append(p)

    pairs: list[tuple[Path, str]] = []
    for label in sorted(by_class):
        imgs = by_class[label]
        if max_per_class:
            imgs = imgs[:max_per_class]
        pairs.extend((p, label) for p in imgs)
    return pairs


def _batch_predict(
    model,
    all_feats: np.ndarray,
) -> tuple[list[str], list[float], list[float], list[dict]]:
    """Run predict + predict_proba on the full feature matrix at once."""
    # model.classes_ works for a bare Pipeline and a CalibratedClassifierCV wrapper.
    classes = list(model.classes_)
    preds = model.predict(all_feats)
    probas = model.predict_proba(all_feats)  # (N, n_classes)

    pred_labels, confs, margins, all_probs_list = [], [], [], []
    for i in range(len(preds)):
        pred_labels.append(str(preds[i]))
        p = probas[i]
        sorted_idx = np.argsort(p)[::-1]
        conf = float(p[sorted_idx[0]])
        margin = float(p[sorted_idx[0]] - p[sorted_idx[1]]) if len(sorted_idx) > 1 else 1.0
        confs.append(conf)
        margins.append(margin)
        all_probs_list.append({classes[j]: round(float(p[j]) * 100, 1) for j in range(len(classes))})

    return pred_labels, confs, margins, all_probs_list


def _confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
    classes: list[str],
) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def _print_confusion_matrix(cm: np.ndarray, classes: list[str], file=None) -> None:
    def pr(*args, **kw):
        print(*args, **kw)
        if file:
            print(*args, **kw, file=file)

    col_w = max(len(c) for c in classes) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
    pr(header)
    pr("-" * len(header))
    for i, row_label in enumerate(classes):
        row = f"{row_label:<{col_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(classes)))
        pr(row)


def audit(root: Path, max_per_class: int | None = None) -> None:
    if not _DISEASE_MODEL.exists():
        raise FileNotFoundError(f"Disease model not found: {_DISEASE_MODEL}")

    print(f"\nLoading disease model ...")
    model = joblib.load(_DISEASE_MODEL)
    # model.classes_ works for a bare Pipeline and a CalibratedClassifierCV wrapper.
    classes = list(model.classes_)
    print(f"  Classes: {classes}")

    pairs = _collect_images(root, max_per_class)
    if not pairs:
        raise RuntimeError(f"No labeled disease images found under {root}")

    label_counts = Counter(label for _, label in pairs)
    print(f"\nDataset: {len(pairs)} images across {len(label_counts)} classes")
    for cls, cnt in sorted(label_counts.items()):
        print(f"  {cls:<22} {cnt:>5} images")

    # ── Feature extraction (cached) ───────────────────────────────────────────
    _cache_key = f"audit_feats_{len(pairs)}_{max_per_class or 'all'}.npy"
    _feat_cache = _ML_DIR / "saved_models" / "feature_cache" / _cache_key
    _path_cache = _feat_cache.with_suffix(".paths.txt")

    if _feat_cache.exists() and _path_cache.exists():
        cached_paths = _path_cache.read_text(encoding="utf-8").splitlines()
        if cached_paths == [str(p) for p, _ in pairs]:
            print(f"  Loading cached features from {_feat_cache.name} ...")
            X = np.load(str(_feat_cache))
            print(f"  Feature matrix (cached): {X.shape}")
        else:
            _feat_cache.unlink(missing_ok=True)
            X = None
    else:
        X = None

    if X is None:
        print(f"\nExtracting features ...")
        from PIL import Image

        all_feats: list[np.ndarray] = []
        t0 = time.perf_counter()
        step = max(1, len(pairs) // 40)
        for i, (path, _) in enumerate(pairs):
            try:
                with Image.open(path) as img:
                    base = preprocess_leaf(img)
                feats = extract_features_from_pil(base, preprocess=False)
            except Exception as exc:
                print(f"  SKIP {path.name}: {exc}")
                feats = np.zeros(427, dtype=np.float32)
            all_feats.append(feats)
            if (i + 1) % step == 0 or i + 1 == len(pairs):
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                print(f"  {i+1:>5}/{len(pairs)}  elapsed {elapsed:.0f}s  eta {eta:.0f}s", flush=True)

        X = np.vstack(all_feats)
        _feat_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(_feat_cache), X)
        _path_cache.write_text("\n".join(str(p) for p, _ in pairs), encoding="utf-8")
        print(f"  Feature matrix: {X.shape}  (cached to {_feat_cache.name})")

    # ── Batch prediction ──────────────────────────────────────────────────────
    print("\nRunning batch prediction ...")
    pred_labels, confs, margins, all_probs_list = _batch_predict(model, X)

    # ── Per-image analysis ────────────────────────────────────────────────────
    true_labels = [label for _, label in pairs]
    paths = [str(path) for path, _ in pairs]

    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / len(true_labels)

    mislabel_candidates: list[dict] = []
    ambiguous_images: list[dict] = []
    rows: list[dict] = []

    for i in range(len(pairs)):
        match = true_labels[i] == pred_labels[i]
        row = {
            "path": paths[i],
            "folder_label": true_labels[i],
            "predicted": pred_labels[i],
            "correct": match,
            "confidence": round(confs[i], 4),
            "margin": round(margins[i], 4),
        }
        row.update({f"prob_{c.replace(' ', '_')}": all_probs_list[i].get(c, 0.0) for c in classes})
        rows.append(row)

        if not match and confs[i] >= _MISLABEL_CONF_THRESHOLD:
            mislabel_candidates.append(row)

        if confs[i] < _AMBIGUOUS_CONF_THRESHOLD:
            ambiguous_images.append(row)

    # ── Per-class accuracy ────────────────────────────────────────────────────
    per_class: dict[str, dict] = {}
    for cls in classes:
        cls_true = [i for i, t in enumerate(true_labels) if t == cls]
        cls_correct = sum(1 for i in cls_true if pred_labels[i] == cls)
        per_class[cls] = {
            "total": len(cls_true),
            "correct": cls_correct,
            "accuracy": cls_correct / len(cls_true) if cls_true else 0.0,
        }

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = _confusion_matrix(true_labels, pred_labels, classes)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "folder_label", "predicted", "correct", "confidence", "margin"] + [
        f"prob_{c.replace(' ', '_')}" for c in classes
    ]
    with open(_OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults CSV -> {_OUT_CSV}")

    # ── Build report ──────────────────────────────────────────────────────────
    report_lines: list[str] = []

    def pr(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    pr("=" * 65)
    pr("DATASET AUDIT REPORT")
    pr(f"Root  : {root}")
    pr(f"Images: {len(pairs)}  (max_per_class={max_per_class or 'none'})")
    pr("=" * 65)
    pr()
    pr(f"Overall accuracy : {accuracy:.2%}  ({correct}/{len(pairs)} correct)")
    pr()
    pr("Per-class accuracy:")
    pr(f"  {'Class':<22} {'Total':>6} {'Correct':>8} {'Accuracy':>9}")
    pr(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*9}")
    for cls in sorted(per_class):
        d = per_class[cls]
        pr(f"  {cls:<22} {d['total']:>6} {d['correct']:>8} {d['accuracy']:>8.2%}")
    pr()
    pr("Confusion matrix (rows=true, cols=predicted):")
    pr()
    _print_confusion_matrix(cm, classes)
    pr()
    pr(f"Potential mislabels  (model >={_MISLABEL_CONF_THRESHOLD:.0%} confident of DIFFERENT label): "
       f"{len(mislabel_candidates)}")
    pr(f"Ambiguous images     (model confidence < {_AMBIGUOUS_CONF_THRESHOLD:.0%})               : "
       f"{len(ambiguous_images)}")
    pr()

    if mislabel_candidates:
        pr("-" * 65)
        pr("TOP MISLABEL CANDIDATES (showing <=30):")
        pr(f"  {'Image':<35} {'True':<20} {'Pred':<20} {'Conf':>6}")
        pr(f"  {'-'*35} {'-'*20} {'-'*20} {'-'*6}")
        for r in sorted(mislabel_candidates, key=lambda x: -x["confidence"])[:30]:
            name = Path(r["path"]).name
            pr(f"  {name:<35} {r['folder_label']:<20} {r['predicted']:<20} {r['confidence']:>5.1%}")
        pr()

    pr("-" * 65)
    pr("MOST CONFUSED PAIRS (class -> misclassified as):")
    confused_pairs: list[tuple[str, str, int]] = []
    for i, ti in enumerate(classes):
        for j, tj in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((ti, tj, cm[i, j]))
    for ti, tj, cnt in sorted(confused_pairs, key=lambda x: -x[2])[:10]:
        total_ti = per_class.get(ti, {}).get("total", 1)
        pct = cnt / total_ti
        pr(f"  {ti:<22} -> {tj:<22} {cnt:>5}  ({pct:.1%} of class)")
    pr()

    pr("-" * 65)
    pr("RECOMMENDATIONS:")
    low_acc_classes = [(c, d["accuracy"]) for c, d in per_class.items() if d["accuracy"] < 0.85]
    if low_acc_classes:
        pr("  Classes with accuracy < 85% (check for mislabeling or visual ambiguity):")
        for c, acc in sorted(low_acc_classes, key=lambda x: x[1]):
            pr(f"    * {c:<22} {acc:.1%}")
    if len(mislabel_candidates) > 20:
        pct = len(mislabel_candidates) / len(pairs)
        pr(f"  {len(mislabel_candidates)} potential mislabels ({pct:.1%} of dataset).")
        pr("  Run with --write-exclusions to create a 'bad_images.txt' exclusion list.")
    if accuracy < 0.80:
        pr("  Overall accuracy < 80% on training data - consider checking label quality.")
    pr()
    pr(f"Full per-image results: {_OUT_CSV}")

    with open(_OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved -> {_OUT_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset images vs trained disease model")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/sources/hf_amdmqd/rice_leaf_disease_dataset"),
        help="Dataset root with class subfolders",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limit images per class (for quick runs)",
    )
    args = parser.parse_args()
    audit(args.root, args.max_per_class)


if __name__ == "__main__":
    main()
