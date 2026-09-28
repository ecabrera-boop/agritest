"""
Per-image, per-category disease analysis.

Analyzes each image one-by-one, scoring it against every disease category
and explaining what features drive each classification. Useful for understanding
model behavior, diagnosing misclassifications, and comparing disease signatures.

What it shows per image:
  - Probability scores across ALL disease categories (ranked)
  - Confidence tier (high / medium / low)
  - Feature group importances projected back from PCA to raw feature space
  - Discriminative gap: what separates the top prediction from the runner-up
  - Per-category feature profile: which feature groups push toward each label

Usage:
    # Single image
    python ml/analyze_per_category.py --image path/to/leaf.jpg

    # All images in a folder (recursive)
    python ml/analyze_per_category.py --dir data/raw/

    # Limit to a specific true-label folder
    python ml/analyze_per_category.py --dir data/raw/blast/

    # Summary table only (no per-image detail)
    python ml/analyze_per_category.py --dir data/raw/ --summary-only

    # Save JSON report
    python ml/analyze_per_category.py --dir data/raw/ --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.preprocessor import extract_features_from_pil, prepare_image, _IMG_SIZE
from ml.dataset_io import IMAGE_SUFFIXES

from ml.model_paths import model_path

# Feature group definitions matching preprocessor.py output order.
# Index ranges: color(0:27) lbp(27:45) glcm(45:85) hog(85:409) edge(409:427)
FEATURE_GROUPS = [
    ("Color Moments",  0,   27,  "RGB/HSV/LAB mean, std, skew — captures leaf color and hue"),
    ("LBP Texture",   27,   45,  "Local Binary Patterns — micro-texture patterns on leaf surface"),
    ("GLCM Texture",  45,   85,  "Gray co-occurrence — coarseness, regularity, contrast of spots/lesions"),
    ("HOG Shape",     85,  409,  "Histogram of Oriented Gradients — lesion shape and edge directionality"),
    ("Edge Energy",  409,  427,  "Sobel edge distribution — density and spread of boundary regions"),
]

CONFIDENCE_TIERS = [
    (0.80, "HIGH    ", "Strong prediction — model is certain"),
    (0.60, "MEDIUM  ", "Moderate confidence — review the image"),
    (0.00, "LOW     ", "Weak signal — image may be ambiguous or at disease boundary"),
]


def _load_model():
    path = model_path("rf_disease")
    if not path.is_file():
        raise FileNotFoundError(
            f"Disease model not found at {path}. "
            "Run: python ml/train_model_01.py --data data/sources"
        )
    return joblib.load(path)


def _get_classes(model) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        classes = getattr(clf, "classes_", None)
    if classes is None:
        raise AttributeError("Model has no classes_. Re-train rf_disease.pkl.")
    return [str(c) for c in classes]


def _group_importance_from_rf(model) -> np.ndarray | None:
    """
    Project RF feature importances (in PCA space) back to original 427-dim space.

    result[i] = sum of |pca_component[k, i]| * rf_importance[k]  for all k

    This approximates how much each original feature contributes to RF splits.
    Returns None if RF or PCA is unavailable.
    """
    try:
        pca = model.named_steps["pca"]
        voting = model.named_steps["clf"]

        # named_estimators_ is a dict (preferred); estimators_ is a flat list
        rf = None
        if hasattr(voting, "named_estimators_"):
            rf = voting.named_estimators_.get("rf")
        if rf is None:
            for est in (voting.estimators_ or []):
                if hasattr(est, "feature_importances_"):
                    rf = est
                    break

        if rf is None or not hasattr(rf, "feature_importances_"):
            return None

        importances = rf.feature_importances_  # (n_pca_components,)
        components = pca.components_           # (n_pca_components, 427)
        # Weighted absolute projection back to original feature space
        orig = np.abs(components.T @ importances)  # (427,)
        orig = orig / (orig.sum() + 1e-12)
        return orig
    except Exception:
        return None


def _summarize_feature_groups(orig_importances: np.ndarray | None) -> list[dict]:
    """Aggregate raw-feature importances into the 5 named groups."""
    groups = []
    for name, start, end, description in FEATURE_GROUPS:
        if orig_importances is not None:
            weight = float(orig_importances[start:end].sum())
        else:
            weight = float("nan")
        groups.append({"name": name, "weight": weight, "description": description})
    # Sort descending by weight
    groups.sort(key=lambda g: g["weight"] if not np.isnan(g["weight"]) else -1, reverse=True)
    return groups


def _tier(confidence: float) -> str:
    for threshold, label, _ in CONFIDENCE_TIERS:
        if confidence >= threshold:
            return label
    return "LOW     "


def _bar(value: float, width: int = 30) -> str:
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _load_image_features(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        raw_feat  : (427,)  before scaler/PCA — for feature-group analysis
        tta_feat  : (5, 427) TTA variants — for probability averaging
    """
    from ml.preprocessor import _tta_variants, extract_features_from_pil, preprocess_leaf_image

    with image_path.open("rb") as f:
        from PIL import Image
        img = Image.open(f)
        img.load()

    base = preprocess_leaf_image(img)
    raw_feat = extract_features_from_pil(base, preprocess=False)  # (427,)

    variants = _tta_variants(base)
    tta_feat = np.stack([
        extract_features_from_pil(v, preprocess=False) for v in variants
    ])  # (5, 427)

    return raw_feat, tta_feat


def analyze_image(
    image_path: Path,
    model,
    classes: list[str],
    orig_importances: np.ndarray | None,
    *,
    verbose: bool = True,
) -> dict:
    """
    Analyze one image against all disease categories.

    Returns a dict with keys:
        image, probabilities, top_class, confidence, margin,
        tier, feature_groups, raw_feat
    """
    raw_feat, tta_feat = _load_image_features(image_path)

    # Average predict_proba across TTA variants
    avg_proba = model.predict_proba(tta_feat).mean(axis=0)  # (n_classes,)

    sorted_idx = np.argsort(avg_proba)[::-1]
    top_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx

    top_class = classes[top_idx]
    confidence = float(avg_proba[top_idx])
    margin = float(avg_proba[top_idx] - avg_proba[second_idx])
    runner_up = classes[second_idx]
    runner_up_conf = float(avg_proba[second_idx])

    probabilities = {
        classes[i]: round(float(avg_proba[i]) * 100, 2)
        for i in sorted_idx
    }

    feature_groups = _summarize_feature_groups(orig_importances)

    result = {
        "image": str(image_path),
        "top_class": top_class,
        "confidence": confidence,
        "margin": margin,
        "runner_up": runner_up,
        "runner_up_conf": runner_up_conf,
        "tier": _tier(confidence).strip(),
        "probabilities": probabilities,
        "feature_groups": feature_groups,
        "raw_feat": raw_feat,
    }

    if verbose:
        _print_image_report(result, classes, sorted_idx, avg_proba)

    return result


def _print_image_report(
    result: dict,
    classes: list[str],
    sorted_idx: np.ndarray,
    avg_proba: np.ndarray,
) -> None:
    img_name = Path(result["image"]).name
    print(f"\n{'=' * 65}")
    print(f"  IMAGE : {img_name}")
    print(f"{'=' * 65}")

    print(f"\n  Category Scores (all {len(classes)} classes, ranked):")
    print(f"  {'Category':<22} {'Score':>7}  {'Bar':32}  {'Tier'}")
    print(f"  {'-' * 22}  {'-' * 7}  {'-' * 32}  {'-' * 7}")
    for i, idx in enumerate(sorted_idx):
        cls = classes[idx]
        prob = float(avg_proba[idx])
        marker = " <-- PREDICTED" if i == 0 else (f"  (runner-up, gap={result['margin']:.3f})" if i == 1 else "")
        tier_str = _tier(prob).strip() if i == 0 else ""
        print(f"  {cls:<22}  {prob:>6.2%}  {_bar(prob, 32)}  {tier_str}{marker}")

    print(f"\n  Confidence : {result['confidence']:.2%}  [{result['tier']}]")
    print(f"  Margin vs runner-up ({result['runner_up']:<18}): {result['margin']:.4f}")

    # Feature group breakdown
    fgs = result["feature_groups"]
    valid = [g for g in fgs if not np.isnan(g["weight"])]
    if valid:
        print(f"\n  Feature Group Importances (RF->PCA projection):")
        print(f"  {'Group':<18}  {'Weight':>7}  {'Bar':24}")
        print(f"  {'-' * 18}  {'-' * 7}  {'-' * 24}")
        for g in fgs:
            w = g["weight"] if not np.isnan(g["weight"]) else 0.0
            print(f"  {g['name']:<18}  {w:>7.4f}  {_bar(w, 24)}")

    # Discrimination hint between top-2
    raw = result["raw_feat"]
    _print_discrimination_hint(raw, result["top_class"], result["runner_up"])


def _print_discrimination_hint(
    raw_feat: np.ndarray,
    top_class: str,
    runner_up: str,
) -> None:
    """
    Show which raw feature groups have the most extreme values for this image,
    giving a clue about what visual signature drove the prediction.
    """
    print(f"\n  Discriminative Feature Profile (this image's raw feature magnitudes):")
    print(f"  (High z-score = this image's feature is far from the overall mean)")

    # Z-score each group's mean absolute value (rough proxy without class centroids)
    group_means = []
    for name, start, end, _ in FEATURE_GROUPS:
        segment = raw_feat[start:end].astype(np.float64)
        group_means.append((name, float(np.abs(segment).mean())))

    all_vals = [v for _, v in group_means]
    pop_mean = np.mean(all_vals)
    pop_std = np.std(all_vals) + 1e-8

    print(f"  {'Group':<18}  {'Abs Mean':>9}  {'Z-score':>8}  Interpretation")
    print(f"  {'-' * 18}  {'-' * 9}  {'-' * 8}  {'-' * 30}")
    for name, mean_val in sorted(group_means, key=lambda x: -x[1]):
        z = (mean_val - pop_mean) / pop_std
        interp = ""
        if z > 1.0:
            interp = "strongly activated in this image"
        elif z < -1.0:
            interp = "below average — less relevant"
        print(f"  {name:<18}  {mean_val:>9.4f}  {z:>+8.3f}  {interp}")

    print(f"\n  >> Predicted: [{top_class}]  Runner-up: [{runner_up}]")
    print(f"    Review the feature groups with the highest z-scores for")
    print(f"    visual clues: why [{top_class}] was chosen over [{runner_up}].")


def _collect_images(root: Path) -> list[Path]:
    images = []
    if root.is_file():
        if root.suffix.lower() in IMAGE_SUFFIXES:
            images.append(root)
    else:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                images.append(p)
    return images


def _print_category_summary(results: list[dict], classes: list[str]) -> None:
    """Print an aggregate summary table across all analyzed images."""
    print(f"\n{'=' * 65}")
    print("  CATEGORY SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total images analyzed: {len(results)}")

    predictions = [r["top_class"] for r in results]
    pred_counts = Counter(predictions)

    tiers = Counter(r["tier"] for r in results)
    print(f"\n  Confidence tier breakdown:")
    for tier, count in sorted(tiers.items()):
        pct = count / len(results) * 100
        print(f"    {tier:<10}: {count:>4}  ({pct:>5.1f}%)")

    print(f"\n  Prediction distribution across categories:")
    print(f"  {'Category':<22}  {'Count':>6}  {'% of images':>12}  {'Avg Confidence':>15}")
    print(f"  {'-' * 22}  {'-' * 6}  {'-' * 12}  {'-' * 15}")
    for cls in sorted(classes):
        cls_results = [r for r in results if r["top_class"] == cls]
        count = len(cls_results)
        pct = count / len(results) * 100 if results else 0
        avg_conf = np.mean([r["confidence"] for r in cls_results]) if cls_results else 0.0
        print(f"  {cls:<22}  {count:>6}  {pct:>11.1f}%  {avg_conf:>14.2%}")

    avg_margins = {
        cls: np.mean([r["margin"] for r in results if r["top_class"] == cls])
        for cls in classes
        if any(r["top_class"] == cls for r in results)
    }
    if avg_margins:
        print(f"\n  Average decision margin by predicted category:")
        print(f"  (Higher margin = model is more certain and less ambiguous)")
        for cls, margin in sorted(avg_margins.items(), key=lambda x: -x[1]):
            print(f"    {cls:<22}: {margin:.4f}")

    # Cross-category confusion: where does each image's runner-up land?
    print(f"\n  Most common runner-up pairings (predicted -> confused with):")
    pairs = Counter(
        (r["top_class"], r["runner_up"])
        for r in results
        if r["top_class"] != r["runner_up"]
    )
    for (top, runner), count in pairs.most_common(8):
        print(f"    {top:<22} -> {runner:<22}: {count:>4} images")


def run(
    images: list[Path],
    *,
    summary_only: bool = False,
    out: Path | None = None,
) -> list[dict]:
    disease_model = model_path("rf_disease")
    print(f"\nLoading model from {disease_model} ...")
    model = _load_model()
    classes = _get_classes(model)
    orig_importances = _group_importance_from_rf(model)

    if orig_importances is not None:
        print(f"  Feature importance projection: OK ({len(orig_importances)}-dim)")
    else:
        print("  Feature importance projection: unavailable (model may lack RF or PCA step)")

    print(f"  Disease categories: {', '.join(classes)}")
    print(f"  Images to analyze: {len(images)}")

    results = []
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        try:
            r = analyze_image(
                img_path,
                model,
                classes,
                orig_importances,
                verbose=not summary_only,
            )
            results.append(r)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({
                "image": str(img_path),
                "error": str(exc),
                "top_class": "",
                "confidence": 0.0,
                "margin": 0.0,
                "runner_up": "",
                "runner_up_conf": 0.0,
                "tier": "ERROR",
                "probabilities": {},
                "feature_groups": [],
            })

    _print_category_summary(results, classes)

    if out is not None:
        export = [
            {k: v for k, v in r.items() if k != "raw_feat"}
            for r in results
        ]
        out.write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"\nReport saved → {out}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-image per-category disease analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image path")
    src.add_argument("--dir", type=Path, help="Directory to scan recursively")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-image detail, print only aggregate summary",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save JSON report (e.g. report.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to analyze (useful for large dirs)",
    )
    args = parser.parse_args()

    root = args.image if args.image else args.dir
    images = _collect_images(root)
    if not images:
        print(f"No images found under {root}")
        sys.exit(1)

    if args.limit and len(images) > args.limit:
        print(f"Limiting to first {args.limit} of {len(images)} images.")
        images = images[: args.limit]

    run(images, summary_only=args.summary_only, out=args.out)


if __name__ == "__main__":
    main()
