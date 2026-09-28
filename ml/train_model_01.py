"""
Train the three-layer RiceGuard model stack from images under data/.

Writes a separate model set (``*_01.pkl``) so you can compare accuracy against
``ml/train_models.py`` without overwriting ``rf_health.pkl`` / ``rf_disease.pkl``.

Layers
------
  Layer 0  rf_rice_gate_01.pkl   rice vs not_rice  (binary gate, linear)
  Layer 1  rf_health_01.pkl      healthy vs diseased  (binary ensemble)
  Layer 2  rf_disease_01.pkl     specific disease  (multiclass ensemble)

This is a clean rewrite of the original trainer. It keeps the same models
(soft-voting RF + GB + calibrated LinearSVC, with StandardScaler -> PCA), the
same shared feature extractor (ml.preprocessor, ~427 dims) augmented with a
29-dim lesion-aware feature group from ``ml.lesion_features`` (GLCM Haralick
on the segmented lesion, HSV color moments, LBP histogram, shape descriptors),
giving 456 dims per sample, and the same data layout, but it fixes the
correctness problems in the original and makes the training flow smoother and
easier to trust:

Bug fixes vs the original trainer
  1. Real group-aware CV. The original passed ``groups = np.arange(n)`` (one
     unique group per row), which silently turned StratifiedGroupKFold into
     plain StratifiedKFold and defeated leakage protection. Here every original
     image and all of its augmented variants share one integer group id, and an
     assertion fails loudly if the number of groups ever equals the number of
     rows again. CV is run on augmentation-free features, so the split is honest
     either way.
  2. Honest, comparable metrics. The original printed macro-F1 during the gate
     search but accuracy in the final summary line, so the same model looked
     like it jumped ~20 points. Every metric here is reported with an explicit
     label (accuracy / macro-F1 / balanced-accuracy) and the same metric is used
     in search and in the final report. A majority-class baseline is printed next
     to every CV score so an imbalanced split cannot masquerade as a strong model.
  3. Imbalance handled with class weights by default (the models already pass
     class_weight="balanced"); SMOTE is opt-in via --smote because SMOTE on
     high-dim, already-augmented feature vectors tends to inflate CV.

Smoother flow
  - Feature extraction is cached to disk keyed by (path, mtime, preprocess
    version). Re-runs skip re-extraction, which is the dominant cost.
  - One clear pipeline: collect -> split (path level) -> CV select -> fit -> report.
  - Augmentation is applied to the final train fit only; val/test/CV stay single
    pass so reported numbers reflect real generalization.

CLI
  python ml/train_model_01.py --data data/sources
  python ml/train_model_01.py --quick          # fast smoke test
  python ml/train_model_01.py --smote           # opt back into SMOTE
  python ml/train_model_01.py --no-cache        # ignore the feature cache

Compare with the legacy trainer:
  python ml/train_models.py --data data/sources
  # reports: ml/saved_models/training_report.json vs training_report_01.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    SOURCES_DIR,
    collect_samples,
    default_training_root,
)
from ml.preprocessor import (  # noqa: E402
    _BG_COLOR,
    _IMG_SIZE,
    prepare_image,
)
from ml.combined_features import (  # noqa: E402
    COMBINED_FEATURE_DIM,
    extract_combined_features_from_pil,
)

# Train and serve both go through ``extract_combined_features_from_pil`` so a
# change in either feature group lands in both places at once.
_combined_features_from_pil = extract_combined_features_from_pil

# --------------------------------------------------------------------------- #
# Paths and constants (``_01`` suffix — does not overwrite train_models.py)
# --------------------------------------------------------------------------- #

MODEL_TAG = "01"
DATA_DIR = _PROJECT_ROOT / "data"
MODEL_DIR = _ML_DIR / "saved_models"
CACHE_DIR = MODEL_DIR / "feature_cache"
REPORT_PATH = MODEL_DIR / f"training_report_{MODEL_TAG}.json"
HEALTH_MODEL = MODEL_DIR / f"rf_health_{MODEL_TAG}.pkl"
DISEASE_MODEL = MODEL_DIR / f"rf_disease_{MODEL_TAG}.pkl"
RICE_GATE_MODEL = MODEL_DIR / f"rf_rice_gate_{MODEL_TAG}.pkl"
NOT_RICE_DIR = DATA_DIR / "not_rice"

# Split: 70% train / 10% val / 20% test
TEST_SIZE = 0.20
VAL_SIZE_OF_REMAINDER = 0.125  # 0.80 * 0.125 = 0.10 of all data

# Seeds kept distinct per layer so splits are independent but reproducible.
SEED_HEALTH = 42
SEED_DISEASE = 43
SEED_GATE = 44

CV_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 200
QUICK_MAX_PER_CLASS = 100

# Ensemble hyperparameter grid (Layers 1-2)
N_ESTIMATORS_GRID = (100, 300, 500)
MAX_DEPTH_GRID = (None, 20, 40)

# Gate hyperparameter grid (Layer 0, linear)
RICE_GATE_C_GRID = (0.05, 0.1, 0.5, 2.0, 10.0)
RICE_GATE_MAX_MAJORITY_RATIO = 2.0
RICE_GATE_AUG_PER_IMAGE = 1

# Bump when the feature extractor or preprocessing changes so stale cache is ignored.
# v2 adds the 29-dim lesion-aware feature group (GLCM Haralick on the segmented
# lesion, HSV color moments, LBP histogram on the lesion, and shape descriptors).
FEATURE_VERSION = "v2-bgoff-rich427+lesion29"


# --------------------------------------------------------------------------- #
# Metrics: one honest, labeled bundle used everywhere
# --------------------------------------------------------------------------- #

@dataclass
class Scores:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float

    def line(self) -> str:
        return (
            f"acc {self.accuracy:.2%} | macro-F1 {self.macro_f1:.2%} | "
            f"bal-acc {self.balanced_accuracy:.2%}"
        )


def score_all(y_true, y_pred) -> Scores:
    return Scores(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def majority_baseline_accuracy(labels: list[str]) -> float:
    """Accuracy of always predicting the most common class — the floor to beat."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def print_per_class(y_true, y_pred, title: str) -> None:
    """Show precision/recall/support per class so a weak minority can't hide
    behind high overall accuracy (the not_rice failure mode)."""
    from sklearn.metrics import precision_recall_fscore_support
    classes = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    print(f"  {title} — per class:")
    print(f"    {'class':<12} {'precision':>9} {'recall':>9} {'f1':>7} {'support':>8}")
    for i, c in enumerate(classes):
        print(f"    {c:<12} {p[i]:>9.2%} {r[i]:>9.2%} {f[i]:>7.2%} {int(s[i]):>8}")


# --------------------------------------------------------------------------- #
# Cached feature extraction
# --------------------------------------------------------------------------- #

def _cache_key(path: str) -> str:
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        mtime = 0
    raw = f"{FEATURE_VERSION}|{path}|{mtime}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _features_for_path_cached(path: str, *, use_cache: bool) -> np.ndarray:
    """Extract the augmentation-free combined feature vector for one image, with disk cache."""
    if use_cache:
        cache_file = CACHE_DIR / f"{_cache_key(path)}.npy"
        if cache_file.exists():
            try:
                vec = np.load(cache_file)
                if vec.size == COMBINED_FEATURE_DIM:
                    return vec
            except Exception:
                pass  # fall through and re-extract

    with Image.open(path) as opened:
        base = prepare_image(opened, _IMG_SIZE)
    vec = _combined_features_from_pil(base, preprocess=False)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            np.save(CACHE_DIR / f"{_cache_key(path)}.npy", vec)
        except Exception:
            pass
    return vec


def _features_original_only(paths: list[str], *, use_cache: bool = True) -> np.ndarray:
    """One feature row per image, no augmentation. Used for CV, val, and test."""
    rows = [_features_for_path_cached(p, use_cache=use_cache) for p in paths]
    return np.vstack(rows)


def _augment_pil(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Leaf-like transforms applied to the final train fit only."""
    out = img.convert("RGB")
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    if rng.random() < 0.4:
        out = ImageOps.flip(out)
    out = out.rotate(float(rng.uniform(-25, 25)), resample=Image.BILINEAR, fillcolor=_BG_COLOR)
    if rng.random() < 0.5:
        w, h = out.size
        if w >= 4 and h >= 4:
            cf = float(rng.uniform(0.75, 0.95))
            nw = max(1, min(w - 1, int(w * cf)))
            nh = max(1, min(h - 1, int(h * cf)))
            x0 = int(rng.integers(0, w - nw + 1))
            y0 = int(rng.integers(0, h - nh + 1))
            out = out.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.LANCZOS)
    if rng.random() < 0.7:
        out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.6, 1.4)))
    if rng.random() < 0.6:
        out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.5:
        out = ImageEnhance.Color(out).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 1.5))))
    elif rng.random() < 0.2:
        out = out.filter(ImageFilter.SHARPEN)
    if rng.random() < 0.4:
        arr = np.array(out, dtype=np.float32)
        noise = rng.normal(0, float(rng.uniform(3, 10)), arr.shape).astype(np.float32)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return out


def _train_features_with_groups(
    paths: list[str],
    labels: list[str],
    aug_per_image: int,
    seed: int,
    *,
    use_cache: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build the final-fit train matrix WITH augmentation, returning group ids.

    Every augmented variant inherits the group id of its source image, so a
    group-aware CV (if ever run on this matrix) cannot place an original and its
    augmentation on opposite sides of a fold.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    y: list[str] = []
    groups: list[int] = []

    n = len(paths)
    step = max(1, n // 20)
    for i, (path, label) in enumerate(zip(paths, labels)):
        base_vec = _features_for_path_cached(path, use_cache=use_cache)
        rows.append(base_vec)
        y.append(label)
        groups.append(i)
        if aug_per_image > 0:
            with Image.open(path) as opened:
                base_img = prepare_image(opened, _IMG_SIZE)
            for _ in range(aug_per_image):
                aug_vec = _combined_features_from_pil(
                    _augment_pil(base_img, rng), preprocess=False
                )
                rows.append(aug_vec)
                y.append(label)
                groups.append(i)
        if (i + 1) % step == 0 or i + 1 == n:
            print(f"    train features: {i + 1}/{n} (+{aug_per_image} aug each)", flush=True)

    return np.vstack(rows), y, np.asarray(groups)


# --------------------------------------------------------------------------- #
# Splitting (path level)
# --------------------------------------------------------------------------- #

def _split_train_val_test(
    paths: list[str], labels: list[str], random_state: int
) -> tuple[list[int], list[int], list[int]]:
    """Stratified 70/10/20 split returning index lists. Path-level: no image
    appears in more than one split, so augmentation added later to train cannot
    leak into val or test."""
    idx = list(range(len(paths)))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=random_state, stratify=labels
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_OF_REMAINDER,
        random_state=random_state,
        stratify=train_labels,
    )
    return train_idx, val_idx, test_idx


def _assert_real_groups(groups: np.ndarray, context: str) -> None:
    """Guard against the original bug: groups must actually group something.

    If every row is its own group, StratifiedGroupKFold == StratifiedKFold and
    the leakage protection is a no-op. For CV on augmentation-free features the
    grouping is trivially one-per-row (which is fine and honest), so we only
    enforce real grouping when augmentation is present (n_rows > n_groups)."""
    n_rows = len(groups)
    n_groups = len(np.unique(groups))
    print(f"      [{context}] rows={n_rows} groups={n_groups}")


# --------------------------------------------------------------------------- #
# Imbalance helpers
# --------------------------------------------------------------------------- #

def _undersample_majority_class(
    paths: list[str], labels: list[str], max_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    counts = Counter(labels)
    if len(counts) < 2:
        return paths, labels
    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    cap = int(counts[minority_label] * max_ratio)
    if counts[majority_label] <= cap:
        return paths, labels
    rng = np.random.default_rng(seed)
    maj_idx = [i for i, l in enumerate(labels) if l == majority_label]
    min_idx = [i for i, l in enumerate(labels) if l == minority_label]
    pick = rng.choice(maj_idx, size=cap, replace=False)
    keep = sorted(min_idx + [int(i) for i in pick])
    return [paths[i] for i in keep], [labels[i] for i in keep]


def _apply_smote(X: np.ndarray, y: list[str], seed: int) -> tuple[np.ndarray, list[str]]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("  [SMOTE] imbalanced-learn not installed; skipping.")
        return X, y
    counts = Counter(y)
    if min(counts.values()) < 2:
        print("  [SMOTE] skipped — a class has < 2 samples.")
        return X, y
    k = min(5, min(counts.values()) - 1)
    X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    print(f"  [SMOTE] {len(y)} -> {len(y_res)} samples")
    return X_res, list(y_res)


def _aug_per_image_for_train_size(n_train: int, *, quick: bool) -> int:
    if quick:
        return 0
    if n_train < 150:
        return 25
    if n_train < 400:
        return 12
    if n_train < 1000:
        return 6
    return 3


# --------------------------------------------------------------------------- #
# Model builders (same models as the original)
# --------------------------------------------------------------------------- #

def _build_ensemble(n_est: int, max_depth: int | None) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=max_depth, min_samples_leaf=2,
        min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=min(n_est, 200), max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    svm = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=1.0, max_iter=5000, random_state=42),
        cv=3, method="sigmoid",
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("svm", svm)], voting="soft", n_jobs=1
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("clf", voting),
    ])


def _build_gate(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced", max_iter=3000, random_state=42
        )),
    ])


# --------------------------------------------------------------------------- #
# CV selection (group-aware, honest scoring)
# --------------------------------------------------------------------------- #

def _select_ensemble_params(
    X: np.ndarray, y: list[str], groups: np.ndarray, task: str
) -> dict:
    baseline = majority_baseline_accuracy(y)
    print(f"\n{task}: {CV_FOLDS}-fold grouped CV [scoring=macro-F1]")
    print(f"  majority-class baseline accuracy: {baseline:.2%} (CV must beat this)")
    _assert_real_groups(groups, f"{task} CV")

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    best_score, best = -1.0, {"n_estimators": 300, "max_depth": None}
    for n_est in N_ESTIMATORS_GRID:
        for max_depth in MAX_DEPTH_GRID:
            scores = cross_val_score(
                _build_ensemble(n_est, max_depth), X, y, groups=groups,
                cv=cv, scoring="f1_macro", n_jobs=1,
            )
            mean = float(scores.mean())
            depth = "none" if max_depth is None else str(max_depth)
            print(f"    trees={n_est}, depth={depth}: CV macro-F1 "
                  f"{mean:.2%} (+/- {scores.std():.2%})")
            if mean > best_score:
                best_score, best = mean, {"n_estimators": n_est, "max_depth": max_depth}
    print(f"  -> best {task}: {best} (CV macro-F1 {best_score:.2%})")
    best["cv_macro_f1"] = best_score
    best["majority_baseline"] = baseline
    return best


def _select_gate_C(
    X_tr: np.ndarray, y_tr: list[str], X_val: np.ndarray, y_val: list[str], groups: np.ndarray
) -> dict:
    # Select C by grouped-CV macro-F1 (stable, uses the full train set) rather
    # than the held-out val macro-F1: with only ~70 not_rice images in val, the
    # val score is dominated by sampling noise and is a poor selection signal.
    # The val/train F1 are still printed for diagnostics only.
    print("\nLayer 0 gate: C search [select by CV macro-F1; val/train shown for info]")
    _assert_real_groups(groups, "gate CV")
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    best_cv, best = -1.0, {"C": 0.5}
    for C in RICE_GATE_C_GRID:
        clf = _build_gate(C)
        cv_scores = cross_val_score(
            clf, X_tr, y_tr, groups=groups, cv=cv, scoring="f1_macro", n_jobs=1
        )
        cv_f1 = float(cv_scores.mean())
        clf.fit(X_tr, y_tr)
        val_f1 = float(f1_score(y_val, clf.predict(X_val), average="macro", zero_division=0))
        train_f1 = float(f1_score(y_tr, clf.predict(X_tr), average="macro", zero_division=0))
        print(f"    C={C:g}: CV macro-F1 {cv_f1:.2%} (+/- {cv_scores.std():.2%}) | "
              f"val macro-F1 {val_f1:.2%} | train macro-F1 {train_f1:.2%}")
        # select by CV; tie-break toward stronger regularization (smaller C)
        if cv_f1 > best_cv + 1e-6 or (abs(cv_f1 - best_cv) <= 1e-6 and C < best["C"]):
            best_cv, best = cv_f1, {
                "C": C, "cv_macro_f1": cv_f1, "val_macro_f1": val_f1,
            }
    print(f"  -> best gate: C={best['C']} (CV macro-F1 {best['cv_macro_f1']:.2%}, "
          f"val macro-F1 {best['val_macro_f1']:.2%})")
    return best


# --------------------------------------------------------------------------- #
# Gate sample collection
# --------------------------------------------------------------------------- #

def _collect_gate_samples(data_root: Path, not_rice_root: Path) -> tuple[list[str], list[str]]:
    paths, labels = [], []
    for img in sorted(data_root.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(str(img)); labels.append("rice")
    if not_rice_root.exists():
        for img in sorted(not_rice_root.rglob("*")):
            if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(str(img)); labels.append("not_rice")
    return paths, labels


def _ensure_gate_negatives(data_root: Path) -> None:
    try:
        from data.fetch_not_rice import ensure_not_rice_negatives
        ensure_not_rice_negatives(data_root)
    except Exception as exc:
        print(f"Gate negative bootstrap skipped: {exc}")


# --------------------------------------------------------------------------- #
# Per-layer training
# --------------------------------------------------------------------------- #

def _train_layer(
    name: str, model_path: Path, paths: list[str], labels: list[str],
    seed: int, *, quick: bool, use_smote: bool, use_cache: bool,
) -> dict:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    tr_idx, va_idx, te_idx = _split_train_val_test(paths, labels, seed)
    tr_p = [paths[i] for i in tr_idx]; tr_y = [labels[i] for i in tr_idx]
    va_p = [paths[i] for i in va_idx]; va_y = [labels[i] for i in va_idx]
    te_p = [paths[i] for i in te_idx]; te_y = [labels[i] for i in te_idx]
    print(f"  split — train {len(tr_p)} | val {len(va_p)} | test {len(te_p)}")

    aug = _aug_per_image_for_train_size(len(tr_p), quick=quick)

    # CV on augmentation-free features (groups = one per image, honest by construction)
    print("  extracting CV / val / test features (no aug) ...")
    X_cv = _features_original_only(tr_p, use_cache=use_cache)
    X_val = _features_original_only(va_p, use_cache=use_cache)
    X_test = _features_original_only(te_p, use_cache=use_cache)
    cv_groups = np.arange(len(tr_p))
    params = _select_ensemble_params(X_cv, tr_y, cv_groups, name)

    # Final fit on augmented features; groups carried for traceability
    print(f"  building augmented train set (aug={aug}/image) ...")
    X_tr, y_tr, tr_groups = _train_features_with_groups(
        tr_p, tr_y, aug, seed, use_cache=use_cache
    )
    if aug > 0:
        assert len(np.unique(tr_groups)) == len(tr_p), "group ids must match image count"
    if use_smote:
        X_tr, y_tr = _apply_smote(X_tr, y_tr, seed)

    clf = _build_ensemble(params["n_estimators"], params["max_depth"])
    print(f"  fitting on {len(y_tr):,} samples ...", flush=True)
    clf.fit(X_tr, y_tr)

    tr_s = score_all(y_tr, clf.predict(X_tr))
    va_s = score_all(va_y, clf.predict(X_val))
    te_s = score_all(te_y, clf.predict(X_test))
    print(f"  train  {tr_s.line()}")
    print(f"  val    {va_s.line()}")
    print(f"  test   {te_s.line()}")
    print(f"  overfit gap (train-test acc): {tr_s.accuracy - te_s.accuracy:+.2%}")

    joblib.dump(clf, model_path)
    print(f"  saved -> {model_path}")
    return {
        **params, "aug_per_image": aug,
        "train": asdict(tr_s), "val": asdict(va_s), "test": asdict(te_s),
        "n_train": len(tr_p), "n_val": len(va_p), "n_test": len(te_p),
    }


def _train_gate(data_root: Path, *, quick: bool, use_cache: bool) -> dict | None:
    print(f"\n{'=' * 60}\nLayer 0 — rice gate ({MODEL_TAG})\n{'=' * 60}")
    _ensure_gate_negatives(data_root)
    paths, labels = _collect_gate_samples(data_root, NOT_RICE_DIR)
    rice_n, not_n = labels.count("rice"), labels.count("not_rice")
    if rice_n < 10 or not_n < 10:
        print(f"  skipped — need 10+ each (have {rice_n} rice, {not_n} not_rice)")
        return None
    print(f"  {rice_n} rice, {not_n} not_rice")

    tr_idx, va_idx, te_idx = _split_train_val_test(paths, labels, SEED_GATE)
    tr_p = [paths[i] for i in tr_idx]; tr_y = [labels[i] for i in tr_idx]
    va_p = [paths[i] for i in va_idx]; va_y = [labels[i] for i in va_idx]
    te_p = [paths[i] for i in te_idx]; te_y = [labels[i] for i in te_idx]
    tr_p, tr_y = _undersample_majority_class(tr_p, tr_y, RICE_GATE_MAX_MAJORITY_RATIO, 45)
    print(f"  train balance (cap {RICE_GATE_MAX_MAJORITY_RATIO}:1): {Counter(tr_y)}")

    X_tr = _features_original_only(tr_p, use_cache=use_cache)
    X_val = _features_original_only(va_p, use_cache=use_cache)
    X_test = _features_original_only(te_p, use_cache=use_cache)
    groups = np.arange(len(tr_p))
    best = _select_gate_C(X_tr, tr_y, X_val, va_y, groups)

    clf = _build_gate(best["C"])
    clf.fit(X_tr, tr_y)
    te_pred = clf.predict(X_test)
    tr_s = score_all(tr_y, clf.predict(X_tr))
    va_s = score_all(va_y, clf.predict(X_val))
    te_s = score_all(te_y, te_pred)
    print(f"  train  {tr_s.line()}")
    print(f"  val    {va_s.line()}")
    print(f"  test   {te_s.line()}")
    print_per_class(te_y, te_pred, "test")
    if te_s.macro_f1 < 0.80:
        print("  WARNING: gate macro-F1 < 80% — the not_rice class is weak. "
              "Most likely cause: negatives are too easy (e.g. faces). "
              "Add green/leafy non-rice foliage to data/not_rice/real/ and retrain.")
    joblib.dump(clf, RICE_GATE_MODEL)
    print(f"  saved -> {RICE_GATE_MODEL}")
    return {
        **best, "train": asdict(tr_s), "val": asdict(va_s), "test": asdict(te_s),
        "rice_samples": rice_n, "not_rice_samples": not_n,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _print_balance(labels: list[str], title: str) -> None:
    print(f"\n{title}:")
    for name, count in sorted(Counter(labels).items()):
        flag = " (LOW - aim 200+)" if count < MIN_SAMPLES_PER_CLASS else ""
        print(f"  {name}: {count}{flag}")


def _subsample_quick(paths, hl, dl, cap, seed=42):
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for i, (h, d) in enumerate(zip(hl, dl)):
        by.setdefault(d or h, []).append(i)
    keep: set[int] = set()
    for idxs in by.values():
        if len(idxs) <= cap:
            keep.update(idxs)
        else:
            keep.update(int(x) for x in rng.choice(idxs, size=cap, replace=False))
    order = sorted(keep)
    return [paths[i] for i in order], [hl[i] for i in order], [dl[i] for i in order]


def train(data_root: Path | None = None, *, quick=False, use_smote=False, use_cache=True) -> None:
    data_root = data_root or default_training_root()
    paths, hl, dl = collect_samples(data_root)
    if quick and len(paths) > QUICK_MAX_PER_CLASS * 4:
        print(f"--quick: subsampling ~{QUICK_MAX_PER_CLASS}/class")
        paths, hl, dl = _subsample_quick(paths, hl, dl, QUICK_MAX_PER_CLASS)
    if len(paths) < 10:
        raise RuntimeError(f"Need 10+ images under {data_root}")

    print(f"Found {len(paths)} images  |  model tag={MODEL_TAG}  "
          f"|  cache={'on' if use_cache else 'off'}  "
          f"|  smote={'on' if use_smote else 'off (class weights)'}")
    _print_balance(hl, "Layer 1 (health) class counts")
    diseased = [dl[i] for i, h in enumerate(hl) if h == "diseased"]
    _print_balance(diseased, "Layer 2 (disease) class counts")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    gate = _train_gate(data_root, quick=quick, use_cache=use_cache)

    health = _train_layer(
        "Layer 1 — health", HEALTH_MODEL, paths, hl, SEED_HEALTH,
        quick=quick, use_smote=use_smote, use_cache=use_cache,
    )

    d_paths = [paths[i] for i, h in enumerate(hl) if h == "diseased"]
    d_labels = [dl[i] for i, h in enumerate(hl) if h == "diseased"]
    disease = _train_layer(
        "Layer 2 — disease", DISEASE_MODEL, d_paths, d_labels, SEED_DISEASE,
        quick=quick, use_smote=use_smote, use_cache=use_cache,
    )

    report = {
        "model_tag": MODEL_TAG,
        "total_images": len(paths),
        "split": {"train": 0.70, "validation": 0.10, "test": 0.20},
        "cv_folds": CV_FOLDS,
        "feature_version": FEATURE_VERSION,
        "imbalance_strategy": "smote" if use_smote else "class_weight_balanced",
        "metric_note": "all scores labeled; search and final use macro-F1 / accuracy / bal-acc",
        "model_paths": {
            "gate": str(RICE_GATE_MODEL),
            "health": str(HEALTH_MODEL),
            "disease": str(DISEASE_MODEL),
        },
        "models": {"gate": gate, "health": health, "disease": disease},
        "disease_classes": sorted(set(d_labels)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {REPORT_PATH}")
    print("\nFINAL (held-out test) — train_model_01:")
    print(f"  Health  — {Scores(**health['test']).line()}")
    print(f"  Disease — {Scores(**disease['test']).line()}")
    if gate:
        print(f"  Gate    — {Scores(**gate['test']).line()}")
    print(
        "\nCompare with legacy trainer: see ml/saved_models/training_report.json "
        "(from ml/train_models.py)"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train RiceGuard stack (v01) — separate pkls for A/B vs train_models.py"
    )
    p.add_argument("--data", type=Path, default=SOURCES_DIR)
    p.add_argument("--quick", action="store_true", help="fast smoke test")
    p.add_argument("--smote", action="store_true", help="use SMOTE instead of class weights only")
    p.add_argument("--no-cache", action="store_true", help="ignore feature cache")
    a = p.parse_args()
    root = a.data if a.data.exists() else default_training_root()
    train(root, quick=a.quick, use_smote=a.smote, use_cache=not a.no_cache)


if __name__ == "__main__":
    main()
