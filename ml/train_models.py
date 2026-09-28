"""
Train BOTH model sets from images under leaf/ (preferred) or data/.

One command trains everything:
  STAGE 1 — the fallback RandomForest models (rf_health/rf_disease/rf_rice_gate).
  STAGE 2 — the neural-network CNN heads (cnn_*), via ml/train_cnn.py.
The app uses the CNN at inference and falls back to the RF models (ML_ENGINE=auto).
Use --no-cnn to train only the RF models; the CNN stage auto-skips if torch is absent.

Expected layout (any depth under the data root):
  leaf/data_1|data_2|data_3/<class_name>/.../*.jpg
  leaf/not_rice/.../*.jpg   (Layer 0 negatives)
  or data/sources/<dataset>/<class_name>/.../*.jpg
  or data/raw/<class_name>/*.jpg

Layer 1 (health): folders named "healthy" vs everything else (diseased).
Layer 2 (disease): only diseased-class folders; labels match folder names.

Enhancements over baseline RF:
  - Rich features via ml.preprocessor (~427 dims: color, LBP, GLCM, HOG, edges)
  - Soft-voting ensemble centred on Random Forest:
      RandomForest (weight 2) + ExtraTrees + GradientBoosting + calibrated LinearSVC
  - StandardScaler → PCA(95% variance) before classification
  - SMOTE on final train fit only (Layers 1–2); Layer 0 gate skips SMOTE
  - Stronger augmentation: crop, colour jitter, blur/sharpen, Gaussian noise

RF-specific tuning (Layers 1–2):
  - Hyperparameter CV searches RF alone (fast) over n_estimators × max_depth
    × min_samples_leaf, scored with macro-F1 (handles imbalanced disease classes
    better than plain accuracy).
  - The tuned RF settings are reused inside the final voting ensemble, where RF
    carries 2× the voting weight to centre predictions on the strongest tree
    learner while still benefiting from ExtraTrees / GB / SVM disagreement.
  - Post-fit: per-feature-block importance (color, LBP, GLCM, HOG, edges) is
    projected back from PCA components and printed for interpretability.

Default run trains once on the full dataset (duration scales with image count):
  - 70% train / 10% validation / 20% test (stratified)
  - 5-fold StratifiedGroupKFold CV on path-level features (no aug/SMOTE leakage)
  - Augmentation per image scales with train-set size (more aug when data is scarce)
  - On-the-fly augmentation on the final train fit only (val/test stay single-pass)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    RAW_DIR,
    SOURCES_DIR,
    STRESS_CLASS,
    collect_samples,
    count_images,
    default_training_root,
    list_health_extra_positives,
    list_stress_extra_positives,
    resolve_not_rice_dir,
)
from ml.preprocessor import (  # noqa: E402
    _BG_COLOR,
    _IMG_SIZE,
    extract_features,
    extract_features_from_pil,
    prepare_image,
    prepare_image_raw,
)

DATA_DIR = _PROJECT_ROOT / "data"
MODEL_DIR = _ML_DIR / "saved_models"
REPORT_PATH = MODEL_DIR / "training_report.json"
HEALTH_MODEL = MODEL_DIR / "rf_health.pkl"
DISEASE_MODEL = MODEL_DIR / "rf_disease.pkl"
RICE_GATE_MODEL = MODEL_DIR / "rf_rice_gate.pkl"
NOT_RICE_DIR = DATA_DIR / "not_rice"  # legacy fallback; prefer resolve_not_rice_dir()

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/486/rice+leaf+diseases.zip"
UCI_ZIP_PATH = DATA_DIR / "rice_leaf_diseases.zip"

DISEASE_LABEL_MAP = {
    "bacterial leaf blight": "bacterial blight",
    "brown spot": "brown spot",
    "leaf smut": "leaf smut",
}

# 70% train, 10% validation, 20% test
TEST_SIZE = 0.20
VAL_SIZE_OF_REMAINDER = 0.125  # 0.8 * 0.125 = 0.10 of all data

MIN_SAMPLES_PER_CLASS = 200
QUICK_MAX_PER_CLASS = 100  # smoke-test cap when --quick (full data kept on disk)
N_ESTIMATORS_GRID = (200, 400, 600)
MAX_DEPTH_GRID = (None, 20, 40)
# RF-specific regularization grid — biggest lever for overfit control after depth.
# leaf=1 = high-variance / leaf=5 = strong smoothing; mid value is usually best
# but it depends on dataset noise, so we let CV decide.
MIN_SAMPLES_LEAF_GRID = (1, 3, 5)
RF_CV_SCORING = "f1_macro"  # rewards minority-disease recall, not just majority hits
RF_VOTING_WEIGHT = 2  # RF gets 2x weight in the final soft-vote ensemble
RICE_GATE_C_GRID = (0.05, 0.1, 0.5, 2.0, 10.0)
RICE_GATE_MAX_MAJORITY_RATIO = 2.0
RICE_GATE_AUG_PER_IMAGE = 1
CV_FOLDS = 5

# Probability calibration for the Layer 1 / Layer 2 voting ensembles.
# The soft-voting RF+ET+GB+SVM ensemble is a strong classifier but a poorly
# *calibrated* one: its predict_proba is diffuse, so even correct predictions
# report middling confidence. We fit a Platt (sigmoid) calibrator on the clean
# held-out validation split (no aug, no SMOTE) so the confidence numbers the
# pipeline gates on actually reflect real single-pass accuracy. Sigmoid (2
# parameters) is deliberately gentler than isotonic — it sharpens confidence
# without over-committing to the tiny validation set.
CALIBRATE_ENSEMBLE = True
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_MIN_PER_CLASS = 20  # need enough val samples per class to fit reliably

# Feature-block layout in the 427-dim vector produced by ml.preprocessor.
# Used to summarize which signal types the trained RF actually relies on.
_FEATURE_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("color (RGB+HSV+LAB)", 0, 27),
    ("LBP texture",         27, 45),
    ("GLCM texture",        45, 85),
    ("HOG shape",           85, 409),
    ("edge energy",         409, 427),
)


def _download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


def _ensure_uci_dataset() -> None:
    if count_images(RAW_DIR) > 0:
        return

    if not UCI_ZIP_PATH.exists() or not zipfile.is_zipfile(UCI_ZIP_PATH):
        if UCI_ZIP_PATH.exists():
            UCI_ZIP_PATH.unlink()
        _download_file(UCI_ZIP_URL, UCI_ZIP_PATH)

    if not zipfile.is_zipfile(UCI_ZIP_PATH):
        raise RuntimeError(
            f"Downloaded file is not a valid zip: {UCI_ZIP_PATH}. "
            "Delete it and run this script again."
        )

    extract_root = DATA_DIR / "uci_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {UCI_ZIP_PATH} ...")
    with zipfile.ZipFile(UCI_ZIP_PATH, "r") as zf:
        zf.extractall(extract_root)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    healthy_dir = RAW_DIR / "healthy"
    healthy_dir.mkdir(exist_ok=True)

    for class_dir in extract_root.rglob("*"):
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            continue

        class_name = class_dir.name.strip().lower()
        if class_name == "healthy":
            target = healthy_dir
        else:
            label = DISEASE_LABEL_MAP.get(class_name, class_name)
            target = RAW_DIR / label
            target.mkdir(exist_ok=True)

        for img in images:
            shutil.copy2(img, target / img.name)

    _ensure_healthy_samples(healthy_dir)


def _ensure_healthy_samples(healthy_dir: Path) -> None:
    existing = list(healthy_dir.glob("*"))
    if existing:
        return

    print("No healthy images found — generating synthetic healthy samples ...")
    rng = np.random.default_rng(42)
    for i in range(40):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 1] = rng.integers(120, 200, size=(64, 64))
        arr[:, :, 0] = rng.integers(40, 100, size=(64, 64))
        arr[:, :, 2] = rng.integers(30, 80, size=(64, 64))
        Image.fromarray(arr).save(healthy_dir / f"synthetic_healthy_{i:03d}.jpg")


def _collect_gate_samples(
    data_root: Path,
    not_rice_root: Path,
) -> tuple[list[str], list[str]]:
    """Layer 0: rice (all class folders) vs not_rice negatives."""
    paths: list[str] = []
    labels: list[str] = []
    not_rice_resolved = not_rice_root.resolve() if not_rice_root.exists() else None

    for img_path in sorted(data_root.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "not_rice" in img_path.parts:
            continue
        if not_rice_resolved is not None:
            try:
                img_path.resolve().relative_to(not_rice_resolved)
                continue
            except ValueError:
                pass
        paths.append(str(img_path))
        labels.append("rice")

    if not_rice_root.exists():
        for img_path in sorted(not_rice_root.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            paths.append(str(img_path))
            labels.append("not_rice")

    return paths, labels


def _ensure_gate_negatives(data_root: Path) -> None:
    """
    Use the not_rice negatives already on disk under the training root
    (e.g. leaf/not_rice/). No external fetch — training runs entirely off leaf/.
    """
    not_rice_root = resolve_not_rice_dir(data_root)
    if count_images(not_rice_root) > 0:
        return
    print(
        f"No not_rice negatives found under {not_rice_root} — "
        "Layer 0 rice gate will be skipped. "
        "Add images under leaf/not_rice/ to train it."
    )


def _train_rice_gate(data_root: Path, quick: bool = False) -> dict | None:
    """
    Train Layer 0 binary rice-leaf gate; returns metrics dict or None.

    The gate is trained on **raw** features (``preprocess=False``) — no luma
    normalization, no background removal. This matches the inference path in
    ``layer0_validator`` so the gate cannot get an artificial accuracy boost
    from enhancement that wouldn't run on real uploads.
    """
    _ensure_gate_negatives(data_root)
    not_rice_root = resolve_not_rice_dir(data_root)
    paths, labels = _collect_gate_samples(data_root, not_rice_root)
    rice_count = labels.count("rice")
    not_rice_count = labels.count("not_rice")
    if rice_count < 10 or not_rice_count < 10:
        print(
            f"Layer 0 skipped — need 10+ rice and 10+ not_rice "
            f"(have {rice_count} rice, {not_rice_count} not_rice)."
        )
        return None

    print(
        f"\nLayer 0 — rice gate: {rice_count} rice, {not_rice_count} not_rice "
        f"(training on RAW features — no enhancement, matches inference)"
    )
    if quick and len(paths) > QUICK_MAX_PER_CLASS * 2:
        rng = np.random.default_rng(44)
        by_label: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            by_label.setdefault(label, []).append(i)
        keep: set[int] = set()
        for indices in by_label.values():
            cap = min(len(indices), QUICK_MAX_PER_CLASS)
            pick = rng.choice(indices, size=cap, replace=False)
            keep.update(int(x) for x in pick)
        order = sorted(keep)
        paths = [paths[i] for i in order]
        labels = [labels[i] for i in order]

    # Legacy random split: the gate is rice-vs-not_rice (not per-disease), not_rice
    # is single-source, and CV already uses StratifiedGroupKFold — source-grouping
    # here would just force the not_rice fallback without adding honesty.
    train_idx, val_idx, test_idx = _split_train_val_test(
        paths, labels, 44, strategy="random"
    )
    train_paths = [paths[i] for i in train_idx]
    train_y = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_y = [labels[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]
    test_y = [labels[i] for i in test_idx]

    train_paths, train_y = _undersample_majority_class(
        train_paths, train_y, RICE_GATE_MAX_MAJORITY_RATIO, 45
    )
    print(
        f"  Gate train balance (cap {RICE_GATE_MAX_MAJORITY_RATIO}:1): "
        f"{Counter(train_y)}"
    )

    aug = 0 if quick else RICE_GATE_AUG_PER_IMAGE
    # preprocess=False keeps train and serve aligned on the RAW upload pixels
    # — see the module docstring of layer0_validator.py.
    X_train_paths = _features_original_only(train_paths, preprocess=False)
    X_val = _features_original_only(val_paths, preprocess=False)
    X_test = _features_original_only(test_paths, preprocess=False)

    params = _select_rice_gate_hyperparameters(
        X_train_paths,
        train_y,
        X_val,
        val_y,
        groups=np.arange(len(train_paths)),
    )

    X_train, y_train = _features_for_paths(
        train_paths, train_y, aug, 45, preprocess=False
    )
    clf, train_acc, val_acc, test_acc = _fit_and_score(
        params,
        X_train,
        y_train,
        X_val,
        val_y,
        X_test,
        test_y,
        builder=_build_rice_gate_classifier,
    )
    joblib.dump(clf, RICE_GATE_MODEL)
    gap = train_acc - test_acc
    print(
        f"Rice gate — train {train_acc:.2%} | val {val_acc:.2%} | test {test_acc:.2%} "
        f"(overfit gap train-test: {gap:.2%}) -> {RICE_GATE_MODEL}"
    )
    if test_acc >= 0.995 and gap < 0.02:
        print(
            "  Tip: add real-world rejects under leaf/not_rice/ (or data/not_rice/) "
            "(soil, tools, other crops) "
            "if uploads are still misclassified in production."
        )
    return {
        **params,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "rice_samples": rice_count,
        "not_rice_samples": not_rice_count,
        "classifier": "LogisticRegression",
        # Marker: the gate uses raw uploads at both train and serve time.
        # If you ever rebuild the feature pipeline, keep this contract.
        "input_preprocessing": "raw (RGB + letterbox resize only — no enhancement)",
        "enhancement_at_inference": False,
    }


def _print_class_balance(labels: list[str], title: str) -> None:
    counts = Counter(labels)
    print(f"\n{title}:")
    for name, count in sorted(counts.items()):
        flag = " (LOW - aim for 200+)" if count < MIN_SAMPLES_PER_CLASS else ""
        print(f"  {name}: {count}{flag}")


def _aug_per_image_for_train_size(n_train: int, *, quick: bool) -> int:
    """Scale augmentation with train split size (fewer images → more variants per file)."""
    if quick:
        return 0
    if n_train < 150:
        return 25
    if n_train < 400:
        return 12
    if n_train < 1000:
        return 6
    return 3


# Dataset source folders (leaf/data_1 … data_6). Used to keep whole sources on
# one side of the train/test split so the reported test accuracy is not inflated
# by same-source leakage.
_SOURCE_FOLDERS = ("data_1", "data_2", "data_3", "data_4", "data_5", "data_6")


def _source_of(path: str) -> str:
    """Which dataset source (data_1…data_6) a path belongs to; else 'other'."""
    parts = Path(path).parts
    for s in _SOURCE_FOLDERS:
        if s in parts:
            return s
    return "other"


def _random_split_indices(
    idx: list[int],
    labels: list[str],
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified random image-level split (legacy behaviour)."""
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=labels,
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_OF_REMAINDER,
        random_state=random_state,
        stratify=train_labels,
    )
    return train_idx, val_idx, test_idx


def _stratified_by_source_indices(
    idx: list[int],
    paths: list[str],
    labels: list[str],
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Source-stratified split: every (source, class) group is split ~70/10/20, so
    the model TRAINS ON ALL SOURCES and is TESTED on a held-out slice drawn from
    all of them. After de-duplication no image appears in more than one split, so
    this is a legitimate (leakage-free) in-distribution estimate — the primary
    accuracy for the study. It does NOT measure generalisation to a brand-new
    source; ml/cross_source_eval.py reports that separately as the limitation.
    """
    rng = np.random.default_rng(random_state)
    groups: dict[tuple[str, str], list[int]] = {}
    for i in idx:
        groups.setdefault((_source_of(paths[i]), labels[i]), []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for key in sorted(groups):
        members = list(groups[key])
        rng.shuffle(members)
        n = len(members)
        n_test = int(round(n * TEST_SIZE))
        # Keep tiny groups learnable: don't let test/val swallow a whole group.
        n_test = min(n_test, max(0, n - 1))
        rest = members[n_test:]
        n_val = int(round(len(rest) * VAL_SIZE_OF_REMAINDER))
        n_val = min(n_val, max(0, len(rest) - 1))
        test_idx += members[:n_test]
        val_idx += rest[:n_val]
        train_idx += rest[n_val:]
    return train_idx, val_idx, test_idx


def _grouped_by_source_indices(
    paths: list[str],
    labels: list[str],
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Leave-whole-source-out split (diagnostic / pessimistic): per class the single
    LARGEST source trains and every OTHER source becomes the test set, so the test
    measures generalisation to sources the model never saw. Used for the honest
    cross-source limitation figure, not the deployed model.
    """
    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    fallback_classes: list[str] = []

    by_class: dict[str, list[int]] = {}
    for i in range(len(paths)):
        by_class.setdefault(labels[i], []).append(i)

    for label in sorted(by_class):
        members = by_class[label]
        by_source: dict[str, list[int]] = {}
        for i in members:
            by_source.setdefault(_source_of(paths[i]), []).append(i)

        if len(by_source) < 2:
            fallback_classes.append(label)
            tr, va, te = _random_split_indices(members, [label] * len(members), random_state)
            train_idx += tr
            val_idx += va
            test_idx += te
            continue

        ordered = sorted(by_source, key=lambda s: len(by_source[s]), reverse=True)
        train_pool = list(by_source[ordered[0]])
        for s in ordered[1:]:
            test_idx += by_source[s]
        rng.shuffle(train_pool)
        n_val = max(1, int(round(len(train_pool) * VAL_SIZE_OF_REMAINDER)))
        n_val = min(n_val, len(train_pool) - 1)
        val_idx += train_pool[:n_val]
        train_idx += train_pool[n_val:]

    if fallback_classes:
        print(
            "  split: source-grouped (largest source→train, other sources→test); "
            f"single-source (fallback→random): {', '.join(sorted(fallback_classes))}"
        )
    else:
        print("  split: source-grouped (largest source→train, other sources→test)")
    return train_idx, val_idx, test_idx


def _split_train_val_test(
    paths: list[str],
    labels: list[str],
    random_state: int,
    *,
    strategy: str = "source_stratified",
) -> tuple[list[int], list[int], list[int]]:
    """
    Return index lists for train (~70%), val (~10%), test (~20%).

    strategy:
      - ``source_stratified`` (default): split each (source, class) group ~70/10/20
        so the model trains on ALL sources and is tested on a leakage-free held-out
        slice from all of them. This is the deployed model's honest in-distribution
        estimate (see _stratified_by_source_indices).
      - ``grouped``: leave whole sources out for a cross-source generalisation test
        (pessimistic diagnostic — see _grouped_by_source_indices).
      - ``random``: legacy stratified-by-class random split (used by the rice gate).

    IMPORTANT: train_models.py and evaluate_models.py must call this with the same
    strategy + seeds so their test sets match.
    """
    idx = list(range(len(paths)))
    if strategy == "random":
        return _random_split_indices(idx, labels, random_state)
    if strategy == "grouped":
        return _grouped_by_source_indices(paths, labels, random_state)
    if strategy == "source_stratified":
        return _stratified_by_source_indices(idx, paths, labels, random_state)
    raise ValueError(f"unknown split strategy: {strategy!r}")


def _augment_pil(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Stronger leaf-like transforms; same feature pipeline as inference."""
    out = img.convert("RGB")
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    if rng.random() < 0.4:
        out = ImageOps.flip(out)
    angle = float(rng.uniform(-25, 25))
    out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=_BG_COLOR)
    if rng.random() < 0.5:
        w, h = out.size
        if w >= 4 and h >= 4:
            crop_frac = float(rng.uniform(0.75, 0.95))
            new_w = max(1, min(w - 1, int(w * crop_frac)))
            new_h = max(1, min(h - 1, int(h * crop_frac)))
            x0 = int(rng.integers(0, w - new_w + 1))
            y0 = int(rng.integers(0, h - new_h + 1))
            out = out.crop((x0, y0, x0 + new_w, y0 + new_h)).resize((w, h), Image.LANCZOS)
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


def _features_for_paths(
    paths: list[str],
    labels: list[str],
    aug_per_image: int,
    seed: int,
    *,
    preprocess: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix; aug_per_image extra variants per file (train only).

    Set ``preprocess=False`` to skip the gentle luminance normalization /
    background-removal step — used by Layer 0 (rice gate) training so the
    gate is trained and served on raw uploads (no enhancement).
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    y: list[str] = []

    # When preprocess=True we use prepare_image (luma-normalize + letterbox).
    # When preprocess=False we use prepare_image_raw (letterbox only) so the
    # augmentations and the served raw upload have matched distributions.
    base_prep = prepare_image if preprocess else prepare_image_raw

    n_paths = len(paths)
    for i, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as opened:
            base = base_prep(opened, _IMG_SIZE)
        rows.append(extract_features_from_pil(base, preprocess=False).ravel())
        y.append(label)
        for _ in range(aug_per_image):
            rows.append(extract_features_from_pil(_augment_pil(base, rng), preprocess=False).ravel())
            y.append(label)
        step = max(1, n_paths // 20)
        if (i + 1) % step == 0 or i + 1 == n_paths:
            print(f"  features: {i + 1}/{n_paths} (+{aug_per_image} aug each)", flush=True)

    return np.vstack(rows), y


def _features_original_only(paths: list[str], *, preprocess: bool = True) -> np.ndarray:
    """
    Path-level features for CV / val / test (no augmentation, no SMOTE).

    ``preprocess=False`` is the gate path — see ``_features_for_paths``.
    """
    rows = [extract_features(p, preprocess=preprocess).ravel() for p in paths]
    return np.vstack(rows)


def _undersample_majority_class(
    paths: list[str],
    labels: list[str],
    max_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Cap majority-class count to max_ratio × minority (path-level, before aug)."""
    counts = Counter(labels)
    if len(counts) < 2:
        return paths, labels

    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    minority_n = counts[minority_label]
    cap = int(minority_n * max_ratio)
    if counts[majority_label] <= cap:
        return paths, labels

    rng = np.random.default_rng(seed)
    maj_idx = [i for i, label in enumerate(labels) if label == majority_label]
    min_idx = [i for i, label in enumerate(labels) if label == minority_label]
    pick = rng.choice(maj_idx, size=cap, replace=False)
    keep = sorted(min_idx + list(int(i) for i in pick))
    return [paths[i] for i in keep], [labels[i] for i in keep]


def _apply_smote(X: np.ndarray, y: list[str], random_state: int = 42) -> tuple[np.ndarray, list[str]]:
    """
    SMOTEENN: oversample minority classes with SMOTE, then remove ambiguous
    boundary samples with Edited Nearest Neighbours. This reduces both
    class imbalance bias and noisy decision-boundary samples that cause
    overfitting. Falls back to plain SMOTE if SMOTEENN is unavailable.
    """
    try:
        from imblearn.over_sampling import SMOTE

        counts = Counter(y)
        min_count = min(counts.values())
        if min_count < 2:
            print("  [SMOTEENN] skipped — a class has < 2 samples.")
            return X, y
        k = min(5, min_count - 1)
        smote = SMOTE(random_state=random_state, k_neighbors=k)

        try:
            from imblearn.combine import SMOTEENN

            resampler = SMOTEENN(smote=smote, random_state=random_state)
            X_res, y_res = resampler.fit_resample(X, y)
            print(f"  [SMOTEENN] {len(y)} -> {len(y_res)} samples "
                  f"(oversampled + boundary cleaned)")
        except Exception:
            X_res, y_res = smote.fit_resample(X, y)
            print(f"  [SMOTE fallback] {len(y)} -> {len(y_res)} samples")

        return X_res, list(y_res)
    except ImportError:
        print("  [SMOTE] imbalanced-learn not installed; skipping.")
        return X, y


def _build_rice_gate_classifier(n_est: int, max_depth: int | None, *, C: float = 0.5) -> Pipeline:
    """Strongly regularized linear gate (n_est/max_depth kept for API compatibility)."""
    del n_est, max_depth
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


def _select_rice_gate_hyperparameters(
    X_train: np.ndarray,
    y_train: list[str],
    X_val: np.ndarray,
    y_val: list[str],
    *,
    groups: np.ndarray,
) -> dict:
    """Pick C by held-out val F1 (not inflated CV on easy negatives)."""
    best_val_f1 = -1.0
    best: dict = {"C": 0.5, "cv_f1": 0.0, "val_f1": 0.0}

    print("\nRice gate: hyperparameter search [val F1 + grouped CV] ...")
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for C in RICE_GATE_C_GRID:
        clf = _build_rice_gate_classifier(0, None, C=C)
        cv_scores = cross_val_score(
            clf,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        cv_f1 = float(cv_scores.mean())
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        train_f1 = float(f1_score(y_train, clf.predict(X_train), average="macro", zero_division=0))
        print(
            f"  C={C:g}: CV F1 {cv_f1:.2%} (+/- {cv_scores.std():.2%}) | "
            f"val F1 {val_f1:.2%} | train F1 {train_f1:.2%}"
        )
        better_val = val_f1 > best_val_f1 + 1e-6
        tie_val = abs(val_f1 - best_val_f1) <= 1e-6
        prefer = better_val or (tie_val and C < best.get("C", float("inf")))
        if prefer:
            best_val_f1 = val_f1
            best = {
                "C": C,
                "cv_f1": cv_f1,
                "val_f1": val_f1,
                "n_estimators": 0,
                "max_depth": None,
            }

    print(f"  -> best Rice gate: C={best['C']} (val F1 {best['val_f1']:.2%}, CV F1 {best['cv_f1']:.2%})")
    best["cv_accuracy"] = best["cv_f1"]
    return best


def _build_random_forest(
    n_est: int,
    max_depth: int | None,
    *,
    min_samples_leaf: int = 3,
) -> RandomForestClassifier:
    """Centralised RF config so CV-tuning and final ensemble stay in sync."""
    # Tie min_samples_split to leaf so increasing leaf also tightens splits;
    # a too-small split lets the tree carve noise even with a larger leaf bound.
    min_split = max(8, 2 * min_samples_leaf + 2)
    return RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,   # CV-tuned (1/3/5) — biggest lever vs overfit
        min_samples_split=min_split,
        max_features="sqrt",                 # sqrt(n_features) per split — classic RF
        class_weight="balanced_subsample",   # rebalances each bootstrap independently
        oob_score=True,                      # free held-out estimate per tree
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )


def _build_rf_pipeline(
    n_est: int,
    max_depth: int | None,
    *,
    min_samples_leaf: int = 3,
) -> Pipeline:
    """RF-only pipeline used during hyperparameter CV (fast, no ensemble overhead)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("clf", _build_random_forest(n_est, max_depth, min_samples_leaf=min_samples_leaf)),
    ])


def _build_ensemble(
    n_est: int,
    max_depth: int | None,
    *,
    min_samples_leaf: int = 3,
) -> Pipeline:
    """
    StandardScaler → PCA(95%) → soft-voting RF + ExtraTrees + GB + calibrated SVM.

    RF carries 2x voting weight: it is the strongest single learner on these
    leaf features and we want it to dominate when the other estimators
    disagree, while still benefiting from their independent error patterns.
    ExtraTrees adds tree-level diversity (random thresholds) which reduces
    correlation between trees and helps separate visually similar diseases
    (e.g. brown spot vs leaf smut).
    """
    rf = _build_random_forest(n_est, max_depth, min_samples_leaf=min_samples_leaf)
    et = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=max(8, 2 * min_samples_leaf + 2),
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=False,             # ExtraTrees: full sample + random thresholds
        random_state=43,             # different seed → different randomness from RF
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=min(n_est, 200),
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=max(3, min_samples_leaf),  # follow RF regularization floor
        min_samples_split=max(8, 2 * min_samples_leaf + 2),
        random_state=42,
    )
    # LinearSVC scales to SMOTE-sized train sets; RBF SVC on 30k+ rows OOMs or hangs.
    svm_base = LinearSVC(
        class_weight="balanced",
        C=0.5,                     # slightly more regularized than C=1.0
        max_iter=5000,
        random_state=42,
    )
    svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")
    voting = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb), ("svm", svm)],
        voting="soft",
        weights=[RF_VOTING_WEIGHT, 1, 1, 1],   # emphasize RF in the final probability
        n_jobs=1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ("clf", voting),
    ])


def _select_hyperparameters(
    X: np.ndarray,
    y: list[str],
    task_name: str,
    *,
    groups: np.ndarray | None = None,
    builder=_build_rf_pipeline,
    n_estimators_grid=N_ESTIMATORS_GRID,
    max_depth_grid=MAX_DEPTH_GRID,
    min_samples_leaf_grid: tuple[int, ...] = MIN_SAMPLES_LEAF_GRID,
    scoring: str = RF_CV_SCORING,
) -> dict:
    """
    5-fold CV on one row per image; optional group splits prevent aug leakage.

    Tunes Random Forest hyperparameters (n_estimators × max_depth × min_samples_leaf)
    on the RF-only pipeline by default. The tuned settings are then plugged into
    the larger voting ensemble for the final fit, so CV stays fast while the
    deployed model still benefits from ensemble diversity.

    Scoring defaults to ``f1_macro`` because:
      - Layer 1 (binary): equal weight to "healthy" and "diseased" classes.
      - Layer 2 (multi-class disease): does not let majority diseases hide
        poor recall on minority diseases (e.g. leaf smut).
    """
    best_score = -1.0
    best_params: dict = {
        "n_estimators": n_estimators_grid[0],
        "max_depth": max_depth_grid[0],
        "min_samples_leaf": min_samples_leaf_grid[0],
    }

    cv_mode = "StratifiedGroupKFold (path-level)" if groups is not None else "StratifiedKFold"
    print(
        f"\n{task_name}: 5-fold CV hyperparameter search "
        f"[{cv_mode}, scoring={scoring}] ..."
    )
    rf_builder = builder is _build_rf_pipeline
    for n_est in n_estimators_grid:
        for max_depth in max_depth_grid:
            for min_leaf in min_samples_leaf_grid:
                if rf_builder:
                    clf = builder(n_est, max_depth, min_samples_leaf=min_leaf)
                else:
                    # Legacy callers (e.g. the rice gate) don't take min_samples_leaf.
                    clf = builder(n_est, max_depth)
                if groups is not None:
                    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                    scores = cross_val_score(
                        clf, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=1
                    )
                else:
                    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)
                mean = float(scores.mean())
                depth_label = "none" if max_depth is None else str(max_depth)
                leaf_label = f"leaf={min_leaf}" if rf_builder else ""
                print(
                    f"  trees={n_est}, max_depth={depth_label}{(' ' + leaf_label) if leaf_label else ''}: "
                    f"CV {scoring} {mean:.2%} (+/- {scores.std():.2%})"
                )
                if mean > best_score:
                    best_score = mean
                    best_params = {
                        "n_estimators": n_est,
                        "max_depth": max_depth,
                    }
                    if rf_builder:
                        best_params["min_samples_leaf"] = min_leaf
                # Non-RF builders ignore the leaf axis — only iterate once.
                if not rf_builder:
                    break

    leaf_part = (
        f", min_samples_leaf={best_params['min_samples_leaf']}"
        if "min_samples_leaf" in best_params
        else ""
    )
    print(
        f"  -> best {task_name}: trees={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}{leaf_part} "
        f"(CV {scoring} {best_score:.2%})"
    )
    best_params["cv_accuracy"] = best_score      # kept for report back-compat
    best_params["cv_score"] = best_score
    best_params["cv_scoring"] = scoring
    return best_params


def _print_per_class_metrics(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: list[str],
    task_name: str,
) -> None:
    """Print per-class precision/recall/F1 and confusion matrix for bias detection."""
    y_pred = clf.predict(X_test)
    classes = sorted(set(y_test))
    print(f"\n{task_name} — Per-Class Report (test set):")
    print(classification_report(y_test, y_pred, labels=classes, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"{task_name} — Confusion Matrix (rows=true, cols=predicted):")
    header = f"{'':20s}" + "".join(f"{c[:10]:>12s}" for c in classes)
    print(header)
    for i, row_label in enumerate(classes):
        row = f"{row_label[:20]:20s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(classes)))
        print(row)


def _unwrap_pipeline(clf):
    """
    Return the underlying fitted Pipeline.

    After calibration the saved model is a ``CalibratedClassifierCV`` wrapping a
    ``FrozenEstimator`` that holds the real ``Pipeline``. Introspection helpers
    (OOB score, feature-block importance) need that inner pipeline; predict/score
    still work on the wrapper directly.
    """
    if isinstance(clf, CalibratedClassifierCV):
        calibrated = getattr(clf, "calibrated_classifiers_", None)
        inner = getattr(calibrated[0], "estimator", None) if calibrated else None
        if isinstance(inner, FrozenEstimator):
            inner = getattr(inner, "estimator", None)
        if inner is not None:
            return inner
    return clf


def _calibrate_on_validation(
    clf: Pipeline,
    X_val: np.ndarray,
    y_val: list[str],
) -> Pipeline | CalibratedClassifierCV:
    """
    Wrap an already-fitted pipeline with a sigmoid probability calibrator fit on
    the clean validation split. Returns ``clf`` unchanged if disabled or if any
    class has too few validation samples to calibrate reliably.
    """
    if not CALIBRATE_ENSEMBLE:
        return clf
    counts = Counter(y_val)
    if len(counts) < 2 or min(counts.values()) < CALIBRATION_MIN_PER_CLASS:
        print(
            f"  [calibration] skipped — need {CALIBRATION_MIN_PER_CLASS}+ val "
            f"samples/class (have {dict(counts)})."
        )
        return clf
    # sklearn cannot calibrate a class that has zero validation samples: the
    # internal cross_val_predict crashes when the frozen model predicts more
    # classes than y_val contains. Skip instead of crashing.
    missing = [c for c in getattr(clf, "classes_", []) if c not in counts]
    if missing:
        print(
            f"  [calibration] skipped — model classes {missing} have no "
            f"validation samples."
        )
        return clf
    # FrozenEstimator keeps clf fixed; the calibrator only learns the sigmoid map.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(clf), method=CALIBRATION_METHOD
    )
    calibrated.fit(X_val, y_val)
    print(f"  [calibration] fitted {CALIBRATION_METHOD} map on {len(y_val)} val samples")
    return calibrated


def _get_inner_rf(clf: Pipeline) -> RandomForestClassifier | None:
    """Return the trained RandomForestClassifier inside the pipeline, if any."""
    clf = _unwrap_pipeline(clf)
    try:
        inner = clf.named_steps["clf"]
    except Exception:
        return None
    if isinstance(inner, RandomForestClassifier):
        return inner
    rf = None
    if hasattr(inner, "named_estimators_"):
        rf = inner.named_estimators_.get("rf")
    if rf is None and hasattr(inner, "estimators_"):
        for est in (inner.estimators_ or []):
            if isinstance(est, RandomForestClassifier):
                rf = est
                break
    return rf


def _extract_oob_score(clf: Pipeline) -> float | None:
    """Pull OOB score from the RF estimator inside the voting pipeline, if available."""
    rf = _get_inner_rf(clf)
    if rf is not None and hasattr(rf, "oob_score_"):
        return float(rf.oob_score_)
    return None


def _print_feature_block_importance(clf: Pipeline, task_name: str) -> dict | None:
    """
    Project RF importance from PCA-space back onto the 427-dim feature blocks
    so we can see whether the trained RF leans on color, LBP, GLCM, HOG, or
    edge features when classifying this task.

    The projection ``|PCA.components_|.T @ rf.feature_importances_`` is an
    approximation (PCA mixes original features), but it is the standard way
    to attribute downstream-classifier importance back to input groups and
    is plenty useful for diagnosing whether (e.g.) HOG shape dominates or
    color/texture features are doing the heavy lifting.
    """
    rf = _get_inner_rf(clf)
    if rf is None or not hasattr(rf, "feature_importances_"):
        return None
    try:
        pca = _unwrap_pipeline(clf).named_steps["pca"]
        components = np.abs(pca.components_)
    except Exception:
        return None

    rf_importance = np.asarray(rf.feature_importances_, dtype=np.float64)
    if components.shape[0] != rf_importance.shape[0]:
        return None

    # importance_original[j] = sum_k |components[k, j]| * rf_importance[k]
    importance_original = components.T @ rf_importance
    total = float(importance_original.sum()) or 1.0
    importance_original /= total

    rows: list[tuple[str, float, float]] = []
    for name, start, end in _FEATURE_BLOCKS:
        block_sum = float(importance_original[start:end].sum())
        per_dim = block_sum / max(1, end - start)
        rows.append((name, block_sum, per_dim))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\n{task_name} — RF feature-block importance (back-projected through PCA):")
    print(f"  {'block':<22s}{'total share':>14s}{'per-dim share':>18s}")
    for name, block_sum, per_dim in rows:
        print(f"  {name:<22s}{block_sum:>14.2%}{per_dim:>18.4%}")

    return {name: round(block_sum, 6) for name, block_sum, _ in rows}


def _fit_and_score(
    params: dict,
    X_train: np.ndarray,
    y_train: list[str],
    X_val: np.ndarray,
    y_val: list[str],
    X_test: np.ndarray,
    y_test: list[str],
    *,
    builder=_build_ensemble,
) -> tuple[Pipeline, float, float, float]:
    if builder is _build_rice_gate_classifier:
        clf = builder(
            params.get("n_estimators", 0),
            params.get("max_depth"),
            C=params.get("C", 0.5),
        )
    elif builder in (_build_ensemble, _build_rf_pipeline):
        # Forward the CV-tuned min_samples_leaf (falls back to a safe default
        # if the caller skipped that axis, e.g. legacy reports).
        clf = builder(
            params["n_estimators"],
            params["max_depth"],
            min_samples_leaf=params.get("min_samples_leaf", 3),
        )
    else:
        clf = builder(params["n_estimators"], params["max_depth"])
    clf.fit(X_train, y_train)

    # OOB is a property of the fitted RF — read it before calibration wraps the
    # pipeline (the gap uses the base pipeline's own train accuracy).
    oob = _extract_oob_score(clf)
    if oob is not None:
        base_train_acc = float(clf.score(X_train, y_train))
        print(f"  RF OOB score: {oob:.4f}  (train-OOB gap: {base_train_acc - oob:+.4f})")

    # Calibrate the voting ensemble on the clean validation split so the
    # confidence the pipeline gates on reflects real single-pass accuracy.
    if builder is _build_ensemble:
        clf = _calibrate_on_validation(clf, X_val, y_val)

    val_acc = float(clf.score(X_val, y_val))
    test_acc = float(clf.score(X_test, y_test))
    train_acc = float(clf.score(X_train, y_train))

    return clf, train_acc, val_acc, test_acc


def _subsample_for_quick(
    paths: list[str],
    health_labels: list[str],
    disease_labels: list[str],
    max_per_class: int,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Cap each of the 4 folder classes (healthy + 3 diseases) for fast --quick runs."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    for i, (h, d) in enumerate(zip(health_labels, disease_labels)):
        key = d if d else h
        by_class.setdefault(key, []).append(i)

    keep: set[int] = set()
    for indices in by_class.values():
        if len(indices) <= max_per_class:
            keep.update(indices)
        else:
            pick = rng.choice(indices, size=max_per_class, replace=False)
            keep.update(int(x) for x in pick)

    order = sorted(keep)
    return (
        [paths[i] for i in order],
        [health_labels[i] for i in order],
        [disease_labels[i] for i in order],
    )


def _train_cnn_stage(quick: bool) -> None:
    """
    Train the CNN (neural-network) heads after the RF models, so one command
    (`python ml/train_models.py`) produces BOTH model sets. The app prefers the
    CNN at inference (ML_ENGINE=auto) and falls back to the RF models.

    Skipped gracefully if torch/torchvision are not installed — the RF models
    are already trained and usable on their own.
    """
    print("\n" + "#" * 60)
    print("# STAGE 2/2 — training NEURAL NETWORK (CNN) heads")
    print("#" * 60)
    try:
        from ml.cnn_embedder import torch_available
    except Exception as exc:
        print(f"CNN stage skipped — cannot import embedder ({exc}).")
        return
    if not torch_available():
        print(
            "CNN stage skipped — PyTorch not installed. The RF models are ready.\n"
            "  To enable the neural network, install it and re-run:\n"
            "    python -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        return
    if quick:
        print("(--quick) CNN heads still train on the full cached embeddings — fast.")
    try:
        from ml.train_cnn import main as train_cnn_main

        train_cnn_main()
    except Exception as exc:
        print(f"CNN stage failed (RF models are still saved): {exc}")


def train(
    data_root: Path | None = None,
    quick: bool = False,
    train_cnn: bool = True,
) -> None:
    data_root = data_root or default_training_root()
    print("#" * 60)
    print("# STAGE 1/2 — training FALLBACK RandomForest (rf_*) models")
    print("#" * 60)
    print(f"Training data root: {data_root.resolve()}")
    print(f"Not-rice negatives: {resolve_not_rice_dir(data_root).resolve()}")
    if data_root.resolve() == SOURCES_DIR.resolve():
        try:
            from data.expand_dataset import balance_leaf_smut_archive

            balance_leaf_smut_archive(data_root, skip_download=True)
        except Exception as exc:
            print(f"Warning: leaf smut balance skipped ({exc})")
    paths, health_labels, disease_labels = collect_samples(data_root)
    if quick and len(paths) > QUICK_MAX_PER_CLASS * 4:
        print(f"--quick: subsampling to ~{QUICK_MAX_PER_CLASS} images per class for speed")
        paths, health_labels, disease_labels = _subsample_for_quick(
            paths, health_labels, disease_labels, QUICK_MAX_PER_CLASS
        )
    if len(paths) < 10:
        raise RuntimeError(
            f"Need at least 10 images under {data_root}. "
            "Place class folders (healthy, blast, brown spot, ...) there."
        )

    print(f"Found {len(paths)} images")
    _print_class_balance(health_labels, "Layer 1 (health) class counts")
    diseased_labels = [disease_labels[i] for i, h in enumerate(health_labels) if h == "diseased"]
    _print_class_balance(diseased_labels, "Layer 2 (disease) class counts")

    gate_metrics = _train_rice_gate(data_root, quick=quick)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset: {len(paths)} images — split 70% train / 10% validation / 20% test (stratified)")

    h_train_idx, h_val_idx, h_test_idx = _split_train_val_test(paths, health_labels, 42)
    d_paths_all = [paths[i] for i, h in enumerate(health_labels) if h == "diseased"]
    d_labels_all = [disease_labels[i] for i, h in enumerate(health_labels) if h == "diseased"]
    d_train_idx, d_val_idx, d_test_idx = _split_train_val_test(d_paths_all, d_labels_all, 43)

    def _subset(indices: list[int], source_paths: list[str], source_labels: list[str]):
        return [source_paths[i] for i in indices], [source_labels[i] for i in indices]

    h_train_paths, h_train_y = _subset(h_train_idx, paths, health_labels)
    h_val_paths, h_val_y = _subset(h_val_idx, paths, health_labels)
    h_test_paths, h_test_y = _subset(h_test_idx, paths, health_labels)

    # Senescent-healthy positives (green body + dry brown tip = healthy, not
    # diseased). TRAIN split only — never val/test — so metrics stay honest.
    # Skipped in --quick mode so the smoke subsample isn't swamped by them.
    if not quick:
        senescent_paths = list_health_extra_positives(data_root)
        if senescent_paths:
            print(
                f"  Health train set: +{len(senescent_paths)} senescent-healthy "
                f"positives (brown tips, train-only)"
            )
            h_train_paths = h_train_paths + senescent_paths
            h_train_y = h_train_y + ["healthy"] * len(senescent_paths)

    d_train_paths, d_train_y = _subset(d_train_idx, d_paths_all, d_labels_all)
    d_val_paths, d_val_y = _subset(d_val_idx, d_paths_all, d_labels_all)
    d_test_paths, d_test_y = _subset(d_test_idx, d_paths_all, d_labels_all)

    # 4th Layer 2 class: "physiological stress" (brown tip / yellowing / water
    # marks — no disease lesions). Mostly TRAIN, but a small held-out slice
    # goes to VAL so probability calibration sees every class the model
    # predicts (calibration is skipped for classes absent from validation).
    if not quick:
        stress_paths = list_stress_extra_positives(data_root)
        if stress_paths:
            rng = np.random.default_rng(43)
            stress_paths = [stress_paths[i] for i in rng.permutation(len(stress_paths))]
            n_stress_val = 0
            if len(stress_paths) >= 3 * CALIBRATION_MIN_PER_CLASS:
                n_stress_val = max(
                    CALIBRATION_MIN_PER_CLASS, round(0.15 * len(stress_paths))
                )
            stress_val = stress_paths[:n_stress_val]
            stress_train = stress_paths[n_stress_val:]
            print(
                f"  Disease train set: +{len(stress_train)} '{STRESS_CLASS}' positives"
                + (
                    f" (+{len(stress_val)} held out to val for calibration)"
                    if stress_val
                    else " (train-only)"
                )
            )
            d_train_paths = d_train_paths + stress_train
            d_train_y = d_train_y + [STRESS_CLASS] * len(stress_train)
            if stress_val:
                d_val_paths = d_val_paths + stress_val
                d_val_y = d_val_y + [STRESS_CLASS] * len(stress_val)

    print(
        f"  Health  — train {len(h_train_paths)} | val {len(h_val_paths)} | test {len(h_test_paths)}"
    )
    print(
        f"  Disease — train {len(d_train_paths)} | val {len(d_val_paths)} | test {len(d_test_paths)} "
        f"(diseased only)"
    )

    h_aug = _aug_per_image_for_train_size(len(h_train_paths), quick=quick)
    d_aug = _aug_per_image_for_train_size(len(d_train_paths), quick=quick)
    print(
        f"Augmentation (data-driven): health={h_aug}/image, disease={d_aug}/image"
    )

    X_h_val = _features_original_only(h_val_paths)
    X_h_test = _features_original_only(h_test_paths)
    X_d_val = _features_original_only(d_val_paths)
    X_d_test = _features_original_only(d_test_paths)

    print("\n=== Training pass (full dataset split) ===")

    print("Layer 1 — CV features (path-level, no aug/SMOTE) ...")
    X_h_cv = _features_original_only(h_train_paths)
    h_groups_cv = np.arange(len(h_train_paths))
    print("Layer 2 — CV features (path-level, no aug/SMOTE) ...")
    X_d_cv = _features_original_only(d_train_paths)
    d_groups_cv = np.arange(len(d_train_paths))

    h_params = _select_hyperparameters(
        X_h_cv, h_train_y, "Health ensemble", groups=h_groups_cv
    )
    d_params = _select_hyperparameters(
        X_d_cv, d_train_y, "Disease ensemble", groups=d_groups_cv
    )

    print("Layer 1 — extracting train features ...")
    X_h_train, y_h_train = _features_for_paths(h_train_paths, h_train_y, h_aug, 42)
    X_h_train, y_h_train = _apply_smote(X_h_train, y_h_train, 42)
    print("Layer 2 — extracting train features ...")
    X_d_train, y_d_train = _features_for_paths(d_train_paths, d_train_y, d_aug, 43)
    X_d_train, y_d_train = _apply_smote(X_d_train, y_d_train, 43)

    print(
        f"\nFitting health ensemble on {len(y_h_train):,} samples "
        f"(may take several minutes) ...",
        flush=True,
    )
    h_clf, h_tr, h_va, h_te = _fit_and_score(
        h_params, X_h_train, y_h_train, X_h_val, h_val_y, X_h_test, h_test_y
    )
    print(
        f"Fitting disease ensemble on {len(y_d_train):,} samples "
        f"(may take several minutes) ...",
        flush=True,
    )
    d_clf, d_tr, d_va, d_te = _fit_and_score(
        d_params, X_d_train, y_d_train, X_d_val, d_val_y, X_d_test, d_test_y
    )

    print(
        f"Health  — train {h_tr:.2%} | val {h_va:.2%} | test {h_te:.2%} "
        f"(overfit gap train-test: {h_tr - h_te:.2%})"
    )
    print(
        f"Disease — train {d_tr:.2%} | val {d_va:.2%} | test {d_te:.2%} "
        f"(overfit gap train-test: {d_tr - d_te:.2%})"
    )

    _print_per_class_metrics(h_clf, X_h_test, h_test_y, "Health")
    _print_per_class_metrics(d_clf, X_d_test, d_test_y, "Disease")

    h_block_importance = _print_feature_block_importance(h_clf, "Health")
    d_block_importance = _print_feature_block_importance(d_clf, "Disease")

    joblib.dump(h_clf, HEALTH_MODEL)
    joblib.dump(d_clf, DISEASE_MODEL)
    print(f"Saved health model -> {HEALTH_MODEL}")
    print(f"Saved disease model -> {DISEASE_MODEL}")

    best_health = {
        **h_params,
        "train_accuracy": h_tr,
        "val_accuracy": h_va,
        "test_accuracy": h_te,
        "aug_per_image": h_aug,
        "round": 1,
        "rf_feature_block_importance": h_block_importance,
    }
    best_disease = {
        **d_params,
        "train_accuracy": d_tr,
        "val_accuracy": d_va,
        "test_accuracy": d_te,
        "aug_per_image": d_aug,
        "round": 1,
        "rf_feature_block_importance": d_block_importance,
    }

    report = {
        "total_images": len(paths),
        "train_health_samples": len(h_train_paths),
        "train_disease_samples": len(d_train_paths),
        "split": {"train": 0.70, "validation": 0.10, "test": 0.20},
        "split_strategy": (
            "source-stratified (health + disease): every (source, class) group is "
            "split ~70/10/20 so the model trains on ALL sources and is tested on a "
            "leakage-free held-out slice from all of them (de-dup guarantees no image "
            "crosses splits). This is the in-distribution accuracy. Cross-source "
            "generalisation to an UNSEEN source is reported separately by "
            "ml/cross_source_eval.py. Rice gate uses a stratified random split."
        ),
        "cv_folds": CV_FOLDS,
        "n_estimators_grid": list(N_ESTIMATORS_GRID),
        "max_depth_grid": [str(d) for d in MAX_DEPTH_GRID],
        "min_samples_leaf_grid": list(MIN_SAMPLES_LEAF_GRID),
        "cv_scoring": RF_CV_SCORING,
        "feature_extractor": "bg_removal + light_luma_norm + rich (color+LBP+GLCM+HOG+edges)",
        "preprocessing": "shared preprocess_leaf_image (train + inference)",
        "ensemble": (
            "RandomForest (weight 2) + ExtraTrees + GradientBoosting + "
            "CalibratedSVM (soft voting)"
        ),
        "rf_voting_weight": RF_VOTING_WEIGHT,
        "pipeline": "StandardScaler → PCA(95%) → VotingClassifier",
        "probability_calibration": (
            f"{CALIBRATION_METHOD} on held-out validation split (FrozenEstimator)"
            if CALIBRATE_ENSEMBLE
            else "none"
        ),
        "rf_tuning_strategy": (
            "CV-tune RF alone on (n_estimators × max_depth × min_samples_leaf) "
            "with macro-F1, then plug tuned RF into the voting ensemble for the "
            "final fit."
        ),
        "smote": True,
        "rice_gate_smote": False,
        "rice_gate_cv": "StratifiedGroupKFold on path-level features (no aug/SMOTE)",
        "rice_gate_model": gate_metrics,
        "health_model": best_health,
        "disease_model": best_disease,
        "disease_classes": sorted(set(d_labels_all)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    from ml.training_report import print_validation_summary

    try:
        from ml.evaluate_models import save_validation_summary

        save_validation_summary(data_root)
    except Exception as exc:
        print(f"Validation metrics cache skipped: {exc}")
    print_validation_summary()

    if train_cnn:
        _train_cnn_stage(quick)
        print(
            "\nDone. Trained BOTH model sets:\n"
            "  • rf_*  (RandomForest, fallback)\n"
            "  • cnn_* (neural network, used by the app when available)\n"
            "The app picks the CNN automatically (ML_ENGINE=auto)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RiceGuard models (RF + CNN)")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Root folder with class subfolders (default: leaf/ if present, else data/sources)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: subsample classes, no augmentation (~minutes)",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help="Train only the fallback RF models; skip the neural-network (CNN) stage.",
    )
    args = parser.parse_args()
    train(
        args.data or default_training_root(),
        quick=args.quick,
        train_cnn=not args.no_cnn,
    )


if __name__ == "__main__":
    main()
