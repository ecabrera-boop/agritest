"""
Train the CNN (transfer-learning) heads on top of frozen MobileNetV3 embeddings.

Pipeline built here (all on the SAME source-stratified split + seeds as the rf_*
models, so the numbers are directly comparable and leakage-free):

  cnn_rice_gate.pkl  LogisticRegression rice-vs-not_rice on embeddings + an
                     out-of-distribution cosine threshold to the rice database.
  cnn_health.pkl     LogisticRegression healthy-vs-diseased on embeddings.
  cnn_disease.pkl    LogisticRegression over the disease classes on embeddings.
  cnn_embed_db.npz   diseased training embeddings + labels + paths for the
                     k-NN "compare this upload to the database" explanation.
  cnn_meta.json      thresholds, dims, class order, metrics.

Strong train-time augmentation (TRAIN splits only):
  rotate ±20°, zoom in up to 20%, brightness/contrast ±15%.
Each training image contributes its original embedding plus
``TRAIN_AUG_PER_IMAGE`` augmented views so the heads learn leaf texture, not
exact pose — matching the deterministic TTA used at serve time in
``cnn_pipeline``. Val/test stay unaugmented for honest metrics.

Embeddings are cached in saved_models/cnn_embed_cache.npz keyed by path, so a
re-run only embeds new images. Augmented views use a separate cache.
First run downloads the ~20 MB backbone weights.

Run:  python ml/train_cnn.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.cnn_embedder import (  # noqa: E402
    AUG_BRIGHTNESS_DELTA,
    AUG_CONTRAST_DELTA,
    AUG_ROTATE_DEG,
    AUG_ZOOM_MAX,
    EMBED_DIM,
    TRAIN_AUG_PER_IMAGE,
    embed_images,
    make_strong_augmented_views,
    torch_available,
)
from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    STRESS_CLASS,
    collect_samples,
    default_training_root,
    list_gate_extra_positives,
    list_health_extra_positives,
    list_stress_extra_positives,
    resolve_gate_positives_dir,
    resolve_health_positives_dir,
    resolve_not_rice_dir,
    resolve_stress_positives_dir,
)
from ml.train_models import _split_train_val_test  # noqa: E402

MODEL_DIR = _ML_DIR / "saved_models"
CACHE_PATH = MODEL_DIR / "cnn_embed_cache.npz"
AUG_CACHE_PATH = MODEL_DIR / "cnn_embed_aug_cache.npz"
DB_PATH = MODEL_DIR / "cnn_embed_db.npz"
HEALTH_DB_PATH = MODEL_DIR / "cnn_health_db.npz"
META_PATH = MODEL_DIR / "cnn_meta.json"
SUMMARY_PATH = MODEL_DIR / "cnn_validation_summary.json"

HEALTH_SEED = 42  # match rf health split
DISEASE_SEED = 43  # match rf disease split
AUG_SEED = 77     # train-aug RNG family (separate from split seeds)


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #
def _load_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.is_file():
        return {}
    data = np.load(CACHE_PATH, allow_pickle=True)
    return {p: e for p, e in zip(data["paths"], data["embeddings"])}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    paths = np.array(list(cache.keys()), dtype=object)
    embs = (
        np.vstack(list(cache.values())).astype(np.float32)
        if cache
        else np.zeros((0, EMBED_DIM), np.float32)
    )
    np.savez(CACHE_PATH, paths=paths, embeddings=embs)


def embed_with_cache(paths: list[str]) -> np.ndarray:
    """Embed paths, reusing cached vectors; embed only the misses."""
    cache = _load_cache()
    missing = [p for p in paths if p not in cache]
    if missing:
        print(f"Embedding {len(missing)} new image(s) (cached: {len(paths) - len(missing)}) ...")
        for start in range(0, len(missing), 256):
            chunk = missing[start : start + 256]
            vecs = embed_images(chunk)
            for p, v in zip(chunk, vecs):
                cache[p] = v.astype(np.float32)
            print(f"  embedded {min(start + 256, len(missing))}/{len(missing)}")
        _save_cache(cache)
    else:
        print(f"All {len(paths)} embeddings served from cache.")
    return np.vstack([cache[p] for p in paths]).astype(np.float32)


def _aug_cache_key(path: str, aug_idx: int) -> str:
    return f"{path}::strongaug{TRAIN_AUG_PER_IMAGE}::{AUG_SEED}::{aug_idx}"


def _load_aug_cache() -> dict[str, np.ndarray]:
    if not AUG_CACHE_PATH.is_file():
        return {}
    data = np.load(AUG_CACHE_PATH, allow_pickle=True)
    return {p: e for p, e in zip(data["paths"], data["embeddings"])}


def _save_aug_cache(cache: dict[str, np.ndarray]) -> None:
    paths = np.array(list(cache.keys()), dtype=object)
    embs = (
        np.vstack(list(cache.values())).astype(np.float32)
        if cache
        else np.zeros((0, EMBED_DIM), np.float32)
    )
    np.savez(AUG_CACHE_PATH, paths=paths, embeddings=embs)


def embed_strong_aug_views(
    paths: list[str],
    *,
    n_aug: int = TRAIN_AUG_PER_IMAGE,
    seed: int = AUG_SEED,
) -> np.ndarray:
    """
    Embed ``n_aug`` strong-aug views per path → shape (len(paths)*n_aug, D).

    Order is path-major: for each path, its n_aug variants appear consecutively.
    Cached under ``cnn_embed_aug_cache.npz`` so re-runs stay fast.
    """
    if not paths or n_aug < 1:
        return np.zeros((0, EMBED_DIM), np.float32)

    cache = _load_aug_cache()
    keys = [_aug_cache_key(p, k) for p in paths for k in range(n_aug)]
    missing_meta: list[tuple[str, int, str]] = []
    for p in paths:
        for k in range(n_aug):
            key = _aug_cache_key(p, k)
            if key not in cache:
                missing_meta.append((p, k, key))

    if missing_meta:
        print(
            f"Strong-aug embedding {len(missing_meta)} view(s) "
            f"({n_aug}/image × {len(paths)} train images; "
            f"cached hits: {len(keys) - len(missing_meta)}) ..."
        )
        # Build PIL views in chunks, then embed.
        chunk_size = 64  # images → up to 64*n_aug tensors per batch cycle
        save_every = 20  # persist every ~1280 views so a kill doesn't lose the run
        for start in range(0, len(missing_meta), chunk_size):
            chunk = missing_meta[start : start + chunk_size]
            views = []
            for path, aug_idx, _key in chunk:
                # Stable salt (md5) so cache keys always map to the same view.
                path_salt = int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)
                views.extend(
                    make_strong_augmented_views(
                        path,
                        n_aug=1,
                        seed=seed + path_salt * 17 + aug_idx * 997,
                    )
                )
            vecs = embed_images(views)
            for (_path, _aug_idx, key), vec in zip(chunk, vecs):
                cache[key] = vec.astype(np.float32)
            done = min(start + chunk_size, len(missing_meta))
            print(f"  aug-embedded {done}/{len(missing_meta)}", flush=True)
            chunk_i = start // chunk_size
            if chunk_i > 0 and chunk_i % save_every == 0:
                _save_aug_cache(cache)
                print(f"  (checkpointed aug cache: {len(cache)} vectors)", flush=True)
        _save_aug_cache(cache)
    else:
        print(f"All {len(keys)} strong-aug embeddings served from cache.")

    return np.vstack([cache[k] for k in keys]).astype(np.float32)


def expand_train_with_strong_aug(
    X: np.ndarray,
    y: list[str],
    paths: list[str],
    *,
    n_aug: int = TRAIN_AUG_PER_IMAGE,
    seed: int = AUG_SEED,
) -> tuple[np.ndarray, list[str]]:
    """
    Append strong-aug embeddings for each train path (labels duplicated).

    Val/test must NOT call this — metrics stay on clean originals.
    """
    if n_aug < 1 or not paths:
        return X, y
    print(
        f"  Strong aug: +{n_aug}/image "
        f"(rotate ±{AUG_ROTATE_DEG:.0f}°, zoom ≤{AUG_ZOOM_MAX:.0%}, "
        f"brightness/contrast ±{AUG_BRIGHTNESS_DELTA:.0%}) "
        f"on {len(paths)} train images"
    )
    X_aug = embed_strong_aug_views(paths, n_aug=n_aug, seed=seed)
    y_aug = [lab for lab in y for _ in range(n_aug)]
    if len(X_aug) != len(y_aug):
        raise RuntimeError(
            f"Augmentation size mismatch: embeddings={len(X_aug)} labels={len(y_aug)}"
        )
    return np.vstack([X, X_aug]).astype(np.float32), y + y_aug


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_true: list[str], y_pred: list[str], proba: np.ndarray, classes: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    out = {
        "accuracy": float(acc),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test_samples": len(y_true),
        "classes": list(classes),
    }
    try:
        if len(classes) == 2:
            pos = classes[1]
            y_bin = [1 if v == pos else 0 for v in y_true]
            out["roc_auc_weighted"] = float(roc_auc_score(y_bin, proba[:, 1]))
        else:
            yb = label_binarize(y_true, classes=classes)
            out["roc_auc_weighted"] = float(
                roc_auc_score(yb, proba, average="weighted", multi_class="ovr")
            )
    except Exception:
        out["roc_auc_weighted"] = None
    return out


# Per-head inverse regularisation strength, selected on the val split
# (ml/experiment_health_heads.py sweep). C=1.0 badly underfit the 960-dim
# embeddings: healthy field photos (hand-held, paddy background) were pulled
# to "diseased" because the head leaned on scene context instead of lesion
# texture. Weaker regularisation fixed exactly those slices:
#   health  C=30: val 0.961→0.978, test 0.956→0.974 (data_5 healthy 0.89→0.94)
#   disease C=30: val 0.842→0.903, test 0.837→0.889
#   gate    C=3 : val peak (higher C starts overfitting rice vs not_rice)
HEALTH_C = 30.0
DISEASE_C = 30.0
GATE_C = 3.0


def _fit_head(X: np.ndarray, y: list[str], *, C: float = 1.0) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    clf.fit(X, y)
    return clf


def _proba_in_class_order(clf: LogisticRegression, X: np.ndarray, classes: list[str]) -> np.ndarray:
    """Return predict_proba columns reordered to match `classes`."""
    p = clf.predict_proba(X)
    idx = [list(clf.classes_).index(c) for c in classes]
    return p[:, idx]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if not torch_available():
        raise SystemExit(
            "PyTorch/torchvision not installed. Install with:\n"
            "  python -m pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cpu"
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data_root = default_training_root()
    paths, health, disease = collect_samples(data_root)
    print(f"Dataset: {len(paths)} unique rice images from {data_root}")

    # not_rice images for the gate
    not_rice_dir = resolve_not_rice_dir(data_root)
    not_rice_paths = [
        str(p)
        for p in sorted(not_rice_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ] if not_rice_dir.exists() else []
    print(f"not_rice images: {len(not_rice_paths)} from {not_rice_dir}")

    # Gate-only tolerant positives (imperfect-but-healthy rice: brown tips,
    # droplets, mud, yellowing, bent leaves). Extra "rice" evidence for the
    # gate classifier ONLY — never fed to the health/disease heads.
    extra_pos_paths = list_gate_extra_positives(data_root)
    if extra_pos_paths:
        print(
            f"gate-only tolerant rice positives: {len(extra_pos_paths)} "
            f"from {resolve_gate_positives_dir(data_root)}"
        )

    # Senescent-healthy positives (green body + dry brown tip, no lesions).
    # Labeling rule: brown tip = normal senescence, NOT disease. Appended to
    # the health head's TRAIN split only — never to val/test — so metrics
    # stay honest. Also counted as "rice" by the gate.
    senescent_paths = list_health_extra_positives(data_root)
    if senescent_paths:
        print(
            f"senescent-healthy train-only positives: {len(senescent_paths)} "
            f"from {resolve_health_positives_dir(data_root)}"
        )

    # -- embed everything (cached) --
    X_all = embed_with_cache(paths)
    X_not = embed_with_cache(not_rice_paths) if not_rice_paths else np.zeros((0, EMBED_DIM), np.float32)
    X_extra = (
        embed_with_cache(extra_pos_paths)
        if extra_pos_paths
        else np.zeros((0, EMBED_DIM), np.float32)
    )
    X_senescent = (
        embed_with_cache(senescent_paths)
        if senescent_paths
        else np.zeros((0, EMBED_DIM), np.float32)
    )

    # 4th Layer 2 class: "physiological stress" (brown tip / slight yellowing /
    # water marks, no disease lesions). TRAIN-only for the disease head, so a
    # mildly stressed leaf gets a confident class instead of a low-margin
    # forced disease guess -> "unknown".
    stress_paths = list_stress_extra_positives(data_root)
    if stress_paths:
        print(
            f"physiological-stress train-only class: {len(stress_paths)} "
            f"from {resolve_stress_positives_dir(data_root)}"
        )
    X_stress = (
        embed_with_cache(stress_paths)
        if stress_paths
        else np.zeros((0, EMBED_DIM), np.float32)
    )

    # ===================== HEALTH (healthy vs diseased) ===================== #
    h_tr, h_va, h_te = _split_train_val_test(paths, health, HEALTH_SEED)
    Xtr, ytr = X_all[h_tr], [health[i] for i in h_tr]
    Xte, yte = X_all[h_te], [health[i] for i in h_te]
    h_train_paths = [paths[i] for i in h_tr]
    if len(X_senescent):
        # Train-only: brown-tip leaves labeled healthy (senescence, not disease).
        Xtr = np.vstack([Xtr, X_senescent])
        ytr = ytr + ["healthy"] * len(X_senescent)
        h_train_paths = h_train_paths + list(senescent_paths)
        print(f"HEALTH train set: +{len(X_senescent)} senescent-healthy positives")
    Xtr, ytr = expand_train_with_strong_aug(Xtr, ytr, h_train_paths)
    print(f"HEALTH train size after strong aug: {len(ytr)}")
    h_clf = _fit_head(Xtr, ytr, C=HEALTH_C)
    h_classes = ["diseased", "healthy"]
    h_proba = _proba_in_class_order(h_clf, Xte, h_classes)
    h_pred = list(h_clf.predict(Xte))
    h_metrics = _metrics(yte, h_pred, h_proba, h_classes)
    print(f"\nHEALTH  test acc = {h_metrics['accuracy']:.4f}  (n={len(yte)})")

    # ===================== DISEASE (multiclass) ===================== #
    d_paths = [p for p, hl in zip(paths, health) if hl == "diseased"]
    d_labels = [d for d, hl in zip(disease, health) if hl == "diseased"]
    d_emb = np.vstack([X_all[i] for i, hl in enumerate(health) if hl == "diseased"]).astype(np.float32)

    d_tr, d_va, d_te = _split_train_val_test(d_paths, d_labels, DISEASE_SEED)
    Xdtr, ydtr = d_emb[d_tr], [d_labels[i] for i in d_tr]
    Xdte, ydte = d_emb[d_te], [d_labels[i] for i in d_te]
    d_tr_paths_all = [d_paths[i] for i in d_tr]
    if len(X_stress):
        # Train-only 4th class: mild non-pathogenic stress. Test metrics below
        # remain over the real disease classes only.
        Xdtr = np.vstack([Xdtr, X_stress])
        ydtr = ydtr + [STRESS_CLASS] * len(X_stress)
        d_tr_paths_all = d_tr_paths_all + list(stress_paths)
        print(f"DISEASE train set: +{len(X_stress)} '{STRESS_CLASS}' positives")
    # Snapshot unaugmented train rows for the k-NN "similar examples" DB.
    db_emb_clean = Xdtr.astype(np.float32)
    db_labels_clean = list(ydtr)
    db_paths_clean = list(d_tr_paths_all)
    # Augment disease TRAIN only; k-NN DB keeps originals (+ stress), not augs.
    Xdtr, ydtr = expand_train_with_strong_aug(Xdtr, ydtr, d_tr_paths_all)
    print(f"DISEASE train size after strong aug: {len(ydtr)}")
    d_clf = _fit_head(Xdtr, ydtr, C=DISEASE_C)
    d_classes = sorted(set(d_labels))
    d_proba = _proba_in_class_order(d_clf, Xdte, d_classes)
    d_pred = list(d_clf.predict(Xdte))
    d_metrics = _metrics(ydte, d_pred, d_proba, d_classes)
    print(f"DISEASE test acc = {d_metrics['accuracy']:.4f}  (n={len(ydte)})  classes={d_classes}")

    # ===================== RICE GATE (rice vs not_rice) ===================== #
    gate_metrics = None
    ood_threshold = None
    reject_floor = None
    if len(X_not):
        # Rice side = labeled dataset images + gate-only tolerant positives +
        # senescent-healthy + physiological-stress positives, so brown tips /
        # droplets / mud / bent leaves are learned as "rice".
        Xg = np.vstack([X_all, X_extra, X_senescent, X_stress, X_not])
        yg = (
            ["rice"] * (len(X_all) + len(X_extra) + len(X_senescent) + len(X_stress))
            + ["not_rice"] * len(X_not)
        )
        gi = np.arange(len(Xg))
        g_tr, g_va, g_te = _split_train_val_test(list(map(str, gi)), yg, 44, strategy="random")
        y_gtr = [yg[i] for i in g_tr]
        # Path list aligned with Xg rows (rice paths then not_rice paths).
        g_paths_all = (
            list(paths)
            + list(extra_pos_paths)
            + list(senescent_paths)
            + list(stress_paths)
            + list(not_rice_paths)
        )
        g_train_paths = [g_paths_all[i] for i in g_tr]
        Xg_tr, y_gtr = expand_train_with_strong_aug(Xg[g_tr], y_gtr, g_train_paths)
        print(f"RICE GATE train size after strong aug: {len(y_gtr)}")
        # Ensemble gate: LogisticRegression + RandomForest on the CNN embeddings.
        # RF adds a non-linear boundary; averaging their not_rice probabilities is
        # steadier than either head alone at telling rice from other green leaves.
        g_clf = _fit_head(Xg_tr, y_gtr, C=GATE_C)
        g_rf = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        g_rf.fit(Xg_tr, y_gtr)
        joblib.dump(g_rf, MODEL_DIR / "cnn_rice_gate_rf.pkl")

        g_classes = ["not_rice", "rice"]
        ni_lr = list(g_clf.classes_).index("not_rice")
        ni_rf = list(g_rf.classes_).index("not_rice")
        ens_pnot = 0.5 * (
            g_clf.predict_proba(Xg[g_te])[:, ni_lr]
            + g_rf.predict_proba(Xg[g_te])[:, ni_rf]
        )
        g_pred = ["not_rice" if p >= 0.5 else "rice" for p in ens_pnot]
        g_proba = np.column_stack([ens_pnot, 1 - ens_pnot])  # [not_rice, rice]
        gate_metrics = _metrics([yg[i] for i in g_te], g_pred, g_proba, g_classes)
        print(f"RICE GATE (LR+RF ensemble) test acc = {gate_metrics['accuracy']:.4f}  (n={len(g_te)})")

        # Out-of-distribution reference = a sample of ALL rice embeddings (healthy
        # AND diseased). A real rice leaf from any source is visually close to SOME
        # rice image → high nearest-neighbour cosine; faces/objects/other plants are
        # far. (The disease k-NN DB alone can't do this — healthy leaves are far
        # from diseased examples, which caused false 'not_rice' rejections.)
        rng = np.random.default_rng(0)
        ref_n = min(4000, len(X_all))
        rice_ref = X_all[rng.choice(len(X_all), size=ref_n, replace=False)].astype(np.float32)

        def _nn_sim_to_ref(mat: np.ndarray) -> np.ndarray:
            sims = mat @ rice_ref.T
            sims[sims > 0.9999] = -1.0  # drop self-matches
            return sims.max(axis=1)

        rice_nn = _nn_sim_to_ref(X_all)
        not_nn = _nn_sim_to_ref(X_not)
        # Accept-override similarity: the gate only overrides a 'not_rice' classifier
        # verdict when the upload is VERY close to real rice. Set near the 10th
        # percentile of real-rice nearest-neighbour similarity (~0.88) so nearly all
        # real rice is protected, while non-rice LEAVES (which sit lower, ~0.56-0.77)
        # stay rejected by the classifier. A lower threshold (e.g. 1st pct ~0.78)
        # wrongly rescued ~half the rice-look-alike leaves.
        ood_threshold = float(max(np.percentile(rice_nn, 10), 0.85))
        # Reject floor: an upload whose nearest real-rice similarity is below this is
        # too far from EVERY known rice leaf to be rice — reject it outright, even if
        # the classifier is fooled (broad/compound leaves score ~0.44-0.68 here, well
        # under real rice's ~0.75 p1). Set at the 2nd percentile of real rice so only
        # ~2% of genuine rice is affected while non-rice leaves are caught.
        reject_floor = float(np.percentile(rice_nn, 2))
        print(
            f"  gate sim — rice p1/p2/p10={np.percentile(rice_nn,1):.3f}/"
            f"{np.percentile(rice_nn,2):.3f}/{np.percentile(rice_nn,10):.3f}  "
            f"not_rice p50/p90={np.percentile(not_nn,50):.3f}/{np.percentile(not_nn,90):.3f}  "
            f"reject_floor={reject_floor:.3f} accept_override={ood_threshold:.3f}"
        )
        np.savez(
            MODEL_DIR / "cnn_rice_ref.npz",
            embeddings=rice_ref,
        )
        joblib.dump(g_clf, MODEL_DIR / "cnn_rice_gate.pkl")
    else:
        print("No not_rice images — skipping CNN rice gate (keeps rf_rice_gate).")

    # -- save heads + k-NN database + meta --
    joblib.dump(h_clf, MODEL_DIR / "cnn_health.pkl")
    joblib.dump(d_clf, MODEL_DIR / "cnn_disease.pkl")

    # Disease k-NN DB = unaugmented TRAIN diseased embeddings only
    # (never expose test leaves or synthetic aug views).
    db_emb = db_emb_clean
    db_labels = np.array(db_labels_clean, dtype=object)
    db_paths = np.array(db_paths_clean, dtype=object)
    np.savez(DB_PATH, embeddings=db_emb, labels=db_labels, paths=db_paths)
    # Health k-NN library = all labeled rice embeddings (+ senescent-healthy).
    # Serve-time Layer 1 uses this to rescue field-healthy paddy photos that
    # the logistic head still calls diseased.
    if len(X_senescent):
        health_db_emb = np.vstack([X_all, X_senescent]).astype(np.float32)
        health_db_labels = list(health) + ["healthy"] * len(X_senescent)
        health_db_paths = list(paths) + list(senescent_paths)
    else:
        health_db_emb = X_all.astype(np.float32)
        health_db_labels = list(health)
        health_db_paths = list(paths)
    np.savez(
        HEALTH_DB_PATH,
        embeddings=health_db_emb,
        labels=np.array(health_db_labels, dtype=object),
        paths=np.array(health_db_paths, dtype=object),
    )

    summary = {"dataset_images": len(paths), "health": h_metrics, "disease": d_metrics}
    if gate_metrics:
        summary["rice_gate"] = gate_metrics
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    meta = {
        "backbone": "mobilenet_v3_large (IMAGENET1K_V2, frozen)",
        "embed_dim": EMBED_DIM,
        "health_classes": h_classes,
        "disease_classes": sorted(set(ydtr)),  # trained classes (incl. stress)
        "disease_eval_classes": d_classes,     # real classes used for test metrics
        "gate_classes": ["not_rice", "rice"] if gate_metrics else None,
        "gate_ensemble": "LogisticRegression + RandomForest (mean not_rice proba)",
        "head_C": {"health": HEALTH_C, "disease": DISEASE_C, "gate": GATE_C},
        "ood_cosine_threshold": ood_threshold,   # accept-override: >= → trust as rice
        "rice_reject_floor": reject_floor,        # < → reject outright (far from all rice)
        "split": "source_stratified (matches rf_* seeds 42/43)",
        "train_augmentation": {
            "rotate_deg": AUG_ROTATE_DEG,
            "zoom_max": AUG_ZOOM_MAX,
            "brightness_delta": AUG_BRIGHTNESS_DELTA,
            "contrast_delta": AUG_CONTRAST_DELTA,
            "aug_per_image": TRAIN_AUG_PER_IMAGE,
            "applied_to": "train splits only (health / disease / gate)",
            "matches_serve_tta": True,
        },
        "train_disease_counts": dict(Counter(db_labels_clean)),
        "train_disease_counts_with_aug": dict(Counter(ydtr)),
        "test_disease_counts": dict(Counter(ydte)),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("CNN VALIDATION SUMMARY (held-out, source-stratified)")
    print("=" * 60)
    for name, m in (("Health", h_metrics), ("Disease", d_metrics)):
        auc = m["roc_auc_weighted"]
        auc_s = "—" if auc is None else f"{auc:.4f}"
        print(f"{name:8s} acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}  auc={auc_s}  n={m['n_test_samples']}")
    if gate_metrics:
        print(f"RiceGate acc={gate_metrics['accuracy']:.4f}  f1={gate_metrics['f1_weighted']:.4f}  n={gate_metrics['n_test_samples']}")
    print(f"\nSaved heads + db + meta under {MODEL_DIR}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
