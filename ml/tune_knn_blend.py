"""
Offline experiment: blend the disease head's probabilities with a k-NN vote
over the train-only embedding DB (cnn_embed_db.npz) and measure the effect on
the held-out disease test split (same seed-43 split as train_cnn.py).

Run:  python ml/tune_knn_blend.py <optional_probe_image>
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_ROOT = _ML_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dataset_io import collect_samples, default_training_root  # noqa: E402
from ml.train_models import _split_train_val_test  # noqa: E402

MODEL_DIR = _ML_DIR / "saved_models"
DISEASE_SEED = 43


def knn_proba(
    X: np.ndarray,
    db_emb: np.ndarray,
    db_labels: list[str],
    classes: list[str],
    *,
    k: int,
    temp: float,
) -> np.ndarray:
    """Similarity-softmax k-NN class distribution for each row of X."""
    lab_idx = np.array([classes.index(l) for l in db_labels])
    out = np.zeros((len(X), len(classes)), np.float64)
    sims_all = X @ db_emb.T  # cosine (embeddings are L2-normalized)
    for r in range(len(X)):
        sims = sims_all[r]
        top = np.argpartition(sims, -k)[-k:]
        s = sims[top]
        w = np.exp((s - s.max()) / temp)
        w /= w.sum()
        np.add.at(out[r], lab_idx[top], w)
    return out


def main() -> None:
    # --- load artifacts ---
    d_clf = joblib.load(MODEL_DIR / "cnn_disease.pkl")
    raw = np.load(MODEL_DIR / "cnn_embed_db.npz", allow_pickle=True)
    db_emb = raw["embeddings"].astype(np.float32)
    db_labels = [str(x) for x in raw["labels"]]

    cache = np.load(MODEL_DIR / "cnn_embed_cache.npz", allow_pickle=True)
    cache_map = {p: e for p, e in zip(cache["paths"], cache["embeddings"])}

    # --- rebuild the seed-43 disease test split ---
    root = default_training_root()
    paths, health, disease = collect_samples(root)
    d_paths = [p for p, hl in zip(paths, health) if hl == "diseased"]
    d_labels = [d for d, hl in zip(disease, health) if hl == "diseased"]
    _tr, _va, te = _split_train_val_test(d_paths, d_labels, DISEASE_SEED)
    te_paths = [d_paths[i] for i in te]
    yte = [d_labels[i] for i in te]
    Xte = np.vstack([cache_map[p] for p in te_paths]).astype(np.float32)
    print(f"Test split: {len(yte)} images")

    classes = [str(c) for c in d_clf.classes_]
    head = d_clf.predict_proba(Xte)

    def acc(proba: np.ndarray) -> float:
        pred = [classes[i] for i in proba.argmax(axis=1)]
        return float(np.mean([p == t for p, t in zip(pred, yte)]))

    base = acc(head)
    print(f"\nBaseline head-only test accuracy: {base:.4f}")

    # --- probe image (the misclassified brown spot photo) ---
    probe = sys.argv[1] if len(sys.argv) > 1 else None
    probe_emb = None
    if probe:
        from ml.cnn_embedder import embed_image_tta

        probe_emb = embed_image_tta(probe).astype(np.float32)[0].reshape(1, -1)
        hp = d_clf.predict_proba(probe_emb)[0]
        print(f"Probe head-only: " + ", ".join(f"{c}={p:.3f}" for c, p in zip(classes, hp)))

    print(f"\n{'k':>4} {'temp':>6} {'w_knn':>6} {'test_acc':>9} {'delta':>8}  probe_top")
    for k in (5, 15, 25, 50):
        for temp in (0.02, 0.05, 0.10):
            kn = knn_proba(Xte, db_emb, db_labels, classes, k=k, temp=temp)
            kn_probe = (
                knn_proba(probe_emb, db_emb, db_labels, classes, k=k, temp=temp)
                if probe_emb is not None
                else None
            )
            for w in (0.3, 0.5, 0.7):
                blend = (1 - w) * head + w * kn
                a = acc(blend)
                probe_s = ""
                if kn_probe is not None:
                    bp = (1 - w) * d_clf.predict_proba(probe_emb) + w * kn_probe
                    order = np.argsort(bp[0])[::-1]
                    probe_s = ", ".join(f"{classes[i]}={bp[0][i]:.3f}" for i in order[:2])
                print(f"{k:>4} {temp:>6.2f} {w:>6.1f} {a:>9.4f} {a - base:>+8.4f}  {probe_s}")


if __name__ == "__main__":
    main()
