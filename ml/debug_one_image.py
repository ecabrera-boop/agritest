"""One-off diagnostic: why does the CNN pipeline mislabel a given image?

Usage: python ml/debug_one_image.py <image_path>
Prints health-head probabilities (original + TTA) and the nearest training
images from the full embedding cache (healthy AND diseased sides).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_ROOT = _ML_DIR.parent
sys.path.insert(0, str(_ROOT))

from ml.cnn_embedder import embed_image_tta  # noqa: E402
from ml.dataset_io import collect_samples, default_training_root  # noqa: E402

MODEL_DIR = _ML_DIR / "saved_models"


def main() -> None:
    img = sys.argv[1]
    health = joblib.load(MODEL_DIR / "cnn_health.pkl")

    emb_tta = embed_image_tta(img).astype(np.float32)
    emb = emb_tta[0]

    print("health classes:", list(health.classes_))
    p_orig = health.predict_proba(emb.reshape(1, -1))[0]
    p_tta = health.predict_proba(emb_tta).mean(axis=0)
    print("original :", {c: round(float(p), 4) for c, p in zip(health.classes_, p_orig)})
    print("tta-mean :", {c: round(float(p), 4) for c, p in zip(health.classes_, p_tta)})

    # Nearest neighbours over the whole labeled dataset via the training cache.
    cache = np.load(MODEL_DIR / "cnn_embed_cache.npz", allow_pickle=True)
    cache_map = {p: e for p, e in zip(cache["paths"], cache["embeddings"])}

    root = default_training_root()
    paths, health_lab, disease_lab = collect_samples(root)
    rows, labs, kept_paths = [], [], []
    for p, h, d in zip(paths, health_lab, disease_lab):
        e = cache_map.get(p)
        if e is None:
            continue
        rows.append(e)
        labs.append(h if h == "healthy" else d)
        kept_paths.append(p)
    mat = np.vstack(rows).astype(np.float32)
    sims = mat @ emb
    top = np.argsort(sims)[::-1][:25]
    print(f"\nTop-25 nearest of {len(kept_paths)} cached training images:")
    for i in top:
        rel = Path(kept_paths[i]).relative_to(root)
        print(f"  {sims[i]:.4f}  {labs[i]:<18s} {rel}")

    by = {}
    for lab in set(labs):
        idx = [i for i, l in enumerate(labs) if l == lab]
        s = sims[idx]
        by[lab] = (float(np.max(s)), float(np.mean(np.sort(s)[-10:])))
    print("\nPer-class nearest / mean-top10 similarity:")
    for lab, (mx, m10) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"  {lab:<18s} max={mx:.4f}  top10mean={m10:.4f}")


if __name__ == "__main__":
    main()
