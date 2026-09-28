"""
Build cnn_health_db.npz from the embedding cache (no full CNN retrain).

Layer 1 uses this k-NN library to keep field-healthy paddy photos (already
in leaf/**/healthy) from being called diseased by the logistic head.

Run:  python ml/build_health_knn_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ML_DIR = Path(__file__).resolve().parent
_ROOT = _ML_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dataset_io import normalize_label  # noqa: E402
from ml.train_cnn import CACHE_PATH, HEALTH_DB_PATH, MODEL_DIR  # noqa: E402


def _health_label_for_path(path: str) -> str | None:
    p = Path(path)
    parts_l = {part.lower() for part in p.parts}
    if "not_rice" in parts_l:
        return None
    for part in p.parts[::-1]:
        lab = normalize_label(part)
        if lab:
            return "healthy" if lab == "healthy" else "diseased"
    if p.parent.name == "senescent_healthy_positives":
        return "healthy"
    return None


def main() -> None:
    if not CACHE_PATH.is_file():
        raise SystemExit(f"Missing {CACHE_PATH}. Train once with python ml/train_cnn.py")
    raw = np.load(CACHE_PATH, allow_pickle=True)
    embs, labels, paths = [], [], []
    for path, vec in zip(raw["paths"], raw["embeddings"]):
        lab = _health_label_for_path(str(path))
        if lab is None:
            continue
        embs.append(vec.astype(np.float32))
        labels.append(lab)
        paths.append(str(path))
    if not embs:
        raise SystemExit("No labeled rice embeddings found in the cache.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        HEALTH_DB_PATH,
        embeddings=np.vstack(embs).astype(np.float32),
        labels=np.array(labels, dtype=object),
        paths=np.array(paths, dtype=object),
    )
    n_h = sum(1 for x in labels if x == "healthy")
    n_d = sum(1 for x in labels if x == "diseased")
    print(f"Wrote {HEALTH_DB_PATH}  n={len(labels)}  healthy={n_h}  diseased={n_d}")


if __name__ == "__main__":
    main()
