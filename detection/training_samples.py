"""Load sample images from the training dataset for result-page reference."""

from pathlib import Path

import numpy as np
from django.conf import settings

from ml.dataset_io import SOURCES_DIR, normalize_label, resolve_class_for_image
from ml.preprocessor import extract_features

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_DIR = settings.BASE_DIR / "data" / "raw"

_SOURCE_DIRS_CACHE: dict[str, list[Path]] = {}
_FEATURE_CACHE: dict[str, np.ndarray] = {}


def _class_folder(health: str, disease_name: str) -> str:
    if health == "unknown":
        return ""
    if health == "healthy":
        return "healthy"
    label = (disease_name or "").strip().lower()
    return label if label else ""


def _raw_samples(folder_name: str, limit: int) -> list[dict]:
    class_dir = RAW_DIR / folder_name
    if not class_dir.is_dir():
        return []

    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return [
        {"disease": folder_name, "filename": p.name}
        for p in images[:limit]
    ]


def _source_dirs_for_class(label: str) -> list[Path]:
    if label in _SOURCE_DIRS_CACHE:
        return _SOURCE_DIRS_CACHE[label]

    if not SOURCES_DIR.is_dir():
        _SOURCE_DIRS_CACHE[label] = []
        return []

    dirs = sorted(
        d for d in SOURCES_DIR.rglob("*")
        if d.is_dir() and normalize_label(d.name) == label
    )
    _SOURCE_DIRS_CACHE[label] = dirs
    return dirs


def _sources_samples(folder_name: str, limit: int) -> list[dict]:
    samples: list[dict] = []
    for class_dir in _source_dirs_for_class(folder_name):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in images:
            samples.append({
                "disease": folder_name,
                "filename": path.name,
                "relpath": path.relative_to(SOURCES_DIR).as_posix(),
            })
            if len(samples) >= limit:
                return samples
    return samples


def get_training_samples(health: str, disease_name: str, limit: int = 5) -> list[dict]:
    """
    Return up to `limit` training images for the given health/disease label.

    Each item: {"disease": str, "filename": str, "relpath": str | omitted}
    Prefers flat data/raw/<class>/; falls back to nested data/sources/.
    """
    folder_name = _class_folder(health, disease_name)
    if not folder_name:
        return []

    raw = _raw_samples(folder_name, limit)
    if raw:
        return raw
    return _sources_samples(folder_name, limit)


def _sample_to_path(sample: dict) -> Path | None:
    disease = str(sample.get("disease") or "").strip().lower()
    filename = str(sample.get("filename") or "").strip()
    relpath = sample.get("relpath")
    if not disease or not filename:
        return None
    return resolve_training_image(disease, filename, relpath=relpath)


def _feature_for_path(path: Path) -> np.ndarray | None:
    key = str(path.resolve())
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        vec = extract_features(key).reshape(-1).astype(np.float32)
    except Exception:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    vec = vec / norm
    _FEATURE_CACHE[key] = vec
    return vec


def get_most_similar_training_sample(
    image_path: str,
    health: str,
    disease_name: str,
    *,
    pool_limit: int = 300,
) -> dict | None:
    """
    Return the most similar training sample (same predicted class) to `image_path`.

    Similarity is cosine similarity on the same feature extractor used by inference.
    """
    samples = get_training_samples(health, disease_name, limit=pool_limit)
    if not samples:
        return None

    try:
        query = extract_features(image_path).reshape(-1).astype(np.float32)
    except Exception:
        return None
    q_norm = float(np.linalg.norm(query))
    if q_norm <= 0.0:
        return None
    query = query / q_norm

    best_sample = None
    best_sim = -1.0
    for sample in samples:
        sample_path = _sample_to_path(sample)
        if sample_path is None:
            continue
        sample_vec = _feature_for_path(sample_path)
        if sample_vec is None:
            continue
        sim = float(np.clip(np.dot(query, sample_vec), -1.0, 1.0))
        if sim > best_sim:
            best_sim = sim
            best_sample = sample

    if best_sample is None:
        return None

    out = dict(best_sample)
    out["similarity"] = round((best_sim + 1.0) * 50.0, 1)
    out["matched_class"] = _class_folder(health, disease_name)
    return out


def resolve_training_image(
    disease: str,
    filename: str,
    relpath: str | None = None,
) -> Path | None:
    """Resolve a safe training image path; None if invalid."""
    if not disease or not filename:
        return None
    if ".." in disease or ".." in filename:
        return None
    if "/" in filename or "\\" in filename:
        return None

    folder = disease.strip().lower()

    if relpath:
        if ".." in relpath:
            return None
        rel = Path(relpath.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            return None
        candidate = (SOURCES_DIR / rel).resolve()
        sources_root = SOURCES_DIR.resolve()
        if sources_root not in candidate.parents and candidate != sources_root:
            return None
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
            return None
        if resolve_class_for_image(candidate, SOURCES_DIR) != folder:
            return None
        if candidate.name != filename:
            return None
        return candidate

    if "/" in disease or "\\" in disease:
        return None

    class_dir = (RAW_DIR / folder).resolve()
    if not class_dir.is_dir() or RAW_DIR.resolve() not in class_dir.parents:
        return None

    candidate = (class_dir / filename).resolve()
    if class_dir not in candidate.parents:
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return candidate
