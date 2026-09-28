"""Resolve active ML model files (default vs ``*_01`` trainer)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent
MODEL_DIR = _ML_DIR / "saved_models"

_MODEL_STEMS = ("rf_rice_gate", "rf_health", "rf_disease")


def _report_path(tag: str) -> Path:
    if tag:
        return MODEL_DIR / f"training_report_{tag}.json"
    return MODEL_DIR / "training_report.json"


def _report_total_images(report_path: Path) -> int:
    if not report_path.exists():
        return 0
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    total = data.get("total_images")
    if total is not None:
        return int(total)
    health = data.get("train_health_samples")
    disease = data.get("train_disease_samples")
    if health is not None and disease is not None:
        return int(health) + int(disease)
    return 0


def _model_set_complete(tag: str) -> bool:
    suffix = f"_{tag}" if tag else ""
    return all((MODEL_DIR / f"{stem}{suffix}.pkl").is_file() for stem in _MODEL_STEMS)


@lru_cache(maxsize=1)
def get_model_tag() -> str:
    """
    Return the active model tag: ``""`` for ``rf_health.pkl`` or ``"01"`` for ``rf_health_01.pkl``.

    Override with env ``ML_MODEL_TAG`` (use ``""`` or ``default`` for legacy names).
    Otherwise picks the **most recently trained** complete model set; ties go to
    the set trained on more images.

    Recency (not dataset size) wins because deduplication in
    ``dataset_io.collect_samples`` intentionally shrinks the training set — a
    freshly retrained, leakage-free model on 5k unique images must not be shadowed
    by a stale set whose report still counts 11k duplicated images.
    """
    env = os.environ.get("ML_MODEL_TAG")
    if env is not None:
        tag = env.strip()
        if tag.lower() in ("", "default", "legacy"):
            return ""
        return tag

    candidates: list[tuple[float, int, str]] = []
    for tag in ("01", ""):
        if not _model_set_complete(tag):
            continue
        images = _report_total_images(_report_path(tag))
        disease_path = MODEL_DIR / (f"rf_disease_{tag}.pkl" if tag else "rf_disease.pkl")
        candidates.append((disease_path.stat().st_mtime, images, tag))

    if not candidates:
        return ""

    # Most recent mtime first; break ties by larger dataset.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def model_path(stem: str) -> Path:
    """Path to ``rf_health.pkl`` or ``rf_health_<tag>.pkl`` for the active tag."""
    tag = get_model_tag()
    suffix = f"_{tag}" if tag else ""
    return MODEL_DIR / f"{stem}{suffix}.pkl"


def report_path() -> Path:
    return _report_path(get_model_tag())


def validation_summary_path() -> Path:
    tag = get_model_tag()
    if tag:
        return MODEL_DIR / f"validation_summary_{tag}.json"
    return MODEL_DIR / "validation_summary.json"


def active_model_label() -> str:
    tag = get_model_tag()
    return f"tag={tag or 'default'}"
