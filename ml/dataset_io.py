"""
Load labeled rice leaf images from a folder tree (e.g. data/sources).

Supports nested layouts from Hugging Face / UCI extracts where class names
appear on ancestor folders, not only the immediate parent.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"
RAW_DIR = DATA_DIR / "raw"
LEAF_DIR = _PROJECT_ROOT / "leaf"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folders to skip when inferring class from path (not disease labels).
_SKIP_PARTS = frozenset(
    {
        "train",
        "test",
        "val",
        "validation",
        "background",
        "cache",
        "download",
        "dataset_clean_6class",
        "dataset clean 6class",
        "sources",
        "_supplement",
        "hf_amdmqd",
        "hf_notnova",
        "uci",
        "kaggle",
        "raw",
        "data",
        "data_1",
        "data_2",
        "data_3",
        "data_4",
        "data_5",
        "data_6",
        "leaf",
        "not_rice",
        "real",
        "images",
        "image",
        "img",
    }
)

_CLASS_ALIASES: dict[str, str] = {
    "healthy": "healthy",
    "health": "healthy",
    "normal": "healthy",
    "bacterial blight": "bacterial blight",
    "bacterial leaf blight": "bacterial blight",
    "bacterial_blight": "bacterial blight",
    "bacterialblight": "bacterial blight",
    "bacterailblight": "bacterial blight",
    "bb": "bacterial blight",
    "blast": "blast",
    "rice blast": "blast",
    "leaf blast": "blast",
    "leafblast": "blast",
    "brown spot": "brown spot",
    "brown_spot": "brown spot",
    "brownspot": "brown spot",
    "bs": "brown spot",
    "tungro": "tungro",
    "rice tungro": "tungro",
    "rice turngo": "tungro",
    "turngo": "tungro",
    "sheath blight": "sheath blight",
    "sheath blight disease": "sheath blight",
    "sheath_blight": "sheath blight",
    "steath blight": "sheath blight",
    "shath blight": "sheath blight",
    "leaf smut": "leaf smut",
    "leaf_smut": "leaf smut",
    "leafsmut": "leaf smut",
    "ls": "leaf smut",
    "hispa": "hispa",
    "rice hispa": "hispa",
    "physiological stress": "physiological stress",
    "physiological_stress": "physiological stress",
    "minor physiological stress": "physiological stress",
    "minor_physiological_stress": "physiological stress",
}


def normalize_label(name: str) -> str | None:
    key = re.sub(r"[_\-]+", " ", name.strip().lower())
    key = re.sub(r"\s+", " ", key)
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    compact = key.replace(" ", "")
    if compact in _CLASS_ALIASES:
        return _CLASS_ALIASES[compact]
    for alias, target in _CLASS_ALIASES.items():
        if alias.replace(" ", "") == compact:
            return target
    return None


def resolve_class_for_image(img_path: Path, search_root: Path) -> str | None:
    """Walk parents under search_root until a known class folder is found."""
    root = search_root.resolve()
    for parent in img_path.parents:
        if parent.resolve() == root:
            break
        part = parent.name.strip().lower()
        if part in _SKIP_PARTS:
            continue
        label = normalize_label(part)
        if label:
            return label
    return None


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


LEAF_SMUT_ARCHIVE = "leaf_smut_all"
LEAF_SMUT_CLASS = "leaf smut"


def _leaf_smut_archive_image_count(data_root: Path) -> int:
    dest = data_root / LEAF_SMUT_ARCHIVE / LEAF_SMUT_CLASS
    if not dest.exists():
        return 0
    return sum(
        1 for p in dest.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _skip_duplicate_leaf_smut(
    img_path: Path,
    data_root: Path,
    *,
    leaf_smut_archive_count: int,
) -> bool:
    """
    When leaf_smut_all has images, train only from that folder.

    Skips leaf smut copies under UCI, HF mirrors, _supplement, etc.
    """
    img_path = img_path.resolve()
    data_root = data_root.resolve()
    if leaf_smut_archive_count < 1:
        return False
    if LEAF_SMUT_ARCHIVE in img_path.parts:
        return False
    if resolve_class_for_image(img_path, data_root) != LEAF_SMUT_CLASS:
        return False
    return True


def _content_digest(path: Path) -> str:
    """MD5 of file bytes — drops byte-identical duplicate images."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _perceptual_digest(path: Path, hash_size: int = 16) -> bytes | None:
    """
    256-bit difference hash (dHash) of the image content.

    Catches *re-encoded* duplicates that MD5 misses — the same photo saved with
    different JPEG compression (different bytes, identical pixels). Two genuinely
    distinct leaf photos essentially never collide on a 256-bit dHash, so an
    exact match is a reliable duplicate signal. Returns None if unreadable.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            gray = im.convert("L").resize((hash_size + 1, hash_size))
        arr = np.asarray(gray, dtype=np.int32)
        return (arr[:, 1:] > arr[:, :-1]).tobytes()
    except Exception:
        return None


def collect_samples(
    data_root: Path,
    *,
    deduplicate: bool = True,
    perceptual_dedup: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Return (paths, health_labels, disease_labels).

    health: healthy | diseased
    disease: class name for diseased rows, else ""

    Deduplication (both flags default True) keeps each real image once:
      - ``deduplicate``      drops byte-identical copies (MD5).
      - ``perceptual_dedup`` also drops re-encoded/resized copies (dHash) — the
        same picture saved at a different JPEG quality, which MD5 cannot detect.

    The merged leaf/data_1|2|3 sources overlap massively (the same photos appear
    across folders, and many are stored twice at different compression). Without
    dedup the random train/val/test split scatters identical copies across
    splits, leaking the answer into the test set and inflating accuracy toward an
    unrealistic ~100% / ROC-AUC 1.0. Dedup keeps train_models and evaluate_models
    in sync because both call this function and split with the same seeds.
    """
    paths: list[str] = []
    health_labels: list[str] = []
    disease_labels: list[str] = []
    skipped = 0
    exact_dupes = 0
    near_dupes = 0
    seen_digests: set[str] = set()
    seen_phashes: set[bytes] = set()

    root = data_root.resolve()
    leaf_smut_archive_count = _leaf_smut_archive_image_count(root)
    for img_path in sorted(root.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "not_rice" in img_path.parts:
            continue
        if _skip_duplicate_leaf_smut(img_path, root, leaf_smut_archive_count=leaf_smut_archive_count):
            skipped += 1
            continue
        class_name = resolve_class_for_image(img_path, root)
        if not class_name:
            skipped += 1
            continue
        if deduplicate:
            try:
                digest = _content_digest(img_path)
            except OSError:
                digest = None
            if digest is not None:
                if digest in seen_digests:
                    exact_dupes += 1
                    continue
                seen_digests.add(digest)
            if perceptual_dedup:
                phash = _perceptual_digest(img_path)
                if phash is not None:
                    if phash in seen_phashes:
                        near_dupes += 1
                        continue
                    seen_phashes.add(phash)
        paths.append(str(img_path))
        if class_name == "healthy":
            health_labels.append("healthy")
            disease_labels.append("")
        else:
            health_labels.append("diseased")
            disease_labels.append(class_name)

    if deduplicate and (exact_dupes or near_dupes):
        print(
            f"collect_samples: removed {exact_dupes} byte-identical + "
            f"{near_dupes} re-encoded duplicate image(s); kept {len(paths)} unique."
        )

    return paths, health_labels, disease_labels


def default_training_root() -> Path:
    """Prefer leaf/ when present; else data/sources; else data/raw."""
    if count_images(LEAF_DIR) > 0:
        return LEAF_DIR
    if count_images(SOURCES_DIR) > 0:
        return SOURCES_DIR
    return RAW_DIR


# Gate-only tolerant positives: healthy rice with brown tips, water droplets,
# mud splashes, slight yellowing, bent leaves. These teach Layer 0 that an
# imperfect rice leaf is still rice. The folder name resolves to NO class
# label, so collect_samples (Layers 1-2) skips it, while the rice-gate
# collectors pick it up as "rice".
GATE_POSITIVES_DIRNAME = "rice_gate_positives"


def resolve_gate_positives_dir(data_root: Path | None = None) -> Path:
    """Folder holding gate-only 'rice' positives (tolerance training images)."""
    root = data_root if data_root is not None else default_training_root()
    return root / GATE_POSITIVES_DIRNAME


def list_gate_extra_positives(data_root: Path | None = None) -> list[str]:
    """All gate-only positive image paths (empty list if the folder is absent)."""
    folder = resolve_gate_positives_dir(data_root)
    if not folder.exists():
        return []
    return [
        str(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


# Train-only "senescent healthy" positives: mostly-green rice leaves with a
# dry brown TIP but no spreading lesions. Labeling rule: brown tip + green
# body = healthy (normal senescence / physiological stress), NOT diseased.
# The folder name resolves to no class label so collect_samples skips it;
# the Layer 1 trainers append these to their TRAIN split only (never to
# validation/test), keeping reported metrics honest.
HEALTH_POSITIVES_DIRNAME = "senescent_healthy_positives"


def resolve_health_positives_dir(data_root: Path | None = None) -> Path:
    """Folder holding train-only 'healthy' brown-tip positives for Layer 1."""
    root = data_root if data_root is not None else default_training_root()
    return root / HEALTH_POSITIVES_DIRNAME


def list_health_extra_positives(data_root: Path | None = None) -> list[str]:
    """All senescent-healthy image paths (empty list if the folder is absent)."""
    folder = resolve_health_positives_dir(data_root)
    if not folder.exists():
        return []
    return [
        str(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


# Train-only Layer 2 class: "physiological stress" — minor discoloration
# (brown tip, slight yellowing, small water marks) that matches NO known
# disease pattern. Gives Layer 2 a confident landing spot for mild-stress
# leaves instead of a forced low-margin disease guess -> "unknown".
# Appended to the disease head's TRAIN split only (never val/test).
STRESS_CLASS = "physiological stress"
STRESS_POSITIVES_DIRNAME = "physiological_stress_positives"


def resolve_stress_positives_dir(data_root: Path | None = None) -> Path:
    """Folder holding train-only 'physiological stress' images for Layer 2."""
    root = data_root if data_root is not None else default_training_root()
    return root / STRESS_POSITIVES_DIRNAME


def list_stress_extra_positives(data_root: Path | None = None) -> list[str]:
    """All physiological-stress image paths (empty list if the folder is absent)."""
    folder = resolve_stress_positives_dir(data_root)
    if not folder.exists():
        return []
    return [
        str(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def resolve_not_rice_dir(data_root: Path | None = None) -> Path:
    """Prefer <data_root>/not_rice (e.g. leaf/not_rice); else data/not_rice."""
    if data_root is not None:
        candidate = data_root / "not_rice"
        if candidate.exists():
            return candidate
    legacy = DATA_DIR / "not_rice"
    if legacy.exists():
        return legacy
    if data_root is not None:
        return data_root / "not_rice"
    return legacy


def available_disease_classes(data_root: Path | None = None) -> set[str]:
    """
    Return canonical disease labels (excluding "healthy") that currently have
    at least one image on disk under *data_root* (default: data/sources).

    Used by Layer 2 to mask out predictions for classes whose folders have
    been removed since the pickle was trained (e.g. a stale "tungro" class
    that the live model still believes in but has no training images for).
    """
    root = (data_root or SOURCES_DIR).resolve()
    if not root.exists():
        return set()
    found: set[str] = set()
    for entry in root.rglob("*"):
        if not entry.is_dir():
            continue
        label = normalize_label(entry.name)
        if not label or label == "healthy" or label in found:
            continue
        for child in entry.rglob("*"):
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                found.add(label)
                break
    return found
