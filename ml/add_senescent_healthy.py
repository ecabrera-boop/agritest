"""
Teach Layer 1 (healthy vs diseased) that "brown tip + green body = Healthy".

Labeling rule implemented here:
    A leaf is "diseased" only if it has distinct, spreading lesions (spots,
    streaks, or powdery growth) across the green tissue. A mostly-green leaf
    with a dry, brown TIP is "healthy" — normal senescence or physiological
    stress, not disease.

The health classifier currently confuses a physiological dry tip with disease
because its "healthy" training photos are almost all pristine. This script
generates senescent-healthy training images: it samples real healthy rice
photos and adds a dry brown tip (plus an occasional bend) — deliberately NOT
cropping the brown tips out — and saves them into

    leaf/senescent_healthy_positives/

That folder is appended to the Layer 1 TRAIN split only (both the CNN health
head and the RF fallback), labeled "healthy". It is never placed in the
validation/test splits, so reported metrics stay honest. It is invisible to
the Layer 2 disease model. The Layer 0 rice gate also counts these as "rice".

Unlike the gate's tolerance set (ml/add_tolerant_rice.py), these images get NO
mud splashes or droplets — mud specks resemble brown-spot lesions, and labeling
them "healthy" would blur the real disease boundary.

Examples
--------
  # generate 500 senescent-healthy positives, then retrain the CNN heads:
  python ml/add_senescent_healthy.py --retrain

  # just generate (retrain later):
  python ml/add_senescent_healthy.py --count 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.add_tolerant_rice import (  # noqa: E402
    _MAX_EDGE,
    _healthy_source_images,
    add_bend,
    add_brown_tips,
)
from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    default_training_root,
    resolve_health_positives_dir,
)


def make_senescent_healthy(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Healthy leaf + dry brown tip (and sometimes a bend). No lesions added."""
    img = img.convert("RGB")
    if max(img.size) > _MAX_EDGE:
        ratio = _MAX_EDGE / max(img.size)
        img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.LANCZOS,
        )
    arr = np.array(img)
    arr = add_brown_tips(arr, rng)
    if rng.random() < 0.4:
        arr = add_bend(arr, rng)

    out = Image.fromarray(arr)
    out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.9, 1.1)))
    out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.92, 1.08)))
    return out


def generate(count: int, seed: int) -> Path:
    data_root = default_training_root()
    dest = resolve_health_positives_dir(data_root)
    dest.mkdir(parents=True, exist_ok=True)

    sources = _healthy_source_images(data_root)
    if not sources:
        raise SystemExit(f"No healthy rice images found under {data_root} — nothing to augment.")
    print(f"Healthy source photos: {len(sources)} (root: {data_root})")
    print(f"Destination: {dest}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sources))
    written = 0
    i = 0
    while written < count:
        src = sources[int(order[i % len(sources)])]
        i += 1
        target = dest / f"senescent_{written:04d}_{src.stem[:32]}.jpg"
        if target.exists():
            written += 1
            continue
        try:
            with Image.open(src) as im:
                out = make_senescent_healthy(im, rng)
        except Exception as exc:
            print(f"  (skipped, unreadable): {src} — {exc}")
            continue
        out.save(target, quality=92)
        written += 1
        if written % 50 == 0 or written == count:
            print(f"  generated {written}/{count}")

    total = sum(1 for q in dest.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
    print(f"\nSenescent-healthy folder now holds {total} train-only image(s).")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate brown-tip healthy rice positives for Layer 1 training."
    )
    parser.add_argument("--count", type=int, default=500, help="Images to generate (default 500).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default 123).")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="After generating, retrain the CNN heads (health/disease/gate).",
    )
    args = parser.parse_args()

    generate(args.count, args.seed)

    if args.retrain:
        try:
            from ml.cnn_embedder import torch_available
        except Exception:
            torch_available = None
        if torch_available is not None and torch_available():
            print("\nRetraining CNN heads (health head now sees senescent-healthy leaves) ...")
            from ml.train_cnn import main as train_cnn_main

            train_cnn_main()
            print(
                "\nCNN heads updated. To also refresh the RF fallback models, run:\n"
                "  python ml/train_models.py --no-cnn"
            )
        else:
            print("\nPyTorch not available — run: python ml/train_models.py --no-cnn")
    else:
        print(
            "\nNext step — retrain so Layer 1 learns the rule:\n"
            "  python ml/add_senescent_healthy.py --retrain   (CNN heads, fast)\n"
            "  python ml/train_models.py                      (full retrain incl. RF)"
        )


if __name__ == "__main__":
    main()
