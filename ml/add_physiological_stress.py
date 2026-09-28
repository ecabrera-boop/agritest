"""
Add a 4th Layer 2 class: "physiological stress" (mild, non-pathogenic damage).

Labeling rule implemented here:
    If a leaf has minor discoloration (a brown tip, slight yellowing, or small
    water marks) but does not match any known disease pattern (blast, brown
    spot, bacterial blight), do NOT force it into a disease class — it belongs
    to "physiological stress".

Why: Layer 2 is a closed-set classifier. When a mildly stressed leaf reaches
it, the probability mass gets split between the three diseases, the top-2
margin falls under DISEASE_MARGIN_MIN, and the pipeline returns "unknown".
Giving the model an explicit stress class concentrates that probability
(e.g. stress 92% vs blast 4% -> margin 88%), so the margin check passes and
the farmer gets a correct "this is stress, not disease" recommendation.

This script generates stress training images from real healthy rice photos:
pronounced dry brown tips, broader chlorosis-style yellowing, and small dried
water marks — deliberately NO spot/streak lesion patterns. Saved into

    leaf/physiological_stress_positives/

That folder is appended to the Layer 2 TRAIN split only (CNN disease head,
its k-NN example database, and the RF fallback), labeled "physiological
stress". It never enters validation/test splits, so disease metrics stay
honest. Layer 1 training is untouched. The Layer 0 gate counts these as rice.

Examples
--------
  # generate 500 stress positives, then retrain the CNN heads:
  python ml/add_physiological_stress.py --retrain

  # just generate (retrain later):
  python ml/add_physiological_stress.py --count 500
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
    add_slight_yellowing,
    add_water_droplets,
)
from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    default_training_root,
    resolve_stress_positives_dir,
)


def make_stress_image(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """
    Moderate physiological stress: pronounced tip dieback and/or chlorosis and/or
    dried water marks. More visible than the "senescent healthy" set (so it can
    plausibly trip Layer 1), but never spot/streak lesion patterns.
    """
    img = img.convert("RGB")
    if max(img.size) > _MAX_EDGE:
        ratio = _MAX_EDGE / max(img.size)
        img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.LANCZOS,
        )
    arr = np.array(img)

    # Always at least two stress signs; brown tip is applied twice in half the
    # cases for a larger dieback area than the senescent-healthy set.
    effects = [add_brown_tips, add_slight_yellowing, add_water_droplets]
    picks = rng.choice(3, size=int(rng.integers(2, 4)), replace=False)
    for k in picks:
        arr = effects[int(k)](arr, rng)
    if 0 in picks and rng.random() < 0.5:
        arr = add_brown_tips(arr, rng)
    if rng.random() < 0.35:
        arr = add_bend(arr, rng)

    out = Image.fromarray(arr)
    out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.88, 1.12)))
    out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.9, 1.1)))
    return out


def generate(count: int, seed: int) -> Path:
    data_root = default_training_root()
    dest = resolve_stress_positives_dir(data_root)
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
        target = dest / f"stress_{written:04d}_{src.stem[:32]}.jpg"
        if target.exists():
            written += 1
            continue
        try:
            with Image.open(src) as im:
                out = make_stress_image(im, rng)
        except Exception as exc:
            print(f"  (skipped, unreadable): {src} — {exc}")
            continue
        out.save(target, quality=92)
        written += 1
        if written % 50 == 0 or written == count:
            print(f"  generated {written}/{count}")

    total = sum(1 for q in dest.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
    print(f"\nPhysiological-stress folder now holds {total} train-only image(s).")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 'physiological stress' training images for Layer 2."
    )
    parser.add_argument("--count", type=int, default=500, help="Images to generate (default 500).")
    parser.add_argument("--seed", type=int, default=321, help="Random seed (default 321).")
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
            print("\nRetraining CNN heads (disease head gains the stress class) ...")
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
            "\nNext step — retrain so Layer 2 learns the class:\n"
            "  python ml/add_physiological_stress.py --retrain   (CNN heads, fast)\n"
            "  python ml/train_models.py                         (full retrain incl. RF)"
        )


if __name__ == "__main__":
    main()
