"""
Teach the Layer 0 rice gate to tolerate imperfect-but-healthy rice leaves.

Labeling rule implemented here:
    An image is "rice" if the main subject is a rice leaf, regardless of
    1) brown tips, 2) water droplets, 3) mud splashes, 4) slight yellowing,
    5) bent leaves. Only a completely different plant, soil, or an inanimate
    object is "not rice".

The gate currently rejects healthy plants over minor imperfections because it
never saw imperfect positives in training. This script fixes that by generating
hard-positive training images: it samples real *healthy* rice photos from the
training set and applies realistic field imperfections (brown tips, water
droplets, mud splashes, slight yellowing, bent leaves), then saves them into

    leaf/rice_gate_positives/

That folder is used ONLY by the Layer 0 gate (both rf_rice_gate and the CNN
gate) as extra "rice" positives — it is invisible to the Layer 1/2 health and
disease models, so the synthetic brown tips can never contaminate the
healthy-vs-diseased training. You can also drop real imperfect-rice photos into
the same folder by hand; they will be picked up on the next retrain.

Examples
--------
  # generate 500 hard positives, then retrain both gates:
  python ml/add_tolerant_rice.py --retrain

  # just generate (retrain later with: python ml/train_models.py):
  python ml/add_tolerant_rice.py --count 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    default_training_root,
    resolve_class_for_image,
    resolve_gate_positives_dir,
)
from ml.preprocessor_shared import plant_mask_from_rgb  # noqa: E402

# Keep augmentation resolution bounded — training letterboxes to 128/224 anyway.
_MAX_EDGE = 1024


# --------------------------------------------------------------------------- #
# Imperfection effects (all operate on HxWx3 uint8 arrays)
# --------------------------------------------------------------------------- #
def _leaf_pixels(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = plant_mask_from_rgb(arr)
    ys, xs = np.where(mask)
    return mask, ys, xs


def add_brown_tips(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dry brown patch(es) at the extremities of the leaf, feathered inward."""
    mask, ys, xs = _leaf_pixels(arr)
    if len(ys) < 100:
        return arr
    h, w = arr.shape[:2]
    cy, cx = ys.mean(), xs.mean()
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    # candidate tip pixels = farthest 2% of leaf pixels from the leaf centroid
    far = np.argsort(d2)[-max(1, len(ys) // 50):]

    out = arr.astype(np.float32)
    yy, xx = np.ogrid[:h, :w]
    for _ in range(int(rng.integers(1, 3))):
        j = int(rng.choice(far))
        ty, tx = ys[j], xs[j]
        radius = rng.uniform(0.05, 0.16) * max(h, w)
        dist = np.sqrt((yy - ty) ** 2 + (xx - tx) ** 2)
        weight = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 1.3
        weight *= mask * rng.uniform(0.55, 0.9)
        brown = np.array(
            [rng.uniform(105, 145), rng.uniform(70, 95), rng.uniform(30, 55)],
            np.float32,
        )
        out += weight[..., None] * (brown - out)
        # subtle dry-texture noise inside the browned area
        noise = rng.normal(0, 7, (h, w, 1)).astype(np.float32)
        out += (weight > 0.15)[..., None] * noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_water_droplets(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small translucent droplets with a darker rim and a specular highlight."""
    mask, ys, xs = _leaf_pixels(arr)
    if len(ys) < 100:
        return arr
    h, w = arr.shape[:2]
    out = arr.astype(np.float32)
    scale = max(max(h, w) / 512.0, 0.5)
    for _ in range(int(rng.integers(4, 12))):
        j = int(rng.integers(len(ys)))
        cy, cx = float(ys[j]), float(xs[j])
        rx = rng.uniform(3, 9) * scale
        ry = rx * rng.uniform(0.75, 1.25)
        pad = int(max(rx, ry) * 1.6) + 2
        y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad)
        x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
        body = np.clip(1.0 - d, 0.0, 1.0)
        rim = np.clip(1.0 - np.abs(d - 0.9) / 0.18, 0.0, 1.0)
        region = out[y0:y1, x0:x1]
        region *= 1.0 - 0.10 * body[..., None]          # water slightly darkens
        region -= 22.0 * rim[..., None]                  # refraction rim
        # off-center specular highlight
        hy, hx = cy - 0.35 * ry, cx - 0.35 * rx
        spec = np.exp(-(((yy - hy) / (0.28 * ry)) ** 2 + ((xx - hx) / (0.28 * rx)) ** 2))
        region += 90.0 * spec[..., None]
        out[y0:y1, x0:x1] = region
    return np.clip(out, 0, 255).astype(np.uint8)


def add_mud_splashes(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clusters of small dark-brown speckles, as splashed by rain from paddy soil."""
    mask, ys, xs = _leaf_pixels(arr)
    if len(ys) < 100:
        return arr
    h, w = arr.shape[:2]
    out = arr.astype(np.float32)
    scale = max(max(h, w) / 512.0, 0.5)
    for _ in range(int(rng.integers(1, 4))):
        j = int(rng.integers(len(ys)))
        cy, cx = float(ys[j]), float(xs[j])
        sigma = rng.uniform(0.02, 0.06) * max(h, w)
        for _ in range(int(rng.integers(15, 60))):
            dy, dx = rng.normal(0, sigma), rng.normal(0, sigma)
            py, px = int(cy + dy), int(cx + dx)
            r = max(1, int(rng.uniform(1, 3.5) * scale))
            y0, y1 = max(0, py - r), min(h, py + r + 1)
            x0, x1 = max(0, px - r), min(w, px + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dot = np.clip(1.0 - np.sqrt((yy - py) ** 2 + (xx - px) ** 2) / r, 0, 1)
            alpha = dot[..., None] * rng.uniform(0.55, 0.95)
            mud = np.array(
                [rng.uniform(70, 110), rng.uniform(50, 80), rng.uniform(30, 55)],
                np.float32,
            )
            out[y0:y1, x0:x1] += alpha * (mud - out[y0:y1, x0:x1])
    return np.clip(out, 0, 255).astype(np.uint8)


def add_slight_yellowing(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gentle chlorosis-like yellow shift over part of the leaf (still healthy)."""
    mask, ys, xs = _leaf_pixels(arr)
    if len(ys) < 100:
        return arr
    h, w = arr.shape[:2]
    # spatial gradient so only one side/end of the leaf yellows
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    proj = np.cos(angle) * (xx / max(w, 1)) + np.sin(angle) * (yy / max(h, 1))
    proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
    strength = rng.uniform(0.15, 0.35)
    weight = (proj ** 1.5) * strength * mask

    out = arr.astype(np.float32)
    target = np.empty_like(out)
    target[..., 0] = np.clip(out[..., 1] * 1.12 + 24, 0, 255)  # R up toward yellow
    target[..., 1] = out[..., 1]                                # keep G (luminance)
    target[..., 2] = out[..., 2] * 0.62                         # B down
    out += weight[..., None] * (target - out)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_bend(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal row shift + small rotation — a bent / drooping leaf."""
    h, w = arr.shape[:2]
    amp = rng.uniform(0.02, 0.07) * w
    freq = rng.uniform(0.5, 1.5)
    phase = rng.uniform(0, 2 * np.pi)
    shifts = (amp * np.sin(np.linspace(0, freq * 2 * np.pi, h) + phase)).astype(int)
    pad = int(np.abs(shifts).max()) + 1
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(arr)
    for y in range(h):
        s = int(shifts[y])
        out[y] = padded[y, pad + s : pad + s + w]

    angle = float(rng.uniform(-8, 8))
    edge = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]]).mean(axis=0)
    pil = Image.fromarray(out).rotate(
        angle, resample=Image.BILINEAR, fillcolor=tuple(int(c) for c in edge)
    )
    return np.array(pil)


_SPOT_EFFECTS = [add_brown_tips, add_water_droplets, add_mud_splashes, add_slight_yellowing]


def make_hard_positive(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply 1-3 imperfections (+ optional bend + mild photo jitter) to a leaf photo."""
    img = img.convert("RGB")
    if max(img.size) > _MAX_EDGE:
        ratio = _MAX_EDGE / max(img.size)
        img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.LANCZOS,
        )
    arr = np.array(img)

    n_effects = int(rng.integers(1, 4))
    picks = rng.choice(len(_SPOT_EFFECTS), size=n_effects, replace=False)
    for k in picks:
        arr = _SPOT_EFFECTS[int(k)](arr, rng)
    if rng.random() < 0.5:
        arr = add_bend(arr, rng)

    out = Image.fromarray(arr)
    # mild capture-condition jitter, same spirit as training-time augmentation
    out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.85, 1.15)))
    out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.9, 1.1)))
    return out


# --------------------------------------------------------------------------- #
# Source selection + generation
# --------------------------------------------------------------------------- #
def _healthy_source_images(data_root: Path) -> list[Path]:
    root = data_root.resolve()
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "not_rice" in p.parts or "rice_gate_positives" in p.parts:
            continue
        if resolve_class_for_image(p, root) == "healthy":
            out.append(p)
    return out


def generate(count: int, seed: int) -> Path:
    data_root = default_training_root()
    dest = resolve_gate_positives_dir(data_root)
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
        target = dest / f"tolerant_{written:04d}_{src.stem[:32]}.jpg"
        if target.exists():
            written += 1
            continue
        try:
            with Image.open(src) as im:
                hard = make_hard_positive(im, rng)
        except Exception as exc:
            print(f"  (skipped, unreadable): {src} — {exc}")
            continue
        hard.save(target, quality=92)
        written += 1
        if written % 50 == 0 or written == count:
            print(f"  generated {written}/{count}")

    total = sum(1 for q in dest.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
    print(f"\nleaf gate-positive folder now holds {total} tolerant rice image(s).")
    return dest


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
def retrain_gates() -> None:
    from ml.train_models import REPORT_PATH, _train_rice_gate

    data_root = default_training_root()
    print("\n" + "=" * 60)
    print("Retraining Layer 0 — RF rice gate (rf_rice_gate.pkl)")
    print("=" * 60)
    metrics = _train_rice_gate(data_root)
    if metrics and REPORT_PATH.is_file():
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            report["rice_gate_model"] = metrics
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Updated rice_gate_model metrics in {REPORT_PATH.name}")
        except Exception as exc:
            print(f"(could not update training report: {exc})")

    try:
        from ml.cnn_embedder import torch_available
    except Exception:
        torch_available = None
    if torch_available is not None and torch_available():
        print("\n" + "=" * 60)
        print("Retraining CNN heads (includes the CNN rice gate)")
        print("=" * 60)
        from ml.train_cnn import main as train_cnn_main

        train_cnn_main()
    else:
        print("\nPyTorch not available — skipped the CNN gate (rf gate updated).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate imperfect-but-healthy rice positives for the Layer 0 gate."
    )
    parser.add_argument("--count", type=int, default=500, help="Images to generate (default 500).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="After generating, retrain the RF rice gate and the CNN gate.",
    )
    args = parser.parse_args()

    generate(args.count, args.seed)

    if args.retrain:
        retrain_gates()
    else:
        print(
            "\nNext step — retrain so the gate learns tolerance:\n"
            "  python ml/add_tolerant_rice.py --retrain   (gates only)\n"
            "  python ml/train_models.py                  (full retrain)"
        )


if __name__ == "__main__":
    main()
