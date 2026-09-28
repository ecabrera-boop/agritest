"""
CNN image embedder (transfer learning).

A pretrained MobileNetV3-Large backbone is used *frozen* as a feature extractor:
each image becomes a 960-dim L2-normalized "fingerprint" that captures shape,
texture and lesion patterns — features that transfer across cameras/lighting far
better than the hand-crafted colour/LBP/GLCM histograms used by the rf_* models.

These embeddings power the deep-learning pipeline:
  - rice gate      : out-of-distribution distance to the rice-leaf database
  - health/disease : light classifier heads + k-NN "compare to database"

Strong train-time augmentation (and matching deterministic TTA at serve time)
teaches the heads to recognise leaf *texture*, not exact pose:
  - rotate ±20°
  - zoom in randomly by up to 20%
  - brightness / contrast ±15%

No GPU and no backbone training are required — inference on CPU embeds a few
thousand images in minutes. Only the small heads are trained (ml/train_cnn.py).

The backbone weights (~20 MB) download from the torchvision hub on first use and
are then cached under ~/.cache/torch. If torch is not installed the module raises
on import of the heavy bits only when actually used, so the rest of the app keeps
working with the hand-crafted models.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

EMBED_DIM = 960
_BATCH = 32

# Strong-aug / TTA contract — keep train (ml/train_cnn.py) and serve
# (ml/cnn_pipeline.py) on the same family of transforms.
AUG_ROTATE_DEG = 20.0
AUG_ZOOM_MAX = 0.20          # zoom-in factor up to +20%
AUG_BRIGHTNESS_DELTA = 0.15  # ±15%
AUG_CONTRAST_DELTA = 0.15    # ±15%
TRAIN_AUG_PER_IMAGE = 3      # extra variants per training image (original kept)
N_TTA = 5                    # original + 4 deterministic variants at inference


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_backbone():
    """Frozen MobileNetV3-Large feature extractor + its ImageNet transforms."""
    import torch
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        mobilenet_v3_large,
    )

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
    # Drop the classifier; keep convolutional features + global pooling.
    model.classifier = torch.nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    return model, weights.transforms()


def _to_rgb(image) -> "PILImage":  # noqa: F821
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def embed_images(images: Iterable) -> np.ndarray:
    """
    Embed an iterable of image paths / PIL images.

    Returns an (N, EMBED_DIM) L2-normalized float32 array. Cosine similarity is
    then just a dot product between rows.
    """
    import torch

    model, transform = _load_backbone()
    items = list(images)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(items), _BATCH):
            batch = items[start : start + _BATCH]
            tensors = torch.stack([transform(_to_rgb(im)) for im in batch])
            feats = model(tensors).cpu().numpy().astype(np.float32)
            out.append(feats)
    if not out:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return _l2_normalize(np.vstack(out))


def embed_image(image) -> np.ndarray:
    """Embed a single image → (EMBED_DIM,) L2-normalized float32 vector."""
    vec = embed_images([image])
    return vec[0] if len(vec) else np.zeros(EMBED_DIM, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Strong augmentation (train) + matching TTA (serve)
# --------------------------------------------------------------------------- #
def _mean_fill(img: "PILImage") -> tuple[int, int, int]:  # noqa: F821
    arr = np.asarray(img)
    if arr.size == 0:
        return (235, 240, 235)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(c) for c in mean)


def _zoom_in(img: "PILImage", zoom: float) -> "PILImage":  # noqa: F821
    """Center-crop then resize back — zoom > 1.0 means zoom *in*."""
    from PIL import Image as _Image

    if zoom <= 1.001:
        return img
    w, h = img.size
    nw = max(1, int(round(w / zoom)))
    nh = max(1, int(round(h / zoom)))
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), _Image.LANCZOS)


def strong_augment_pil(img, rng: np.random.Generator):
    """
    Strong train-time augmentation for CNN heads.

    Rotate ±20°, zoom in randomly by up to 20%, brightness/contrast ±15%.
    Forces the model to learn leaf texture rather than exact pose or framing —
    so water ripples and slight camera angles at inference do not fool it.
    """
    from PIL import Image as _Image
    from PIL import ImageEnhance

    out = _to_rgb(img)
    angle = float(rng.uniform(-AUG_ROTATE_DEG, AUG_ROTATE_DEG))
    out = out.rotate(
        angle,
        resample=_Image.BILINEAR,
        expand=False,
        fillcolor=_mean_fill(out),
    )
    out = _zoom_in(out, float(rng.uniform(1.0, 1.0 + AUG_ZOOM_MAX)))
    out = ImageEnhance.Brightness(out).enhance(
        float(rng.uniform(1.0 - AUG_BRIGHTNESS_DELTA, 1.0 + AUG_BRIGHTNESS_DELTA))
    )
    out = ImageEnhance.Contrast(out).enhance(
        float(rng.uniform(1.0 - AUG_CONTRAST_DELTA, 1.0 + AUG_CONTRAST_DELTA))
    )
    return out


def tta_variants_pil(img) -> list:
    """
    Deterministic TTA variants in the same transform family as ``strong_augment_pil``.

    Returns N_TTA images: original, ±20° rotate, zoom+brightness, zoom+contrast.
    Averaging head probabilities over these matches train-time exposure.
    """
    from PIL import Image as _Image
    from PIL import ImageEnhance

    base = _to_rgb(img)
    fill = _mean_fill(base)
    return [
        base,
        base.rotate(AUG_ROTATE_DEG, resample=_Image.BILINEAR, expand=False, fillcolor=fill),
        base.rotate(-AUG_ROTATE_DEG, resample=_Image.BILINEAR, expand=False, fillcolor=fill),
        ImageEnhance.Brightness(_zoom_in(base, 1.0 + 0.75 * AUG_ZOOM_MAX)).enhance(
            1.0 + 0.67 * AUG_BRIGHTNESS_DELTA
        ),
        ImageEnhance.Contrast(_zoom_in(base, 1.0 + 0.50 * AUG_ZOOM_MAX)).enhance(
            1.0 - 0.67 * AUG_CONTRAST_DELTA
        ),
    ]


def embed_image_tta(image) -> np.ndarray:
    """Embed deterministic TTA variants → shape (N_TTA, EMBED_DIM)."""
    return embed_images(tta_variants_pil(image))


def make_strong_augmented_views(
    path: str,
    n_aug: int = TRAIN_AUG_PER_IMAGE,
    *,
    seed: int,
) -> list:
    """Build ``n_aug`` strong-aug PIL views of ``path`` (deterministic given seed)."""
    rng = np.random.default_rng(seed)
    base = _to_rgb(path)
    return [strong_augment_pil(base, rng) for _ in range(n_aug)]
