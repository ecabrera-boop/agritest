"""
Lesion-emphasis view for the CNN disease head.

Frozen ImageNet embeddings of a whole field photo often match canopy *scene*
(overlapping blades, soil, glare) instead of lesion shape. Blast and brown
spot then swap because both are "green rice with brown spots".

This helper fades non-lesion pixels toward the leaf colour so MobileNet's
next pass is forced to look at the spots. Used only as a blast ↔ brown-spot
referee — the full-frame embedding remains the primary signal.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

from .lesion_features import segment_lesion

PathOrImage = Union[str, Image.Image]

# Downscale before masking — MobileNet input is 224px anyway, and full-res
# field photos made the HSV mask noisy.
EMPHASIS_MAX_EDGE = 384
EMPHASIS_FADE = 0.25  # keep this fraction of the original non-lesion pixel
_MIN_LESION_PIXELS = 80


def _to_rgb(image: PathOrImage) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def make_lesion_emphasis_image(
    image: PathOrImage,
    *,
    max_edge: int = EMPHASIS_MAX_EDGE,
    fade: float = EMPHASIS_FADE,
) -> Image.Image:
    """
    Return a RGB image with non-lesion pixels faded toward the leaf mean.

    If too few lesion pixels are found, the (resized) original is returned
    unchanged so the referee becomes a no-op rather than a hallucinated mask.
    """
    img = _resize_max_edge(_to_rgb(image), max_edge)
    arr = np.asarray(img)
    lesion, leaf = segment_lesion(arr)
    if int(lesion.sum()) < _MIN_LESION_PIXELS:
        return img

    out = arr.astype(np.float32)
    leaf_mean = (
        arr[leaf].mean(axis=0) if leaf.any() else arr.reshape(-1, 3).mean(axis=0)
    )
    fade_mask = ~lesion
    keep = float(np.clip(fade, 0.0, 1.0))
    out[fade_mask] = out[fade_mask] * keep + leaf_mean.astype(np.float32) * (1.0 - keep)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
