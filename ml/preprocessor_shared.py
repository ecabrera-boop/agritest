"""
Shared leaf preprocessing (train + inference).

Background removal is disabled by default (``REMOVE_BG_DEFAULT = False``) so the model
analyses the full original scene. Segmentation helpers remain for Layer 0 and optional
``remove_bg=True`` experiments.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import binary_fill_holes as _scipy_binary_fill_holes

    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from skimage.measure import label as sk_label
    from skimage.morphology import closing, disk, opening

    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

_IMG_SIZE = 128
REMOVE_BG_DEFAULT = False

# Neutral backdrop after background removal (not pure white — avoids blown highlights)
_BG_COLOR = (235, 240, 235)
# Gentle luminance pull on masked leaf pixels only (no saturation/contrast boosts)
_LUMA_TARGET = 118.0
_LUMA_BLEND = 0.12
_CROP_MARGIN_FRAC = 0.06
# Segment on a downscaled copy for speed; final letterbox still targets ``size``.
_MAX_SEG_EDGE = 384


def _leaf_color_masks(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HSV-based leaf colour bands: green, yellow-brown, g-dominant, skin-like."""
    r = arr[:, :, 0].astype(np.float32) / 255.0
    g = arr[:, :, 1].astype(np.float32) / 255.0
    b = arr[:, :, 2].astype(np.float32) / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(r)
    mask_r = cmax == r
    mask_g = cmax == g
    mask_b = cmax == b
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)) + 4)

    sat = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
    val = cmax

    green_leaf = (hue >= 35) & (hue <= 95) & (sat >= 0.18) & (val >= 0.15)
    yellow_brown = (
        (hue >= 18) & (hue < 38) & (sat >= 0.12) & (val >= 0.12) & (g >= r * 0.82)
    )
    g_dominant = (g > 0.31) & (g > r * 1.05) & (g > b * 1.05)
    skin_like = (hue < 28) & (sat >= 0.12) & (val >= 0.30) & (r > g * 0.88) & (r > b)
    return green_leaf, yellow_brown, g_dominant, skin_like


def plant_mask_from_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Boolean mask of plant-like pixels (green + yellow-brown leaf, excludes skin).

    Shared by background removal and Layer 0 heuristics so train/serve/gate agree.
    """
    green_leaf, yellow_brown, g_dominant, skin_like = _leaf_color_masks(arr)
    return (green_leaf | yellow_brown | g_dominant) & ~skin_like


def green_leaf_fraction(arr: np.ndarray) -> float:
    """
    Fraction of pixels in the green-leaf HSV band (excludes tan water/mud).

    Used by the CNN rice gate to recognise a flooded young paddy (lots of
    green blades) without treating brown objects as vegetation.
    """
    green_leaf, _, _, skin_like = _leaf_color_masks(arr)
    return float((green_leaf & ~skin_like).mean())


def _refine_plant_mask(mask: np.ndarray) -> np.ndarray:
    if not _SKIMAGE:
        return mask
    m = opening(mask, disk(1))
    m = closing(m, disk(2))
    return m


def _resize_for_segmentation(arr: np.ndarray, max_edge: int = _MAX_SEG_EDGE) -> tuple[np.ndarray, float]:
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return arr, 1.0
    scale = max_edge / long_edge
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = np.array(Image.fromarray(arr).resize((nw, nh), Image.LANCZOS))
    return small, scale


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    if not _SKIMAGE:
        return mask
    labeled = sk_label(mask)
    if labeled.max() == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def _fill_holes_by_border_flood(mask: np.ndarray) -> np.ndarray:
    """Treat background pixels unreachable from the image border as interior leaf."""
    bg = ~mask
    h, w = mask.shape
    reached = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        if bg[0, x] and not reached[0, x]:
            reached[0, x] = True
            queue.append((0, x))
        if bg[h - 1, x] and not reached[h - 1, x]:
            reached[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if bg[y, 0] and not reached[y, 0]:
            reached[y, 0] = True
            queue.append((y, 0))
        if bg[y, w - 1] and not reached[y, w - 1]:
            reached[y, w - 1] = True
            queue.append((y, w - 1))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not reached[ny, nx]:
                reached[ny, nx] = True
                queue.append((ny, nx))

    return mask | (bg & ~reached)


def _fill_holes_in_mask(mask: np.ndarray) -> np.ndarray:
    if _SCIPY:
        return _scipy_binary_fill_holes(mask)
    return _fill_holes_by_border_flood(mask)


def build_leaf_region_mask(arr: np.ndarray) -> np.ndarray:
    """
    Plant colour mask → largest blob → fill interior holes.

    Lesion pixels that are not green/brown but sit fully inside the leaf outline
    are included in the returned region.
    """
    mask = _refine_plant_mask(plant_mask_from_rgb(arr))
    mask = _largest_component(mask)
    if mask.sum() < 32:
        return np.ones(arr.shape[:2], dtype=bool)
    return _fill_holes_in_mask(mask)


def remove_background(
    arr: np.ndarray,
    bg: tuple[int, int, int] = _BG_COLOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blank pixels outside the filled leaf region; keep the entire leaf body.

    Returns composited RGB array and the boolean leaf-region mask used.
    """
    mask = build_leaf_region_mask(arr)
    return _composite_background(arr, mask, bg), mask


def _composite_background(arr: np.ndarray, mask: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    out = np.empty_like(arr)
    out[:] = np.array(bg, dtype=np.uint8)
    out[mask] = arr[mask]
    return out


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray, margin_frac: float) -> np.ndarray:
    if not mask.any():
        return arr
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = arr.shape[:2]
    margin = int(max(y1 - y0 + 1, x1 - x0 + 1) * margin_frac) + 1
    y0 = max(0, y0 - margin)
    y1 = min(h - 1, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(w - 1, x1 + margin)
    return arr[y0 : y1 + 1, x0 : x1 + 1].copy()


def _gentle_luminance_normalize(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Slight per-image luminance alignment on leaf pixels; preserves hue ratios."""
    if mask.sum() < 64:
        return arr
    rgb = arr.astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    leaf_luma = luma[mask]
    mean_luma = float(leaf_luma.mean())
    if abs(mean_luma - _LUMA_TARGET) < 4.0:
        return arr
    scale = 1.0 + (_LUMA_TARGET - mean_luma) / 255.0 * _LUMA_BLEND
    scaled = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    out = arr.copy()
    out[mask] = scaled[mask]
    return out


def _letterbox_resize(img: Image.Image, size: int, bg: tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    if w < 1 or h < 1:
        raise ValueError(f"Invalid image size {img.size!r}; cannot resize.")
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess_leaf(
    img: Image.Image,
    size: int = _IMG_SIZE,
    *,
    remove_bg: bool | None = None,
) -> Image.Image:
    """
    Canonical train/serve preprocessing: optional background removal + light normalize + resize.

    Background removal is controlled by ``remove_bg`` (defaults to ``REMOVE_BG_DEFAULT``).
    """
    if remove_bg is None:
        remove_bg = REMOVE_BG_DEFAULT

    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)

    if remove_bg:
        seg_arr, _ = _resize_for_segmentation(arr)
        seg_arr, mask = remove_background(seg_arr, _BG_COLOR)
        seg_arr = _crop_to_mask(seg_arr, mask, _CROP_MARGIN_FRAC)
        mask = plant_mask_from_rgb(seg_arr)
        if not mask.any():
            mask = np.ones(seg_arr.shape[:2], dtype=bool)
    else:
        seg_arr = arr
        mask = np.ones(seg_arr.shape[:2], dtype=bool)

    seg_arr = _gentle_luminance_normalize(seg_arr, mask)
    return _letterbox_resize(Image.fromarray(seg_arr), size, _BG_COLOR)
