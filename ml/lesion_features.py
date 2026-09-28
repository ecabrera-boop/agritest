"""
Lesion-aware feature extractor for rice leaf disease classification.

Implements the four formula groups from the literature, computed AFTER the
lesion is segmented out of the leaf so the statistics describe the spot, not
the whole green leaf:

  1. GLCM (Haralick) texture features
       Build p(i,j) on the lesion region (gray quantized to 8 levels, masked).
       Distances d=1,2,3 px; angles 0deg, 45deg, 90deg, 135deg; properties
       averaged across distances and angles.

         Contrast      = sum_ij (i-j)^2 * p(i,j)
         Correlation   = sum_ij [(i-mu_i)(j-mu_j) * p(i,j)] / (sigma_i*sigma_j)
         Energy (ASM)  = sum_ij p(i,j)^2
         Homogeneity   = sum_ij p(i,j) / (1 + |i-j|)
         Entropy       = -sum_ij p(i,j) * log(p(i,j))

  2. HSV color moments per H, S, V channel (computed on lesion pixels only):
         mu    = (1/N) sum p_k
         sigma = sqrt((1/N) sum (p_k - mu)^2)
         skew  = cbrt((1/N) sum (p_k - mu)^3)        # signed cube root

  3. LBP (uniform, P=8, R=1) restricted to the lesion mask; returned as the
     normalized histogram (illumination-robust texture).

  4. Shape descriptors of the lesion mask:
         Circularity  = 4*pi * Area / Perimeter^2
         Eccentricity = best-fit-ellipse eccentricity (skimage regionprops)
         AspectRatio  = major_axis / minor_axis
         Solidity     = Area / ConvexHullArea
         Extent       = Area / BoundingBoxArea

Output: 1-D float32 vector of length ``LESION_FEATURE_DIM`` (= 29).

Segmentation strategy
---------------------
Lesions are off-green patches (brown/yellow/dark) sitting inside the green
leaf. The mask is built in two stages:

  a) Plant mask: re-use ``plant_mask_from_rgb`` so we never sample background.
  b) Within the plant region, mark a pixel as lesion when its HSV signature is
     non-green-leaf (brown/yellow hue, OR low V dark spot, OR low S desaturated
     pale spot). If that yields too few pixels, fall back to Otsu on the value
     channel within the plant region (darker class = lesion).

If even the fallback finds < ``_MIN_LESION_PIXELS`` pixels, the leaf is treated
as essentially healthy: GLCM / HSV / LBP are computed on the full plant region
instead, and the shape descriptors are returned as zeros (an explicit "no
lesion" signal the classifier can learn from).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    from skimage.feature import graycomatrix, local_binary_pattern
    from skimage.filters import threshold_otsu
    from skimage.measure import label as sk_label
    from skimage.measure import regionprops
    from skimage.morphology import closing, disk, opening, remove_small_objects

    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

from .preprocessor_shared import plant_mask_from_rgb

# ---------------------------------------------------------------------------
# Tunables (kept as module constants so they show up in code review easily)
# ---------------------------------------------------------------------------

_GLCM_LEVELS = 8                       # gray-level quantization for p(i,j)
_GLCM_DISTANCES = (1, 2, 3)
_GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

_LBP_P = 8
_LBP_R = 1
_LBP_BINS = _LBP_P + 2                 # uniform method -> P+2 bins

_MIN_LESION_PIXELS = 60                # below this, treat leaf as healthy
_MIN_LEAF_PIXELS = 64

# Final feature vector layout (kept here so callers can sanity-check):
#   GLCM    : 5
#   HSV mom : 9
#   LBP hist: 10  (= _LBP_BINS)
#   shape   : 5
LESION_FEATURE_DIM = 5 + 9 + _LBP_BINS + 5  # = 29


# ---------------------------------------------------------------------------
# Color space helper
# ---------------------------------------------------------------------------

def _rgb_to_hsv_array(arr_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> HSV float32 (H,W,3) with H in [0,360), S,V in [0,1]."""
    r = arr_uint8[..., 0].astype(np.float32) / 255.0
    g = arr_uint8[..., 1].astype(np.float32) / 255.0
    b = arr_uint8[..., 2].astype(np.float32) / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(r)
    mr, mg, mb = (cmax == r) & (delta > 0), (cmax == g) & (delta > 0), (cmax == b) & (delta > 0)
    h[mr] = 60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    h[mg] = 60.0 * (((b[mg] - r[mg]) / delta[mg]) + 2)
    h[mb] = 60.0 * (((r[mb] - g[mb]) / delta[mb]) + 4)

    s = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
    v = cmax
    return np.stack([h, s, v], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Lesion segmentation
# ---------------------------------------------------------------------------

def _refine_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological cleanup; tolerant if skimage is missing."""
    if not _SKIMAGE or mask.size == 0:
        return mask
    m = opening(mask, disk(1))
    m = closing(m, disk(2))
    m = m.astype(bool)
    try:
        # skimage >= 0.26 renamed the parameter to ``max_size`` with inclusive
        # semantics (size <= max_size removed). Use ``max_size=7`` so the
        # behaviour matches the older ``min_size=8`` (size < 8 removed) on both.
        m = remove_small_objects(m, max_size=7)
    except TypeError:
        try:
            m = remove_small_objects(m, min_size=8)
        except Exception:
            pass
    except Exception:
        pass
    return m


def segment_lesion(arr_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lesion_mask, leaf_mask) — both bool arrays the same shape as ``arr_uint8``.

    Lesion = inside leaf AND (off-green hue OR dark OR pale).
    If that gives < ``_MIN_LESION_PIXELS``, an Otsu fallback on the V channel
    selects the darker pixel class inside the leaf. If even that fails, the
    returned lesion mask is all-False so the caller can fall back gracefully.
    """
    leaf_mask = plant_mask_from_rgb(arr_uint8)
    leaf_mask = _refine_binary_mask(leaf_mask)
    if leaf_mask.sum() < _MIN_LEAF_PIXELS:
        leaf_mask = np.ones(arr_uint8.shape[:2], dtype=bool)

    hsv = _rgb_to_hsv_array(arr_uint8)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Healthy rice leaf foliage sits near H in [35, 95] with strong S. A pixel
    # inside the leaf qualifies as a lesion candidate when it falls outside that
    # band in a clearly disease-like way:
    #   brown_yellow : hue rotated toward red/orange while still saturated
    #                  (Brown Spot / Blast / Bacterial Blight)
    #   dark_necrotic: very dark AND saturated (rules out plain shadow on green)
    #   pale_spot    : washed-out highlight (chlorotic patches, Leaf Smut tips)
    brown_yellow = (((h < 35) | (h > 95)) & (s >= 0.25) & (v >= 0.15))
    dark_necrotic = (v < 0.28) & (s >= 0.18)
    pale_spot = (s < 0.15) & (v >= 0.65)
    lesion = leaf_mask & (brown_yellow | dark_necrotic | pale_spot)
    lesion = _refine_binary_mask(lesion)

    if lesion.sum() >= _MIN_LESION_PIXELS:
        return lesion, leaf_mask

    # Otsu fallback inside the leaf, darker class wins.
    if _SKIMAGE and leaf_mask.any():
        v_inside = v[leaf_mask]
        if v_inside.size >= 32 and float(v_inside.max() - v_inside.min()) > 0.05:
            try:
                thr = float(threshold_otsu(v_inside))
                otsu_mask = leaf_mask & (v < thr)
                otsu_mask = _refine_binary_mask(otsu_mask)
                if otsu_mask.sum() >= _MIN_LESION_PIXELS:
                    return otsu_mask, leaf_mask
            except Exception:
                pass

    return np.zeros_like(leaf_mask, dtype=bool), leaf_mask


# ---------------------------------------------------------------------------
# Feature group 1: GLCM Haralick texture
# ---------------------------------------------------------------------------

def _glcm_features(gray_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Mask-aware GLCM. Pixels outside ``mask`` are quantized to level 0, then the
    [0,:] / [:,0] entries are dropped from p(i,j) so co-occurrences that
    include a non-lesion neighbour are not counted.
    """
    if not _SKIMAGE or not mask.any():
        return np.zeros(5, dtype=np.float32)

    # Quantize gray to [1, _GLCM_LEVELS-1] inside the mask; 0 reserved for outside.
    q = (gray_uint8.astype(np.int32) * (_GLCM_LEVELS - 1)) // 255  # in [0, L-1]
    q = q + 1                                                      # in [1, L]
    q = np.clip(q, 1, _GLCM_LEVELS - 1)
    q_masked = np.where(mask, q, 0).astype(np.uint8)

    # Crop to mask bbox so the matrix is dense.
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    patch = q_masked[y0:y1 + 1, x0:x1 + 1]
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return np.zeros(5, dtype=np.float32)

    gcm = graycomatrix(
        patch,
        distances=list(_GLCM_DISTANCES),
        angles=list(_GLCM_ANGLES),
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=False,                  # we re-normalize after dropping level 0
    )
    # Drop the dummy level (index 0) -> shape (L-1, L-1, D, A).
    gcm = gcm[1:, 1:, :, :].astype(np.float64)

    # Per (distance, angle), renormalize so p sums to 1 over the lesion-only matrix.
    sums = gcm.sum(axis=(0, 1), keepdims=True)
    sums[sums == 0] = 1.0
    p = gcm / sums

    L = p.shape[0]
    i_idx = np.arange(L).reshape(L, 1, 1, 1).astype(np.float64)
    j_idx = np.arange(L).reshape(1, L, 1, 1).astype(np.float64)

    contrast = np.sum((i_idx - j_idx) ** 2 * p, axis=(0, 1))
    energy = np.sum(p ** 2, axis=(0, 1))                                  # ASM
    homogeneity = np.sum(p / (1.0 + np.abs(i_idx - j_idx)), axis=(0, 1))  # IDM
    # Entropy with the convention 0 * log(0) = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=(0, 1))

    # Correlation:
    px = p.sum(axis=1)   # (L, D, A) — marginal over j
    py = p.sum(axis=0)   # (L, D, A) — marginal over i
    levels = np.arange(L).reshape(L, 1, 1).astype(np.float64)
    mu_i = np.sum(levels * px, axis=0)
    mu_j = np.sum(levels * py, axis=0)
    sig_i = np.sqrt(np.sum(((levels - mu_i) ** 2) * px, axis=0))
    sig_j = np.sqrt(np.sum(((levels - mu_j) ** 2) * py, axis=0))
    cov = np.sum(
        (i_idx - mu_i.reshape(1, 1, *mu_i.shape))
        * (j_idx - mu_j.reshape(1, 1, *mu_j.shape))
        * p,
        axis=(0, 1),
    )
    denom = sig_i * sig_j
    correlation = np.where(denom > 1e-12, cov / np.where(denom == 0, 1.0, denom), 0.0)

    # Average across distances and angles -> 5 scalar features.
    feats = np.array(
        [
            contrast.mean(),
            correlation.mean(),
            energy.mean(),
            homogeneity.mean(),
            entropy.mean(),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Feature group 2: HSV color moments
# ---------------------------------------------------------------------------

def _hsv_color_moments(arr_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """3 moments x 3 channels = 9 features. Hue normalized to [0,1] for scale parity."""
    if not mask.any():
        return np.zeros(9, dtype=np.float32)
    hsv = _rgb_to_hsv_array(arr_uint8)
    hsv[..., 0] = hsv[..., 0] / 360.0
    feats = []
    for c in range(3):
        ch = hsv[..., c][mask].astype(np.float64)
        if ch.size == 0:
            feats.extend([0.0, 0.0, 0.0])
            continue
        mu = float(ch.mean())
        sigma = float(np.sqrt(np.mean((ch - mu) ** 2)))
        m3 = float(np.mean((ch - mu) ** 3))
        skew = float(np.sign(m3) * np.cbrt(abs(m3)))   # signed cube root, real
        feats.extend([mu, sigma, skew])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature group 3: LBP histogram on the lesion
# ---------------------------------------------------------------------------

def _lbp_features(gray_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not _SKIMAGE or not mask.any():
        return np.zeros(_LBP_BINS, dtype=np.float32)
    # LBP expects integer images; cast explicitly to silence skimage's UserWarning.
    lbp = local_binary_pattern(gray_uint8.astype(np.uint8), P=_LBP_P, R=_LBP_R, method="uniform")
    values = lbp[mask]
    if values.size == 0:
        return np.zeros(_LBP_BINS, dtype=np.float32)
    hist, _ = np.histogram(values, bins=_LBP_BINS, range=(0, _LBP_BINS))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist = hist / s
    return hist


# ---------------------------------------------------------------------------
# Feature group 4: shape descriptors
# ---------------------------------------------------------------------------

def _shape_features(mask: np.ndarray) -> np.ndarray:
    """Five shape descriptors for the (largest) lesion blob."""
    if not _SKIMAGE or not mask.any():
        return np.zeros(5, dtype=np.float32)

    labeled = sk_label(mask.astype(np.uint8))
    regions = regionprops(labeled)
    if not regions:
        return np.zeros(5, dtype=np.float32)
    region = max(regions, key=lambda r: r.area)

    area = float(region.area)
    perimeter = float(region.perimeter) if region.perimeter else 0.0
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    eccentricity = float(region.eccentricity) if region.eccentricity is not None else 0.0

    def _first_attr(obj, names, default=0.0):
        # ``getattr(obj, x, default)`` would eagerly evaluate ``default`` even when
        # ``x`` is present, which trips skimage's deprecation warning for the old
        # alias. Iterate manually instead so only the first available name is touched.
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                return val if val is not None else default
        return default

    major = float(_first_attr(region, ("axis_major_length", "major_axis_length")) or 0.0)
    minor = float(_first_attr(region, ("axis_minor_length", "minor_axis_length")) or 0.0)
    aspect_ratio = (major / minor) if minor > 1e-6 else 0.0

    convex_area = float(_first_attr(region, ("area_convex", "convex_area")) or 0.0)
    solidity = (area / convex_area) if convex_area > 0 else 0.0

    minr, minc, maxr, maxc = region.bbox
    bbox_area = float(max(0, maxr - minr) * max(0, maxc - minc))
    extent = (area / bbox_area) if bbox_area > 0 else 0.0

    return np.array(
        [circularity, eccentricity, aspect_ratio, solidity, extent],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_lesion_features_from_array(arr_uint8: np.ndarray) -> np.ndarray:
    """
    Compute the four-group lesion feature vector from a (H,W,3) uint8 RGB array.

    The image is assumed to already be in the canonical 128x128 letterboxed
    form produced by ``ml.preprocessor.prepare_image`` so the lesion area /
    perimeter values are comparable across samples.
    """
    if arr_uint8.ndim != 3 or arr_uint8.shape[-1] != 3:
        raise ValueError(f"Expected RGB uint8 array; got shape {arr_uint8.shape}")

    lesion_mask, leaf_mask = segment_lesion(arr_uint8)
    gray = arr_uint8.mean(axis=2).astype(np.uint8)

    if lesion_mask.sum() >= _MIN_LESION_PIXELS:
        stat_mask = lesion_mask
        shape_vec = _shape_features(lesion_mask)
    else:
        # No identifiable lesion: describe the whole leaf so texture/color
        # features stay informative, but emit zero shape descriptors as an
        # explicit "no lesion" signal.
        stat_mask = leaf_mask
        shape_vec = np.zeros(5, dtype=np.float32)

    glcm_vec = _glcm_features(gray, stat_mask)
    hsv_vec = _hsv_color_moments(arr_uint8, stat_mask)
    lbp_vec = _lbp_features(gray, stat_mask)

    vec = np.concatenate([glcm_vec, hsv_vec, lbp_vec, shape_vec]).astype(np.float32)
    if vec.size != LESION_FEATURE_DIM:
        raise ValueError(
            f"Lesion feature vector has {vec.size} dims; expected {LESION_FEATURE_DIM}."
        )
    return vec


def extract_lesion_features_from_pil(img: Image.Image) -> np.ndarray:
    """PIL convenience wrapper; the image is assumed already preprocessed."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return extract_lesion_features_from_array(arr)
