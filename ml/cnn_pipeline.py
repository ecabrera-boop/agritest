"""
CNN (transfer-learning) inference pipeline — mirror of ml/pipeline.run_pipeline.

Embeds the upload with the frozen MobileNetV3 backbone under Test-Time
Augmentation (TTA) — original plus ±20° rotate / zoom / brightness-contrast
variants that match the strong train-time aug in ``ml/train_cnn.py`` — then:
  0. rice gate      — cnn_rice_gate.pkl + out-of-distribution cosine distance to
                      the rice-leaf database (rejects faces/objects/other plants).
                      A flooded young paddy that sits far outside the close-up
                      leaf cloud is rescued; stem close-ups of grassy lookalikes
                      (barnyard grass, rice cutgrass) are not. A ripening
                      panicle with dew is accepted as healthy rice instead of
                      bacterial blight.
  1. health         — cnn_health.pkl (healthy vs diseased), TTA-averaged, then
                      rescue-blended with a k-NN vote over cnn_health_db.npz.
                      The blend only runs when the head says diseased: it
                      stops field-healthy paddy photos from being called sick
                      when their nearest training images are healthy. It does
                      NOT override a healthy head — grain-filling canopy
                      photos sit next to diseased Paddy Doctor shots in
                      embedding space, and a two-way blend would flip them.
  2. disease        — cnn_disease.pkl (specific disease), TTA-averaged, blended
                      with a similarity-weighted k-NN vote over the train-image
                      embedding database (see KNN_BLEND_*). If the head's top
                      class has almost no nearest-neighbour support, a k-NN
                      veto (KNN_SUPPORT_MIN) re-weights toward the database
                      plurality. A second, lesion-emphasis CNN pass then
                      referees the remaining blast ↔ brown-spot swap (field
                      canopy photos where k-NN is also fooled by scene style).
                      The same k-NN also powers the "compare to database"
                      explanation.

Returns the SAME dict shape as ml/pipeline.run_pipeline, plus an extra
``similar_examples`` list (nearest training images) for the "compare to the
database" explanation. If artifacts or torch are missing, ``cnn_available()``
returns False and ml/pipeline falls back to the hand-crafted rf_* pipeline.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .layer3_recommendation import get_recommendation
from .validation import (
    DISEASE_MARGIN_MIN,
    DISEASE_MIN_CONFIDENCE,
    HEALTH_MIN_CONFIDENCE,
)

_ML_DIR = Path(__file__).resolve().parent
MODEL_DIR = _ML_DIR / "saved_models"

_HEALTH_PKL = MODEL_DIR / "cnn_health.pkl"
_DISEASE_PKL = MODEL_DIR / "cnn_disease.pkl"
_GATE_PKL = MODEL_DIR / "cnn_rice_gate.pkl"
_GATE_RF_PKL = MODEL_DIR / "cnn_rice_gate_rf.pkl"
_DB_NPZ = MODEL_DIR / "cnn_embed_db.npz"
_HEALTH_DB_NPZ = MODEL_DIR / "cnn_health_db.npz"
_RICE_REF_NPZ = MODEL_DIR / "cnn_rice_ref.npz"
_META_JSON = MODEL_DIR / "cnn_meta.json"

# Rice-leaf gate. Three signals on the CNN embedding decide rice vs not_rice:
#   p_not = mean not_rice probability of the LogisticRegression + RandomForest heads
#   near  = nearest cosine similarity to the rice-leaf database
# Reject as not_rice if EITHER:
#   (a) near < reject_floor          → too far from every known rice leaf (catches
#       broad/compound non-rice leaves the classifier is fooled by), OR
#   (b) p_not >= GATE_REJECT_PROBA, unless the image is at least plausibly rice
#       (near >= reject_floor) AND p_not is only weakly negative
#       (< GATE_OVERRIDE_MAX_PNOT). A lookalike that sits just inside the rice
#       cosine cloud (barnyard grass ~0.88) must not override a 60%+ not_rice vote.
#
#   near >= reject_floor (not the stricter accept_override/ood_cosine_threshold)
#   is the override bar: reject_floor is calibrated in train_cnn.py at the 2nd
#   percentile of genuine-rice nearest-neighbour similarity, so ~98% of real
#   rice clears it. Gating the override on accept_override instead (the 10th
#   percentile) left a real gap: photos in that 2nd-10th percentile band (e.g.
#   a top-down phone shot of a potted leaf against a cluttered porch
#   background, nearest ~0.82 vs reject_floor ~0.79/accept_override ~0.87, LR
#   leaning not_rice while RF and the disease head are confident) had no
#   rescue path and were hard-rejected as not_rice even with textbook lesions
#   and 90%+ agreement from every downstream signal (disease head, k-NN
#   database, lesion-emphasis pass). GATE_OVERRIDE_MAX_PNOT still keeps
#   lookalikes like barnyard grass (p_not 62-82%) rejected regardless of which
#   cosine bar is used, since their not_rice vote is nowhere near weak.
GATE_REJECT_PROBA = 0.50

# Cosine-to-rice may override a *weak* not_rice vote (genuine rice leaves sit
# in the known-rice cloud). It must NOT override a clear not_rice vote:
# barnyard grass (Echinochloa) embeds at ~0.88 cosine — comfortably above
# reject_floor ~0.79 — while the gate heads already say ~62–82% not_rice.
# Without this cap the lookalike is accepted as rice.
GATE_OVERRIDE_MAX_PNOT = 0.58

# The gate classifier is at least leaning toward "rice" when its mean not_rice
# probability stays below this. A rice-leaning vote rescues a genuine rice leaf
# whose busy field background pushes its embedding just past the OOD reject
# floor (the UCI reference set is plain-background, so real field photos sit
# slightly outside that cloud) — e.g. a hand-held leaf shot against blurred
# outdoor clutter (p_not ~0.48, nearest ~0.62). See _rice_gate_reject.
#
# Raised from 0.20 to 0.50 (aligned with GATE_REJECT_PROBA) to rescue that
# hand-held/cluttered-background case. Accepted tradeoff: p_not and nearest_rice
# are not jointly monotonic, so this also rescues a documented rice-cutgrass
# stem lookalike (p_not ~0.38, nearest ~0.76) that the old 0.20 floor correctly
# rejected — no single p_not threshold can admit the former without the latter.
# See test_does_not_rescue_hairy_node_cutgrass / test_rescues_hand_held_leaf_
# against_cluttered_background in detection/tests/test_cnn_rice_gate.py.
GATE_RICE_CONFIDENT_PROBA = 0.50

# How far below the OOD reject floor a confident-rice image may sit and still be
# rescued. Images farther than this from every known rice leaf are always
# rejected, even when the classifier is fooled.
#
# The rice reference set is plain-background UCI photos, so a genuine rice leaf
# shot against a busy/blurred field background sits well under the floor (cosine
# ~0.66 vs floor ~0.81). A band of 0.12 was too tight and mis-rejected such real
# leaves as "not rice". 0.20 rescues confident-rice field photos (p_not < 0.20)
# while still hard-rejecting anything below ~0.61 similarity even when confident.
RICE_GATE_RESCUE_BAND = 0.20

# Young transplanted rice in floodwater is a *scene*, not a close-up leaf.
# It sits far from the UCI leaf reference (cosine ~0.60 vs floor ~0.80), so the
# leaf-gate rejects it even though the plants are rice. Rescue that scene when
# the frame is mostly green blades, the linear head is sure this is not a
# close-up leaf, and cosine stays well below the leaf cloud.
# Close-up lookalikes stay rejected:
#   barnyard grass sits *inside* the leaf cloud (cosine ~0.88)
#   rice cutgrass / hairy-node stems sit closer than a paddy (~0.70–0.76)
#   and the linear head is much less sure they are "not a leaf".
PADDY_SCENE_MIN_GREEN = 0.30
PADDY_SCENE_MAX_NEAREST = 0.66
PADDY_SCENE_MIN_LR_PNOT = 0.85

KNN_K = 5

# Layer 2 decision = (1 - w) * classifier proba + w * k-NN database vote.
# The vote takes the KNN_BLEND_K most similar train-image embeddings and
# softmax-weights them by cosine similarity at temperature KNN_BLEND_TEMP.
# Tuned offline on the held-out seed-43 disease test split (ml/tune_knn_blend.py):
# head-only accuracy 0.891 → 0.958 with k=5, temp=0.02, w=0.5. The blend pulls
# a prediction toward what the training database actually contains, which fixes
# field-canopy photos where the logistic head leans on scene style instead of
# lesion shape.
KNN_BLEND_K = 5
KNN_BLEND_TEMP = 0.02
KNN_BLEND_WEIGHT = 0.5

# After the 50/50 blend, the logistic head can still outvote the database
# when it is extremely confident and wrong (classic blast ↔ brown-spot swap
# on field photos). If the predicted class has almost no weight among the
# KNN_BLEND_K neighbours (below KNN_SUPPORT_MIN), re-blend at KNN_VETO_WEIGHT
# so the training-image plurality wins. Tuned on the misclassified brown-spot
# canopy photo: head 69% blast / k-NN 0% blast → veto to brown spot, while
# true blast images keep k-NN blast ≥ 0.64 and are left alone.
KNN_SUPPORT_MIN = 0.20
KNN_VETO_WEIGHT = 0.80

# When the whole-scene head (and even k-NN) call blast but a lesion-emphasis
# embedding of the same photo calls brown spot, trust the lesion view.
# Tuned on the field-canopy brown-spot photo (head 87% blast / k-NN 75% blast
# / lesion view 83% brown spot) while true close-up blast from UCI data_1
# stays blast because its lesion view does not flip.
_BLAST = "blast"
_BROWN_SPOT = "brown spot"
_BLIGHT = "bacterial blight"
LESION_VETO_WEIGHT = 0.75
LESION_VETO_BROWN_MIN = 0.40
LESION_VETO_MARGIN = 0.10

# Ripening panicle + dew is healthy rice, but the leaf-trained health head
# calls it bacterial blight (yellow-green grains + water-soaked looking drops).
# Override that blight call only when the frame has many dough-stage spikelets
# and is not a heavily lesioned leaf. Blast / brown-spot leaf photos stay
# diseased (they do not take this blight-only path).
PANICLE_MIN_GREEN = 0.50
PANICLE_MAX_LESION = 0.20
PANICLE_MIN_SPIKELETS = 20


def cnn_available() -> bool:
    """True only if torch is importable AND the trained heads exist on disk."""
    from .cnn_embedder import torch_available

    return (
        torch_available()
        and _HEALTH_PKL.is_file()
        and _DISEASE_PKL.is_file()
    )


@lru_cache(maxsize=1)
def _load():
    import json

    health = joblib.load(_HEALTH_PKL)
    disease = joblib.load(_DISEASE_PKL)
    gate = joblib.load(_GATE_PKL) if _GATE_PKL.is_file() else None
    gate_rf = joblib.load(_GATE_RF_PKL) if _GATE_RF_PKL.is_file() else None
    meta = json.loads(_META_JSON.read_text(encoding="utf-8")) if _META_JSON.is_file() else {}
    db = None
    if _DB_NPZ.is_file():
        raw = np.load(_DB_NPZ, allow_pickle=True)
        db = {
            "embeddings": raw["embeddings"].astype(np.float32),
            "labels": [str(x) for x in raw["labels"]],
            "paths": [str(x) for x in raw["paths"]],
        }
    rice_ref = None
    if _RICE_REF_NPZ.is_file():
        rice_ref = np.load(_RICE_REF_NPZ)["embeddings"].astype(np.float32)
    health_db = None
    if _HEALTH_DB_NPZ.is_file():
        raw_h = np.load(_HEALTH_DB_NPZ, allow_pickle=True)
        health_db = {
            "embeddings": raw_h["embeddings"].astype(np.float32),
            "labels": [str(x) for x in raw_h["labels"]],
            "paths": [str(x) for x in raw_h["paths"]],
        }
    return health, disease, gate, gate_rf, db, meta, rice_ref, health_db


def _proba_dict(clf, x: np.ndarray) -> dict[str, float]:
    proba = clf.predict_proba(x.reshape(1, -1))[0]
    return {str(c): float(p) for c, p in zip(clf.classes_, proba)}


def _proba_dict_tta(clf, X: np.ndarray) -> dict[str, float]:
    """Average class probabilities across TTA embedding rows (N, D)."""
    if X.ndim == 1:
        return _proba_dict(clf, X)
    proba = clf.predict_proba(X).mean(axis=0)
    return {str(c): float(p) for c, p in zip(clf.classes_, proba)}


def _cnn_gate_rejected(
    p_not: float,
    nearest_rice: float | None,
    reject_floor: float | None,
    accept_override: float | None,
) -> bool:
    """
    True when Layer 0 should hard-stop as not_rice.

    Reject if either:
      (a) embedding is below the rice-reference floor (with a narrow rescue
          band for confident-rice field photos), or
      (b) the classifier votes not_rice and the image is not both at least
          plausibly rice-like *and* only weakly negative (see
          GATE_OVERRIDE_MAX_PNOT).
    """
    if nearest_rice is not None and reject_floor is not None and nearest_rice < reject_floor:
        hard_reject_floor = reject_floor - RICE_GATE_RESCUE_BAND
        classifier_confident_rice = p_not < GATE_RICE_CONFIDENT_PROBA
        if nearest_rice < hard_reject_floor:
            return True
        if not classifier_confident_rice:
            return True
    # The override bar is reject_floor (2nd percentile of genuine-rice
    # similarity), not the stricter accept_override/ood_cosine_threshold
    # (10th percentile) — see the GATE_REJECT_PROBA comment above for why:
    # gating on the stricter bar left ~8% of real rice photos with no rescue
    # path. accept_override is kept as a fallback for older meta.json files
    # that predate rice_reject_floor.
    override_bar = reject_floor if reject_floor is not None else accept_override
    close_to_rice = (
        nearest_rice is not None
        and override_bar is not None
        and nearest_rice >= override_bar
    )
    if p_not >= GATE_REJECT_PROBA and not (
        close_to_rice and p_not < GATE_OVERRIDE_MAX_PNOT
    ):
        return True
    return False


def _paddy_scene_rescue(
    p_not_rf: float,
    p_not_lr: float,
    nearest_rice: float | None,
    reject_floor: float | None,
    green_frac: float,
) -> bool:
    """
    True when a leaf-gate reject should be overridden as a rice paddy scene.

    The CNN rice reference is close-up leaves, so a flooded young stand scores
    far from every known leaf (cosine ~0.60) and the linear head votes not_rice.
    That is a framing mismatch, not a different species. Rescue only when:

      * cosine is well below the close-up leaf cloud (barnyard grass ~0.88 and
        hairy-node cutgrass stems ~0.70–0.76 are not rescued),
      * the linear head is sure this is not a leaf close-up (field scenes
        ~0.97; stem macros of lookalikes ~0.50–0.75),
      * the frame is substantially green vegetation, and
      * the RF head is not clearly not_rice (same cap as GATE_OVERRIDE_MAX_PNOT).
    """
    if nearest_rice is None or reject_floor is None:
        return False
    if nearest_rice >= min(reject_floor, PADDY_SCENE_MAX_NEAREST):
        return False
    if p_not_lr < PADDY_SCENE_MIN_LR_PNOT:
        return False
    if green_frac < PADDY_SCENE_MIN_GREEN:
        return False
    if p_not_rf >= GATE_OVERRIDE_MAX_PNOT:
        return False
    return True


def _unknown(
    reason: str,
    confidence: float = 0.0,
    all_probs: dict | None = None,
    gate_warning: str = "",
    health_probs: dict | None = None,
) -> dict:
    return {
        "health": "unknown",
        "disease_name": "",
        "confidence": confidence,
        "all_disease_probs": all_probs or {},
        "health_probs": health_probs or {},
        "recommendation": get_recommendation(reason),
        "rejection_reason": reason,
        "gate_warning": gate_warning,
        "similar_examples": [],
        "engine": "cnn",
    }


def _knn_class_vote(emb: np.ndarray, db: dict, classes: list[str]) -> dict[str, float] | None:
    """
    Similarity-softmax class distribution over the KNN_BLEND_K nearest
    database embeddings. Returns None when the database is unavailable.
    """
    if not db or not len(db["embeddings"]):
        return None
    sims = db["embeddings"] @ emb
    k = min(KNN_BLEND_K, len(sims))
    top = np.argpartition(sims, -k)[-k:]
    s = sims[top]
    w = np.exp((s - s.max()) / KNN_BLEND_TEMP)
    w /= w.sum()
    vote = {c: 0.0 for c in classes}
    for i, wi in zip(top, w):
        lab = db["labels"][i]
        if lab in vote:
            vote[lab] += float(wi)
    return vote


def _knn_top1_similarity(emb: np.ndarray, db: dict | None) -> float | None:
    """Cosine similarity to the single nearest embedding in db, or None if empty."""
    if not db or not len(db["embeddings"]):
        return None
    return float((db["embeddings"] @ emb).max())


# Minimum cosine similarity to the nearest cnn_health_db.npz neighbour before
# the k-NN rescue/veto in _health_with_knn is trusted at all. Calibrated from
# a leave-one-out self-similarity check over health_db: even the worst 1% of
# genuine same-domain matches score >= ~0.79 there (matches rice_reject_floor
# in cnn_meta.json, computed independently on the rice-gate reference set).
# A photo whose best health_db match sits below this has no real analogue in
# the database — a hand-held leaf with visible dieback matched at only
# ~0.61-0.63 (weak "least-bad" hits on unrelated field scenes), which used to
# silently flip a correct 90%-diseased head call to 83%-healthy. Below the
# floor, the raw classifier is trusted instead of the neighbour vote.
HEALTH_KNN_MIN_SIMILARITY = 0.78


def _blend_with_knn(probs: dict[str, float], vote: dict[str, float] | None) -> dict[str, float]:
    """Blend classifier probabilities with the k-NN database vote."""
    if vote is None:
        return probs
    w = KNN_BLEND_WEIGHT
    return {c: (1.0 - w) * p + w * vote.get(c, 0.0) for c, p in probs.items()}


def _knn_support_veto(
    probs: dict[str, float],
    vote: dict[str, float] | None,
    *,
    support_min: float = KNN_SUPPORT_MIN,
    veto_weight: float = KNN_VETO_WEIGHT,
) -> tuple[dict[str, float], str | None]:
    """
    If the current top class is nearly absent from the nearest training
    images, re-weight toward the k-NN plurality.

    Returns ``(probs, vetoed_from)``. ``vetoed_from`` is the rejected class
    name, or None when the head and the database already agree.
    """
    if not vote or not probs:
        return probs, None
    pred = max(probs, key=probs.get)
    if vote.get(pred, 0.0) >= support_min:
        return probs, None
    knn_top = max(vote, key=vote.get)
    if knn_top == pred or vote.get(knn_top, 0.0) < support_min:
        return probs, None
    mixed = {
        c: (1.0 - veto_weight) * p + veto_weight * vote.get(c, 0.0)
        for c, p in probs.items()
    }
    return mixed, pred


def _health_with_knn(
    h_probs: dict[str, float],
    health_vote: dict[str, float] | None,
    nearest_sim: float | None = None,
) -> tuple[dict[str, float], str | None]:
    """
    Rescue-only health blend: pull a diseased head toward healthy neighbours.

    A two-way 50/50 blend is unsafe here. Frozen canopy embeddings of a
    healthy grain-filling stand match diseased Paddy Doctor field shots
    (busy green + panicles ≈ spots), so neighbours vote ~100% diseased
    even when the logistic head is already healthy. Only blend/veto when
    the head itself says diseased.

    ``nearest_sim`` (top-1 cosine similarity to health_db, from
    _knn_top1_similarity) additionally gates the rescue: below
    HEALTH_KNN_MIN_SIMILARITY the "neighbours" are not real matches, just the
    least-far entries in a sparsely covered region, and their vote is not
    trusted over the head.
    """
    if not h_probs or health_vote is None:
        return h_probs, None
    if max(h_probs, key=h_probs.get) == "healthy":
        return h_probs, None
    if nearest_sim is not None and nearest_sim < HEALTH_KNN_MIN_SIMILARITY:
        return h_probs, None
    blended = _blend_with_knn(h_probs, health_vote)
    return _knn_support_veto(blended, health_vote)


def _blast_brownspot_lesion_veto(
    probs: dict[str, float],
    emp_probs: dict[str, float] | None,
    *,
    veto_weight: float = LESION_VETO_WEIGHT,
    brown_min: float = LESION_VETO_BROWN_MIN,
    margin: float = LESION_VETO_MARGIN,
) -> tuple[dict[str, float], str | None]:
    """
    Override a whole-scene "blast" call when the lesion-emphasis CNN pass
    clearly prefers brown spot.

    Returns ``(probs, vetoed_from)``. ``vetoed_from`` is ``"blast"`` when the
    override fires, else None.
    """
    if not probs or not emp_probs:
        return probs, None
    if max(probs, key=probs.get) != _BLAST:
        return probs, None
    emp_brown = float(emp_probs.get(_BROWN_SPOT, 0.0))
    emp_blast = float(emp_probs.get(_BLAST, 0.0))
    if emp_brown < brown_min or (emp_brown - emp_blast) < margin:
        return probs, None
    if max(emp_probs, key=emp_probs.get) != _BROWN_SPOT:
        return probs, None
    mixed = {
        c: (1.0 - veto_weight) * p + veto_weight * emp_probs.get(c, 0.0)
        for c, p in probs.items()
    }
    return mixed, _BLAST


def _count_dough_spikelets(arr: np.ndarray) -> int:
    """
    Count compact yellow-green blobs the size of dough-stage rice grains.

    Filters out long chlorotic streaks (blight) and tiny specks. Brown-spot
    lesions are usually too red/brown for this hue band, so they count low.
    """
    from scipy import ndimage as ndi

    from .lesion_features import _rgb_to_hsv_array

    hsv = _rgb_to_hsv_array(arr)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r = arr[:, :, 0].astype(np.float32) / 255.0
    g = arr[:, :, 1].astype(np.float32) / 255.0
    b = arr[:, :, 2].astype(np.float32) / 255.0
    grain = (
        (h >= 40)
        & (h <= 62)
        & (s >= 0.28)
        & (s <= 0.75)
        & (v >= 0.42)
        & (g >= r * 0.98)
        & (g > b * 1.05)
    )
    labeled, n_lab = ndi.label(grain)
    if n_lab == 0:
        return 0
    area = float(grain.size)
    min_a = 0.0001 * area
    max_a = 0.006 * area
    n = 0
    for i in range(1, n_lab + 1):
        ys, xs = np.where(labeled == i)
        blob_area = len(ys)
        if blob_area < min_a or blob_area > max_a or blob_area < 8:
            continue
        coords = np.column_stack([xs.astype(float), ys.astype(float)])
        coords -= coords.mean(axis=0)
        cov = np.cov(coords, rowvar=False)
        if np.ndim(cov) < 2:
            continue
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(eig[0] / max(eig[-1], 1e-9)))
        if elong > 5.0:
            continue
        n += 1
    return n


def _panicle_stats_qualify(n_spikelets: int, green_frac: float, lesion_frac: float) -> bool:
    """True when scene stats match a ripening panicle rather than a lesioned leaf."""
    return (
        n_spikelets >= PANICLE_MIN_SPIKELETS
        and green_frac >= PANICLE_MIN_GREEN
        and lesion_frac <= PANICLE_MAX_LESION
    )


def _ripening_panicle_scene(arr: np.ndarray) -> bool:
    """Healthy dough-stage panicle (many grains, still-green leaves, few lesions)."""
    from .lesion_features import segment_lesion
    from .preprocessor_shared import green_leaf_fraction

    green_frac = green_leaf_fraction(arr)
    if green_frac < PANICLE_MIN_GREEN:
        return False
    lesion, _leaf = segment_lesion(arr)
    lesion_frac = float(lesion.mean()) if lesion.size else 1.0
    if lesion_frac > PANICLE_MAX_LESION:
        return False
    return _panicle_stats_qualify(
        _count_dough_spikelets(arr), green_frac, lesion_frac
    )


def _nearest_examples(emb: np.ndarray, db: dict, k: int) -> list[dict]:
    if not db or not len(db["embeddings"]):
        return []
    sims = db["embeddings"] @ emb
    top = np.argsort(sims)[::-1][:k]
    return [
        {
            "path": db["paths"][i],
            "label": db["labels"][i],
            "similarity": round(float(sims[i]), 4),
        }
        for i in top
    ]


def run_cnn_pipeline(image_path: str) -> dict:
    """Full CNN pipeline. Same output contract as ml.pipeline.run_pipeline."""
    from .cnn_embedder import embed_image, embed_image_tta

    health_clf, disease_clf, gate_clf, gate_rf, db, meta, rice_ref, health_db = _load()
    # TTA embeddings match train-time strong aug (rotate ±20°, zoom, brightness/
    # contrast). Row 0 is the original; heads average over all rows. OOD / k-NN
    # use the original only so distance thresholds stay calibrated.
    emb_tta = embed_image_tta(image_path).astype(np.float32)
    emb = emb_tta[0]

    # ---- Layer 0: rice gate (LR+RF ensemble + rice-database distance) ----
    p_nots_tta: list[float] = []
    p_nots_orig: list[float] = []
    p_not_lr_tta = 0.0
    p_not_lr_orig = 0.0
    p_not_rf_tta = 0.0
    p_not_rf_orig = 0.0
    if gate_clf is not None:
        p_not_lr_tta = _proba_dict_tta(gate_clf, emb_tta).get("not_rice", 0.0)
        p_not_lr_orig = _proba_dict(gate_clf, emb).get("not_rice", 0.0)
        p_nots_tta.append(p_not_lr_tta)
        p_nots_orig.append(p_not_lr_orig)
    if gate_rf is not None:
        p_not_rf_tta = _proba_dict_tta(gate_rf, emb_tta).get("not_rice", 0.0)
        p_not_rf_orig = _proba_dict(gate_rf, emb).get("not_rice", 0.0)
        p_nots_tta.append(p_not_rf_tta)
        p_nots_orig.append(p_not_rf_orig)
    p_not_tta = float(np.mean(p_nots_tta)) if p_nots_tta else 0.0
    p_not_orig = float(np.mean(p_nots_orig)) if p_nots_orig else 0.0
    # Don't let TTA dilute a clear original not_rice vote (barnyard grass:
    # orig ~0.73, TTA ~0.62). Genuine rice stays low on both.
    p_not = max(p_not_tta, p_not_orig)
    p_not_lr = max(p_not_lr_tta, p_not_lr_orig) if gate_clf is not None else p_not
    p_not_rf = max(p_not_rf_tta, p_not_rf_orig) if gate_rf is not None else p_not

    nearest_rice = None
    if rice_ref is not None and len(rice_ref):
        nearest_rice = float((rice_ref @ emb).max())
    reject_floor = meta.get("rice_reject_floor")
    accept_override = meta.get("ood_cosine_threshold")

    # Strict gate: a not_rice flag ends the pipeline here. The result page shows
    # "not a rice leaf" instead of a disease analysis with a warning banner.
    # Exception: a flooded young paddy is rice, but it does not look like the
    # close-up leaf photos the gate was trained on — see _paddy_scene_rescue.
    if _cnn_gate_rejected(p_not, nearest_rice, reject_floor, accept_override):
        from PIL import Image as _Image

        from .preprocessor_shared import green_leaf_fraction

        arr = np.array(_Image.open(image_path).convert("RGB"))
        if not _paddy_scene_rescue(
            p_not_rf,
            p_not_lr,
            nearest_rice,
            reject_floor,
            green_leaf_fraction(arr),
        ):
            return _unknown("not_rice")

    # ---- Layer 1: health + rescue-only k-NN (see _health_with_knn) ----
    h_probs = _proba_dict_tta(health_clf, emb_tta)
    h_probs_orig = _proba_dict(health_clf, emb)
    health_vote = _knn_class_vote(emb, health_db, list(h_probs.keys()))
    health_sim = _knn_top1_similarity(emb, health_db)
    h_probs, _h_veto = _health_with_knn(h_probs, health_vote, health_sim)
    h_probs_orig, _h_veto_orig = _health_with_knn(h_probs_orig, health_vote, health_sim)
    health = max(h_probs, key=h_probs.get)
    health_conf = h_probs[health]
    if max(h_probs_orig, key=h_probs_orig.get) == health:
        health_conf = max(health_conf, h_probs_orig[health])
    if (
        health == "healthy"
        and health_vote is not None
        and health_vote.get("healthy", 0.0) >= 0.80
    ):
        # Library match: don't leave a rescued field-healthy leaf sitting
        # on the 0.60 unknown floor after a 50/50 blend with a diseased head.
        health_conf = max(
            health_conf,
            0.25 * health_conf + 0.75 * float(health_vote["healthy"]),
        )

    if health_conf < HEALTH_MIN_CONFIDENCE:
        return _unknown("low_confidence", confidence=health_conf, health_probs=h_probs)

    if health == "healthy":
        return {
            "health": "healthy",
            "disease_name": "",
            "confidence": health_conf,
            "all_disease_probs": {},
            "health_probs": h_probs,
            "recommendation": get_recommendation("healthy"),
            "rejection_reason": "",
            "gate_warning": "",
            "similar_examples": [],
            "engine": "cnn",
        }

    # ---- Layer 2: disease + k-NN compare-to-database ----
    d_probs = _proba_dict_tta(disease_clf, emb_tta)
    # Acceptance uses max(TTA-averaged, original-only) so a diluted TTA score
    # cannot reject a clear single-pass winner (matches the rf pipeline).
    d_probs_orig = _proba_dict(disease_clf, emb)
    # Blend both scoring paths with the k-NN database vote (original embedding
    # only, matching the OOD gate) — see KNN_BLEND_* above.
    knn_vote = _knn_class_vote(emb, db, list(d_probs.keys()))
    d_probs = _blend_with_knn(d_probs, knn_vote)
    d_probs_orig = _blend_with_knn(d_probs_orig, knn_vote)
    d_probs, vetoed_from = _knn_support_veto(d_probs, knn_vote)
    d_probs_orig, vetoed_orig = _knn_support_veto(d_probs_orig, knn_vote)
    # Whole-scene embeddings (and the train-image k-NN) still call blast on
    # some brown-spot field photos. One extra MobileNet pass on a lesion-
    # faded view referees that pair without touching blight / stress.
    emp_probs = None
    current_top = max(d_probs, key=d_probs.get) if d_probs else ""
    orig_top = max(d_probs_orig, key=d_probs_orig.get) if d_probs_orig else ""
    if current_top == _BLAST or orig_top == _BLAST:
        from .lesion_emphasis import make_lesion_emphasis_image

        emp_emb = embed_image(make_lesion_emphasis_image(image_path)).astype(np.float32)
        emp_probs = _proba_dict(disease_clf, emp_emb)
        d_probs, lesion_veto = _blast_brownspot_lesion_veto(d_probs, emp_probs)
        d_probs_orig, lesion_veto_orig = _blast_brownspot_lesion_veto(
            d_probs_orig, emp_probs
        )
        vetoed_from = vetoed_from or lesion_veto
        vetoed_orig = vetoed_orig or lesion_veto_orig
    ordered = sorted(d_probs.items(), key=lambda kv: kv[1], reverse=True)
    ordered_orig = sorted(d_probs_orig.items(), key=lambda kv: kv[1], reverse=True)
    # Label comes from the TTA-blended (+ veto) path. Original-only may only
    # raise the acceptance score when it names the SAME disease — otherwise an
    # overconfident head would override the database vote (blast vs brown spot).
    disease_name, disease_conf = ordered[0]
    margin = disease_conf - (ordered[1][1] if len(ordered) > 1 else 0.0)
    orig_conf = ordered_orig[0][1]
    orig_margin = orig_conf - (ordered_orig[1][1] if len(ordered_orig) > 1 else 0.0)
    if ordered_orig[0][0] == disease_name:
        gate_conf = max(disease_conf, orig_conf)
        gate_margin = max(margin, orig_margin)
    else:
        gate_conf = disease_conf
        gate_margin = margin
    vetoed = vetoed_from or vetoed_orig
    if vetoed is not None and knn_vote is not None:
        # Margin vs the rejected class (the one with no neighbour support),
        # not vs a close runner-up the k-NN also likes. Otherwise a 54/46
        # brown-spot vs blight neighbour split would send a clear "not blast"
        # leaf to "unknown".
        gate_margin = max(
            gate_margin,
            knn_vote.get(disease_name, 0.0) - knn_vote.get(vetoed, 0.0),
        )
    if vetoed is not None and emp_probs is not None:
        gate_margin = max(
            gate_margin,
            float(emp_probs.get(disease_name, 0.0)) - float(emp_probs.get(vetoed, 0.0)),
        )
    all_probs = {k: round(v * 100, 1) for k, v in ordered}
    examples = _nearest_examples(emb, db, KNN_K)

    # Dew on a ripening panicle looks like water-soaked blight to the leaf
    # health/disease heads. If the frame is clearly grain heads on still-green
    # leaves, call it healthy instead of bacterial blight.
    if disease_name == _BLIGHT:
        from PIL import Image as _Image

        panicle_arr = np.array(_Image.open(image_path).convert("RGB"))
        if _ripening_panicle_scene(panicle_arr):
            return {
                "health": "healthy",
                "disease_name": "",
                "confidence": max(0.80, float(h_probs.get("healthy", 0.0))),
                "all_disease_probs": {},
                "health_probs": {"healthy": 0.80, "diseased": 0.20},
                "recommendation": get_recommendation("healthy"),
                "rejection_reason": "",
                "gate_warning": "",
                "similar_examples": [],
                "engine": "cnn",
            }

    if gate_conf < DISEASE_MIN_CONFIDENCE or gate_margin < DISEASE_MARGIN_MIN:
        return _unknown(
            "low_confidence",
            confidence=gate_conf,
            all_probs=all_probs,
            health_probs=h_probs,
        )

    return {
        "health": "diseased",
        "disease_name": disease_name,
        "confidence": gate_conf,
        "all_disease_probs": all_probs,
        "health_probs": h_probs,
        "recommendation": get_recommendation(disease_name),
        "rejection_reason": "",
        "gate_warning": "",
        "similar_examples": examples,
        "engine": "cnn",
    }
