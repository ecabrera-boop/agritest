"""CNN rice-leaf gate: reject grassy lookalikes, keep genuine rice."""

from django.test import SimpleTestCase

from ml.cnn_pipeline import (
    GATE_OVERRIDE_MAX_PNOT,
    GATE_REJECT_PROBA,
    GATE_RICE_CONFIDENT_PROBA,
    PADDY_SCENE_MAX_NEAREST,
    PADDY_SCENE_MIN_GREEN,
    PADDY_SCENE_MIN_LR_PNOT,
    PANICLE_MAX_LESION,
    PANICLE_MIN_GREEN,
    PANICLE_MIN_SPIKELETS,
    RICE_GATE_RESCUE_BAND,
    _cnn_gate_rejected,
    _paddy_scene_rescue,
    _panicle_stats_qualify,
)

# Production meta (cnn_meta.json) at the time this test was written.
_REJECT_FLOOR = 0.7986
_ACCEPT_OVERRIDE = 0.8683


class CnnRiceGateTests(SimpleTestCase):
    def test_rejects_barnyard_grass_despite_rice_like_cosine(self):
        # This photo: LR+RF mean not_rice ~0.62 (orig ~0.73), nearest rice 0.880.
        # The old override treated 0.880 >= 0.868 as "close enough" and accepted.
        self.assertGreaterEqual(0.621, GATE_REJECT_PROBA)
        self.assertGreaterEqual(0.621, GATE_OVERRIDE_MAX_PNOT)
        self.assertTrue(
            _cnn_gate_rejected(0.621, 0.8802, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )
        self.assertTrue(
            _cnn_gate_rejected(0.731, 0.8802, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_accepts_confident_rice_field_canopy(self):
        # Grain-filling rice stand: p_not ~0.03, nearest 0.917.
        self.assertFalse(
            _cnn_gate_rejected(0.0255, 0.9165, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_weak_not_rice_vote_still_overridden_when_very_close(self):
        # Borderline classifier on an embedding that is firmly in the rice cloud.
        p_not = 0.52
        self.assertGreaterEqual(p_not, GATE_REJECT_PROBA)
        self.assertLess(p_not, GATE_OVERRIDE_MAX_PNOT)
        self.assertFalse(
            _cnn_gate_rejected(p_not, 0.90, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_hard_rejects_far_from_every_rice_leaf(self):
        hard = _REJECT_FLOOR - RICE_GATE_RESCUE_BAND
        self.assertTrue(
            _cnn_gate_rejected(0.01, hard - 0.02, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_rescues_confident_rice_just_under_the_floor(self):
        near = _REJECT_FLOOR - 0.05
        self.assertLess(0.10, GATE_RICE_CONFIDENT_PROBA)
        self.assertGreater(near, _REJECT_FLOOR - RICE_GATE_RESCUE_BAND)
        self.assertFalse(
            _cnn_gate_rejected(0.10, near, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_rescues_hand_held_leaf_against_cluttered_background(self):
        # Hand-held rice leaf photographed against blurred outdoor background:
        # LR+RF mean not_rice ~0.48, nearest known rice leaf only ~0.62 (well
        # under the plain-background reject floor, but inside the rescue band).
        # GATE_RICE_CONFIDENT_PROBA was raised from 0.20 to 0.50 specifically
        # so this case is no longer hard-rejected as not_rice.
        p_not = 0.4832
        nearest = 0.6188
        self.assertLess(p_not, GATE_RICE_CONFIDENT_PROBA)
        self.assertGreater(nearest, _REJECT_FLOOR - RICE_GATE_RESCUE_BAND)
        self.assertFalse(
            _cnn_gate_rejected(p_not, nearest, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_rescues_rice_leaf_between_reject_floor_and_accept_override(self):
        # Real user photo: potted rice leaf shot top-down with a cluttered
        # porch background (wooden decking, floor, shoe visible). The LR head
        # alone leans not_rice (~0.79) but RF leans rice (~0.32), so the
        # ensemble mean is a weak 0.554 not_rice — under GATE_OVERRIDE_MAX_PNOT.
        # Nearest known rice leaf is 0.824: solidly above reject_floor (2nd
        # percentile of genuine-rice similarity) but short of accept_override
        # (10th percentile). The disease head separately called this leaf 96%+
        # blast with every one of its nearest database neighbours also labeled
        # blast — the gate must not be the reason that diagnosis never runs.
        # Regression for the 2nd-10th percentile gap: before this fix, only
        # near >= accept_override qualified for the GATE_OVERRIDE_MAX_PNOT
        # rescue, so this photo (and ~8% of genuine rice by the train_cnn.py
        # percentile calibration) was hard-rejected as not_rice.
        p_not = 0.5541
        nearest = 0.8238
        self.assertGreaterEqual(p_not, GATE_REJECT_PROBA)
        self.assertLess(p_not, GATE_OVERRIDE_MAX_PNOT)
        self.assertGreaterEqual(nearest, _REJECT_FLOOR)
        self.assertLess(nearest, _ACCEPT_OVERRIDE)
        self.assertFalse(
            _cnn_gate_rejected(p_not, nearest, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )


class PaddySceneRescueTests(SimpleTestCase):
    def test_rescues_young_flooded_paddy(self):
        # Transplanted seedlings in floodwater: LR+RF mean not_rice ~0.73,
        # nearest leaf 0.605, LR ~0.97, RF ~0.49, frame ~80% green.
        self.assertTrue(
            _cnn_gate_rejected(0.728, 0.605, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )
        self.assertTrue(
            _paddy_scene_rescue(0.489, 0.967, 0.605, _REJECT_FLOOR, 0.80)
        )

    def test_does_not_rescue_barnyard_grass_closeup(self):
        # Close-up lookalike sits inside the rice-leaf cloud (cosine ~0.88).
        self.assertGreaterEqual(0.8802, _REJECT_FLOOR)
        self.assertFalse(
            _paddy_scene_rescue(0.489, 0.818, 0.8802, _REJECT_FLOOR, 0.96)
        )
        self.assertTrue(
            _cnn_gate_rejected(0.731, 0.8802, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )

    def test_does_not_rescue_hairy_node_cutgrass(self):
        # Rice cutgrass / similar stem macros: closer to leaves than a paddy
        # (cosine ~0.70–0.76) and the linear head is not sure they are "not a leaf".
        self.assertGreater(0.703, PADDY_SCENE_MAX_NEAREST)
        self.assertLess(0.746, PADDY_SCENE_MIN_LR_PNOT)
        self.assertTrue(
            _cnn_gate_rejected(0.565, 0.703, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )
        self.assertFalse(
            _paddy_scene_rescue(0.386, 0.746, 0.703, _REJECT_FLOOR, 0.77)
        )
        # Known, accepted regression: GATE_RICE_CONFIDENT_PROBA was raised from
        # 0.20 to 0.50 to rescue hand-held/cluttered-background rice photos
        # (see test_rescues_hand_held_leaf_against_cluttered_background). p_not
        # and nearest_rice are not jointly monotonic, so this cutgrass photo
        # (p_not 0.377 < the hand-held-leaf case's 0.483) is rescued too — no
        # single p_not threshold can admit one without the other.
        self.assertLess(0.499, PADDY_SCENE_MIN_LR_PNOT)
        self.assertFalse(
            _cnn_gate_rejected(0.377, 0.756, _REJECT_FLOOR, _ACCEPT_OVERRIDE)
        )
        self.assertFalse(
            _paddy_scene_rescue(0.255, 0.499, 0.756, _REJECT_FLOOR, 0.70)
        )

    def test_does_not_rescue_when_rf_also_says_not_rice(self):
        self.assertGreaterEqual(0.667, GATE_OVERRIDE_MAX_PNOT)
        self.assertFalse(
            _paddy_scene_rescue(0.667, 0.967, 0.605, _REJECT_FLOOR, 0.80)
        )

    def test_does_not_rescue_non_green_objects(self):
        self.assertLess(0.02, PADDY_SCENE_MIN_GREEN)
        self.assertFalse(
            _paddy_scene_rescue(0.49, 0.97, 0.41, _REJECT_FLOOR, 0.02)
        )


class RipeningPanicleTests(SimpleTestCase):
    def test_qualifies_dough_stage_panicle(self):
        # Ripening panicle photo: 33 spikelets, 84% green, 12% "lesion" (dew/grains).
        self.assertTrue(_panicle_stats_qualify(33, 0.845, 0.124))

    def test_rejects_brown_spot_leaf_stats(self):
        self.assertGreater(0.395, PANICLE_MAX_LESION)
        self.assertLess(13, PANICLE_MIN_SPIKELETS)
        self.assertFalse(_panicle_stats_qualify(13, 0.343, 0.395))

    def test_rejects_sparse_field_without_grains(self):
        # Young flooded paddy: green but almost no dough-stage spikelets.
        self.assertGreaterEqual(0.798, PANICLE_MIN_GREEN)
        self.assertLess(3, PANICLE_MIN_SPIKELETS)
        self.assertFalse(_panicle_stats_qualify(3, 0.798, 0.081))

    def test_green_floor_blocks_senesced_canopy(self):
        self.assertLess(0.23, PANICLE_MIN_GREEN)
        self.assertFalse(_panicle_stats_qualify(40, 0.23, 0.10))
