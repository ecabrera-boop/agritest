"""Lesion-emphasis referee: don't let a canopy CNN pass call blast on brown spot."""

import numpy as np
from django.test import SimpleTestCase
from PIL import Image

from ml.cnn_pipeline import _blast_brownspot_lesion_veto
from ml.lesion_emphasis import make_lesion_emphasis_image


class BlastBrownspotLesionVetoTests(SimpleTestCase):
    def test_flips_blast_when_lesion_view_is_brown_spot(self):
        # The field-canopy photo: whole-scene 77% blast, lesion view 83% brown spot.
        probs = {
            "blast": 0.771,
            "brown spot": 0.154,
            "bacterial blight": 0.075,
            "physiological stress": 0.0,
        }
        emp = {
            "blast": 0.14,
            "brown spot": 0.83,
            "bacterial blight": 0.03,
            "physiological stress": 0.0,
        }
        out, vetoed = _blast_brownspot_lesion_veto(probs, emp)
        self.assertEqual(vetoed, "blast")
        self.assertEqual(max(out, key=out.get), "brown spot")
        self.assertGreater(out["brown spot"], out["blast"])

    def test_keeps_true_blast_when_lesion_view_agrees(self):
        probs = {"blast": 0.80, "brown spot": 0.18, "bacterial blight": 0.02}
        emp = {"blast": 0.72, "brown spot": 0.22, "bacterial blight": 0.06}
        out, vetoed = _blast_brownspot_lesion_veto(probs, emp)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)
        self.assertEqual(max(out, key=out.get), "blast")

    def test_does_not_flip_when_emphasis_is_weak(self):
        probs = {"blast": 0.60, "brown spot": 0.35, "bacterial blight": 0.05}
        emp = {"blast": 0.38, "brown spot": 0.41, "bacterial blight": 0.21}
        out, vetoed = _blast_brownspot_lesion_veto(probs, emp)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)

    def test_does_not_touch_brown_spot_or_blight(self):
        probs = {"blast": 0.20, "brown spot": 0.70, "bacterial blight": 0.10}
        emp = {"blast": 0.05, "brown spot": 0.90, "bacterial blight": 0.05}
        out, vetoed = _blast_brownspot_lesion_veto(probs, emp)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)

    def test_no_emphasis_leaves_probs_unchanged(self):
        probs = {"blast": 0.9, "brown spot": 0.1}
        out, vetoed = _blast_brownspot_lesion_veto(probs, None)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)


class LesionEmphasisImageTests(SimpleTestCase):
    def test_returns_same_size_rgb(self):
        arr = np.zeros((160, 160, 3), dtype=np.uint8)
        arr[:] = (36, 150, 42)
        arr[50:70, 40:70] = (150, 85, 35)
        img = Image.fromarray(arr)
        out = make_lesion_emphasis_image(img, max_edge=160)
        self.assertEqual(out.size, img.size)
        self.assertEqual(out.mode, "RGB")

    def test_healthy_green_leaf_is_unchanged_when_no_lesion(self):
        arr = np.zeros((80, 80, 3), dtype=np.uint8)
        arr[:] = (36, 150, 42)
        img = Image.fromarray(arr)
        out = np.asarray(make_lesion_emphasis_image(img, max_edge=80))
        np.testing.assert_array_equal(out, arr)
