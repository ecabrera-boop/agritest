"""k-NN support veto: don't let the CNN head call blast when neighbours say brown spot."""

from django.test import SimpleTestCase

from ml.cnn_pipeline import HEALTH_KNN_MIN_SIMILARITY, _health_with_knn, _knn_support_veto


class KnnSupportVetoTests(SimpleTestCase):
    def test_flips_blast_when_neighbours_are_brown_spot(self):
        # The canopy brown-spot photo: head 69% blast, k-NN 0% blast / 54% brown spot.
        probs = {
            "blast": 0.347,
            "brown spot": 0.331,
            "bacterial blight": 0.321,
            "physiological stress": 0.0,
        }
        vote = {
            "blast": 0.0,
            "brown spot": 0.54,
            "bacterial blight": 0.46,
            "physiological stress": 0.0,
        }
        out, vetoed = _knn_support_veto(probs, vote)
        self.assertEqual(vetoed, "blast")
        self.assertEqual(max(out, key=out.get), "brown spot")
        self.assertGreater(out["brown spot"], out["blast"])

    def test_keeps_true_blast_when_neighbours_agree(self):
        probs = {"blast": 0.80, "brown spot": 0.18, "bacterial blight": 0.02}
        vote = {"blast": 0.64, "brown spot": 0.36, "bacterial blight": 0.0}
        out, vetoed = _knn_support_veto(probs, vote)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)
        self.assertEqual(max(out, key=out.get), "blast")

    def test_flips_diseased_when_neighbours_are_healthy(self):
        # Field-healthy paddy photo: head ~78% diseased, k-NN 100% healthy.
        probs = {"diseased": 0.776, "healthy": 0.224}
        vote = {"diseased": 0.0, "healthy": 1.0}
        out, vetoed = _knn_support_veto(probs, vote)
        self.assertEqual(vetoed, "diseased")
        self.assertEqual(max(out, key=out.get), "healthy")
        self.assertGreater(out["healthy"], out["diseased"])

    def test_no_vote_leaves_probs_unchanged(self):
        probs = {"blast": 0.9, "brown spot": 0.1}
        out, vetoed = _knn_support_veto(probs, None)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)

    def test_does_not_veto_when_knn_is_also_unsure(self):
        probs = {"blast": 0.6, "brown spot": 0.4}
        vote = {"blast": 0.05, "brown spot": 0.10, "bacterial blight": 0.05}
        out, vetoed = _knn_support_veto(probs, vote)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)


class HealthKnnRescueTests(SimpleTestCase):
    def test_keeps_healthy_head_when_paddy_neighbours_are_diseased(self):
        # Grain-filling canopy: head ~64% healthy, k-NN 100% diseased
        # (nearest library images are diseased Paddy Doctor field shots).
        probs = {"diseased": 0.357, "healthy": 0.643}
        vote = {"diseased": 1.0, "healthy": 0.0}
        out, vetoed = _health_with_knn(probs, vote)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)
        self.assertEqual(max(out, key=out.get), "healthy")

    def test_still_rescues_diseased_head_when_neighbours_are_healthy(self):
        probs = {"diseased": 0.776, "healthy": 0.224}
        vote = {"diseased": 0.0, "healthy": 1.0}
        out, vetoed = _health_with_knn(probs, vote)
        self.assertEqual(max(out, key=out.get), "healthy")
        self.assertGreater(out["healthy"], out["diseased"])

    def test_keeps_diseased_when_head_and_neighbours_agree(self):
        probs = {"diseased": 0.85, "healthy": 0.15}
        vote = {"diseased": 0.9, "healthy": 0.1}
        out, vetoed = _health_with_knn(probs, vote)
        self.assertEqual(max(out, key=out.get), "diseased")

    def test_skips_rescue_when_neighbours_are_weak_matches(self):
        # Hand-held leaf with real dieback (LeafScan #615): head 90% diseased,
        # health_db votes 85% healthy but the nearest neighbour is only ~0.61
        # cosine away — far below HEALTH_KNN_MIN_SIMILARITY (~0.78), i.e. not
        # a real match. The rescue must not fire; the correct diseased head
        # call is kept.
        probs = {"diseased": 0.9029, "healthy": 0.0971}
        vote = {"diseased": 0.1495, "healthy": 0.8505}
        weak_sim = 0.6134
        self.assertLess(weak_sim, HEALTH_KNN_MIN_SIMILARITY)
        out, vetoed = _health_with_knn(probs, vote, weak_sim)
        self.assertIsNone(vetoed)
        self.assertEqual(out, probs)
        self.assertEqual(max(out, key=out.get), "diseased")

    def test_still_rescues_when_neighbours_are_strong_matches(self):
        # Same head/vote shape as the weak-match case above, but the nearest
        # neighbour this time is a genuinely close match — the rescue should
        # still fire exactly as before HEALTH_KNN_MIN_SIMILARITY was added.
        probs = {"diseased": 0.776, "healthy": 0.224}
        vote = {"diseased": 0.0, "healthy": 1.0}
        strong_sim = 0.95
        self.assertGreaterEqual(strong_sim, HEALTH_KNN_MIN_SIMILARITY)
        out, vetoed = _health_with_knn(probs, vote, strong_sim)
        self.assertEqual(max(out, key=out.get), "healthy")
        self.assertGreater(out["healthy"], out["diseased"])

    def test_nearest_sim_none_preserves_old_behaviour(self):
        # Callers that don't pass nearest_sim (e.g. existing tests above) get
        # the pre-fix behaviour: the vote is trusted unconditionally.
        probs = {"diseased": 0.776, "healthy": 0.224}
        vote = {"diseased": 0.0, "healthy": 1.0}
        out, vetoed = _health_with_knn(probs, vote, None)
        self.assertEqual(max(out, key=out.get), "healthy")
