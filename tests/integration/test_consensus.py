"""Consensus math in kgirl.main: the spectral fallback must score agreement, not concentration."""
import unittest

import numpy as np

import tests.integration  # noqa: F401  (sys.path)

try:
    from kgirl.main import spectral_coherence, spectral_weights
except Exception as exc:          # fastapi / pydantic-settings missing
    spectral_coherence = None
    IMPORT_ERROR = exc


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class ConsensusTests(unittest.TestCase):
    def test_identical_answers_are_fully_coherent(self):
        e = np.tile([[0.3, -0.2, 0.9]], (4, 1))
        self.assertAlmostEqual(spectral_coherence(e), 1.0, places=9)
        self.assertTrue(np.allclose(spectral_weights(e), 0.25))

    def test_orthogonal_answers_score_zero_and_order_is_monotone(self):
        self.assertAlmostEqual(spectral_coherence(np.eye(3)), 0.0, places=9)
        base = np.array([1.0, 0.0, 0.0])
        close = np.stack([base, base + [0, 0.1, 0], base + [0, 0, 0.1]])
        far = np.stack([base, [0.0, 1.0, 0.0], base + [0, 0, 0.1]])
        self.assertGreater(spectral_coherence(close), spectral_coherence(far))

    def test_weights_survive_negative_eigenvector_sign(self):
        e = np.array([[1.0, 0.1], [0.9, 0.2], [1.0, 0.0]])
        for flip in (1.0, -1.0):
            w = spectral_weights(flip * e)
            self.assertAlmostEqual(w.sum(), 1.0, places=9)
            self.assertTrue((w > 0).all())


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class AskEndpointTests(unittest.TestCase):
    def test_ask_runs_with_demo_adapters(self):
        import os
        import sys
        import warnings
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "demo"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
        import kgirl.main as M
        from demo_adapter import get_demo_model_pool
        saved = list(M.g_model_pool)
        M.g_model_pool[:] = get_demo_model_pool()
        try:
            r = TestClient(M.app).post("/ask", json={"prompt": "Explain quantum entanglement"})
        finally:
            M.g_model_pool[:] = saved
        self.assertEqual(r.status_code, 200)           # was a NameError on every call
        self.assertTrue(0.0 <= r.json()["coherence"] <= 1.0)


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class FallbackEnergyTests(unittest.TestCase):
    def test_noise_scores_near_one_structure_near_zero_any_dimension(self):
        from kgirl.evolve.chaos import ChaosGenerator
        from kgirl.main import fallback_energy
        g = ChaosGenerator(0.3)
        for d in (256, 1536, 4096):
            self.assertGreater(fallback_energy(g.normal(0, 1, d)), 0.85)
            focused = np.zeros(d)
            focused[:4] = 1.0
            self.assertLess(fallback_energy(focused), 0.02)

    def test_default_threshold_sits_above_the_text_average(self):
        from kgirl.main import CardyEnergy, calibrate_energy
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "demo"))
        from demo_adapter import DemoAdapter
        a = DemoAdapter("x", "factual")
        texts = ["The surrogate predicts k_inf from 39 rod enrichments with ridge regression.",
                 "Holographic recall ranks stored traces by their normalised overlap with the query.",
                 "The bridge exposes stability, fluctuation and entropy profiles over HTTP.",
                 "Each pull request needs green tests before it waits on review."]
        cal = calibrate_energy([a.embed(t) for t in texts])
        ce = CardyEnergy()
        if not ce.use_cth:
            self.assertGreater(ce.default_max, cal["mean"])
