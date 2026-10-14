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
