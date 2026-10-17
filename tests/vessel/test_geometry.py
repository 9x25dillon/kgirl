import unittest

import tests.vessel  # noqa: F401  (sys.path)

import numpy as np

from kgirl.evolve.chaos import ChaosRNG
from kgirl.vessel import geometry as G


def acts(seed=0.31, n=40, d=12):
    return np.tanh(ChaosRNG(seed).normal((n, d)) @ ChaosRNG(seed + 0.1).normal((d, d)))


class CKATests(unittest.TestCase):
    def test_invariant_under_rotation_scale_translation_and_width(self):
        X = acts()
        Q, _ = np.linalg.qr(ChaosRNG(0.2).normal((12, 12)))
        R, _ = np.linalg.qr(ChaosRNG(0.3).normal((30, 30)))
        for Y in (X @ Q, 7.5 * X, X + 3.0, X @ R[:12, :]):
            self.assertAlmostEqual(G.linear_cka(X, Y), 1.0, places=12)

    def test_detects_anisotropy_and_probe_reordering(self):
        X = acts()
        self.assertLess(G.linear_cka(X, X * np.exp(np.linspace(-2.3, 2.3, 12))), 0.95)
        self.assertLess(G.linear_cka(X, X[::-1]), 0.5)

    def test_rejects_mismatched_probes(self):
        with self.assertRaises(ValueError):
            G.linear_cka(acts(n=10), acts(n=11))

    def test_gradient_matches_finite_differences(self):
        X = acts(n=9, d=5)
        A = G.normalized_gram(acts(0.77, n=9, d=7))
        f, g = G.cka_to_anchor_grad(X, A)
        self.assertAlmostEqual(f, G.gram_alignment(A, G.normalized_gram(X)), places=12)
        num = np.zeros_like(X)
        for idx in np.ndindex(X.shape):
            E = np.zeros_like(X)
            E[idx] = 1e-6
            num[idx] = (G.gram_alignment(A, G.normalized_gram(X + E)) - G.gram_alignment(A, G.normalized_gram(X - E))) / 2e-6
        self.assertLess(np.max(np.abs(num - g)) / np.max(np.abs(g)), 1e-6)

    def test_collapsed_representation_is_zero_not_nan(self):
        X = np.ones((6, 3))
        self.assertEqual(G.linear_cka(X, acts(n=6, d=3)), 0.0)
        f, g = G.cka_to_anchor_grad(X, G.normalized_gram(acts(n=6, d=3)))
        self.assertEqual(f, 0.0)
        self.assertFalse(np.any(g))


class SpectraTests(unittest.TestCase):
    def test_principal_cosines(self):
        I = np.eye(6)
        self.assertTrue(np.allclose(G.principal_cosines(I[:, :2], I[:, :2]), 1.0))
        self.assertTrue(np.allclose(G.principal_cosines(I[:, :2], I[:, 2:4]), 0.0))
        self.assertAlmostEqual(G.subspace_overlap(I[:, :2], I[:, 1:3]), 0.5)

    def test_effective_and_stable_rank(self):
        self.assertAlmostEqual(G.effective_rank(np.ones(7)), 7.0)
        self.assertAlmostEqual(G.effective_rank(np.array([1.0, 0, 0])), 1.0)
        self.assertAlmostEqual(G.stable_rank(np.diag([2.0, 1.0, 1.0])), 1.5)

    def test_tail_exponent_recovers_a_planted_power_law(self):
        alpha = 3.0
        q = (np.arange(400) + 0.5) / 400
        lam = (1 - q) ** (-1 / (alpha - 1))                     # Pareto(α) quantiles, λ_min = 1
        U, _ = np.linalg.qr(ChaosRNG(0.5).normal((400, 400)))
        W = U * np.sqrt(lam * 400)                               # WᵀW / rows has eigenvalues λ
        fit = G.tail_exponent(W)
        self.assertAlmostEqual(fit["alpha"], alpha, delta=0.25)
        self.assertTrue(np.isnan(G.tail_exponent(np.eye(4))["alpha"]))

    def test_gram_entropy_bounds(self):
        self.assertAlmostEqual(G.gram_entropy(G.normalized_gram(np.eye(5))), np.log(4), places=10)


if __name__ == "__main__":
    unittest.main()
