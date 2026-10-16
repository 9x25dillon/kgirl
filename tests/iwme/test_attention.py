"""Attention–memory core: P2 and its closed-loop qualification, P3 and P3′, hysteresis, gated forgetting."""

import math
import unittest

import tests.iwme  # noqa: F401  (sys.path)

import numpy as np

from kgirl.iwme import attention as at


class OpenAndClosedLoopRetention(unittest.TestCase):
    def test_open_loop_impulse_response_is_attention_independent(self):
        rho = 0.3
        for seq in (np.zeros(12), np.sin(np.arange(12)), np.linspace(-3, 3, 12)):
            def run(h):
                for m in seq:
                    h = (1 - rho) * h + rho * m
                return h
            s = (run(1e-3) - run(-1e-3)) / 2e-3
            self.assertAlmostEqual(s, at.open_loop_sensitivity(rho, 12), places=12)

    def test_closed_loop_jacobian_product_matches_finite_differences(self):
        from kgirl.evolve.chaos import ChaosRNG
        rng = ChaosRNG(0.61)
        K = rng.normal((5, 2))
        mem = at.AttentionMemory(K=K, V=K @ np.array([[0.8, 0.3], [-0.2, 1.1]]), T=0.7, rho=0.4)
        H0 = np.array([0.2, -0.4])
        S = at.closed_loop_sensitivity(mem, H0, 8)
        F = at.finite_difference_sensitivity(mem, H0, 8)
        np.testing.assert_allclose(S, F, rtol=1e-6, atol=1e-9)

    def test_closed_loop_retention_depends_on_temperature(self):
        H0 = np.array([0.3])
        sens = [float(at.closed_loop_sensitivity(at.two_point_memory(T, 0.5), H0, 10)[0, 0]) for T in (1.2, 2.0, 50.0)]
        self.assertGreater(sens[0], sens[1])
        self.assertGreater(sens[1], sens[2])
        self.assertGreater(sens[0] / at.open_loop_sensitivity(0.5, 10), 100)

    def test_effective_time_constant_limits(self):
        self.assertTrue(math.isinf(at.effective_time_constant(1.0)))
        self.assertAlmostEqual(at.effective_time_constant(math.exp(-1)), 1.0)


class ContractionAndBifurcation(unittest.TestCase):
    def test_popoviciu_bound_holds_and_is_sharp(self):
        from kgirl.evolve.chaos import ChaosRNG
        rng = ChaosRNG(0.13)
        for _ in range(6):
            K = rng.normal((5, 3))
            V = rng.normal((5, 3))
            mem = at.AttentionMemory(K=K, V=V, T=1.0, rho=0.5)
            self.assertLessEqual(at.sup_cov_norm(mem, rng.normal((200, 3)) * 4), at.cov_bound(K, V) + 1e-12)
        mem = at.two_point_memory(1.0)
        self.assertAlmostEqual(float(mem.jacobian(np.array([0.0]))[0, 0]), 1.0)   # (1−ρ) + ρ·1/T at T = 1

    def test_contraction_above_bound_means_unique_entrainment(self):
        K = np.array([[1.0], [-1.0]])
        mem = at.AttentionMemory(K=K, V=K, T=1.2, rho=0.5)
        L = (1 - 0.5) + 0.5 * at.cov_bound(K, K) / 1.2          # Lipschitz bound of Φ
        a, b = mem.run(np.array([0.9]), 300), mem.run(np.array([-0.9]), 300)
        gaps = np.abs(a[:, 0] - b[:, 0])
        self.assertTrue(np.all(gaps <= 1.8 * L ** np.arange(301) + 1e-15))

    def test_pitchfork(self):
        self.assertEqual(at.fixed_points_tanh(1.0), (0.0,))
        lo, zero, hi = at.fixed_points_tanh(0.5)
        self.assertAlmostEqual(hi, math.tanh(hi / 0.5), places=12)
        self.assertAlmostEqual(lo, -hi)
        self.assertGreater(at.multiplier_tanh(0.0, 0.5, 0.5), 1.0)
        self.assertLess(abs(at.multiplier_tanh(hi, 0.5, 0.5)), 1.0)
        # lock-in: opposite initial conditions end at opposite stable states
        mem = at.two_point_memory(0.5)
        self.assertAlmostEqual(mem.run(np.array([0.01]), 400)[-1, 0], hi, places=9)
        self.assertAlmostEqual(mem.run(np.array([-0.01]), 400)[-1, 0], lo, places=9)

    def test_critical_slowing_is_algebraic(self):
        self.assertAlmostEqual(at.decay_exponent(1.0, 0.5, t1=500, t2=5000), -0.5, delta=0.01)
        self.assertLess(at.decay_exponent(1.3, 0.5), -3.0)

    def test_flip_bifurcation_for_antialigned_values(self):
        Tf = at.flip_threshold(0.5)
        self.assertAlmostEqual(Tf, 1 / 3)
        self.assertGreater(at.period_two_amplitude(0.9 * Tf), 1e-2)
        self.assertLess(at.period_two_amplitude(1.1 * Tf), 1e-10)

    def test_crossing_types_from_cross_covariance(self):
        K = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
        kinds = {name: at.linear_thresholds(K, V, 0.5)[0]["type"].split(" ")[0]
                 for name, V in (("a", K), ("n", -K), ("r", K @ np.array([[0, -1], [1, 0]], float).T))}
        self.assertEqual(kinds, {"a": "pitchfork", "n": "flip", "r": "Neimark–Sacker"})


class HysteresisAndGating(unittest.TestCase):
    def test_static_hysteresis_only_below_tc(self):
        self.assertGreater(at.static_hysteresis_area(0.7), 0.3)
        self.assertLess(abs(at.static_hysteresis_area(1.05)), 1e-12)

    def test_finite_rate_loop_approaches_static_loop(self):
        static = at.static_hysteresis_area(0.6)
        fast = at.hysteresis_area(0.6, settle=20)
        slow = at.hysteresis_area(0.6, settle=400)
        self.assertLess(abs(slow - static), abs(fast - static))

    def test_gated_forgetting(self):
        g = at.GatedMemory(0.05, 0.8)
        self.assertAlmostEqual(g.rate(np.ones(4) / 4), 0.05)
        self.assertAlmostEqual(g.rate(np.array([1.0, 0, 0, 0])), 0.8)
        r = g.impulse_response([np.ones(3) / 3] * 5)
        np.testing.assert_allclose(r, 0.95 ** np.arange(6))


if __name__ == "__main__":
    unittest.main()
