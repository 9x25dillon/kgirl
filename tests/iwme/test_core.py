"""Repaired core (P1 plumbing) and the exact persistence-after-decoupling information."""

import math
import unittest

import tests.iwme  # noqa: F401

import numpy as np

from kgirl.iwme import core
from kgirl.iwme.attention import softmax


def _toy_core(c=0.4):
    K = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])
    return core.IWMECore(
        Y_E=lambda X, Yext, n: np.tanh(X) + Yext + 0.1 * n,
        A=lambda H, Y, E: softmax(K @ (H + Y) / E[0]),
        M=lambda H, A, Y, n: 0.7 * H + 0.3 * (A @ K) + 0.05 * n,
        F_E=lambda X, U, n, act: 0.9 * X + U + act + 0.05 * n,
        C_E=lambda H: c * H,
        T=lambda X: X[:1] ** 2 + X[1:],
    )


class Core(unittest.TestCase):
    def test_augmented_state_reproduces_observables(self):
        from kgirl.evolve.chaos import ChaosRNG
        rng = ChaosRNG(0.123)
        sysm = _toy_core()
        steps = 50
        inputs = [{"U": 0.1 * np.sin([t, t + 1.0]), "E": np.array([0.8]), "Yext": 0.2 * np.cos([t, 2.0 * t])} for t in range(steps)]
        noises = [{"y": rng.normal(2), "h": rng.normal(2), "x": rng.normal(2)} for _ in range(steps)]
        G1 = sysm.run(np.zeros(2), np.zeros(2), inputs, noises)
        Fp = sysm.augmented()
        Z, G2 = np.zeros(4), []
        for inp, nz in zip(inputs, noises):
            Z, g = Fp(Z, 2, inp["U"], inp["E"], inp["Yext"], nz)
            G2.append(g)
        np.testing.assert_array_equal(G1, np.array(G2))

    def test_h1_means_no_persistence(self):
        self.assertEqual(core.persistence_information(core.LinearIWME(c=0.0))["block_bits"], 0.0)

    def test_actuated_memory_persists_until_its_state_is_conditioned_on(self):
        m = core.LinearIWME(c=0.5)
        self.assertGreater(core.persistence_information(m)["block_bits"], 0.5)
        self.assertLess(core.persistence_information(m, condition="XH")["block_bits"], 1e-9)

    def test_lag_profiles_separate_ema_from_accumulator(self):
        ema = core.persistence_information(core.LinearIWME(memory="ema"), horizon=60, lags=[5, 30, 60])["per_lag_bits"]
        acc = core.persistence_information(core.LinearIWME(memory="accumulator"), horizon=60, lags=[5, 30, 60])["per_lag_bits"]
        self.assertTrue(ema[0] > ema[1] > ema[2])
        self.assertLess(ema[2], 1e-6)
        self.assertGreater(min(acc), 1.0)

    def test_conditional_mi_against_direct_formula(self):
        m = core.LinearIWME()
        X, H, G, Y = core._linear_rows(m, 20, 10)
        A, B, C = G[3:4], Y, X[20:21]

        def ld(R):
            return np.linalg.slogdet(R @ R.T)[1]
        direct = 0.5 * (ld(np.vstack([A, C])) + ld(np.vstack([B, C])) - ld(C) - ld(np.vstack([A, B, C]))) / math.log(2)
        self.assertAlmostEqual(core.conditional_mi_bits(A, B, C), direct, places=10)


if __name__ == "__main__":
    unittest.main()
