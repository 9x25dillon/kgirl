"""Cascades: closed forms, box counting, transfer matrices and operators, the Hutchinson fixed point."""

import math
import unittest

import tests.iwme  # noqa: F401

import numpy as np

from kgirl.iwme import cascade as cs


class Binomial(unittest.TestCase):
    def test_report_table(self):
        expected = {-1: 1.1258, 0: 1.0, 0.5: 0.9385, 1: 0.8813, 2: 0.7859, 3: 0.7172, math.inf: 0.5146, -math.inf: 1.7370}
        for q, v in expected.items():
            self.assertAlmostEqual(round(cs.binomial_Dq(0.7, q), 4), v, places=9)

    def test_spectrum_is_nonincreasing(self):
        qs = np.linspace(-5, 5, 41)
        d = [cs.binomial_Dq(0.7, q) for q in qs]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(d, d[1:])))

    def test_box_counting_exact_at_every_level(self):
        mu = cs.binomial_measure(0.62, 14)
        self.assertAlmostEqual(mu.sum(), 1.0, places=12)
        for q in (-2, 0.5, 1, 2, 4):
            self.assertAlmostEqual(cs.estimate_Dq(mu, q, range(3, 15)), cs.binomial_Dq(0.62, q), places=10)

    def test_coarse_graining_is_the_pushforward(self):
        np.testing.assert_allclose(cs.coarse_grain(cs.binomial_measure(0.7, 9)), cs.binomial_measure(0.7, 8), atol=1e-15)


class MarkovAndMemory(unittest.TestCase):
    def test_markov_transfer_matrix(self):
        mc = cs.MarkovCascade()
        for q in (0.3, 1.0, 2.5):
            self.assertAlmostEqual(mc.enumerate_Zn(q, 8) / mc.Zn(q, 8), 1.0, places=12)
        self.assertAlmostEqual(mc.tau(1.0), 0.0, places=12)       # masses sum to one

    def test_iid_limit(self):
        iid = cs.MarkovCascade(p=(0.65, 0.65), P=(((0.3, 0.7), (0.6, 0.4)), ((0.9, 0.1), (0.2, 0.8))))
        for q in (0.5, 2.0):
            self.assertAlmostEqual(iid.tau(q), cs.binomial_tau(0.65, q), places=12)

    def test_ema_cascade_transfer_operator(self):
        ec = cs.EMACascade()
        self.assertAlmostEqual(ec.tau_transfer(0.0, 201), -1.0, places=9)
        self.assertAlmostEqual(ec.tau_transfer(1.0, 201), 0.0, places=9)
        self.assertAlmostEqual(ec.tau_transfer(2.0, 401), ec.tau_enumeration(2.0, 12, 18), delta=1e-2)


class HutchinsonFixedPoint(unittest.TestCase):
    def test_exact_half_contraction(self):
        hc = cs.hutchinson_contraction(0.3, 8)
        for r in hc["ratios"]:
            self.assertAlmostEqual(r, 0.5, places=10)

    def test_w1_exact_on_simple_pair(self):
        # all mass uniform on [0, ½] vs uniform on [0, 1]: F − G = x on [0, ½] and 1 − x on [½, 1], so W1 = ¼
        self.assertAlmostEqual(cs.wasserstein1_dyadic(np.array([1.0, 0.0]), np.array([0.5, 0.5])), 0.25)

    def test_effective_dimension_definition_is_vacuous(self):
        np.testing.assert_allclose(cs.efl_effective_dimension(1.7, 10), 1.0)
        self.assertAlmostEqual(cs.box_dimension_from_branching(2, 3), math.log(2) / math.log(3))


if __name__ == "__main__":
    unittest.main()
