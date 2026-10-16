"""Source D Walsh–Hadamard and multi-field checks; Source B coherence formula."""

import math
import unittest

import tests.iwme  # noqa: F401

import numpy as np

from kgirl.iwme import sources as so


class WalshHadamard(unittest.TestCase):
    def test_transform_matches_definition(self):
        f = np.array([3.0, -1.0, 2.0, 0.5, 1.0, 4.0, -2.0, 0.0])
        n = len(f)
        brute = [sum((-1) ** bin(a & i).count("1") * f[i] for i in range(n)) for a in range(n)]
        np.testing.assert_allclose(so.wht(f), brute)
        np.testing.assert_allclose(so.iwht(so.wht(f)), f)

    def test_parseval_holds_for_every_signal(self):
        from kgirl.evolve.chaos import ChaosRNG
        f = ChaosRNG(0.77).normal(32)
        self.assertTrue(so.parseval_ok(f, so.wht(f)))

    def test_source_d_tier1(self):
        r = so.reported_tier1()
        self.assertEqual(r["computed_natural"], [6.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0])
        self.assertFalse(r["reported_matches_any_ordering_or_scale"])
        self.assertTrue(r["parseval_reported"])
        self.assertEqual(r["signal_whose_transform_is_reported"], [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

    def test_sequency_order_is_a_permutation(self):
        F = so.wht(np.arange(8.0))
        self.assertEqual(sorted(so.sequency_order(F).tolist()), sorted(F.tolist()))

    def test_low_order_concentration(self):
        self.assertEqual(so.low_order_concentration(np.ones(16)), 1.0)
        alt = np.array([(-1) ** i for i in range(16)], float)
        self.assertEqual(so.low_order_concentration(alt), 0.0)


class FieldsAndCoherence(unittest.TestCase):
    def test_z9_z3_check_cannot_fail(self):
        r = so.z9_z3_homomorphism(12)
        self.assertEqual(r["failures"], 0)
        self.assertFalse(r["z4_to_z3_unital_hom_exists"])

    def test_pc_formulas(self):
        r = so.pc_cases()
        self.assertLess(r["identical"]["pc_4_10"], 1e-12)
        self.assertEqual(r["orthogonal"]["pc_4_10"], 1.0)
        self.assertTrue(math.isnan(r["antipodal_pair"]["pc_4_10"]))
        self.assertAlmostEqual(r["identical"]["pc_spectral"], 1.0)
        self.assertEqual(r["antipodal_pair"]["pc_resultant"], 0.0)

    def test_spectral_share_matches_kgirl_main_when_importable(self):
        try:
            from kgirl.main import spectral_coherence
        except Exception:
            self.skipTest("kgirl.main has optional dependencies that are not installed")
        from kgirl.evolve.chaos import ChaosRNG
        E = ChaosRNG(0.4).normal((5, 12))
        self.assertAlmostEqual(spectral_coherence(E), so.pc_spectral_share(E), places=9)


if __name__ == "__main__":
    unittest.main()
