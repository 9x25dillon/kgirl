"""Source E checks: E8 roots, octonions, vortex rings, the proposed PDE, the dipole radius, modular flow."""

import math
import tempfile
import unittest
from pathlib import Path

import tests.iwme  # noqa: F401

import numpy as np

from kgirl.iwme import source_e as se


class Roots(unittest.TestCase):
    def test_counts_norms_and_normalization(self):
        f = se.root_facts()
        self.assertEqual((f["n_roots"], f["n_positive"]), (240, 120))
        self.assertEqual(f["inner_products"], [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.assertAlmostEqual(f["sum_norm2_over_120"], 2.0)
        self.assertAlmostEqual(se.root_facts(1 / math.sqrt(2))["sum_norm2_over_120"], 1.0)

    def test_standard_subsystems(self):
        f = se.root_facts()
        self.assertEqual((f["D8_subsystem"], f["E7_subsystem_orthogonal_to_a_root"]), (112, 126))
        self.assertTrue(se.negation_closed_counts_are_even(se.e8_roots()))


class Octonions(unittest.TestCase):
    def test_unit_and_conjugation(self):
        e0 = np.eye(8)[0]
        r = se._rng(0.11)
        a = r.normal(8)
        np.testing.assert_allclose(se.omul(e0, a), a)
        np.testing.assert_allclose(se.omul(a, se.conj(a)), np.dot(a, a) * e0, atol=1e-12)

    def test_jordan_identity_threshold(self):
        self.assertLess(se.jordan_identity_residual(3, trials=1), 1e-12)
        self.assertGreater(se.jordan_identity_residual(4, trials=1), 1e-3)


class Vortices(unittest.TestCase):
    def test_thomson_threshold(self):
        self.assertLess(se.ring_max_growth(6), 1e-4)
        self.assertGreater(se.ring_max_growth(8), 0.1)

    def test_triad_hessian(self):
        np.testing.assert_allclose(se.chordal_hessian_reduced(3), [[2 / 3, 1 / 3], [1 / 3, 2 / 3]], atol=1e-5)
        np.testing.assert_allclose(se.chordal_hessian_reduced(3, V0=2.0), 2 * np.array([[2, 1], [1, 2]]) / 3, atol=1e-5)


class PDE(unittest.TestCase):
    def test_radial_solver_matches_linear_closed_form(self):
        # β = μ = 0, uniform disk source q on r < a, u(R) = 0: u(r) = q/(2πα) ln(R/r) outside the core
        R, a = 10.0, 0.5
        r, u, _ = se.radial_steady(R, 2000, se.PDEParams(beta=0.0), q=1.0, core=a)
        for rr in (2.0, 5.0, 8.0):
            i = np.searchsorted(r, rr)
            self.assertAlmostEqual(u[i], math.log(R / r[i]) / (2 * math.pi), delta=2e-3)

    def test_far_field_amplitude(self):
        # Δu = (β/α)u³ ⇒ u → √(α/β)/r; at r = 40 of R = 160 the product r·u approaches 1/√2
        r, u, _ = se.radial_steady(160, 3200, se.PDEParams(beta=2.0), core=0.5)
        i = np.searchsorted(r, 40)
        self.assertGreater(r[i] * u[i], 0.5)
        self.assertLess(r[i] * u[i], 1 / math.sqrt(2) + 0.05)

    def test_lamb_term_is_orthogonal_to_M(self):
        Mx, My, h = se.initial_field(32)
        Lx, Ly = se.lamb_term(Mx, My, h)
        self.assertLess(np.max(np.abs(Mx * Lx + My * Ly)), 1e-12)


class FormFactor(unittest.TestCase):
    def test_radius_and_angle(self):
        r = se.dipole_radius_fm(se.dipole_lambda(30.0))
        self.assertAlmostEqual(r, 4 * se.HBARC_GEV_FM / se.M_PROTON_GEV, places=12)
        self.assertAlmostEqual(se.angle_for_radius(r), 30.0, places=9)

    def test_shape_moments(self):
        r = 0.84
        r2 = (r / se.HBARC_GEV_FM) ** 2
        h = 1e-4
        for name, g in se.shapes(r).items():
            d1 = (g(h) - g(-h)) / (2 * h)
            d2 = (g(h) - 2 * g(0.0) + g(-h)) / h ** 2
            self.assertAlmostEqual(-6 * d1 / r2, 1.0, places=5)                  # same ⟨r²⟩
            self.assertAlmostEqual(60 * d2 / r2 ** 2, se.r4_over_r2sq(name), places=3)   # ⟨r⁴⟩ = 60 G″

    def test_compare_bins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bins.csv"
            g = se.shapes(se.dipole_radius_fm(se.dipole_lambda()))["dipole (exponential density)"]
            p.write_text("Q2_GeV2,GE,sigma\n0.01,%.8f,0.001\n0.05,%.8f,0.002\n" % (g(0.01), g(0.05) + 0.004))
            rows = se.compare_bins(str(p))
            self.assertAlmostEqual(rows[0]["pull"], 0.0, places=4)       # CSV keeps 8 decimals
            self.assertAlmostEqual(rows[1]["pull"], -2.0, places=4)


class Modular(unittest.TestCase):
    def test_modular_flow_is_heisenberg(self):
        self.assertLess(se.modular_vs_heisenberg(n=4, beta=1.3, tau=-0.8), 1e-12)


if __name__ == "__main__":
    unittest.main()
