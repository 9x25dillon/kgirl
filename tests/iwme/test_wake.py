"""Wake solver and the Source B §4 checks."""

import math
import unittest

import tests.iwme  # noqa: F401

import numpy as np

from kgirl.iwme import wake as wk


class Schemes(unittest.TestCase):
    def test_cfl_closed_forms(self):
        self.assertAlmostEqual(wk.cfl_limit("centered", 1.2, 2), 1 / math.sqrt(2))
        self.assertAlmostEqual(wk.cfl_limit("backward", 0.5, 1), math.sqrt(0.75))
        self.assertEqual(wk.cfl_limit("backward", 2.5, 1), 0.0)
        self.assertAlmostEqual(wk.cfl_limit_numeric("backward", 0.5, 1), math.sqrt(0.75), places=7)

    def test_orders(self):
        self.assertAlmostEqual(wk.temporal_order("centered"), 2.0, delta=0.05)
        self.assertAlmostEqual(wk.temporal_order("backward"), 1.0, delta=0.05)

    def test_stability_boundary_is_sharp(self):
        n = 128
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        W0 = np.cos(2 * np.pi * x) + 0.01 * np.cos(2 * np.pi * 63 * x)
        for scheme, gamma in (("centered", 40.0), ("backward", 40.0)):
            for factor, expect in ((0.98, True), (1.02, False)):
                dt = dx * factor * wk.cfl_limit(scheme, gamma * dx * factor, 1)
                dt = dx * factor * wk.cfl_limit(scheme, gamma * dt, 1)
                run = wk.run_wake(W0, np.zeros(n), gamma=gamma, dx=dx, dt=dt, steps=3000, scheme=scheme)
                self.assertEqual(run.stable, expect, (scheme, factor))

    def test_energy_identity_with_forcing_and_damping(self):
        n = 64
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        run = wk.run_wake(np.exp(-((x - 0.3) / 0.05) ** 2), np.zeros(n), gamma=1.3, dx=dx, dt=0.8 * dx, steps=500,
                          force=lambda k, t: np.sin(4 * np.pi * x) * math.sin(5 * t))
        np.testing.assert_allclose(run.energy - run.energy[0], run.work - run.dissipated, atol=1e-13)

    def test_unforced_energy_never_increases(self):
        n = 64
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        run = wk.run_wake(np.exp(-((x - 0.5) / 0.05) ** 2), np.zeros(n), gamma=0.4, dx=dx, dt=0.9 * dx, steps=400)
        self.assertTrue(np.all(np.diff(run.energy) <= 1e-15))

    def test_hybrid_zero_order_hold(self):
        n = 32
        x = (np.arange(n) + 0.5) / n
        hw = wk.HybridWake(basis=np.stack([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]), eta=2.0, hold=5)
        f = hw.forcing(np.array([[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(f(4, 0.0), 2 * np.sin(2 * np.pi * x))
        np.testing.assert_allclose(f(5, 0.0), 2 * np.cos(2 * np.pi * x))
        np.testing.assert_allclose(f(99, 0.0), 2 * np.cos(2 * np.pi * x))      # held after the last memory step


class SourceBChecks(unittest.TestCase):
    def test_defect_lives_in_the_curl(self):
        for m in (1, -1, 2):
            d = wk.defect_identity(48, m, loops=(2, 4, 8, 16))
            np.testing.assert_allclose(d["circulation"], 2 * math.pi * m, atol=1e-9)
            self.assertLess(d["max_abs_flux"], 1e-9)

    def test_coherence_leaves_unit_interval(self):
        ce = wk.coherence_equation(0.8, 2.0, 0.3, 4.0)
        self.assertLess(ce["c_infinity"], 0.0)
        self.assertAlmostEqual(ce["t_cross_zero"], math.log(1 + 2.0 * 0.8 / (0.3 * 4.0)) / 2.0)
        self.assertTrue(np.all(np.diff(wk.coherence_near_vortex(np.array([0.1, 0.2, 0.4]))) > 0))

    def test_static_source_pumps_energy(self):
        r = wk.static_source_energy(steps=200)
        self.assertFalse(r["monotone_decay"])

    def test_steady_wake(self):
        _, W = wk.steady_point_source(3.0, mass=0.0, n=601)
        self.assertAlmostEqual(W[300], 1.5, places=8)
        x, W = wk.steady_point_source(8.0, mass=1.5, n=8001)       # e^{−√1.5·8} keeps the Dirichlet ends negligible
        np.testing.assert_allclose(W, wk.eq_4_9(x, 1.0, 1.5, 1.0), atol=2e-4)

    def test_tridiagonal_solver(self):
        n = 50
        lower, upper = -np.ones(n - 1), -np.ones(n - 1)
        diag = 3 * np.ones(n)
        rhs = np.arange(n, dtype=float)
        A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
        np.testing.assert_allclose(wk.solve_tridiagonal(lower, diag, upper, rhs), np.linalg.solve(A, rhs), atol=1e-12)

    def test_pulse_displacement(self):
        r = wk.decoherence_pulse(beta=0.7, sigma=1.0, gamma=0.35, T_end=120.0, dt=2e-3)
        self.assertAlmostEqual(r["W_final"], 2.0, places=5)

    def test_point_source_energy_log_divergence(self):
        self.assertAlmostEqual(wk.point_source_energy_2d(1e-6) / wk.point_source_energy_2d(1e-3), 2.0)


@unittest.skipUnless(__import__("importlib").util.find_spec("sympy"), "sympy not installed")
class EulerLagrange(unittest.TestCase):
    def test_lagrangian_4_4(self):
        r = wk.euler_lagrange_4_4()
        self.assertTrue(r["has_restoring_term"])
        self.assertFalse(r["coherence_source_is_time_derivative"])


if __name__ == "__main__":
    unittest.main()
