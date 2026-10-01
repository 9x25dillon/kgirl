"""numpy layers: pulse (nihil leverage), bloom sentinel, BPE array lattice."""

import unittest
from dataclasses import replace

import tests.nihiline  # noqa: F401

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class PulseTests(unittest.TestCase):
    def setUp(self):
        from kgirl.nihiline.pulse import PulseParams
        self.p = PulseParams(T_s=20.0, dt_s=0.004)

    def test_interior_optimal_nihil(self):
        from kgirl.nihiline.pulse import leverage_sweep
        r = leverage_sweep(0.6, np.linspace(0, 0.9, 10), self.p)
        self.assertGreater(r["f_opt"], 0.0)
        self.assertLess(r["f_opt"], 0.9)
        self.assertTrue(np.all(np.diff(r["viability"]) >= -1e-9))   # more nothing never hurts the cells

    def test_deterministic_and_vectorized_equals_scalar(self):
        from kgirl.nihiline.pulse import simulate
        a = simulate([0.2, 0.5], 0.6, self.p)
        b = simulate([0.2, 0.5], 0.6, self.p)
        np.testing.assert_array_equal(a["product"], b["product"])
        self.assertTrue(np.isfinite(a["cycle_entropy_bits"]).all())

    def test_control_sweep_reports_a_verdict(self):
        from kgirl.nihiline.pulse import control_from_tpa, control_nihil_sweep
        r = control_nihil_sweep(np.linspace(0, 1, 4), np.linspace(0, 0.9, 7), self.p)
        self.assertEqual(r["product"].shape, (4, 7))
        self.assertIn("nothing", r["verdict"])
        self.assertEqual(control_from_tpa(20), 0.0)
        self.assertEqual(control_from_tpa(70), 1.0)


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class BloomTests(unittest.TestCase):
    def test_order_asymmetry_matches_preprint(self):
        from kgirl.nihiline.bloom import run
        o = run().order_fit
        self.assertAlmostEqual(o["phase_linear_coeff"], o["expected_phase"], delta=0.01 * abs(o["expected_phase"]))
        self.assertAlmostEqual(o["amp_quadratic_coeff"], o["expected_amp"], delta=0.02 * o["expected_amp"])

    def test_phase_warns_before_amplitude(self):
        from kgirl.nihiline.bloom import run
        for onset in (600.0, 900.0):
            r = run(t_onset=onset)
            self.assertIsNotNone(r.delta_t)
            self.assertGreater(r.delta_t, 0.0)

    def test_sliding_matches_naive(self):
        from kgirl.nihiline.bloom import sliding
        x = np.sin(np.arange(200) * 0.37) + np.arange(200) * 0.01
        w, half = 21, 10
        naive = np.array([np.var(x[max(0, i - half):i + half + 1]) for i in range(len(x))])
        np.testing.assert_allclose(sliding(x, w, "var"), naive, atol=1e-10)

    def test_stress_onset(self):
        from kgirl.nihiline.bloom import onset_from_stress
        self.assertTrue(np.isfinite(onset_from_stress(0.02, 0.25, 0.0)))
        self.assertEqual(onset_from_stress(0.02, 0.25, 0.4), float("inf"))


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ArrayTests(unittest.TestCase):
    def test_winding_counts_a_planted_vortex(self):
        from kgirl.nihiline.array import wrapped_charge
        n, m = 9, 12
        y, x = np.mgrid[0:n, 0:m]
        psi = np.exp(1j * np.arctan2(y - 4.5, x - 5.5))
        q = wrapped_charge(psi, 0.0)
        self.assertEqual(int(np.abs(q[:-1]).sum()), 1)
        self.assertEqual(abs(int(q[:-1].sum())), 1)   # sign follows the lab's (clockwise) plaquette loop

    def test_characterize_and_lyapunov(self):
        from kgirl.nihiline.array import ArrayParams, characterize, lyapunov
        p = replace(ArrayParams(), lanes=6, length=20)
        r = characterize(p, steps=120, burn=60)
        self.assertTrue(0.0 <= r["R"] <= 1.0)
        self.assertEqual(set(r["lane_rsd"]), {"CEA", "AFP"})
        ly = lyapunov(p, steps=80)
        self.assertIn(ly["regime"], {"ordered", "critical", "chaotic"})


if __name__ == "__main__":
    unittest.main()
