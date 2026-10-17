"""kgirl.resonance against known answers. Margins sit around measured values (docs/resonance)."""

import math
import unittest

import tests.resonance  # noqa: F401  (sys.path)

import numpy as np

from kgirl.evolve.chaos import ChaosRNG
from kgirl.resonance import dfa, distances_from_gram, h0, mfdfa, pairwise_distances, recognize, rescaled_range, \
    sorted_death_distance, track, trace_dimension

FS = 1000.0
T = np.arange(0, 2, 1 / FS)


class FractalTests(unittest.TestCase):
    def test_white_noise_and_random_walk(self):
        w = ChaosRNG(0.3).normal(16384)
        self.assertAlmostEqual(dfa(w), 0.5, delta=0.04)              # measured 0.501
        self.assertAlmostEqual(dfa(np.cumsum(w)), 1.5, delta=0.05)   # measured 1.506
        self.assertGreater(rescaled_range(w), dfa(w))                # R/S is biased upward (0.545)
        self.assertAlmostEqual(trace_dimension(w), 1.5, delta=0.05)
        self.assertLess(mfdfa(w).width, 0.15)                         # monofractal

    def test_mfdfa_recovers_the_exact_cascade_spectrum(self):
        from kgirl.iwme.cascade import binomial_measure, binomial_tau
        qs = np.arange(-4, 4.01, 0.5)
        r = mfdfa(binomial_measure(0.3, 16), qs, dyadic=True)
        exact = np.array([binomial_tau(0.3, q) for q in qs])
        exact_alpha = np.gradient(exact, qs)
        self.assertAlmostEqual(r.width, exact_alpha.max() - exact_alpha.min(), delta=0.005)   # 1.1239 vs 1.1239
        resid = r.tau - exact
        self.assertLess(np.max(np.abs(resid - np.polyval(np.polyfit(qs, resid, 1), qs))), 0.005)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            mfdfa(np.ones(10))
        with self.assertRaises(ValueError):
            mfdfa(np.r_[np.ones(100), np.nan])


class TopologyTests(unittest.TestCase):
    def test_prim_equals_kruskal(self):
        D = pairwise_distances(ChaosRNG(0.4).normal((30, 4)))
        iu = np.triu_indices(30, 1)
        parent = list(range(30))

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        deaths = []
        for k in np.argsort(D[iu]):
            i, j = find(iu[0][k]), find(iu[1][k])
            if i != j:
                parent[i] = j
                deaths.append(D[iu[0][k], iu[1][k]])
        np.testing.assert_allclose(h0(D).deaths, np.sort(deaths))

    def test_clusters_betti_entropy_from_points_and_gram(self):
        from kgirl.vessel.geometry import normalized_gram
        r = ChaosRNG(0.2)
        X = np.concatenate([r.normal((20, 3)) * 0.1 + c for c in ([0, 0, 0], [5, 0, 0], [0, 5, 0])])
        bar = h0(pairwise_distances(X))
        self.assertEqual(bar.betti0(1.0), 3)
        self.assertEqual(sorted(np.bincount(bar.clusters(1.0))), [20, 20, 20])
        self.assertEqual(bar.n, 60)
        self.assertTrue(0 < bar.normalized_entropy() < 1)
        g = h0(distances_from_gram(normalized_gram(X)))
        self.assertEqual(g.betti0(g.gap()[1] + 1e-12), 3)            # the Gram alone recovers the families
        self.assertEqual(sorted_death_distance(bar, bar), 0.0)
        with self.assertRaises(ValueError):
            sorted_death_distance(bar, h0(pairwise_distances(X[:10])))

    def test_equal_bars_have_maximal_entropy(self):
        D = np.abs(np.subtract.outer(np.arange(6.0), np.arange(6.0)))
        self.assertAlmostEqual(h0(D).normalized_entropy(), 1.0)
        with self.assertRaises(ValueError):
            h0(np.array([[0, 1], [2, 0.0]]))


class PLLTests(unittest.TestCase):
    def test_acquires_tracks_and_refuses_noise(self):
        tr = track(np.cos(2 * np.pi * 40.3 * np.arange(0, 4, 1 / FS) + 0.7), FS, 40.0)
        self.assertTrue(tr.locked)
        self.assertLess(tr.lock_time(), 0.6)                          # measured 0.27 s
        self.assertAlmostEqual(tr.frequency_estimate(), 40.3, delta=0.02)
        t4 = np.arange(0, 4, 1 / FS)
        far = track(np.cos(2 * np.pi * 55.0 * t4), FS, 40.0)          # 15 Hz pull-in (type-2 loop)
        self.assertTrue(far.locked)
        self.assertAlmostEqual(far.frequency_estimate(), 55.0, delta=0.1)
        step = track(np.cos(2 * np.pi * np.cumsum(np.where(t4 < 2, 40.0, 41.0)) / FS), FS, 40.0)
        self.assertAlmostEqual(step.frequency_estimate(), 41.0, delta=0.01)
        noise = track(ChaosRNG(0.9).normal(len(t4)), FS, 40.0)
        self.assertFalse(noise.locked)
        self.assertTrue(math.isnan(noise.lock_time()))

    def test_config_guards(self):
        from kgirl.resonance import PLLConfig
        with self.assertRaises(ValueError):
            PLLConfig(fs=100, f0=60)                                  # above Nyquist
        with self.assertRaises(ValueError):
            PLLConfig(fs=1000, f0=10, bandwidth=5)                    # loop too wide for the carrier


class SpikingTests(unittest.TestCase):
    def test_single_tones_are_recognized_across_the_band(self):
        for f in (7.3, 12.0, 40.0, 150.0):
            r = recognize(np.sin(2 * np.pi * f * T), FS, lo=5, hi=200)
            self.assertAlmostEqual(r.dominant, f, delta=0.012 * f, msg=f)   # measured within 0.5 %

    def test_amplitude_invariance_two_tones_snr_and_silence(self):
        a = recognize(np.sin(2 * np.pi * 40 * T), FS, lo=5, hi=200)
        b = recognize(100 * np.sin(2 * np.pi * 40 * T), FS, lo=5, hi=200)
        self.assertEqual(a.summary(), b.summary())
        two = recognize(np.sin(2 * np.pi * 12 * T) + np.sin(2 * np.pi * 80 * T), FS, lo=5, hi=200).peaks()
        self.assertEqual(len(two), 2)
        self.assertAlmostEqual(two[0], 12, delta=1)
        self.assertAlmostEqual(two[1], 80, delta=3)
        snr1 = recognize(np.sin(2 * np.pi * 40 * T) + ChaosRNG(0.5).normal(len(T)), FS, lo=5, hi=200)
        self.assertAlmostEqual(snr1.dominant, 40, delta=0.5)
        for seed in (0.11, 0.37, 0.63, 0.89):
            silent = recognize(ChaosRNG(seed).normal(len(T)), FS, lo=5, hi=200)
            self.assertEqual(silent.spikes, 0)
            self.assertTrue(math.isnan(silent.dominant))
        self.assertEqual(recognize(np.zeros(100), FS).spikes, 0)


class MCPToolTests(unittest.TestCase):
    def test_resonance_analyze(self):
        import json

        from kgirl.harness.mcp_server import _DESCRIPTIONS, _tools
        tools = _tools(None)
        self.assertIn("resonance_analyze", _DESCRIPTIONS)
        fn = tools["resonance_analyze"][1]
        out = json.loads(fn({"series": list(np.cos(2 * np.pi * 40 * np.arange(3000) / FS)), "fs": FS, "f0": 39}))
        self.assertTrue(out["pll"]["locked"])
        self.assertAlmostEqual(out["resonators"]["dominant_hz"], 40, delta=0.6)
        for bad in ({"series": [1, 2, 3]}, {"series": [0.0] * 50, "fs": -1}, {"series": [float("nan")] * 50}):
            with self.assertRaises(ValueError):
                fn(bad)


if __name__ == "__main__":
    unittest.main()
