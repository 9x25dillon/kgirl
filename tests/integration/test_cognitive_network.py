"""Regression tests for the repaired emergent cognitive network and the chaos generator it now uses."""
import unittest

import numpy as np

import tests.integration  # noqa: F401  (sys.path)

from kgirl.evolve.chaos import ChaosGenerator


class ChaosGeneratorTests(unittest.TestCase):
    def test_reproducible_and_calibrated(self):
        a, b = ChaosGenerator(0.3), ChaosGenerator(0.3)
        self.assertTrue(np.array_equal(a.normal(0, 1, 64), b.normal(0, 1, 64)))
        u = ChaosGenerator(0.7).random(50_000)
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.01)
        self.assertAlmostEqual(u.var(), 1 / 12, delta=0.003)
        p = ChaosGenerator(0.7).poisson(0.1, 50_000)
        self.assertAlmostEqual(p.mean(), 0.1, delta=0.01)

    def test_scalar_draws_do_not_waste_the_stream(self):
        g, h = ChaosGenerator(0.5), ChaosGenerator(0.5)
        scalars = np.array([g.random() for _ in range(10)])
        self.assertTrue(np.allclose(scalars, h.random(10)))


class CognitiveNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kgirl.cognitive import emergent_cognitive_network as ecn
        cls.ecn = ecn

    def test_neurons_spike_and_short_inputs_are_accepted(self):
        p = self.ecn.NeuromorphicProcessor(200)
        out = p.process_spiking_input(np.full(50, 10.0), timesteps=40)
        self.assertGreater(sum(int(s.sum()) for s in out["spike_trains"]), 0)

    def test_associative_memory_ranks_by_content(self):
        m = self.ecn.HolographicAssociativeMemory(hologram_dim=32)
        x = np.arange(64)
        k_sin = m.store_holographic(np.sin(x / 5), {"emotional_valence": 0.5})
        m.store_holographic(np.cos(x * 1.3), {"emotional_valence": 0.5})
        hits = m.recall_associative(np.roll(np.sin(x / 5), 7), similarity_threshold=0.5)
        self.assertEqual(hits[0]["memory_key"], k_sin)
        self.assertEqual(len(hits), 1)
        self.assertTrue(np.allclose(hits[0]["reconstructed_data"], np.sin(x / 5)))

    def test_data_engine_completes_missing_region(self):
        e = self.ecn.HolographicDataEngine(data_dim=16)
        a = np.outer(np.sin(np.arange(16) / 3), np.cos(np.arange(16) / 4))
        e.encode_holographic(a)
        e.encode_holographic(np.eye(16))
        part = a.copy()
        part[4:9, 4:9] = np.nan
        rec = e.recall_holographic(part, iterations=5)
        self.assertEqual(rec.shape, a.shape)
        self.assertLess(np.mean((rec[4:9, 4:9] - a[4:9, 4:9]) ** 2), 1e-6)
        self.assertTrue(np.allclose(e.readout(), a + np.eye(16)))
        hits = e.associative_recall(a[2:6, 3:8], similarity_threshold=0.99)
        self.assertEqual((hits[0]["pattern_index"], hits[0]["location"]), (0, (2, 3)))

    def test_orchestrator_metrics_are_measured_and_deterministic(self):
        self.ecn.reseed(0.4123)
        o = self.ecn.EmergentTechnologyOrchestrator()
        m1 = o._measure_cognitive_evolution()
        m2 = o._measure_cognitive_evolution()
        self.assertEqual(m1["complexity"], m2["complexity"])      # the probe does not mutate state
        self.assertTrue(0.0 <= m1["complexity"] <= 1.0 and 0.0 <= m1["robustness"] <= 1.0)
        self.assertGreater(m1["mean_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
