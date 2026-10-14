"""kgirl.quantum holographic memories: recall must depend on content (fixed phase keys, normalised scores)."""
import unittest
import warnings

import numpy as np

import tests.integration  # noqa: F401  (sys.path)


class HolographicMemoryTests(unittest.TestCase):
    def test_simple_memory_recalls_degraded_copy_first(self):
        from kgirl.quantum.holographic_memory_simple import HolographicAssociativeMemory
        from kgirl import telemetry
        m = HolographicAssociativeMemory(hologram_dim=32, fractal_dim=16)
        target = telemetry.lattice_matrix(8, 8)
        m.store("map", target)
        m.store("wave", np.sin(np.linspace(0, 4 * np.pi, 64)))
        m.store("ramp", np.arange(10.0))
        q = target.copy()
        q.flat[::5] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = m.recall(q, top_k=3)
        self.assertEqual(res[0]["key"], "map")
        for r in res:
            self.assertLessEqual(r["quantum_similarity"], 1.0 + 1e-9)
            self.assertLessEqual(r["holographic_similarity"], 1.0 + 1e-9)

    def test_system_memory_recall_is_deterministic_and_normalised(self):
        from kgirl.quantum.holographic_memory_system import HolographicAssociativeMemory
        m = HolographicAssociativeMemory(memory_size=8, hologram_dim=64)
        x = np.sin(np.arange(64) / 4.0)
        m.store(x, {})
        m.store(np.cos(np.arange(64) * 1.7), {})
        hits = m.recall(x, threshold=0.9)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["similarity"], 1.0, places=9)
        self.assertTrue(np.allclose(hits[0]["reconstructed_data"], x))

    def test_orchestrator_metrics_are_measured(self):
        from kgirl.quantum.holographic_memory_system import EnhancedCognitiveMemoryOrchestrator
        o = EnhancedCognitiveMemoryOrchestrator()
        base = np.sin(np.linspace(0, 6 * np.pi, 256))
        for k in range(3):
            o.integrated_memory_processing({"data": np.roll(base, 5 * k)}, {"emotional_valence": 0.5})
        self.assertEqual(len(o.cognitive_trajectory), 3)
        self.assertGreater(o.memory_resilience, 0.0)
        rec = o.emergent_memory_recall({"data": base, "similarity_threshold": 0.5})
        self.assertGreater(rec["fractal"]["fractal_completion_confidence"], 0.9)
        self.assertTrue(0.0 <= rec["quantum"][0]["overlap_probability"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
