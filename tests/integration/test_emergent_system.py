"""The LiMp modules rebuilt from their call sites: emergent_cognitive_system, holographic_pipeline_adapter."""
import asyncio
import json
import math
import unittest

import numpy as np

import tests.integration  # noqa: F401  (sys.path)


class EmergentSystemTests(unittest.TestCase):
    def test_steps_measure_what_they_claim(self):
        from kgirl.cognitive.emergent_cognitive_system import (NeuromorphicStep, QuantumOptimizationStep,
                                                               SwarmCognitiveStep)
        q = QuantumOptimizationStep(n_qubits=3)(np.arange(8.0))           # basis 0 has the lowest cost
        self.assertEqual(q["bitstring"], "000")
        self.assertGreater(q["ground_state_probability"], 0.95)
        self.assertAlmostEqual(float(np.sum(np.square(q["state"]))), 1.0, places=9)
        self.assertLess(q["entropy_trajectory"][-1], q["entropy_trajectory"][0])
        s = SwarmCognitiveStep(n_agents=16, search_dim=3)(lambda P: np.sum((P - 0.3) ** 2, axis=1))
        self.assertLess(s["best_cost"], 1e-4)
        self.assertGreater(s["convergence"], 0.5)
        quiet = NeuromorphicStep(n_neurons=20)(np.zeros(20))
        driven = NeuromorphicStep(n_neurons=20)(np.full(20, 10.0))
        self.assertEqual(quiet["mean_rate"], 0.0)
        self.assertGreater(driven["mean_rate"], 0.0)

    def test_cycle_api_used_by_the_bridges(self):
        import torch
        from kgirl.cognitive.emergent_cognitive_system import EmergentCognitiveOrchestrator
        x = torch.tensor(np.sin(np.arange(32) / 3), dtype=torch.float32)
        a = EmergentCognitiveOrchestrator().execute_cognitive_cycle(x)
        b = EmergentCognitiveOrchestrator().execute_cognitive_cycle(x)
        for key in ("quantum_state", "swarm_results", "neural_results", "emergence_metrics"):
            self.assertIn(key, a)
        self.assertEqual(a["emergence_metrics"], b["emergence_metrics"])          # deterministic
        self.assertTrue(all(0.0 <= v <= 1.0 for v in a["emergence_metrics"].values()))

    def test_bridges_now_find_the_module(self):
        from kgirl.cognitive import advanced_cognitive_enhancements as ace
        from kgirl.cognitive import cognitive_integration_bridge as cib
        self.assertTrue(ace.EMERGENT_AVAILABLE)
        self.assertTrue(cib.EMERGENT_AVAILABLE)


class PipelineAdapterTests(unittest.TestCase):
    def test_tools_round_trip_and_errors_are_replies(self):
        from kgirl.cognitive.holographic_pipeline_adapter import HolographicNumbskullAdapter
        a = HolographicNumbskullAdapter()
        x = np.sin(np.linspace(0, 6 * np.pi, 256))
        keys = []
        for shift in (0, 40, 80):
            r = asyncio.run(a.invoke("STORE_HOLOGRAPHIC", [json.dumps(np.roll(x, shift).tolist()), "{}"]))
            self.assertTrue(r["ok"])
            keys.append(r["memory_key"])
        r = asyncio.run(a.invoke("RECALL_ASSOCIATIVE", [json.dumps(np.roll(x, 40).tolist()), "0.99"]))
        self.assertTrue(r["ok"])
        self.assertEqual(r["matches"][0]["memory_key"], keys[1])
        self.assertEqual(asyncio.run(a.invoke("MEMORY_ANALYZE", []))["num_memory_traces"], 3)
        bad = asyncio.run(a.invoke("STORE_HOLOGRAPHIC", ["[]"]))
        self.assertFalse(bad["ok"])
        self.assertFalse(asyncio.run(a.invoke("NOPE", []))["ok"])

    def test_higuchi_fractal_dimension(self):
        from kgirl.cognitive.holographic_pipeline_adapter import HolographicNumbskullAdapter
        from kgirl.evolve.chaos import ChaosGenerator
        a = HolographicNumbskullAdapter()
        noise = ChaosGenerator(0.3).normal(0, 1, 1024)
        d = {name: asyncio.run(a.invoke("ENCODE_FRACTAL", [json.dumps(v.tolist())]))["fractal_dimension"]
             for name, v in (("line", np.linspace(0, 1, 1024)), ("noise", noise), ("brownian", np.cumsum(noise)))}
        self.assertAlmostEqual(d["line"], 1.0, places=2)
        self.assertGreater(d["noise"], 1.9)
        self.assertTrue(math.isclose(d["brownian"], 1.5, abs_tol=0.1))


if __name__ == "__main__":
    unittest.main()
