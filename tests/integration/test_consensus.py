"""Consensus math in kgirl.main: the spectral fallback must score agreement, not concentration."""
import unittest

import numpy as np

import tests.integration  # noqa: F401  (sys.path)

try:
    from kgirl.main import spectral_coherence, spectral_weights
except Exception as exc:          # fastapi / pydantic-settings missing
    spectral_coherence = None
    IMPORT_ERROR = exc


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class ConsensusTests(unittest.TestCase):
    def test_identical_answers_are_fully_coherent(self):
        e = np.tile([[0.3, -0.2, 0.9]], (4, 1))
        self.assertAlmostEqual(spectral_coherence(e), 1.0, places=9)
        self.assertTrue(np.allclose(spectral_weights(e), 0.25))

    def test_orthogonal_answers_score_zero_and_order_is_monotone(self):
        self.assertAlmostEqual(spectral_coherence(np.eye(3)), 0.0, places=9)
        base = np.array([1.0, 0.0, 0.0])
        close = np.stack([base, base + [0, 0.1, 0], base + [0, 0, 0.1]])
        far = np.stack([base, [0.0, 1.0, 0.0], base + [0, 0, 0.1]])
        self.assertGreater(spectral_coherence(close), spectral_coherence(far))

    def test_weights_survive_negative_eigenvector_sign(self):
        e = np.array([[1.0, 0.1], [0.9, 0.2], [1.0, 0.0]])
        for flip in (1.0, -1.0):
            w = spectral_weights(flip * e)
            self.assertAlmostEqual(w.sum(), 1.0, places=9)
            self.assertTrue((w > 0).all())


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class AskEndpointTests(unittest.TestCase):
    def test_ask_runs_with_demo_adapters(self):
        import os
        import sys
        import warnings
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "demo"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
        import kgirl.main as M
        from demo_adapter import get_demo_model_pool
        saved = list(M.g_model_pool)
        M.g_model_pool[:] = get_demo_model_pool()
        try:
            r = TestClient(M.app).post("/ask", json={"prompt": "Explain quantum entanglement"})
        finally:
            M.g_model_pool[:] = saved
        self.assertEqual(r.status_code, 200)           # was a NameError on every call
        self.assertTrue(0.0 <= r.json()["coherence"] <= 1.0)


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class FallbackEnergyTests(unittest.TestCase):
    def test_noise_scores_near_one_structure_near_zero_any_dimension(self):
        from kgirl.evolve.chaos import ChaosGenerator
        from kgirl.main import fallback_energy
        g = ChaosGenerator(0.3)
        for d in (256, 1536, 4096):
            self.assertGreater(fallback_energy(g.normal(0, 1, d)), 0.85)
            focused = np.zeros(d)
            focused[:4] = 1.0
            self.assertAlmostEqual(fallback_energy(focused), 4 * 3 / (d + 2), places=9)   # PR = 4

    def test_default_threshold_sits_above_the_text_average(self):
        from kgirl.main import CardyEnergy, calibrate_energy
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "demo"))
        from demo_adapter import DemoAdapter
        a = DemoAdapter("x", "factual")
        texts = ["The surrogate predicts k_inf from 39 rod enrichments with ridge regression.",
                 "Holographic recall ranks stored traces by their normalised overlap with the query.",
                 "The bridge exposes stability, fluctuation and entropy profiles over HTTP.",
                 "Each pull request needs green tests before it waits on review."]
        cal = calibrate_energy([a.embed(t) for t in texts])
        ce = CardyEnergy()
        if not ce.use_cth:
            self.assertGreater(ce.default_max, cal["mean"])


@unittest.skipIf(spectral_coherence is None, "kgirl.main not importable")
class AskRagAndTokenCapTests(unittest.TestCase):
    """use_rag is best-effort, generation is never capped, and the energy cutoff follows the embedder."""

    def _client_with(self, pool):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
        import kgirl.main as M
        saved = list(M.g_model_pool)
        M.g_model_pool[:] = pool
        self.addCleanup(lambda: M.g_model_pool.__setitem__(slice(None), saved))
        return M, TestClient(M.app)

    def _recording_pool(self, seen, state=None):
        from kgirl.main import BaseAdapter
        fixed_state = np.array([1.0, 0.0, 0.0]) if state is None else state

        class Recording(BaseAdapter):
            def __init__(self, name):
                self.name = name

            def generate(self, prompt):
                seen.append((self.name, prompt))
                return "the same answer"

            def embed(self, text):
                return np.array([1.0, 0.5, 0.25])

            def get_hidden_states(self, text):
                return fixed_state(text) if callable(fixed_state) else fixed_state

        return [Recording("a"), Recording("b")]

    def test_rag_server_down_still_answers_without_context(self):
        from unittest import mock
        seen = []
        M, client = self._client_with(self._recording_pool(seen))
        with mock.patch.object(M, "CHAOS_RAG_URL", "http://127.0.0.1:9"):   # nothing listens on port 9
            r = client.post("/ask", json={"prompt": "q?", "use_rag": True, "max_energy": 1.0})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["rag_hits"], [])
        self.assertEqual({p for _, p in seen}, {"q?"})                  # prompt left unaugmented
        self.assertEqual(r.json()["decision"], "auto")

    def test_rag_context_is_prepended_when_hits_come_back(self):
        from unittest import mock
        seen = []
        M, client = self._client_with(self._recording_pool(seen))
        with mock.patch.object(M, "_fetch_rag_context", return_value=("CTX-LINE", ["CTX-LINE"], {"top_k": 1})):
            r = client.post("/ask", json={"prompt": "q?", "use_rag": True})
        self.assertEqual(r.json()["rag_hits"], ["CTX-LINE"])
        self.assertEqual(r.json()["rag_router"], {"top_k": 1})
        for _, prompt in seen:
            self.assertIn("CTX-LINE", prompt)
            self.assertTrue(prompt.endswith("Question: q?"))

    def test_no_token_cap_is_accepted_or_sent(self):
        import kgirl.main as M
        self.assertNotIn("max_tokens", M.AskRequest.model_fields)
        seen = []
        _, client = self._client_with(self._recording_pool(seen))
        r = client.post("/ask", json={"prompt": "q?"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(seen), [("a", "q?"), ("b", "q?")])

    def _diffuse_pool(self, seen, name):
        # Hidden states shaped like nomic-embed-text: fallback energy around 0.8 for every prompt.
        import hashlib
        def state(text):
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=768)
            v[:8] += 3.0
            return v
        pool = self._recording_pool(seen, state)
        for a in pool:
            a.name = name
        return pool

    def test_diffuse_embedder_is_calibrated_instead_of_always_escalating(self):
        import kgirl.main as M
        M._energy_cutoffs.clear()
        seen = []
        _, client = self._client_with(self._diffuse_pool(seen, "diffuse-test"))
        r = client.post("/ask", json={"prompt": "What is the capital of Peru?"}).json()
        self.assertGreater(r["energy"], 0.5)                     # above both old fixed cutoffs (0.16, 0.5)
        self.assertGreater(r["max_energy"], 0.5)                 # the calibrated cutoff moved with it
        self.assertIn(r["decision"], ("auto", "needs_citations"))   # answered; under 0.16 / 0.5 it escalated
        self.assertEqual(r["answer"], "the same answer")
        self.assertEqual(M._energy_cutoffs["diffuse-test"]["source"], "calibrated:diffuse-test")

    def test_calibrated_cutoffs_order_and_explicit_override_wins(self):
        import kgirl.main as M
        M._energy_cutoffs.clear()
        seen = []
        _, client = self._client_with(self._diffuse_pool(seen, "diffuse-test-2"))
        cut = M.energy_cutoffs(M.g_model_pool[0], M.CardyEnergy())
        self.assertLess(cut["mean"], cut["auto"])
        self.assertLess(cut["auto"], cut["citations"])
        r = client.post("/ask", json={"prompt": "q?", "max_energy": 0.0}).json()
        self.assertEqual(r["max_energy"], 0.0)
        self.assertNotEqual(r["decision"], "auto")

    def test_calibration_failure_falls_back_to_fixed_cutoffs_and_retries(self):
        import kgirl.main as M
        M._energy_cutoffs.clear()

        class Down(M.BaseAdapter):
            name = "down-test"

            def get_hidden_states(self, text):
                raise RuntimeError("embedder down")

        ce = M.CardyEnergy()
        if ce.use_cth:
            self.skipTest("CTH energy has fixed cutoffs")
        self.assertEqual(M.energy_cutoffs(Down(), ce)["auto"], ce.FALLBACK_DEFAULT_MAX)
        self.assertNotIn("down-test", M._energy_cutoffs)        # not cached, so the next request retries
