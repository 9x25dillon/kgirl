"""numbskull integration: the resolver, every component probe, and the emergent-protocol fixes.

Skipped when numbskull is not found (see requirements-numbskull.txt for where kgirl looks).
"""
import os
import tempfile
import unittest
from pathlib import Path

import tests.integration  # noqa: F401  (sys.path)

from kgirl.paths import candidates, find_repo

HAVE_NUMBSKULL = find_repo("numbskull").is_dir()


class PathTests(unittest.TestCase):
    def test_env_var_wins_and_missing_repo_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["ZZ_KGIRL_TEST_REPO_PATH"] = d
            try:
                self.assertEqual(find_repo("zz-kgirl-test-repo"), Path(d))
            finally:
                del os.environ["ZZ_KGIRL_TEST_REPO_PATH"]
        missing = find_repo("no-such-repo-anywhere")
        self.assertFalse(missing.exists())
        self.assertIn(Path("/home/kill/no-such-repo-anywhere"), candidates("no-such-repo-anywhere"))


@unittest.skipUnless(HAVE_NUMBSKULL, "numbskull not found (NUMBSKULL_PATH, <kgirl>/numbskull, ../numbskull)")
class NumbskullTests(unittest.TestCase):
    def test_every_component_probe_passes(self):
        try:
            import emergentnet  # noqa: F401
            import wavecaster  # noqa: F401
        except ImportError:
            self.skipTest("pip install -e numbskull/packages/wavecaster -e numbskull/packages/emergentnet")
        from kgirl.numbskull import status
        rep = status()
        failed = {k: v.get("error") for k, v in rep["components"].items() if not v["ok"]}
        self.assertEqual(failed, {})

    def test_emergent_protocol_runs_on_1d_input(self):
        import numpy as np
        from kgirl.numbskull import emergent_protocol
        res = emergent_protocol()(np.linspace(0.0, 1.0, 32))
        for section in ("holographic", "morphogenetic", "quantum_cognitive", "holographic_memory", "emergence_metrics"):
            self.assertIn(section, res)
        self.assertGreaterEqual(float(res["emergence_metrics"]["morphogenetic_convergence"]), 0.0)

    def test_neuro_symbolic_facade_is_deterministic(self):
        from kgirl.cognitive.neuro_symbolic_engine import FeatureExtractor, NeuroSymbolicEngine
        self.assertEqual(FeatureExtractor().extract("lattice"), FeatureExtractor().extract("lattice"))
        e = NeuroSymbolicEngine()
        for t in ("lattice damping coherence", "quantum swarm optimiser"):
            e.store(t, {"t": t})
        self.assertEqual(e.recall("damping the lattice", k=1)[0]["metadata"]["t"], "lattice damping coherence")
        a = e.analyze("kinetic force damping")
        self.assertTrue(0.0 <= a["pattern_complexity"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
