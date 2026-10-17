"""TorchSubstrate on AURIC (patern-coding). Skipped without torch or a patern-coding checkout."""

import unittest

import tests.vessel  # noqa: F401  (sys.path)

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

from kgirl.paths import find_repo

HAVE_AURIC = (find_repo("patern-coding") / "auric" / "model.py").exists()


@unittest.skipUnless(HAVE_TORCH and HAVE_AURIC, "needs torch and a patern-coding checkout")
class AuricSubstrateTests(unittest.TestCase):
    SMALL = {"dim": 32, "layers": 2, "heads": 2, "context": 16}

    def setUp(self):
        from kgirl.vessel import corpus as C
        from kgirl.vessel.torch_substrate import auric_substrate
        torch.set_num_threads(2)
        self.a = auric_substrate(overrides=self.SMALL)
        self.P = C.probe_set(8, 48)

    def test_formats_and_chaos_init(self):
        from kgirl.vessel.torch_substrate import auric_substrate
        acts = self.a.activations(self.P)
        self.assertEqual(set(acts), {"block0", "block1"})
        self.assertEqual(acts["block1"].shape, (48, 32))
        lp = self.a.log_probs(self.P)
        self.assertEqual(lp.shape, (48, 256))
        np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-5)
        twin = auric_substrate(overrides=self.SMALL)
        np.testing.assert_array_equal(self.a.parameters()["blocks.0.qkv.weight"], twin.parameters()["blocks.0.qkv.weight"])

    def test_invariance_holds_on_transformer_activations(self):
        from kgirl.vessel.invariance import verify_activations
        n, worst, fails, controls, broken, _ = verify_activations(self.a.activations(self.P)["block1"], samples=3)
        self.assertEqual((fails, broken), ([], []))
        self.assertLess(worst, 1e-9)

    def test_rep_gradient_descends_and_restore_is_exact(self):
        from kgirl.vessel import geometry as G
        from kgirl.vessel.substrate import Objective
        acts = self.a.activations(self.P)
        A = {k: G.normalized_gram(np.tanh(v[:, ::-1] * 3)) for k, v in acts.items()}

        def rep():
            now = self.a.activations(self.P)
            return sum(1 - G.gram_alignment(A[k], G.normalized_gram(now[k])) for k in A)

        r0 = rep()
        state = self.a.checkpoint()
        self.a.opt = torch.optim.SGD(self.a.model.parameters(), lr=1e-3)
        self.a.step(None, Objective(lr=1e-3, task_weight=0, lam_rep=1.0, anchors=A, probes=self.P, clip=0))
        self.assertLess(rep(), r0)
        self.a.restore(state)
        self.assertAlmostEqual(rep(), r0, places=10)

    def test_identity_crosses_from_numpy_mlp_into_auric(self):
        from kgirl.vessel import corpus as C
        from kgirl.vessel.matrixdb import MatrixDB
        from kgirl.vessel.persona import genome_for
        from kgirl.vessel.substrate import Objective, ReferenceLM, RefConfig
        from kgirl.vessel.torch_substrate import auric_substrate
        from kgirl.vessel.vessel import Probes, Vessel
        probes = Probes.default(8, 128, 64)
        X, y = C.windows(C.prose(0.11, 400), 8)
        m = ReferenceLM(RefConfig(context=8))
        for b in C.batches(X, y, 128, 0.3, 300):
            m.step(b, Objective(lr=3e-3))
        with MatrixDB() as db:
            v = Vessel.birth(db, "ada", m, probes, genome_for("ada"))
            teach, _ = C.windows(C.prose(0.77, 200), 8)
            out = v.migrate(auric_substrate(overrides=self.SMALL, seed=0.9), steps=200, teach=teach, batch=64, accept=0.0)
            self.assertGreater(out["drift_before"], 0.95)       # measured 0.997
            self.assertLess(out["drift_after"], 0.65)           # measured 0.516 at 200 steps
            self.assertFalse(out["adopted"])
            self.assertEqual(out["to"]["substrate"], "auric")


if __name__ == "__main__":
    unittest.main()
