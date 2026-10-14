"""KGIRL: holographic fractal memory (holo.py) and the recursive self-improvement loop (improve.py)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import tests.tauls  # noqa: F401  (sys.path)

try:
    import numpy as np
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


def clusters(n=6000, d=32, k=6, seed=0.3):
    """k Gaussian clusters; cluster c emits tokens 5c … 5c+4 uniformly (so ln 5 is the ideal NLL)."""
    from kgirl.tauls import chaos
    cen = chaos.normal(k * d, seed, 1).reshape(k, d) * 3
    lab = (chaos.uniform(n, seed, 2) * k).astype(int)
    h = cen[lab] + chaos.normal(n * d, seed, 3).reshape(n, d)
    y = lab * 5 + (chaos.uniform(n, seed, 4) * 5).astype(int)
    return h.astype(np.float32), y


@unittest.skipUnless(HAVE_TORCH, "numpy/torch not installed")
class HoloMemoryTests(unittest.TestCase):
    def test_mdl_recovers_the_clusters_and_reaches_the_ideal_nll(self):
        from kgirl.tauls.holo import HoloMemory, MemoryConfig
        h, y = clusters()
        m = HoloMemory(32, 50, MemoryConfig(rank=32, branching=6, max_depth=3, min_leaf=20))
        m.build(m.project(h[:5000]), y[:5000])
        s = m.summary()
        self.assertEqual(s["leaves"], 6)                     # one leaf per true cluster
        self.assertGreater(s["splits_rejected_by_mdl"], 0)   # uniform tokens inside a cluster: no further split
        p, _ = m.distribution(m.project(h[5000:]))
        self.assertTrue(np.allclose(p.sum(1), 1.0, atol=1e-5))
        nll = -np.log(p[np.arange(1000), y[5000:]]).mean()
        self.assertLess(nll, math.log(5) + 0.05)

    def test_without_mdl_the_tree_keeps_splitting(self):
        from kgirl.tauls.holo import HoloMemory, MemoryConfig
        h, y = clusters()
        m = HoloMemory(32, 50, MemoryConfig(rank=32, branching=6, max_depth=3, min_leaf=20, mdl=False))
        m.build(m.project(h), y)
        self.assertGreater(m.summary()["leaves"], 6)

    def test_holographic_vote_reads_key_similarity(self):
        """Inside one leaf the token is set by position on a circle; counts cannot see it, the trace can."""
        from kgirl.tauls import chaos
        from kgirl.tauls.holo import HoloMemory, MemoryConfig
        n, d = 3000, 16
        ang = chaos.uniform(n, 0.7, 1) * 2 * np.pi
        basis = np.linalg.qr(chaos.normal(d * 2, 0.7, 2).reshape(d, 2))[0]
        h = (np.stack([np.cos(ang), np.sin(ang)], 1) @ basis.T).astype(np.float32)
        y = (ang / (2 * np.pi) * 8).astype(int)               # 8 arcs → 8 tokens
        m = HoloMemory(d, 8, MemoryConfig(rank=64, max_depth=0))   # one cell: counts alone are uniform
        m.build(m.project(h[:2500]), y[:2500])
        rows = np.arange(500)
        p0, _ = m.distribution(m.project(h[2500:]), eta=0.0)
        p1, _ = m.distribution(m.project(h[2500:]), eta=0.5)
        nll0 = -np.log(p0[rows, y[2500:]]).mean()
        nll1 = -np.log(p1[rows, y[2500:]]).mean()
        self.assertAlmostEqual(nll0, math.log(8), delta=0.05)
        self.assertLess(nll1, nll0 - 0.2)

    def test_deterministic_save_load_and_fractal_telemetry(self):
        from kgirl.tauls.holo import HoloMemory, MemoryConfig
        h, y = clusters()
        a = HoloMemory(32, 50, MemoryConfig(rank=16, branching=4, max_depth=3, min_leaf=10, mdl=False))
        b = HoloMemory(32, 50, MemoryConfig(rank=16, branching=4, max_depth=3, min_leaf=10, mdl=False))
        a.build(a.project(h), y)
        b.build(b.project(h), y)
        self.assertTrue(np.array_equal(a.centroid, b.centroid))       # chaos, not a PRNG
        with tempfile.TemporaryDirectory() as td:
            a.save(Path(td) / "m.npz")
            c = HoloMemory.load(Path(td) / "m.npz")
        z = a.project(h[:64])
        self.assertTrue(np.allclose(a.distribution(z, 0.5)[0], c.distribution(z, 0.5)[0]))
        f = a.fractal_dimension()
        self.assertGreater(f["dimension"], 0.0)
        radii = [lv["rms_radius"] for lv in f["levels"]]
        self.assertEqual(radii, sorted(radii, reverse=True))            # finer cells are smaller

    def test_gate_prefers_the_less_uncertain_source(self):
        from kgirl.tauls.holo import gate
        w = gate(np.array([3.0, 0.5]), np.array([0.5, 3.0]), prior=0.5, temp=1.0)
        self.assertGreater(w[0], 0.9)
        self.assertLess(w[1], 0.1)
        self.assertTrue(np.allclose(gate(np.zeros(3), np.ones(3), 0.2, math.inf), 0.2))


@unittest.skipUnless(HAVE_TORCH, "numpy/torch not installed")
class ImproveTests(unittest.TestCase):
    def test_loop_runs_never_worsens_and_saves_a_loadable_checkpoint(self):
        from tests.tauls.test_train import cfg, corpus
        from kgirl.tauls.data import TokenData
        from kgirl.tauls.improve import ImproveConfig, improve
        from kgirl.tauls.train import load_model, train
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            corpus(d)
            train(cfg(d, "a", steps=30), log=lambda *a: None)
            model, ck = load_model(d / "a" / "best.pt")
            data = TokenData(d / "data")
            ic = ImproveConfig(rounds=2, depths=[1, 2], memory_tokens=768, eval_tokens=1024, distil_steps=6,
                               batch=4, distil_positions=64, memory={"rank": 16, "min_leaf": 8, "branching": 4})
            rec = improve(model, data, ic, d / "kg", torch.device("cpu"), log=lambda *a: None, ckpt=ck)
            self.assertEqual(len(rec["rounds"]), 2)
            self.assertLessEqual(rec["final_val_B_nll"], rec["baseline_val_B_nll"])   # monotone acceptance
            for r in rec["rounds"]:
                self.assertTrue(math.isfinite(r["memory_score"]["nll_mixed"]))
                self.assertLessEqual(r["memory_score"]["nll_mixed"], r["memory_score"]["nll_lm"] + 0.05)
            tel = json.loads((d / "kg" / "telemetry.json").read_text())
            self.assertEqual(tel["rounds"][1]["depth"], 2)
            m2, _ = load_model(d / "kg" / "improved.pt")
            self.assertEqual(m2.cfg, model.cfg)

            from kgirl.tauls.holo import HoloMemory
            from kgirl.tauls.improve import generate_with_memory
            mem = HoloMemory.load(d / "kg" / "memory_round2.npz")
            g = rec["rounds"][1]["memory_score"]["gate"]
            a = generate_with_memory(m2, mem, g, [1, 2, 3], 8)
            b = generate_with_memory(m2, mem, g, [1, 2, 3], 8)
            self.assertEqual(len(a), 8)
            self.assertEqual(a, b)                                    # chaos sampler: reproducible
            self.assertTrue(all(0 <= t < m2.cfg.vocab_size for t in a))


if __name__ == "__main__":
    unittest.main()
