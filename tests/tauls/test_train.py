import json
import math
import tempfile
import unittest
from pathlib import Path

import tests.tauls  # noqa: F401  (sys.path)

try:
    import numpy  # noqa: F401  (required by kgirl.tauls.data)
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

TEXT = ("def mean(xs):\n    return sum(xs) / len(xs)\n\n"
        "def total(xs):\n    out = 0\n    for x in xs:\n        out += x\n    return out\n")


def corpus(d: Path):
    from kgirl.tauls.data import build, iter_documents
    from kgirl.tauls.tokenizer import train
    root = d / "repo"
    root.mkdir()
    for i in range(12):
        (root / f"m{i}.py").write_text(f"# module {i}\n" + TEXT * (3 + i % 3))
    tok = train((x.framed() for x in iter_documents([root])), 320)
    build([root], tok, d / "data", val_permille=200, log=lambda *a: None)
    return tok


def cfg(d: Path, out: str, **kw):
    from kgirl.tauls.train import TrainConfig
    base = dict(data=str(d / "data"), out=str(d / out), preset="tiny", model={"ctx": 32}, batch=8, accum=2,
                steps=40, lr=3e-3, min_lr=3e-4, warmup=5, eval_every=20, save_every=20, log_every=10,
                eval_tokens=4096, guard_warmup=5, device="cpu")
    base.update(kw)
    return TrainConfig.from_dict(base)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TrainTests(unittest.TestCase):
    def test_learns_resumes_exactly_and_generates(self):
        from kgirl.tauls import chaos
        from kgirl.tauls.train import load_model, train
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            tok = corpus(d)
            quiet = lambda *a: None  # noqa: E731
            full = train(cfg(d, "a"), log=quiet)
            self.assertEqual(full["step"], 40)
            events = [json.loads(line) for line in (d / "a" / "metrics.jsonl").read_text().splitlines()]
            losses = [e["loss"] for e in events if e["event"] == "train"]
            self.assertLess(losses[-1], losses[0] * 0.8)
            self.assertTrue(math.isfinite(full["last_eval"]["bpb"]))
            self.assertIn("tauls", next(e for e in events if e["event"] == "train"))
            self.assertTrue((d / "a" / "best.pt").exists())

            train(cfg(d, "b", stop_at=13), log=quiet)                # interrupted session
            res = train(cfg(d, "b"), resume=True, log=quiet)         # resumed to the end
            self.assertEqual(res["step"], 40)
            a = torch.load(d / "a" / "latest.pt", weights_only=False)
            b = torch.load(d / "b" / "latest.pt", weights_only=False)
            for k in a["model"]:
                self.assertTrue(torch.equal(a["model"][k], b["model"][k]), k)
            self.assertEqual(a["guard"], b["guard"])

            m, ck = load_model(d / "a" / "latest.pt")
            self.assertEqual(ck["tokenizer_sha"], tok.sha)
            out = m.generate(tok.encode("def "), 20, sampler=chaos.Sampler(0.5))
            self.assertIsInstance(tok.decode(out), str)

    def test_hierarchical_preset_trains(self):
        from kgirl.tauls.train import train
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            corpus(d)
            res = train(cfg(d, "h", preset="tiny-hrm", steps=12, eval_every=12, save_every=100,
                            model={"ctx": 32, "loss": "stablemax"}), log=lambda *a: None)
            self.assertEqual(res["step"], 12)
            self.assertTrue(math.isfinite(res["last_eval"]["loss"]))


class GuardTests(unittest.TestCase):
    def test_coherence_thresholds(self):
        if not HAVE_TORCH:
            self.skipTest("torch not installed")
        from kgirl.tauls.train import CoherenceGuard
        g = CoherenceGuard(beta=0.9, warmup=5)
        for _ in range(5):
            state, _ = g.decide(1.0, 2.0)
            self.assertEqual(state, "STABLE")
            g.absorb(state, 1.0)
        self.assertEqual(g.decide(1.0, 2.0)[0], "HIGH")
        self.assertEqual(g.decide(1.6, 2.0)[0], "STABLE")      # κ ≈ 0.62
        self.assertEqual(g.decide(3.0, 2.0)[0], "LOW")         # κ ≈ 0.33
        self.assertEqual(g.decide(10.0, 2.0)[0], "CRITICAL")   # κ = 0.1
        self.assertEqual(g.decide(float("nan"), 2.0)[0], "CRITICAL")
        self.assertEqual(g.decide(1.0, float("inf"))[0], "CRITICAL")
        ema = g.ema
        g.absorb("CRITICAL", 10.0)                             # a spike never raises its own threshold
        self.assertEqual((g.ema, g.skips, g.run), (ema, 1, 1))


if __name__ == "__main__":
    unittest.main()
