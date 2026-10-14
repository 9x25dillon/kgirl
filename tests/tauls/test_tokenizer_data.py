import json
import tempfile
import unittest
from pathlib import Path

import tests.tauls  # noqa: F401  (sys.path)

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False

from kgirl.tauls.tokenizer import EOT, Tokenizer, train

CORPUS = [
    "def mean(xs):\n    return sum(xs) / len(xs)\n",
    "class Soup:\n    def recall(self, query, budget=600):\n        return self.pool.search(query)\n",
    "# nihil, scar on zero, Bloom — café 😀 ünïcödé\nx = 1234567 ** 2\n",
] * 20


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = train(CORPUS, 400)

    def test_round_trip_any_text(self):
        for text in CORPUS + ["", "\x00\x01 binary-ish ​", "    \n\t\t  trailing  ", "never seen: Ωμέγα ☃ 𝔘",
                              "a" * 5000]:
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_specials_are_single_ids_and_never_merged(self):
        ids = self.tok.encode(f"x{EOT}y<|user|>")
        self.assertIn(self.tok.eot, ids)
        self.assertIn(self.tok.specials["<|user|>"], ids)
        self.assertNotIn(self.tok.eot, self.tok.encode(f"x{EOT}y", allow_special=False))
        self.assertEqual(self.tok.vocab_size, 256 + len(self.tok.merges) + len(self.tok.specials))

    def test_training_is_deterministic_and_compresses(self):
        again = train(CORPUS, 400)
        self.assertEqual(self.tok.merges, again.merges)
        self.assertEqual(self.tok.sha, again.sha)
        text = CORPUS[1]
        self.assertLess(len(self.tok.encode(text)), len(text.encode("utf-8")) / 2)
        self.assertIn(" return", [self.tok.decode([i]) for i in self.tok.encode(text)])

    def test_save_load_and_sha_check(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.tok.save(Path(d) / "tok.json")
            back = Tokenizer.load(p)
            self.assertEqual(back.encode(CORPUS[2]), self.tok.encode(CORPUS[2]))
            data = json.loads(p.read_text())
            data["merges"] = data["merges"][:-1]
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                Tokenizer.load(p)


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ChaosTests(unittest.TestCase):
    def test_statistics_addressability_and_independence(self):
        from kgirl.tauls import chaos
        u = chaos.uniform(200_000, 0.4123, 7)
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.005)
        self.assertAlmostEqual(u.var(), 1 / 12, delta=0.002)
        hist = np.histogram(u, 20, (0, 1))[0]
        self.assertLess((hist.max() - hist.min()) / hist.mean(), 0.05)
        self.assertLess(abs(np.corrcoef(u[:-1], u[1:])[0, 1]), 0.01)
        self.assertLess(abs(np.corrcoef(u[:50_000], chaos.uniform(50_000, 0.4123, 8))[0, 1]), 0.02)
        self.assertTrue(np.array_equal(chaos.uniform(10, 0.4123, 5, start=100), chaos.uniform(110, 0.4123, 5)[100:]))
        z = chaos.normal(200_000, 0.4123, 3)
        self.assertAlmostEqual(z.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(z.std(), 1.0, delta=0.01)
        self.assertEqual(sorted(chaos.permutation(100, 0.1)), list(range(100)))


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class DataTests(unittest.TestCase):
    def test_build_dedupes_filters_and_samples_deterministically(self):
        from kgirl.tauls.data import TokenData, build, iter_documents
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            (root / "pkg").mkdir(parents=True)
            (root / "vendor").mkdir()
            (root / "node_modules").mkdir()
            (root / ".git").mkdir()
            for i, text in enumerate(CORPUS[:3]):
                (root / "pkg" / f"m{i}.py").write_text(text * 30)
            (root / "vendor" / "m0_copy.py").write_text(CORPUS[0] * 30)      # vendored duplicate: read once
            (root / "node_modules" / "x.js").write_text("var a = 1;\n" * 50)  # skipped directory
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")      # git internals
            (root / "blob.py").write_bytes(b"\x00\x01\x02 not text")           # binary
            (root / "min.js").write_text("a" * 5000)                           # minified, one line
            (root / "README").write_text("no extension")                       # not an allowed extension
            docs = list(iter_documents([root]))
            self.assertEqual(sorted(x.path for x in docs), ["pkg/m0.py", "pkg/m1.py", "pkg/m2.py"])
            sft = Path(d) / "sft.jsonl"
            sft.write_text(json.dumps({"instruction": "fix mean", "context": "calc.py", "output": "--- a\n+++ b"}) + "\n")
            tok = train((x.framed() for x in docs), 400)
            meta = build([root], tok, Path(d) / "data", val_permille=0, sft=sft, log=lambda *a: None)
            self.assertEqual(meta["docs"]["train"] + meta["docs"]["val"], 4)
            self.assertEqual(meta["docs"]["val"], 1)                           # guaranteed even at 0 ‰
            manifest = [json.loads(line) for line in (Path(d) / "data" / "manifest.jsonl").read_text().splitlines()]
            self.assertIn("soup", {m["repo"] for m in manifest})
            td = TokenData(Path(d) / "data")
            self.assertEqual(td.tokens("train") + td.tokens("val"), sum(m["tokens"] for m in manifest))
            x1, y1 = td.windows("train", 5, 4, 32, 0.4123)
            x2, y2 = td.windows("train", 5, 4, 32, 0.4123)
            self.assertTrue(np.array_equal(x1, x2))
            self.assertTrue(np.array_equal(x1[:, 1:], y1[:, :-1]))
            self.assertFalse(np.array_equal(x1, td.windows("train", 6, 4, 32, 0.4123)[0]))
            self.assertEqual(x1.shape, (4, 32))
            self.assertIn("# file: repo/pkg/m", tok.decode(td.split["train"][:20].tolist()))


if __name__ == "__main__":
    unittest.main()
