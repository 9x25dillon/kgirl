import os
import sys
import tempfile
import textwrap
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from tests.harness import _fixtures  # noqa: F401  (puts src/ on sys.path)
from kgirl.harness.atlas import Atlas
from kgirl.harness.atlas.instances import register_qincrs
from kgirl.harness.soup import CurationPolicy, Curator, QincrsValidator, Soup
from kgirl.harness.soup.qincrs_gate import ENGINE_MODULE

REAL_ENGINE = os.environ.get("KGIRL_QINCRS_PATH", "")


class Band(Enum):
    DELTA = 1
    THETA = 2
    ALPHA = 3
    BETA = 4
    GAMMA = 5


class StubAuditor:
    """Same quantities as QINCRS's auditor (mean shift, error std), no numpy."""

    def audit(self, orig, recon, t0, t1, capsule):
        errs = [recon[b] - orig[b] for b in Band]
        mean = sum(errs) / len(errs)
        sd = (sum((e - mean) ** 2 for e in errs) / len(errs)) ** 0.5
        return SimpleNamespace(Δκ=mean, D_ω=sd, audit_pass=False)


STUB = SimpleNamespace(FrequencyBand=Band, CollapseIntegrityAuditor=StubAuditor)


def _promote_policy():
    return CurationPolicy(min_support=1, promote_lcb=0.0)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.soup = Soup(Path(self.tmp.name) / "soup.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _frag(self, wins, losses):
        fid = self.soup.add(f"frag {wins}/{losses}", status="staged")
        self.soup.credit([fid], True, weight=wins)
        if losses:
            self.soup.credit([fid], False, weight=losses)
        return next(f for f in self.soup.all(status="staged") if f.id == fid)

    def test_robust_passes_thin_is_held(self):
        v = QincrsValidator(engine=STUB)
        self.assertTrue(v(self._frag(8, 0)))
        self.assertFalse(v(self._frag(1, 0)))
        self.assertEqual([x.ok for x in v.log], [True, False])
        self.assertIn("thin evidence", v.log[1].reason)

    def test_curator_uses_validator_to_gate_promotion(self):
        cur = Curator(self.soup, _promote_policy(), validator=QincrsValidator(engine=STUB))
        strong = self._frag(8, 0)
        weak = self._frag(1, 0)
        rep = cur.step()
        self.assertIn(strong.id, rep.promoted)
        self.assertNotIn(weak.id, rep.promoted)

    def test_fail_open_when_engine_missing(self):
        v = QincrsValidator(engine_path="/nonexistent/QiNCrs.py")
        self.assertTrue(v(self._frag(1, 0)))
        self.assertIn("engine unavailable", v.log[0].reason)

    def test_strict_blocks_when_engine_missing(self):
        v = QincrsValidator(engine_path="/nonexistent/QiNCrs.py", strict=True)
        self.assertFalse(v(self._frag(8, 0)))

    def test_does_not_touch_numbskull(self):
        before = {m for m in sys.modules if "numbskull" in m}
        QincrsValidator(engine=STUB)(self._frag(8, 0))
        self.assertEqual({m for m in sys.modules if "numbskull" in m}, before)

    @unittest.skipUnless(REAL_ENGINE and Path(REAL_ENGINE).is_file(), "set KGIRL_QINCRS_PATH to the real QiNCrs.py")
    def test_real_engine_runs(self):
        try:
            v = QincrsValidator(engine_path=REAL_ENGINE, strict=True)
            ok_strong, ok_weak = v(self._frag(8, 0)), v(self._frag(1, 0))
        finally:
            sys.modules.pop(ENGINE_MODULE, None)
        self.assertNotIn("engine unavailable", v.log[0].reason)
        self.assertTrue(ok_strong)
        self.assertFalse(ok_weak)


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.atlas = Atlas(self.base / "atlas.db")
        self.stage = self.base / "stage"

    def tearDown(self):
        self.atlas.close()
        self.tmp.cleanup()

    def _write(self, rel, text):
        p = self.base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text), encoding="utf-8")
        return p

    def test_identical_copies_collapse_and_variants_split(self):
        src = "def coherence():\n    return 1\n"
        a = self._write("one/QiNCrs.py", src)
        b = self._write("two/QiNCrs.py", src)
        c = self._write("three/qincrs_core.py", "def core():\n    return 2\n")
        inst = register_qincrs(self.atlas, [a, b, c], stage_dir=self.stage)
        self.assertEqual(len(inst), 2)
        self.assertEqual(sorted(len(i.paths) for i in inst), [1, 2])
        names = {r["name"] for r in self.atlas.repos()}
        self.assertEqual(names, {i.repo for i in inst})
        self.assertTrue(any(h.kind == "function" for h in self.atlas.search("coherence")))

    def test_refuses_numbskull_and_non_qincrs(self):
        nb = self._write("numbskull/QiNCrs.py", "x = 1\n")
        with self.assertRaises(ValueError):
            register_qincrs(self.atlas, [nb], stage_dir=self.stage)
        other = self._write("misc/util.py", "x = 1\n")
        with self.assertRaises(ValueError):
            register_qincrs(self.atlas, [other], stage_dir=self.stage)
        self.assertEqual(list(self.atlas.repos()), [])

    def test_does_not_index_sibling_files(self):
        q = self._write("host/QiNCrs.py", "def q():\n    return 1\n")
        self._write("host/neighbor.py", "def neighbor():\n    return 2\n")
        register_qincrs(self.atlas, [q], stage_dir=self.stage)
        self.assertEqual(self.atlas.search("neighbor"), [])


if __name__ == "__main__":
    unittest.main()
