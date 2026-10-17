"""The lifecycle on the reference substrate. Thresholds carry margin around measured values (docs/vessel)."""

import os
import tempfile
import unittest
from pathlib import Path

import tests.vessel  # noqa: F401  (sys.path)

import numpy as np

from kgirl.vessel import corpus as C
from kgirl.vessel.invariance import reparametrize_reference, verify
from kgirl.vessel.matrixdb import MatrixDB
from kgirl.vessel.persona import genome_for, name_seed, validate_soulpack, voice_corpus
from kgirl.vessel.signature import measure
from kgirl.vessel.steer import SteerConfig
from kgirl.vessel.substrate import Objective, ReferenceLM, RefConfig
from kgirl.vessel.vessel import Probes, Vessel, identity_distance, recognize

CTX = 8


def clay(steps=400):
    X, y = C.windows(C.prose(0.11, 400), CTX)
    m = ReferenceLM(RefConfig(context=CTX))
    for b in C.batches(X, y, 128, 0.3, steps):
        m.step(b, Objective(lr=3e-3))
    return m


def individuate(base, name, steps=250):
    g = genome_for(name)
    m = ReferenceLM(RefConfig(context=CTX))
    m.load_parameters(base.parameters())
    X, y = C.windows(voice_corpus(g, 300), CTX)
    for b in C.batches(X, y, 128, g.seed, steps):
        m.step(b, Objective(lr=3e-3))
    return m, g


class PersonaTests(unittest.TestCase):
    def test_genomes_are_deterministic_and_distinct(self):
        self.assertEqual(genome_for("ada"), genome_for("ada"))
        self.assertNotEqual(genome_for("ada").favourites(), genome_for("bix").favourites())
        self.assertTrue(0.05 <= name_seed("anything") <= 0.95)
        g = genome_for("ada")
        self.assertEqual(type(g).from_dict(g.as_dict()), g)
        self.assertIn(g.phrase, voice_corpus(g, 40))

    def test_soulpack_validation(self):
        from kgirl.vessel.persona import soulpack
        doc = soulpack("ada", genome_for("ada"), db_path="m.db", ledger_head="ab" * 32)
        self.assertEqual(validate_soulpack(doc), [])
        self.assertEqual(validate_soulpack({**doc, "version": "1.0"}), ["version must match 0.1.x"])
        self.assertIn("missing signing", validate_soulpack({k: v for k, v in doc.items() if k != "signing"}))


class InvarianceTests(unittest.TestCase):
    def test_declared_group_controls_and_reparametrization(self):
        m = clay(200)
        r = verify(m, C.probe_set(CTX, 96, 0.1414), samples=4)
        self.assertTrue(r.verified, r.render())
        self.assertLess(r.worst_declared, 1e-9)
        self.assertTrue(r.reparametrization["weights_changed"])
        self.assertTrue(r.reparametrization["fingerprint_equal"])
        twin = reparametrize_reference(m)
        self.assertFalse(np.allclose(twin.p["W2"], m.p["W2"]))


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = Probes.default(CTX, 512, 128)
        cls.db = MatrixDB()
        base = clay()
        cls.vessels = {}
        for name in ("ada", "bix", "cyr"):
            m, g = individuate(base, name)
            cls.vessels[name] = Vessel.birth(cls.db, name, m, cls.probes, g)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_1_birth_siblings_and_recognition(self):
        ada = self.vessels["ada"]
        self.assertLess(ada.drift(), 1e-9)
        for a, b in (("ada", "bix"), ("ada", "cyr"), ("bix", "cyr")):
            self.assertGreater(identity_distance(self.vessels[a], self.vessels[b])["identity_gap"], 0.2)
        self.assertTrue(recognize(ada, self.vessels.values())["recognized"])
        self.assertEqual([e.event for e in self.db.events("ada")], ["birth", "anchor"])

    def test_2_drift_steer_consolidate(self):
        ada = self.vessels["ada"]
        before = ada.substrate.parameters()
        X, y = C.windows(C.code(0.22, 400), CTX)
        out = ada.learn(C.batches(X, y, 128, 0.4, 250), anchored=False)
        self.assertGreater(out["drift_after"], 0.9)
        self.assertFalse(recognize(ada, self.vessels.values())["recognized"])
        with self.assertRaises(RuntimeError):
            ada.consolidate(0.2)

        res = ada.steer(SteerConfig())
        self.assertEqual(res.outcome, "coherent")
        self.assertLessEqual(res.drift_end, 0.10)
        rec = recognize(ada, self.vessels.values())
        self.assertTrue(rec["recognized"], rec)
        self.assertIn(b"patient", ada.substrate.generate(b"the ", 48))

        c = ada.consolidate(0.2)
        self.assertLess(c["drift_after"], c["drift_before"])
        events = [e.event for e in self.db.events("ada")]
        self.assertEqual(events, ["birth", "anchor", "learn", "consolidate.refused", "steer", "anchor"])
        self.assertTrue(self.db.verify("ada")["ok"])

        first = self.db.labels("ada")[0]
        ada.rollback(first)
        np.testing.assert_allclose(ada.substrate.parameters()["W1"], before["W1"])

    def test_3_migration_by_live_distillation_and_refusal(self):
        bix = self.vessels["bix"]
        teach, _ = C.windows(C.prose(0.77, 400) + C.code(0.66, 200), CTX)
        old = bix.substrate
        refused = bix.migrate(ReferenceLM(RefConfig(context=CTX, hidden=96, seed=0.9)), steps=60, teach=teach)
        self.assertFalse(refused["adopted"])
        self.assertIs(bix.substrate, old)
        wide = ReferenceLM(RefConfig(context=CTX, hidden=96, seed=0.9))
        out = bix.migrate(wide, steps=900, teach=teach)
        self.assertGreater(out["drift_before"], 0.9)
        self.assertTrue(out["adopted"], out)
        self.assertIs(bix.substrate, wide)
        self.assertEqual(bix.substrate.describe()["hidden"], 96)
        self.assertEqual([e.event for e in self.db.events("bix")][-2:], ["migrate.refused", "migrate"])

    def test_4_open_checks_probes_and_soulpack(self):
        cyr = self.vessels["cyr"]
        again = Vessel.open(self.db, "cyr", cyr.substrate, self.probes)
        self.assertAlmostEqual(again.drift(), cyr.drift(), places=12)
        with self.assertRaises(ValueError):
            Vessel.open(self.db, "cyr", cyr.substrate, Probes.default(CTX, 64, 32))
        doc = cyr.soulpack(expertise=("prose",))
        self.assertEqual(validate_soulpack(doc), [])
        self.assertTrue(doc["signing"]["checksum"].endswith(self.db.head("cyr")))
        r = cyr.observe("check")
        self.assertEqual(self.db.events("cyr")[-1].event, "observe")
        self.assertEqual(set(r.signature.body), {"E", "W1", "W2", "W3"})
        self.assertEqual(measure(cyr.substrate, self.probes.audit).fingerprint(), r.signature.fingerprint())


class InterfaceTests(unittest.TestCase):
    def test_cli_status_verify_and_mcp_tools(self):
        from kgirl.vessel.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("KGIRL_HOME")
            os.environ["KGIRL_HOME"] = tmp
            try:
                path = Path(tmp) / "vessel" / "matrix.db"
                with MatrixDB(path) as db:
                    Vessel.birth(db, "dot", ReferenceLM(RefConfig(context=CTX)), Probes.default(CTX, 32, 16),
                                 genome_for("dot"))
                self.assertEqual(main(["verify", "dot"]), 0)
                self.assertEqual(main(["status"]), 0)
                from kgirl.harness.mcp_server import _DESCRIPTIONS, _tools
                tools = _tools(None)
                self.assertTrue({"vessel_status", "vessel_verify", "vessel_demo"} <= set(tools) <= set(_DESCRIPTIONS))
                self.assertIn('"ok": true', tools["vessel_verify"][1]({"name": "dot"}))
                self.assertIn('"anchor": 1', tools["vessel_status"][1]({"name": "dot"}))
                with self.assertRaises(ValueError):
                    tools["vessel_demo"][1]({"scale": "huge"})
            finally:
                if old is None:
                    os.environ.pop("KGIRL_HOME")
                else:
                    os.environ["KGIRL_HOME"] = old


if __name__ == "__main__":
    unittest.main()
