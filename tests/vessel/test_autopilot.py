"""Autopilot and the RSI slider: permissions per level, breaker, fidelity-gated self-tuning, watchdog."""

import unittest

import tests.vessel  # noqa: F401  (sys.path)

import numpy as np

from kgirl.vessel import corpus as C
from kgirl.vessel.autopilot import RSI, Autopilot, AutopilotConfig
from kgirl.vessel.matrixdb import MatrixDB
from kgirl.vessel.persona import genome_for, voice_corpus
from kgirl.vessel.steer import SteerConfig
from kgirl.vessel.substrate import Objective, ReferenceLM, RefConfig
from kgirl.vessel.vessel import Probes, Vessel

CTX = 8


class AutopilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = C.windows(C.prose(0.11, 400), CTX)
        clay = ReferenceLM(RefConfig(context=CTX))
        for b in C.batches(X, y, 128, 0.3, 400):
            clay.step(b, Objective(lr=3e-3))
        g = genome_for("ada")
        Xp, yp = C.windows(voice_corpus(g, 300), CTX)
        for b in C.batches(Xp, yp, 128, g.seed, 250):
            clay.step(b, Objective(lr=3e-3))
        cls.born = clay.parameters()
        cls.probes = Probes.default(CTX, 512, 128)          # the calibrated size; 256 lets repairs fall short
        cls.code = C.windows(C.code(0.22, 400), CTX)

    def vessel(self):
        m = ReferenceLM(RefConfig(context=CTX))
        m.load_parameters(self.born)
        db = MatrixDB()
        self.addCleanup(db.close)
        return Vessel.birth(db, "ada", m, self.probes, genome_for("ada"))

    def world(self, v, push=60, seed=0.4):
        stream = C.batches(*self.code, 128, seed, 100_000)

        def disturb(_k):
            for _ in range(push):
                v.substrate.step(next(stream), Objective(lr=3e-3))
        return disturb

    def stream(self):
        return C.batches(*self.code, 128, 0.7, 100_000)

    def test_rsi0_observes_and_never_edits(self):
        v = self.vessel()
        before = v.substrate.parameters()
        ap = Autopilot(v, AutopilotConfig(level=RSI.OBSERVE), stream=self.stream())
        ap.run(2)                                                   # no world: nothing should move
        for k, w in v.substrate.parameters().items():
            np.testing.assert_array_equal(w, before[k])
        ap.run(1, self.world(v))
        s = ap.summary()
        self.assertEqual((s["steered"], s["learned_steps"], s["consolidations"]), (0, 0, 0))
        self.assertGreater(s["drift_out"][-1], 0.9)
        events = {e.event for e in v.db.events("ada")}
        self.assertFalse(events & {"steer", "learn", "anchor.consolidate", "autopilot.tune"})
        self.assertIn("autopilot.cycle", events)

    def test_rsi1_steers_rsi2_learns_rsi3_consolidates(self):
        out = {}
        for lv in (RSI.STEER, RSI.LEARN, RSI.CONSOLIDATE):
            v = self.vessel()
            ap = Autopilot(v, AutopilotConfig(level=lv, learn_steps=20, stable_cycles=1), stream=self.stream())
            ap.run(3, self.world(v))
            out[lv] = ap.summary()
            self.assertTrue(all(d <= 0.10 + 1e-9 for d in out[lv]["drift_out"]), out[lv])
            self.assertTrue(v.db.verify("ada")["ok"])
        self.assertEqual(out[RSI.STEER]["learned_steps"], 0)
        self.assertEqual(out[RSI.LEARN]["learned_steps"], 60)
        self.assertEqual(out[RSI.LEARN]["consolidations"], 0)
        self.assertGreaterEqual(out[RSI.CONSOLIDATE]["consolidations"], 1)

    def test_circuit_breaker_restores_the_anchor_and_demotes(self):
        v = self.vessel()
        anchor = v.substrate.parameters()
        ap = Autopilot(v, AutopilotConfig(level=RSI.LEARN, learn_steps=0),
                       steer_cfg=SteerConfig(max_steps=1, fresh_optimizer=True), stream=self.stream())
        rec = ap.cycle(self.world(v, push=200))
        self.assertTrue(rec.breaker)
        self.assertEqual(ap.level, RSI.STEER)
        np.testing.assert_allclose(v.substrate.parameters()["W1"], anchor["W1"])
        self.assertLess(rec.drift_out, 1e-6)
        names = [e.event for e in v.db.events("ada")]
        self.assertIn("autopilot.breaker", names)
        self.assertEqual(v.db.events("ada", "autopilot.level")[-1].payload["to"], 1)

    def test_self_tune_is_gated_on_speed_and_fidelity(self):
        v = self.vessel()
        ap = Autopilot(v, AutopilotConfig(level=RSI.SELF_TUNE, tune_candidates=2, tune_budget=150))
        weights = v.substrate.parameters()
        res = ap.self_tune()
        for k, w in v.substrate.parameters().items():
            np.testing.assert_array_equal(w, weights[k])            # sandboxes leave no trace
        self.assertEqual(res["trials"], ["noise", "noise"])          # no drift experience yet → fallback
        p = v.db.events("ada", "autopilot.tune")[-1].payload
        self.assertEqual(len(p["candidates"]), 2)
        self.assertTrue(all({"faster", "faithful", "kl", "steps"} <= set(c) for c in p["candidates"]))
        adopted = [c for c in p["candidates"] if c["faster"] and c["faithful"]]
        self.assertEqual(p["adopted"], bool(adopted))
        if p["adopted"]:
            self.assertTrue(all(a <= b * 0.85 for a, b in zip(p["new_steps"], p["incumbent_steps"])))
            self.assertLessEqual(min(c["kl"] for c in adopted), p["incumbent_kl"] * 1.05)

    def test_watchdog_reverts_a_controller_that_fails_live(self):
        v = self.vessel()
        good = SteerConfig(fresh_optimizer=True)
        ap = Autopilot(v, AutopilotConfig(level=RSI.SELF_TUNE, tune_every=99, learn_steps=0),
                       steer_cfg=SteerConfig(max_steps=1, fresh_optimizer=True))
        ap._previous = good                                         # as if a tune had just replaced `good`
        ap.cycle(self.world(v))
        self.assertIs(ap.steer_cfg, good)
        self.assertEqual(len(v.db.events("ada", "autopilot.tune.revert")), 1)

    def test_config_and_level_control(self):
        with self.assertRaises(ValueError):
            AutopilotConfig(level=5)
        with self.assertRaises(ValueError):
            AutopilotConfig(beta=0)
        v = self.vessel()
        ap = Autopilot(v, AutopilotConfig(level=0))
        ap.set_level(3, "operator raised the slider")
        self.assertEqual(ap.level, RSI.CONSOLIDATE)
        self.assertEqual(v.db.events("ada", "autopilot.level")[-1].payload, {"from": 0, "to": 3,
                                                                          "reason": "operator raised the slider"})

    def test_cli_and_mcp_slider(self):
        from kgirl.harness.mcp_server import _DESCRIPTIONS, _tools
        self.assertIn("vessel_autopilot", _DESCRIPTIONS)
        fn = _tools(None)["vessel_autopilot"][1]
        for bad in ({"rsi": 5}, {"rsi": 1, "cycles": 0}, {"cycles": 9}):
            with self.assertRaises(ValueError):
                fn(bad)
        text = fn({"rsi": 0, "cycles": 1})
        self.assertIn("RSI 0 (observe and record; never changes weights)", text)
        self.assertIn("ledger ok", text)


if __name__ == "__main__":
    unittest.main()
