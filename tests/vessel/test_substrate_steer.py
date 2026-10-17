import unittest

import tests.vessel  # noqa: F401  (sys.path)

import numpy as np

from kgirl.vessel import corpus as C
from kgirl.vessel import geometry as G
from kgirl.vessel.steer import SteerConfig, steer
from kgirl.vessel.substrate import Objective, ReferenceLM, RefConfig, StepReport


class ReferenceLMTests(unittest.TestCase):
    def test_full_objective_gradient_matches_finite_differences(self):
        m = ReferenceLM(RefConfig(context=4, embed=3, hidden=5))
        P = C.probe_set(4, 16)
        X, y = C.windows(C.prose(0.3, 3), 4)
        anchors = {layer: G.normalized_gram(np.tanh((np.arange(80).reshape(16, 5) * 0.37) % 1.3)) for layer in m.layers}
        obj = Objective(lr=0, lam_rep=0.7, anchors=anchors, probes=P, lam_voice=0.5,
                        voice=np.log(np.full((16, 256), 1 / 256)), lam_w=0.3,
                        anchor_params={k: v + 0.01 for k, v in m.parameters().items()})
        batch = (X[:12], y[:12])

        def total():
            loss, rep, kl, pen, _, _ = m.objective_and_grad(batch, obj)
            return loss + rep + kl + pen

        grads = m.objective_and_grad(batch, obj)[-1]
        worst = 0.0
        for k in m.p:
            for idx in list(np.ndindex(m.p[k].shape))[:20]:
                o = m.p[k][idx]
                m.p[k][idx] = o + 1e-6
                fp = total()
                m.p[k][idx] = o - 1e-6
                fm = total()
                m.p[k][idx] = o
                num = (fp - fm) / 2e-6
                worst = max(worst, abs(num - grads[k][idx]) / max(1e-7, abs(num) + abs(grads[k][idx])))
        self.assertLess(worst, 1e-5)

    def test_deterministic_init_learning_and_exact_restore(self):
        a, b = ReferenceLM(), ReferenceLM()
        np.testing.assert_array_equal(a.p["W1"], b.p["W1"])
        self.assertFalse(np.array_equal(a.p["W1"], ReferenceLM(RefConfig(seed=0.9)).p["W1"]))
        X, y = C.windows(C.prose(0.11, 80), 8)
        before = a.loss((X, y))
        state = a.checkpoint()
        for batch in C.batches(X, y, 64, 0.3, 60):
            a.step(batch, Objective(lr=3e-3))
        self.assertLess(a.loss((X, y)), before - 1.0)
        a.restore(state)
        self.assertAlmostEqual(a.loss((X, y)), before, places=12)

    def test_steering_shift_changes_activations_and_shape_checks(self):
        m = ReferenceLM()
        P = C.probe_set(8, 8)
        base = m.activations(P)["h2"]
        shifted = m.activations(P, shift={"h2": np.ones(64)})["h2"]
        np.testing.assert_allclose(shifted - base, 1.0)
        with self.assertRaises(ValueError):
            m.load_parameters({**m.parameters(), "W1": np.zeros((2, 2))})


class Toy:
    """θ ∈ ℝ, drift = |θ|/10 (capped at 1). A step moves θ toward 0 by lr·100, overshooting when lr is large."""

    layers = ()

    def __init__(self, theta=10.0, bad=False):
        self.theta, self.bad = theta, bad

    def checkpoint(self):
        return self.theta

    def restore(self, s):
        self.theta = s

    def step(self, batch, obj):
        self.theta += obj.lr * 100 if self.bad else -np.sign(self.theta) * obj.lr * 100
        return StepReport(0, 0, 0, 0, {}, 0)

    def drift(self):
        return min(abs(self.theta) / 10, 1.0)


class SteerControllerTests(unittest.TestCase):
    def objective(self, lr):
        return Objective(lr=lr)

    def test_within_band_does_nothing(self):
        t = Toy(theta=1.0)
        r = steer(t, audit=t.drift, objective=self.objective, cfg=SteerConfig())
        self.assertEqual((r.outcome, r.steps, t.theta), ("within_band", 0, 1.0))

    def test_reaches_coherence_and_shrinks_steps_after_overshoot(self):
        t = Toy(theta=10.0)
        r = steer(t, audit=t.drift, objective=self.objective, cfg=SteerConfig(lr=3e-3, block=1, max_steps=500))
        self.assertEqual(r.outcome, "coherent")
        self.assertLessEqual(r.drift_end, 0.10)
        self.assertEqual(r.accepted + r.rejected, len(r.trace))

    def test_harmful_steps_are_rolled_back_until_stalled(self):
        t = Toy(theta=5.0, bad=True)
        r = steer(t, audit=t.drift, objective=self.objective, cfg=SteerConfig(block=2))
        self.assertEqual(r.outcome, "stalled")
        self.assertEqual(r.accepted, 0)
        self.assertEqual(t.theta, 5.0)                           # every block restored

    def test_guard_vetoes_steps_that_cost_too_much(self):
        t = Toy(theta=10.0)
        r = steer(t, audit=t.drift, objective=self.objective, guard=lambda: 10.0 - abs(t.theta),
                  cfg=SteerConfig(block=1, guard_tolerance=0.1))
        self.assertEqual(r.accepted, 0)
        self.assertTrue(all(b.reason == "guard" for b in r.trace))
        self.assertEqual(t.theta, 10.0)

    def test_force_and_config_validation(self):
        t = Toy(theta=1.0)
        r = steer(t, audit=t.drift, objective=self.objective, cfg=SteerConfig(block=1, max_steps=3), force=True)
        self.assertEqual((r.outcome, r.steps), ("coherent", 1))   # force engages inside the band, then releases
        with self.assertRaises(ValueError):
            SteerConfig(engage=0.05, release=0.1)
        with self.assertRaises(ValueError):
            SteerConfig(lr=1.0, lr_max=0.1)


if __name__ == "__main__":
    unittest.main()
