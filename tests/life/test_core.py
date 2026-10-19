import json
import math
import unittest
from dataclasses import replace

import tests.life  # noqa: F401  (sys.path)

from kgirl.life import experiments as X
from kgirl.life.analysis import steady_state
from kgirl.life.chaos import ChaosStream
from kgirl.life.core import (LifeParams, LifeState, Organism, overhead, potential, reach_count, step,
                             window_quality)

try:
    import numpy  # noqa: F401
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


def drive(org, steps, energy=0.5, noise=0.1, seed=0.77, shock_at=None, shock=0.0, amp=0.5):
    rng, recs = ChaosStream(seed), []
    for t in range(steps):
        off = shock if shock_at is not None and t >= shock_at else 0.0
        obs = [amp * math.sin(2 * math.pi * t / (20 + 7 * i)) + off + noise * rng.normal() for i in range(org.state.d)]
        recs.append(org.step(obs, energy))
    return recs


class ParamsTests(unittest.TestCase):
    def test_defaults_are_admissible(self):
        self.assertEqual(LifeParams().violations(), [])

    def test_inadmissible_parameters_are_named(self):
        bad = LifeParams(eta_M=1.2, C_min=0.9, eta_in=1.5, landauer_bits=1.0, w0=0.1, kT=1.0).violations()
        text = " ".join(bad)
        for needle in ("eta_M", "C_min", "eta_in", "Landauer"):
            self.assertIn(needle, text)
        with self.assertRaises(ValueError):
            LifeParams(rho=0.0).check()

    def test_landauer_floor(self):
        kt_ln2 = math.log(2)
        self.assertEqual(LifeParams(w0=kt_ln2 * 8, landauer_bits=8).violations(), [])
        self.assertTrue(LifeParams(w0=kt_ln2 * 7.9, landauer_bits=8).violations())


class InvariantTests(unittest.TestCase):
    def test_deterministic_and_pure(self):
        a, b = Organism(d=4, seed=0.3), Organism(d=4, seed=0.3)
        s0 = a.state
        ra, rb = drive(a, 200), drive(b, 200)
        self.assertEqual([r.action for r in ra], [r.action for r in rb])
        self.assertEqual(a.state, b.state)
        self.assertEqual(s0.t, 0)                                  # frozen: the old state is untouched

    def test_energy_ledger_closes_and_entropy_is_nonnegative(self):
        p = LifeParams()
        s = LifeState.initial(4, p)
        rng = ChaosStream(0.21)
        for t in range(600):
            energy = 0.6 if t < 300 else 0.0                       # feed, then starve to force deficits
            obs = [3 * rng.normal() for _ in range(4)]
            u = t % len(p.action_costs)
            nxt, rec = step(p, s, obs, energy, u, work=0.05 * (t % 3))
            self.assertAlmostEqual(nxt.F, s.F + p.eta_in * energy - rec.work - rec.dissipation + rec.deficit,
                                   places=12)
            self.assertGreaterEqual(rec.heat, 0.0)
            self.assertGreaterEqual(nxt.F, 0.0)
            self.assertTrue(0.0 <= rec.paid <= 1.0)
            if rec.deficit > 0:
                self.assertLess(rec.paid, 1.0)
            s = nxt
        self.assertEqual(s.F, 0.0)

    def test_state_stays_in_its_domain_under_violent_input(self):
        for integrator in ("exact", "euler"):
            p = LifeParams(integrator=integrator)
            org = Organism(p, d=3, seed=0.6)
            rng = ChaosStream(0.13)
            for t in range(800):
                org.step([50 * rng.normal() if t % 97 == 0 else rng.normal() for _ in range(3)],
                         10.0 if t % 50 < 25 else 0.0, reset=t % 211 == 0)
                s = org.state
                self.assertTrue(0 <= s.O <= 1 and 0 <= s.R <= 1 and 0 <= s.A <= p.A_max, (integrator, t, s.O))
                self.assertTrue(p.C_min < s.C < p.C_max)
                self.assertTrue(all(map(math.isfinite, (s.F, s.P, s.model_norm()))))

    def test_serialization_roundtrip(self):
        org = Organism(d=4)
        drive(org, 50)
        back = LifeState.from_dict(json.loads(json.dumps(org.state.to_dict())))
        self.assertEqual(back, org.state)


class FixTests(unittest.TestCase):
    """The reference implementation's measured defects, and that the rebuild removes them."""

    def test_literal_euler_organization_chatters_where_the_exact_step_converges(self):
        out, flips = {}, {}
        for integrator in ("exact", "euler"):
            org = Organism(LifeParams(integrator=integrator), d=4)
            drive(org, 900, noise=0.0, amp=0.0)                  # constant world: the model learns it exactly
            Os = []
            for _ in range(60):
                drive(org, 1, noise=0.0, amp=0.0)
                Os.append(org.state.O)
            out[integrator] = max(Os) - min(Os)
            d = [b - a for a, b in zip(Os, Os[1:])]
            flips[integrator] = sum(1 for a, b in zip(d, d[1:]) if a * b < 0)
        self.assertLess(out["exact"], 1e-6, out)
        self.assertGreater(out["euler"], 1e-2, out)                  # period-2 chatter against the clip at 1
        self.assertGreater(flips["euler"], 40, flips)
        ss = steady_state(LifeParams(), 0.5)
        self.assertLess(ss.euler_multiplier_O, -1.0)             # the published defaults: g + ℓ > 2

    def test_self_model_does_not_collapse(self):
        org = Organism(d=8)
        drive(org, 400, noise=0.1)
        self.assertGreater(min(org.history[k].ceff for k in range(100, 400)), 0.05)
        self.assertGreater(org.state.R, 0.5)                     # the reference reaches R = 0 at step 49

    def test_discontinuity_costs_the_self_model(self):
        a, b = Organism(d=4), Organism(d=4)
        drive(a, 200)
        drive(b, 200)
        a.step([0.0] * 4, 0.5, reset=True)
        b.step([0.0] * 4, 0.5)
        self.assertTrue(a.history[-1].discontinuity)
        self.assertLess(a.state.R, b.state.R)

    def test_potential_tracks_the_budget(self):
        p = LifeParams()
        org = Organism(p, d=4)
        drive(org, 300)
        self.assertAlmostEqual(org.state.P, math.log1p(len(p.action_costs) ** p.horizon))
        drive(org, 600, energy=0.0)
        self.assertEqual(org.state.P, 0.0)
        s = org.state
        curve = [potential(p, replace(s, F=k * 0.002, O=0.9, C=p.C_star)) for k in range(150)]
        self.assertEqual(curve, sorted(curve))                       # more free energy never reaches less
        self.assertGreaterEqual(len(set(curve)), 6)                  # the reference's P takes one value only

    def test_reach_count_matches_brute_force(self):
        from itertools import product
        p = LifeParams(wu=0.3, horizon=3)
        for budget in (0.0, 0.5, 1.0, 2.0, 3.3):
            brute = sum(1 for seq in product(p.action_costs, repeat=3) if sum(0.1 + 0.3 * c for c in seq) <= budget)
            self.assertEqual(reach_count(p, budget, 0.1), brute)

    def test_unpaid_work_is_not_done(self):
        org = Organism(d=4)
        drive(org, 300)
        drive(org, 400, energy=0.0)                               # F is exhausted long before the end
        p, m0, H0 = org.p, org.state.M, org.state.H
        recs = drive(org, 50, energy=0.0, shock_at=0, shock=2.0)
        self.assertEqual(max(r.paid for r in recs), 0.0)
        keep = (1.0 - 2 * p.eta_M * p.lambda_reg) ** 50              # only the passive ridge decay acts
        for ra, rb in zip(org.state.M, m0):
            for a, b in zip(ra, rb):
                self.assertAlmostEqual(a, keep * b, places=12)
        self.assertEqual(org.state.H, H0)                            # no paid memory writes either
        fed = Organism(d=4)
        drive(fed, 300)
        m1 = fed.state.M
        drive(fed, 50, shock_at=0, shock=2.0)
        moved_fed = math.sqrt(sum((a - b) ** 2 for ra, rb in zip(fed.state.M, m1) for a, b in zip(ra, rb)))
        self.assertGreater(moved_fed, 0.3)

    def test_two_sided_window_reaches_rigidity_only_with_v_ref(self):
        for v_ref, above in ((0.0, False), (0.5, True)):
            org = Organism(LifeParams(v_ref=v_ref), d=4)
            drive(org, 300, noise=0.0)
            self.assertEqual(org.state.C > LifeParams().C_star + 0.05, above, (v_ref, org.state.C))

    def test_policy_learns_the_cheaper_action(self):
        org = Organism(LifeParams(wu=0.5, eta_pi=0.2, tau_pi=0.05), d=4, seed=0.2)
        drive(org, 600)
        pi = org.policy()
        self.assertEqual(max(range(len(pi)), key=pi.__getitem__), 0)
        self.assertGreater(pi[0], 0.6)

    def test_window_quality(self):
        p = LifeParams()
        self.assertEqual(window_quality(p, p.C_star), 1.0)
        self.assertAlmostEqual(window_quality(p, p.C_min), 0.0)
        self.assertGreater(overhead(p, LifeState.initial(2, p)), 0.0)


class AblationTests(unittest.TestCase):
    def test_ablations_take_out_what_they_name(self):
        w = X.World(d=4, steps=500, shock_at=250)
        base = X.run(world=w)
        no_mem = X.run(world=w, ablate=("memory",))
        no_model = X.run(world=w, ablate=("model",))
        no_self = X.run(world=w, ablate=("self",))
        pre = slice(150, 250)
        self.assertGreater(X._mean(no_mem.err[pre]), 1.1 * X._mean(base.err[pre]))
        self.assertIsNone(X.adaptation(no_model, 250, pre=100)["adapt_steps"])
        self.assertIsNotNone(X.adaptation(base, 250, pre=100)["adapt_steps"])
        self.assertEqual(set(no_self.R), {0.3})
        with self.assertRaises(ValueError):
            Organism(d=2, ablate=("soul",))


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class TriageTests(unittest.TestCase):
    def test_reference_defects_reproduce(self):
        from kgirl.life.__main__ import triage
        r = triage()
        ref, new = r["reference"], r["rebuild"]
        self.assertEqual(ref["P_values_fed_and_starving"], [round(math.log(2), 6)])
        self.assertIsNotNone(ref["R_first_zero_step"])
        self.assertLess(ref["euler_multiplier_O"], -1.0)
        self.assertEqual(new["share_fed_steps_with_C_eff_zero"], 0.0)
        self.assertGreater(new["P_fed"], new["P_starving_end"])


if __name__ == "__main__":
    unittest.main()
