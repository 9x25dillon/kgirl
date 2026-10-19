import math
import unittest
from dataclasses import replace

import tests.life  # noqa: F401  (sys.path)

from kgirl.life import experiments as X
from kgirl.life.analysis import (critical_energy, early_warning, kendall_tau, lag1, recovery_time, rigidity,
                                 steady_state)
from kgirl.life.chaos import ChaosStream
from kgirl.life.continuous import Exogenous, integrate, paper_confinement
from kgirl.life.core import LifeParams
from kgirl.life.replication import (availability, best_replication, best_stagger, free_running_availability,
                                    marginal_gain, offsets, schedule, simulate, swarm_size, uniform_availability)

try:
    import numpy  # noqa: F401
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


def stationary_stats(r, late):
    return X._mean([e * e for e in r.err[late]]), r.var, r.model_norm


class SteadyStateTests(unittest.TestCase):
    def test_closed_form_matches_simulation(self):
        p = LifeParams()
        for E in (0.3, 0.5):
            r = X.run(p, X.World(d=4, energy=E, shock_at=None, steps=1500))
            late = slice(1000, 1500)
            ss = steady_state(p, E, *stationary_stats(r, late))
            for key, tol in (("F", 0.01), ("O", 0.01), ("C", 0.02), ("A", 0.02)):
                measured = X._mean(getattr(r, key)[late])
                self.assertLess(abs(measured - getattr(ss, key)) / getattr(ss, key), tol, (E, key, measured))

    def test_critical_inflow_brackets_the_collapse(self):
        p = LifeParams()
        top = X.run(p, X.World(d=4, energy=0.5, shock_at=None, steps=1200))
        e2, var, m = stationary_stats(top, slice(800, 1200))
        e_crit = critical_energy(p, e2, var, m)
        self.assertTrue(0 < e_crit < 0.5)
        late = slice(1600, 2400)
        above = X.run(p, X.World(d=4, energy=1.15 * e_crit, shock_at=None, steps=2400))
        below = X.run(p, X.World(d=4, energy=0.85 * e_crit, shock_at=None, steps=2400))
        self.assertGreater(X._mean(above.O[late]), p.O_min)
        self.assertLess(X._mean(below.O[late]), p.O_min)

    def test_starvation_is_reported(self):
        ss = steady_state(LifeParams(), 0.0)
        self.assertTrue(ss.starving)
        self.assertEqual(ss.F, 0.0)
        self.assertEqual(ss.O, 0.0)
        self.assertEqual(critical_energy(replace(LifeParams(), a=1e-6)), math.inf)


class EarlyWarningTests(unittest.TestCase):
    def ar1(self, phis, seed=0.31):
        rng, x, out = ChaosStream(seed), 0.0, []
        for phi in phis:
            x = phi * x + rng.normal()
            out.append(x)
        return out

    def test_statistics(self):
        self.assertAlmostEqual(kendall_tau(list(range(20))), 1.0)
        self.assertAlmostEqual(kendall_tau(list(range(20, 0, -1))), -1.0)
        self.assertAlmostEqual(lag1(self.ar1([0.8] * 20000)), 0.8, delta=0.03)

    def test_critical_slowing_down_is_detected_and_not_imagined(self):
        n = 1500
        ramp = self.ar1([0.1 + 0.87 * t / n for t in range(n)])
        flat = self.ar1([0.5] * n, seed=0.71)
        up, calm = early_warning(ramp, window=300), early_warning(flat, window=300)
        self.assertTrue(up.slowing_down)
        self.assertGreater(up.lag1_tau, 0.6)
        self.assertFalse(calm.slowing_down)
        self.assertLess(abs(calm.lag1_tau), 0.5)
        self.assertIsNone(early_warning([1.0] * 10, window=20))

    def test_recovery_time_of_an_exponential(self):
        tau = 12.0
        x = [1.0] * 100 + [1.0 + math.exp(-k / tau) for k in range(300)]
        self.assertEqual(recovery_time(x, 100), math.ceil(tau * math.log(10)))
        self.assertIsNone(recovery_time([0.0] * 50 + [1.0] * 50, 50))

    def test_rigidity_verdicts(self):
        self.assertIsNone(rigidity([0.0] * 5).verdict)
        stuck = [0.0] * 100 + [1.0] * 100
        self.assertEqual(rigidity(stuck, shocks=[100]).verdict, "rigidifying")
        flat = self.ar1([0.3] * 400, seed=0.55)
        self.assertEqual(rigidity(flat, window=150).verdict, "adaptive")


class ContinuousTests(unittest.TestCase):
    def test_appendix_confinement_is_a_constant_bias(self):
        p = LifeParams()
        vals = {round(paper_confinement(p, c, 0.3), 12) for c in (0.2, 0.4, 0.5, 0.7, 0.84)}
        self.assertEqual(vals, {round(-0.3 * (p.C_max - p.C_min), 12)})

    def test_barrier_confines_where_the_appendix_term_does_not(self):
        p = LifeParams()
        ex = Exogenous(var=2.0)                            # η_V·v = 0.1 drives C down hard
        paper = integrate(p, t_end=200, dt=0.05, ex=ex, gamma_C=0.3, confinement="paper")
        fixed = integrate(p, t_end=200, dt=0.05, ex=ex, gamma_C=0.3)
        self.assertLess(paper.final()["C"], p.C_min)
        self.assertTrue(all(p.C_min < row[2] < p.C_max for row in fixed.x))

    def test_ode_settles_on_the_closed_form_and_produces_entropy(self):
        p = LifeParams()
        ex = Exogenous(energy=0.5, e2=0.4, var=0.03, model_norm=1.5, G=0.5, L=0.1)
        tr = integrate(p, t_end=1500, dt=0.05, ex=ex, every=100)
        ss = steady_state(p, 0.5, 0.4, 0.03, 1.5, G=0.5, L=0.1)
        f = tr.final()
        for k in ("F", "O", "C", "A", "R"):
            self.assertAlmostEqual(f[k], getattr(ss, k), delta=2e-3 * max(1.0, getattr(ss, k)), msg=k)
        self.assertGreater(tr.min_entropy_rate, 0.0)
        self.assertGreater(tr.entropy, 0.0)


class ReplicationTests(unittest.TestCase):
    def test_exact_availability_matches_the_closed_form(self):
        for n in range(1, 7):
            for r in (0.1, 0.35, 0.6, 0.9, 0.99):
                self.assertAlmostEqual(availability(offsets("uniform", n), 1.0, r), uniform_availability(n, 1.0, r),
                                       places=12)
        self.assertEqual(availability(offsets("sync", 4), 1.0, 0.3), 0.7)

    def test_half_cycle_offsets_make_the_third_replica_redundant(self):
        self.assertEqual(offsets("paper", 3), [0.0, 0.5, 0.0])
        for r in (0.3, 0.5, 0.6, 0.8):
            self.assertAlmostEqual(availability(offsets("paper", 3), 1.0, r), uniform_availability(2, 1.0, r))
            if r > 0.5:
                self.assertGreater(uniform_availability(3, 1.0, r), uniform_availability(2, 1.0, r))
        plan = best_replication(1.0, 0.4, value=10.0, c_rep=0.5, c_coord=0.1)
        self.assertEqual(plan.n, 2)

    def test_simulation(self):
        locked = simulate("uniform", 3, reset=0.4, cycles=300)
        self.assertGreater(locked.availability, 0.99)
        for sch in ("sync", "uniform", "random"):
            free = simulate(sch, 3, reset=0.4, cycles=600, phase_lock=False, jitter=0.15)
            self.assertAlmostEqual(free.availability, free_running_availability(3, 1.0, 0.4), delta=0.03)
        self.assertAlmostEqual(simulate("single", 1, reset=0.4).availability, 0.6, delta=0.02)

    def test_swarm_size_matches_brute_force(self):
        for p in (0.05, 0.1, 0.2, 0.37, 0.5, 0.63, 0.8, 0.95):
            for v in (1.5, 3.0, 6.0, 10.0, 25.0, 100.0):
                brute = max([1] + [n for n in range(1, 50) if marginal_gain(p, n, v) > 1.0])
                self.assertEqual(swarm_size(p, v, max_n=50), brute, (p, v))
        self.assertEqual(swarm_size(0.5, 10.0), 3)                 # the harness's old default
        self.assertEqual(swarm_size(0.3, 10.0, max_n=8, affordable=2), 2)

    def test_schedule_expectations(self):
        p, n, tau = 0.4, 3, 10.0
        par = schedule(p, n, tau, 0.0)
        self.assertAlmostEqual(par.cost, n * tau)
        self.assertAlmostEqual(par.latency, tau)
        seq = schedule(p, n, tau, tau)
        self.assertAlmostEqual(seq.cost, tau * (1 - (1 - p) ** n) / p)
        self.assertAlmostEqual(seq.success, 1 - 0.6 ** 3)
        rng, cost, lat, trials, delta = ChaosStream(0.19), 0.0, 0.0, 40000, 4.0
        for _ in range(trials):                                    # Monte Carlo of the same rule
            k = next((i for i in range(n) if rng.uniform() < p), None)
            t_win = k * delta + tau if k is not None else None
            if t_win is None:
                cost, lat = cost + n * tau, lat + (n - 1) * delta + tau
            else:
                cost += sum(min(tau, t_win - j * delta) for j in range(n) if j * delta < t_win)
                lat += t_win
        sc = schedule(p, n, tau, delta)
        self.assertAlmostEqual(cost / trials, sc.cost, delta=0.02 * sc.cost)
        self.assertAlmostEqual(lat / trials, sc.latency, delta=0.02 * sc.latency)

    def test_stagger_follows_the_success_rate(self):
        self.assertGreater(best_stagger(0.8, 3, 60.0, 1.0).delta, 0.0)
        self.assertEqual(best_stagger(0.15, 3, 60.0, 5.0).delta, 0.0)


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class EvolveLifeTests(unittest.TestCase):
    def test_map_elites_over_life_parameters(self):
        from kgirl.evolve.problems import life as L
        w = replace(L.WORLD, steps=240, shock_at=120)
        a = L.evolve(2, 6, 0.33, init=8, world=w)
        b = L.evolve(2, 6, 0.33, init=8, world=w)
        self.assertEqual(a.champion().fitness, b.champion().fitness)
        self.assertGreater(a.history[-1].coverage, 0.0)
        for e in a.elites():
            self.assertEqual(replace(LifeParams(), **{k: e.info[k] for k in L.SPACE.names}).violations(), [])
        self.assertTrue(math.isfinite(L.baseline(w).fitness))


if __name__ == "__main__":
    unittest.main()
