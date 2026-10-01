import unittest

import tests.evolve  # noqa: F401  (sys.path)

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ChaosTests(unittest.TestCase):
    def test_uniform_and_normal_statistics(self):
        from kgirl.evolve import ChaosRNG
        r = ChaosRNG(0.3)
        u = r.uniform(100_000)
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.01)
        self.assertAlmostEqual(u.var(), 1 / 12, delta=0.003)
        self.assertLess(abs(np.corrcoef(u[:-1], u[1:])[0, 1]), 0.02)
        hist = np.histogram(u, 10, (0, 1))[0]
        self.assertLess((hist.max() - hist.min()) / hist.mean(), 0.1)
        z = r.normal(100_000)
        self.assertAlmostEqual(z.std(), 1.0, delta=0.02)
        self.assertAlmostEqual(((z - z.mean()) ** 4).mean() / z.var() ** 2, 3.0, delta=0.1)

    def test_deterministic(self):
        from kgirl.evolve import ChaosRNG
        np.testing.assert_array_equal(ChaosRNG(0.7).uniform(500), ChaosRNG(0.7).uniform(500))
        self.assertFalse(np.array_equal(ChaosRNG(0.7).uniform(50), ChaosRNG(0.71).uniform(50)))


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class MapElitesTests(unittest.TestCase):
    def toy(self, seed=0.4123):
        from kgirl.evolve import Descriptor, Evaluation, MapElites, Param, Space
        space = Space([Param("x", -2, 2), Param("y", -2, 2)])

        def evaluate(X):
            g = space.decode_batch(X)
            out = []
            for x, y in zip(g["x"], g["y"]):
                out.append(Evaluation(-(x * x + y * y), (x, y)))   # behavior = position: easy to cover
            return out
        return MapElites(space, evaluate, [Descriptor("x", -2, 2, 8), Descriptor("y", -2, 2, 8)], seed=seed,
                         batch=32, init=32)

    def test_coverage_grows_and_elite_improves(self):
        me = self.toy().run(25)
        self.assertGreater(me.history[-1].coverage, me.history[0].coverage)
        self.assertGreater(me.history[-1].coverage, 0.9)
        self.assertGreater(me.champion().fitness, -0.2)     # optimum 0 at the origin
        self.assertTrue(np.isnan(me.grid()).sum() < 64)

    def test_lineage_and_reproducibility(self):
        a, b = self.toy().run(10), self.toy().run(10)
        self.assertEqual([h.qd_score for h in a.history], [h.qd_score for h in b.history])
        c = a.champion()
        anc = a.ancestry(c.id)
        self.assertEqual(anc[0], c.id)
        self.assertGreaterEqual(a.stepping_stones(), 1)

    def test_pareto(self):
        from kgirl.evolve import front, ranks
        F = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.2], [1, 1]])
        self.assertEqual(list(front(F)), [4])
        self.assertEqual(list(ranks(F)), [1, 1, 1, 2, 0])


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class AssayProblemTests(unittest.TestCase):
    def test_evolution_beats_continuous_drive_baseline(self):
        from kgirl.evolve.problems import assay
        base = assay.baseline()
        me = assay.evolve(generations=4, batch=16, init=32)
        champ = me.champion()
        self.assertGreater(champ.fitness, base.fitness)
        self.assertGreater(champ.info["f_nihil"], 0.0)       # it discovers the nihil window
        for p in assay.SPACE.params:                          # every elite stays inside the claim ranges
            self.assertTrue(all(p.lo - 1e-9 <= e.info[p.name] <= p.hi + 1e-9 for e in me.elites()))


if __name__ == "__main__":
    unittest.main()
