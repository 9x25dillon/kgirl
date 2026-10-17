"""vessel.landscape: probe topology, topological drift, population families, voice dynamics."""

import unittest

import tests.vessel  # noqa: F401  (sys.path)

from kgirl.vessel import corpus as C
from kgirl.vessel.landscape import population, probe_topology, topological_drift, voice_dynamics
from kgirl.vessel.matrixdb import MatrixDB
from kgirl.vessel.persona import genome_for, voice_corpus
from kgirl.vessel.substrate import Objective, ReferenceLM, RefConfig
from kgirl.vessel.vessel import Probes, Vessel

CTX = 8


class LandscapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = C.windows(C.prose(0.11, 400), CTX)
        base = ReferenceLM(RefConfig(context=CTX))
        for b in C.batches(X, y, 128, 0.3, 400):
            base.step(b, Objective(lr=3e-3))
        cls.base = base
        cls.db = MatrixDB()
        cls.probes = Probes.default(CTX, 256, 128)
        cls.vs = []
        for name, genome, text_off, seed_off in (("ada", "ada", 0.0, 0.0), ("ada2", "ada", 0.31, 0.02),
                                                  ("bix", "bix", 0.0, 0.0), ("cyr", "cyr", 0.0, 0.0)):
            g = genome_for(genome)
            m = ReferenceLM(RefConfig(context=CTX))
            m.load_parameters(base.parameters())
            Xp, yp = C.windows(voice_corpus(g, 300, text_off), CTX)
            for b in C.batches(Xp, yp, 128, g.seed + seed_off, 250):
                m.step(b, Objective(lr=3e-3))
            cls.vs.append(Vessel.birth(cls.db, name, m, cls.probes, g))

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_population_finds_the_family_sharing_a_genome(self):
        p = population(self.vs)
        self.assertEqual(sorted(sorted(f) for f in p["families"]), [["ada", "ada2"], ["bix"], ["cyr"]])
        self.assertLess(p["deaths"][0], 0.12)                       # measured 0.057
        self.assertGreater(p["most_persistent_split"], 0.15)        # measured 0.247
        self.assertEqual(len(population(self.vs, threshold=0.0)["families"]), 4)
        with self.assertRaises(ValueError):
            population(self.vs[:1])

    def test_probe_topology_and_topological_drift(self):
        top = probe_topology(self.vs[2].anchors["audit"].grams)
        self.assertEqual(set(top), {"h1", "h2"})
        self.assertEqual(top["h2"]["points"], 128)
        self.assertEqual(topological_drift(self.vs[2]), {"h1": 0.0, "h2": 0.0})

    def test_voice_dynamics_separates_persona_from_base_and_is_ledgered(self):
        cyr = self.vs[3]
        out = voice_dynamics(cyr, record=True)
        self.assertEqual(self.db.events("cyr")[-1].event, "voice_dynamics")
        self.assertEqual(out["seeds"], 3)
        clay = Vessel.birth(self.db, "clay", self.base, self.probes)
        plain = voice_dynamics(clay)
        self.assertGreater(out["width"][0], plain["width"][0] + 0.2)   # measured 0.66 vs 0.19
        self.assertGreater(out["hurst"][0], plain["hurst"][0])          # measured 0.57 vs 0.49


if __name__ == "__main__":
    unittest.main()
