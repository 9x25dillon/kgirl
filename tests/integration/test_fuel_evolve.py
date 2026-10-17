"""kgirl.evolve fuel: MAP-Elites over 39 rod enrichments on the NuRea_sim surrogate."""
import unittest

import numpy as np

import tests.integration  # noqa: F401  (sys.path)


def _have_data() -> bool:
    try:
        from kgirl.neutronics import dataset
        dataset.load("test")
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_data(), "NuRea_sim lattice data not found")
class FuelEvolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kgirl.evolve.problems import fuel as F
        cls.F = F
        cls.model = F.load_model()

    def test_evaluator_matches_the_surrogate_and_penalises_leaving_its_support(self):
        F = self.F
        ev = F.make_evaluator(self.model)
        mid = np.full((1, F.N_RODS), (2.85 - F.E_LO) / (F.E_HI - F.E_LO))       # every rod at the training mean
        hi = np.ones((1, F.N_RODS))                                             # every rod at 5 wt%: far outside
        a, b = ev(mid)[0], ev(hi)[0]
        preds = [s.predict(np.full((1, F.N_RODS), 2.85))[0] for s in self.model.surrogates.values()]
        self.assertAlmostEqual(a.info["k_inf"], float(np.mean([p[0] for p in preds])), places=9)
        self.assertAlmostEqual(a.info["pppf"], float(max(p[1] for p in preds)), places=9)   # pessimistic
        self.assertLess(a.info["trust_sd"], 1.0)
        self.assertGreater(b.info["trust_sd"], 3.0)
        self.assertLess(b.fitness, -b.info["pppf"] - 1.0)                       # the penalty applies

    def test_search_is_deterministic_and_reports_against_simulated_designs(self):
        F = self.F
        me1, _ = F.evolve(generations=30, batch=64, model=self.model)
        me2, _ = F.evolve(generations=30, batch=64, model=self.model)
        self.assertEqual(me1.champion().fitness, me2.champion().fitness)       # chaos, not a PRNG
        self.assertGreater(me1.history[-1].coverage, 0.2)
        cmp = F.compare_to_data(me1, self.model)
        self.assertGreater(cmp["cells_compared"], 10)
        champ = me1.champion().info
        self.assertEqual(len(champ["enrichments"]), F.N_RODS)
        self.assertTrue(all(F.E_LO <= v <= F.E_HI for v in champ["enrichments"]))

    def test_default_is_the_pessimistic_ensemble_and_poly_is_the_original_search(self):
        F = self.F
        self.assertEqual(self.model.mode, "ensemble")
        self.assertEqual(sorted(self.model.surrogates), ["maxtype", "poly"])
        poly = F.load_model(features="poly")
        self.assertEqual(list(poly.surrogates), ["poly"])
        flat = np.full((1, F.N_RODS), (2.85 - F.E_LO) / (F.E_HI - F.E_LO))   # mean is typical, spread is not
        e_ens, e_poly = F.make_evaluator(self.model)(flat)[0], F.make_evaluator(poly)(flat)[0]
        self.assertGreater(e_ens.info["support_sd"], F.SUPPORT_SD)           # far outside the training designs
        self.assertAlmostEqual(e_poly.fitness, -e_poly.info["pppf"], places=9)          # poly: no support penalty
        self.assertLess(e_ens.fitness, -e_ens.info["pppf"] - 1.0)                       # ensemble: penalised
        with self.assertRaises(ValueError):
            F.load_model(features="fft")

    def test_original_poly_search_is_reproduced_exactly(self):
        me, m = self.F.evolve(300, 128, 0.4123, model=self.F.load_model(features="poly"))
        champ = me.champion().info
        self.assertAlmostEqual(champ["pppf"], 1.4350, places=3)              # docs/neutronics, before this change
        self.assertAlmostEqual(me.history[-1].coverage, 0.59, places=2)
        self.assertEqual(self.F.compare_to_data(me, m)["cells_where_elite_flatter"], 17)

    def test_cli_json(self):
        import contextlib
        import io
        import json
        from kgirl.evolve.__main__ import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["fuel", "-g", "5", "-b", "32", "--json"]), 0)
        out = json.loads(buf.getvalue())
        self.assertIn("versus_simulated_per_cell", out)
        self.assertEqual(out["surrogate"], {"degree": 2, "features": "ensemble"})
        self.assertIn("cells_compared_in_support", out["versus_simulated_per_cell"])
        self.assertLess(out["best_simulated_design"]["pppf"], 2.0)


if __name__ == "__main__":
    unittest.main()
