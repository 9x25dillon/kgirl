"""Neutronics surrogate on the real NuRea_sim lattice-physics data.

Pure-math tests always run; the dataset tests are skipped when NuRea_sim (or its LFS data) is absent.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests.integration  # noqa: F401  (sys.path)

from kgirl.neutronics import dataset
from kgirl.neutronics.dataset import LatticeSplit
from kgirl.neutronics.surrogate import Surrogate, design, metrics, poly_terms, run


def _have_data() -> bool:
    try:
        dataset.load("test")
        return True
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return False


class SurrogateMathTests(unittest.TestCase):
    def test_term_count_matches_closed_form(self):
        d = 39
        self.assertEqual(len(poly_terms(d, 1)), 1 + d)
        self.assertEqual(len(poly_terms(d, 2)), 1 + d + d * (d + 1) // 2)

    def test_degree2_recovers_an_exact_quadratic(self):
        # deterministic design: logistic-map chaos, not a PRNG
        n, d, r, x = 400, 4, 3.99, 0.123
        xs = np.empty((n, d))
        for i in range(n):
            for j in range(d):
                x = r * x * (1.0 - x)
                xs[i, j] = 0.7 + 4.3 * x
        k = 0.8 + 0.05 * xs[:, 0] + 0.01 * xs[:, 1] * xs[:, 2]
        p = 1.1 + 0.02 * xs[:, 3] ** 2
        split = LatticeSplit(k, p, xs)
        s = Surrogate(degree=2).fit(split, lambdas=(1e-8,))
        m = metrics(s.predict(xs), split.targets)
        self.assertLess(m["k_inf"]["rmse"], 1e-6)
        self.assertLess(m["pppf"]["rmse"], 1e-6)

    def test_design_constant_column(self):
        phi = design(np.ones((3, 2)), poly_terms(2, 2))
        self.assertTrue(np.allclose(phi[:, 0], 1.0))


@unittest.skipUnless(_have_data(), "NuRea_sim lattice data not found (NUREA_LATTICE_DIR, NUREA_SIM_PATH, git lfs pull)")
class LatticeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = dataset.load("train"), dataset.load("test")

    def test_shapes_and_physics_checks(self):
        self.assertEqual(self.train.enrichment.shape[1], dataset.N_RODS)
        self.assertGreater(self.train.n, self.test.n)
        for s in (self.train, self.test):
            c = dataset.validate(s)
            self.assertGreater(c["corr_kinf_mean_enrichment"], 0.8)

    def test_degree2_beats_degree1_and_baseline_on_held_out(self):
        with tempfile.TemporaryDirectory() as d:
            rec = run(self.train, self.test, (1, 2), Path(d))
            for f in ("telemetry.json", "surrogate.npz", "limps_payload.json"):
                self.assertTrue((Path(d) / f).is_file(), f)
        base = rec["baseline_mean_predictor"]["k_inf"]["rmse"]
        d1, d2 = (rec["models"][f"degree{k}"]["test"] for k in (1, 2))
        self.assertGreater(d2["k_inf"]["r2"], 0.99)
        self.assertLess(d2["k_inf"]["rmse"], d1["k_inf"]["rmse"])
        self.assertLess(d1["k_inf"]["rmse"], base)
        self.assertGreater(d2["pppf"]["r2"], d1["pppf"]["r2"])
        self.assertEqual(rec["best"], "degree2")


if __name__ == "__main__":
    unittest.main()
