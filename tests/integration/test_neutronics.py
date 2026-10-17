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
from kgirl.neutronics.surrogate import (MAXTYPE_NAMES, Surrogate, cross_validate, design, max_type_features,
                                       metrics, poly_terms, run)


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


def _chaos_designs(n, d, x=0.123, r=3.99):
    xs = np.empty((n, d))
    for i in range(n):
        for j in range(d):
            x = r * x * (1.0 - x)
            xs[i, j] = 0.7 + 4.3 * x
    return xs


class MaxTypeFeatureTests(unittest.TestCase):
    def test_values_and_stable_smooth_max(self):
        f = max_type_features(np.array([[1.0, 2.0, 3.0, 4.0]]))[0]
        self.assertEqual(len(f), len(MAXTYPE_NAMES))
        self.assertAlmostEqual(f[0], 4.0)                                    # max
        self.assertAlmostEqual(f[1], 3.0)                                    # mean of the top three
        self.assertAlmostEqual(f[2], np.std([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(f[3], 0.5 * np.log(np.exp(2.0 * np.arange(1, 5)).sum()))
        self.assertAlmostEqual(f[4], 4.0 / 2.5)
        big = max_type_features(np.array([[400.0, 401.0, 399.0]]))          # naive exp(2x) would overflow
        self.assertTrue(np.isfinite(big).all())
        self.assertGreater(big[0, 3], 401.0)

    def test_maxtype_captures_a_peak_target_that_a_quadratic_cannot(self):
        xs = _chaos_designs(600, 8)
        split = LatticeSplit(0.9 + 0.02 * xs.mean(axis=1), 1.0 + 0.3 * xs.max(axis=1), xs)
        rmse = {}
        for f in ("poly", "maxtype"):
            s = Surrogate(degree=2, features=f).fit(split)
            self.assertEqual(s.n_features, len(s.terms) + (5 if f == "maxtype" else 0))
            rmse[f] = metrics(s.predict(xs), split.targets)["pppf"]["rmse"]
        self.assertLess(rmse["maxtype"], rmse["poly"] / 3)

    def test_unknown_feature_set_is_refused(self):
        with self.assertRaises(ValueError):
            Surrogate(features="fft")

    def test_save_keeps_the_feature_set(self):
        xs = _chaos_designs(120, 4)
        s = Surrogate(degree=2, features="maxtype").fit(LatticeSplit(xs[:, 0], xs[:, 1] + 1.0, xs))
        with tempfile.TemporaryDirectory() as d:
            s.save(Path(d) / "s.npz")
            z = np.load(Path(d) / "s.npz")
            self.assertEqual(str(z["features"]), "maxtype")
            self.assertEqual(z["fmu"].shape, (5,))
            self.assertEqual(z["coef"].shape[0], s.n_features)
        self.assertTrue(s.limps_payload(xs[:2], metrics(s.predict(xs), np.stack([xs[:, 0], xs[:, 1] + 1.0], 1)))
                        ["structure"].endswith("+maxtype"))

    def test_cross_validate_is_deterministic_and_paired(self):
        xs = _chaos_designs(500, 6)
        split = LatticeSplit(0.9 + 0.02 * xs.mean(axis=1), 1.0 + 0.3 * xs.max(axis=1), xs)
        a = cross_validate(split, degree=2, seeds=(0.4123, 0.2718), folds=4, bootstrap=100)
        b = cross_validate(split, degree=2, seeds=(0.4123, 0.2718), folds=4, bootstrap=100)
        self.assertEqual(a["delta_pppf_r2"], b["delta_pppf_r2"])            # chaos, not a PRNG
        self.assertEqual(a["delta_pppf_r2"]["of"], 8)
        self.assertGreater(a["delta_pppf_r2"]["positive"], 6)
        lo, hi = a["bootstrap_delta_pppf_r2"]["ci95"]
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0.0)


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
        d1, d2, mx = (rec["models"][k]["test"] for k in ("degree1", "degree2", "degree2+maxtype"))
        self.assertGreater(d2["k_inf"]["r2"], 0.99)
        self.assertLess(d2["k_inf"]["rmse"], d1["k_inf"]["rmse"])
        self.assertLess(d1["k_inf"]["rmse"], base)
        self.assertGreater(d2["pppf"]["r2"], d1["pppf"]["r2"])
        self.assertGreater(mx["pppf"]["r2"], d2["pppf"]["r2"] + 0.02)          # the max-type lift (CV: +0.05)
        self.assertLessEqual(mx["k_inf"]["rmse_pcm"], d2["k_inf"]["rmse_pcm"] + 1.0)
        self.assertEqual(rec["best"], "degree2+maxtype")
        poly_only = run(self.train, self.test, (2,), None, ("poly",))
        self.assertEqual(list(poly_only["models"]), ["degree2"])


if __name__ == "__main__":
    unittest.main()
