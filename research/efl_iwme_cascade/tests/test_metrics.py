"""Metrics: closure, coherence, description length, interventions."""

import math
import unittest

import numpy as np

from efl_iwme_cascade import checks
from efl_iwme_cascade.metrics import closure as cl
from efl_iwme_cascade.metrics import coherence as co
from efl_iwme_cascade.metrics import description_length as dl
from efl_iwme_cascade.metrics.interventions import Intervention, apply, catch_up


class Closure(unittest.TestCase):
    def test_argument_checks(self):
        for a, b in ((3, 2), (0, 2), (1, 5), (2, 2)):
            with self.assertRaises(ValueError):
                cl.closure_ratio(np.eye(4), a, b)

    def test_ratio_between_inner_scales(self):
        cov = cl.ar1_walk_covariance(0.0, 8)
        self.assertAlmostEqual(cl.closure_ratio(cov, 3, 7), 1.0, places=12)
        self.assertEqual(len(cl.local_profile(cov)), 6)
        self.assertEqual(len(cl.adjacent_coherences(cov)), 7)

    def test_weighted_walk_covariance_equals_repeated_rows(self):
        inc = np.array([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
        repeated = np.concatenate([inc[:1], inc[:1], inc[:1], inc[1:]])
        np.testing.assert_allclose(cl.walk_covariance(inc, np.array([3.0, 1.0, 1.0])), cl.walk_covariance(repeated))

    def test_weighted_second_moments(self):
        x = np.array([[1.0, 2.0], [3.0, -1.0]])
        np.testing.assert_allclose(cl.second_moments(x, np.array([1.0, 3.0])), (x[:1].T @ x[:1] + 3.0 * x[1:].T @ x[1:]) / 4.0)

    def test_classification_rule(self):
        band = {"raw": 0.2, "normal_score": 0.2}
        self.assertEqual(cl.classify({"raw": 0.1, "normal_score": 0.1}, band), cl.ADJACENT_LINEAR)
        self.assertEqual(cl.classify({"raw": 0.5, "normal_score": 0.1}, band), cl.NONLINEAR_ADJACENT)
        self.assertEqual(cl.classify({"raw": 0.5, "normal_score": 0.5}, band), cl.NON_ADJACENT)
        self.assertEqual(cl.classify({"raw": 0.1, "normal_score": 0.5}, band), cl.NON_ADJACENT)
        self.assertEqual(cl.classify({"raw": 0.2, "normal_score": 0.2}, band), cl.ADJACENT_LINEAR)   # at the band: no violation

    def test_surrogate_band_is_the_requested_quantile(self):
        rng = checks.CheckStream(401)
        seen = []

        def draw():
            x = rng.normal((400, 4)).cumsum(axis=1)
            seen.append(cl.sample_defects(x))
            return x

        band = cl.surrogate_band(draw, surrogates=40, level=0.9)
        self.assertAlmostEqual(band["raw"], float(np.quantile([d["raw"] for d in seen], 0.9)))
        self.assertAlmostEqual(band["normal_score"], float(np.quantile([d["normal_score"] for d in seen], 0.9)))

    def test_layered_rows_reproduce_the_recursion(self):
        rows = cl.layered_rows([2.0, 0.5], [0.25, 1.0], skip=0.3)
        np.testing.assert_allclose(rows[1], [2.0, 0.5, 0.0])
        np.testing.assert_allclose(rows[2], [0.5 * 2.0 + 0.3, 0.25, 1.0])


class Coherence(unittest.TestCase):
    def test_one_segment_is_refused(self):
        with self.assertRaises(ValueError):
            co.welch_msc(np.zeros(256), np.zeros(256), 256)

    def test_identical_signals_are_fully_coherent(self):
        x = checks.CheckStream(402).normal(16 * 256)
        np.testing.assert_allclose(co.welch_msc(x, -3.0 * x, 256), 1.0, atol=1e-12)
        self.assertAlmostEqual(co.broadband(x, -3.0 * x), 1.0, places=12)

    def test_debias_and_threshold(self):
        self.assertAlmostEqual(co.debias(1.0 / 8.0, 8), 0.0, places=15)
        self.assertAlmostEqual(co.debias(1.0, 8), 1.0, places=15)
        self.assertAlmostEqual(co.null_threshold(8), 0.3482, places=4)
        self.assertLess(co.null_threshold(64), co.null_threshold(8))

    def test_normal_scores(self):
        x = checks.CheckStream(403).normal((501, 3)) ** 3
        s = co.normal_scores(x)
        self.assertEqual(s.shape, x.shape)
        np.testing.assert_allclose(s.mean(axis=0), 0.0, atol=1e-12)                     # symmetric scores
        np.testing.assert_array_equal(np.argsort(s[:, 0]), np.argsort(x[:, 0]))          # order kept
        np.testing.assert_allclose(np.sort(s[:, 1]), np.sort(s[:, 2]))                   # same scores in every column
        self.assertEqual(co.normal_scores(x[:, 0]).shape, (501,))

    def test_cascade_weights(self):
        self.assertAlmostEqual(co.cascade_level_weight(1), 2.0, places=12)
        self.assertAlmostEqual(co.cascade_level_weight(30) / co.cascade_level_weight(29), 0.5, places=9)


class DescriptionLength(unittest.TestCase):
    def test_regret_ignores_strings_of_zero_probability(self):
        from efl_iwme_cascade import models
        strings = models.all_strings(3)
        law = np.zeros(8)
        law[0] = law[7] = 0.5
        fair = models.Binomial(0.5)
        self.assertAlmostEqual(dl.regret_bits(law, fair, strings), (3.0 - 1.0) / 3.0, places=12)

    def test_singular_model_covariance_has_infinite_regret(self):
        self.assertTrue(math.isinf(dl.gaussian_regret_bits(np.eye(2), np.array([[1.0, 0.0], [0.0, -1.0]]))))

    def test_asymptotic_rate(self):
        self.assertAlmostEqual(dl.asymptotic_regret(15, 2560), 15.0 / (2.0 * 2560.0 * math.log(2.0)))


class Interventions(unittest.TestCase):
    def test_only_the_handle_family_is_edited(self):
        iv = Intervention("I", "G", "change", after=None, handle_family="AM", edit=lambda m: m + 1)
        self.assertEqual(apply(iv, "AM", 10), 11)
        self.assertEqual(apply(iv, "MC3", 10), 10)

    def test_catch_up(self):
        refit = {25: 0.9, 50: 0.4, 100: 0.05, 200: 0.01}
        self.assertEqual(catch_up(refit, 0.1, (25, 50, 100, 200)), 100)
        self.assertEqual(catch_up(refit, 0.4, (25, 50, 100, 200)), 50)
        self.assertIsNone(catch_up(refit, 0.001, (25, 50, 100, 200)))


if __name__ == "__main__":
    unittest.main()
