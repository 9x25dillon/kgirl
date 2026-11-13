"""The exploratory limits module: machinery only. Its numbers carry no verdict."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from efl_iwme_cascade import checks, limits
from efl_iwme_cascade.experiments import partial_observability as part1
from efl_iwme_cascade.metrics import closure as cl
from efl_iwme_cascade.models import efl

TINY = part1.Config("unit_tiny_limits", "pcg", 6, (40, 80), 2, (30,), 1, False)


class Cases(unittest.TestCase):
    def test_laplace_variates(self):
        z = limits._laplace(checks.CheckStream(801).uniform(400000))
        self.assertAlmostEqual(float(z.mean()), 0.0, delta=0.01)
        self.assertAlmostEqual(float(z.var()), 1.0, delta=0.02)
        self.assertAlmostEqual(float(np.mean(z ** 4)), 6.0, delta=0.3)              # Laplace kurtosis
        self.assertTrue(np.isfinite(limits._laplace(np.array([0.0, 0.5]))).all())

    def test_every_case_draws_eight_scales_and_unknown_cases_are_refused(self):
        for case in limits.CASES:
            x = limits.simulate(case, checks.CheckStream(802))(50)
            self.assertEqual(x.shape, (50, efl.N_SCALES))
            self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(KeyError):
            limits.simulate("LINEAR", checks.CheckStream(802))

    def test_quadratic_link_keeps_second_order_closure(self):
        """Derivation behind the QUAD case: with Gaussian input the quadratic part of a link is
        uncorrelated with every linear functional upstream, so linear correlations still multiply."""
        x = limits.simulate("QUAD", checks.CheckStream(803))(400000)
        self.assertLess(cl.closure_defect(cl.second_moments(x)), 0.05)

    def test_reduced_run_has_the_reported_structure(self):
        res = limits.classifier_cases(sample=320, replicates=2, surrogates=10, population=20000)
        self.assertEqual(list(res), list(limits.CASES))
        for case, c in res.items():
            self.assertEqual(sum(c["counts"].values()), 2)
            self.assertEqual(c["adjacent_only"], case in limits.ADJACENT_ONLY)
            self.assertEqual(set(c["large_sample"]["closure_ratio"]), {"raw", "normal_score"})

    def test_closure_margin_and_report(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(part1, "RESULTS_DIR", Path(tmp)), \
                mock.patch.object(part1, "load_config", return_value=TINY):
            margin = limits.closure_margin("unit_tiny_limits")
        self.assertEqual(list(margin["truths"]), ["G1", "G2", "G3", "G4"])
        self.assertEqual(len(margin["truths"]["G2"]["replicates"]), 2)
        self.assertIn("rule_iii_holds", margin["truths"]["G3"])
        self.assertNotIn("rule_iii_holds", margin["truths"]["G1"])
        cases = limits.classifier_cases(sample=320, replicates=1, surrogates=10, population=20000)
        text = limits.render({"closure_margin": [margin], "classifier_cases": cases, "seconds": 0.0,
                              "design": {"sample": 320, "replicates": 1, "surrogates": 10, "population": 20000}})
        self.assertIn("exploratory, not registered", text)
        self.assertIn("wrong label", text)
        self.assertNotIn(chr(124), text)


if __name__ == "__main__":
    unittest.main()
