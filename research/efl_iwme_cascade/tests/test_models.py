"""Model families: conventions, handles, argument checks."""

import math
import unittest

import numpy as np

from efl_iwme_cascade import checks, host, models
from efl_iwme_cascade.models import efl, iwme

AM, HM, MC2, BIN = checks.AM, checks.HM, checks.MC2, checks.BIN


class Conventions(unittest.TestCase):
    def test_signs(self):
        np.testing.assert_array_equal(models.signs(np.array([[0, 1]])), [[1.0, -1.0]])

    def test_markov_contexts_read_oldest_symbol_first(self):
        q = MC2.cond_left(np.array([[0, 1, 1, 0]]))[0]
        np.testing.assert_allclose(q, [0.55, 0.7, 0.4, 0.1])        # start, after L, after LR, after RR

    def test_markov_table_shapes_are_checked(self):
        with self.assertRaises(ValueError):
            models.markov_from_table(2, [[0.5], [0.5, 0.5]])

    def test_memory_trace_starts_at_zero_and_follows_the_update(self):
        np.testing.assert_allclose(models.memory_trace(np.array([[0, 0, 1]]), 0.5)[0], [0.0, 0.5, 0.75])

    def test_family_registry(self):
        self.assertEqual(models.FAMILIES, ("B", "MC1", "MC2", "MC3", "HMM2", "AM"))
        with self.assertRaises(KeyError):
            models.fit_family("XYZ", models.all_strings(3))

    def test_describe_names_the_parameters(self):
        self.assertIn("rho = 0.3000", models.describe(AM))
        self.assertIn("coupling = 0", models.describe(AM.with_coupling(0.0)))
        self.assertIn("p = 0.6300", models.describe(BIN))
        self.assertIn("pi0 = 0.300", models.describe(HM))
        self.assertTrue(models.describe(MC2).startswith("[0.550]"))


class AttentionMemory(unittest.TestCase):
    def test_round_trip_through_the_host_cascade(self):
        back = AM.as_host()
        self.assertAlmostEqual(back.rho, 0.3)
        self.assertAlmostEqual(back.T, 0.6)
        self.assertAlmostEqual(back.bias, -0.2)

    def test_host_cascades_with_a_start_memory_are_refused(self):
        with self.assertRaises(ValueError):
            models.AttentionCascade.from_host(host.EMACascade(h0=0.2))

    def test_only_the_coupled_model_is_a_host_cascade(self):
        with self.assertRaises(ValueError):
            AM.with_coupling(0.0).as_host()
        with self.assertRaises(ValueError):
            models.AttentionCascade(0.3, 0.0, 0.1).as_host()

    def test_actuation_is_linear_in_the_memory_and_in_the_coupling(self):
        h = np.array([-0.5, 0.0, 0.8])
        np.testing.assert_allclose(AM.actuation(h), (1.0 - AM.rho) * AM.gain * h)
        np.testing.assert_allclose(AM.with_coupling(0.25).actuation(h), 0.25 * AM.actuation(h))
        np.testing.assert_array_equal(AM.with_coupling(0.0).actuation(h), np.zeros(3))

    def test_infinite_temperature_is_the_binomial_member(self):
        flat = models.AttentionCascade(0.3, 0.0, 0.2)
        self.assertTrue(math.isinf(flat.T))
        self.assertEqual(flat.bias, 0.0)
        np.testing.assert_allclose(flat.cond_left(models.all_strings(4)), 1.0 / (1.0 + math.exp(-0.2)))

    def test_core_adapter_reports_attention_weights_of_the_two_point_memory(self):
        core = iwme.as_core(AM)
        weights = core.A(np.array([0.2]), np.array([1.0]), np.zeros(1))
        after = (1.0 - AM.rho) * 0.2 + AM.rho * 1.0
        self.assertAlmostEqual(float(weights[0]), 1.0 / (1.0 + math.exp(-(after + AM.bias) / AM.T)), places=14)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)

    def test_fit_recovers_parameters(self):
        strings = models.all_strings(12)
        law = models.string_law(AM, 12, strings)
        fit = models.fit_attention(strings[models.sample_indices(law, 6000, checks.CheckStream(301))])
        self.assertAlmostEqual(fit.rho, 0.3, delta=0.04)
        self.assertAlmostEqual(fit.T, 0.6, delta=0.08)
        self.assertAlmostEqual(fit.bias, -0.2, delta=0.04)
        self.assertEqual(fit.coupling, 1.0)


class HiddenState(unittest.TestCase):
    def test_equivalent_point_is_refused_when_it_leaves_the_unit_interval(self):
        with self.assertRaises(ValueError):
            checks.hidden_markov_equivalent(HM, (0.99, 0.01))

    def test_sampling_matches_the_law(self):
        strings = models.all_strings(4)
        law = models.string_law(HM, 4, strings)
        idx = models.sample_indices(law, 40000, checks.CheckStream(302))
        self.assertLess(float(np.abs(np.bincount(idx, minlength=16) / 40000.0 - law).max()), 0.01)


class LayeredTruths(unittest.TestCase):
    def test_adjacent_design_has_the_registered_coherences_and_closes(self):
        from efl_iwme_cascade.metrics import closure as cl
        cov = efl.truths()["ADJ"].covariance()
        np.testing.assert_allclose(cl.adjacent_coherences(cov), efl.PSI, atol=1e-12)
        self.assertAlmostEqual(cl.closure_ratio(cov), 1.0, places=12)

    def test_nonlinear_design_is_the_adjacent_chain_underneath(self):
        t = efl.truths()
        np.testing.assert_allclose(t["NL"].latent_correlation(), t["ADJ"].latent_correlation(), atol=1e-15)
        z = checks.CheckStream(303).normal((50, efl.N_SCALES))
        adj, nl = t["ADJ"].sample(z), t["NL"].sample(z)
        sd = np.sqrt(np.diag(t["ADJ"].covariance()))
        np.testing.assert_allclose(nl[:, 0::2], (adj / sd)[:, 0::2])
        np.testing.assert_allclose(nl[:, 1::2], ((adj / sd) ** 3)[:, 1::2])

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            efl.fit("banded", np.eye(8))

    def test_draw_has_the_requested_covariance(self):
        cov = efl.truths()["SKIP"].covariance()
        x = efl.draw(cov, np.eye(efl.N_SCALES) * math.sqrt(efl.N_SCALES))
        np.testing.assert_allclose(x.T @ x / efl.N_SCALES, cov, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
