"""The stream adapter: registered seeds, independence, provenance, and the host boundary."""

import unittest

import numpy as np

from efl_iwme_cascade import host, streams


class RegisteredSeeds(unittest.TestCase):
    def test_training_seeds_are_the_registered_formula(self):
        for g in range(5):
            for r in range(12):
                seed, lanes = streams.chaos_seed("train", g, r)
                self.assertAlmostEqual(seed, 0.1 + 0.01 * g + 0.001 * r, places=15)
                self.assertEqual(lanes, 64)

    def test_part2_seeds_are_the_registered_formulas(self):
        self.assertEqual(streams.chaos_seed("layered", 2, 11), (0.3 + 0.01 * 2 + 0.001 * 11, 4096))
        self.assertEqual(streams.chaos_seed("surrogate", 1, 3), (0.45 + 0.01 * 1 + 0.001 * 3, 4096))

    def test_hidden_markov_starts_are_the_four_of_amendment_1(self):
        got = [streams.chaos_seed("hmm_init", 0, r)[0] for r in range(4)]
        np.testing.assert_allclose(got, [0.37, 0.383, 0.396, 0.409], atol=1e-15)

    @staticmethod
    def _registered_design():
        return ([("train", g, r) for g in range(5) for r in range(12)]
                + [("post", i, r) for i in range(5) for r in range(6)]
                + [("hmm_init", 0, r) for r in range(4)]
                + [("layered", t, r) for t in range(3) for r in range(12)]
                + [("surrogate", t, r) for t in range(3) for r in range(12)]
                + [("welch", 0, 0), ("data_surrogate", 0, 0)])

    def test_registered_chaos_streams_are_distinct_within_a_purpose_and_index(self):
        by_group = {}
        for purpose, index, replicate in self._registered_design():
            seed, lanes = streams.chaos_seed(purpose, index, replicate)
            by_group.setdefault((purpose, index), []).append((round(seed, 9), lanes))
        for group, tags in by_group.items():
            self.assertEqual(len(tags), len(set(tags)), group)

    def test_the_registered_formula_reuses_streams_across_indices_exactly_where_documented(self):
        """A defect of the registered seed formula, pinned so that RESULTS.md stays accurate: the step per
        index is 0.01 and the step per replicate 0.001, so with 12 replicates the streams of replicates
        10 and 11 of one index are the streams of replicates 0 and 1 of the next. Comparisons inside one
        ground truth are unaffected; samples of neighbouring ground truths are coupled in those replicates.
        The PCG replication has no such reuse (next test)."""
        seen, reused = {}, []
        for key in self._registered_design():
            seed, lanes = streams.chaos_seed(*key)
            tag = (round(seed, 9), lanes)
            if tag in seen:
                reused.append((seen[tag], key))
            seen[tag] = key
        want = [((p, i, r), (p, i + 1, r - 10))
                for p, top in (("train", 5), ("layered", 3), ("surrogate", 3))
                for i in range(top - 1) for r in (10, 11)]
        self.assertEqual(sorted(reused), sorted(want))
        self.assertEqual(len(reused), 16)

    def test_replication_streams_are_all_distinct(self):
        keys = [(streams.PURPOSES[p], i, r) for p, i, r in self._registered_design()]
        self.assertEqual(len(keys), len(set(keys)))
        first = [float(streams.stream("pcg", p, i, r).uniform(1)[0]) for p, i, r in self._registered_design()]
        self.assertEqual(len(first), len(set(first)))

    def test_chaos_stream_is_the_host_generator_at_that_seed(self):
        direct = np.asarray(host.ChaosRNG(0.1 + 0.01 * 9, lanes=64).uniform(500))
        np.testing.assert_array_equal(streams.stream("chaos", "train", 9, 0).uniform(500), direct)


class Replication(unittest.TestCase):
    def test_pcg_streams_are_spawned_by_purpose_index_and_replicate(self):
        s = streams.stream("pcg", "post", 3, 4)
        self.assertEqual(s.provenance, {"generator": "numpy.random.PCG64", "entropy": 20261112, "spawn_key": [2, 3, 4]})
        want = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261112, spawn_key=(2, 3, 4)))).random(16)
        np.testing.assert_array_equal(s.uniform(16), want)

    def test_generators_share_one_interface(self):
        for kind in streams.KINDS:
            s = streams.stream(kind, "layered", 7, 0)
            self.assertEqual(s.uniform((3, 5)).shape, (3, 5))
            self.assertEqual(s.normal((4, 8)).shape, (4, 8))
            u = s.uniform(1000)
            self.assertTrue(0.0 <= u.min() and u.max() < 1.0)

    def test_unknown_names_are_refused(self):
        with self.assertRaises(KeyError):
            streams.stream("mt19937", "train")
        with self.assertRaises(KeyError):
            streams.stream("chaos", "validation")
        with self.assertRaises(KeyError):
            streams.chaos_seed("validation")

    def test_provenance_names_generator_versions_and_host_state(self):
        for kind in streams.KINDS:
            p = streams.provenance(kind)
            self.assertEqual(p["kind"], kind)
            self.assertEqual(p["numpy"], np.__version__)
            self.assertIn("host_commit", p)
            self.assertIn("host_source_untouched", p)


class HostBoundary(unittest.TestCase):
    def test_adapter_exports(self):
        for name in ("ChaosRNG", "EMACascade", "MarkovCascade", "IWMECore", "two_point_memory"):
            self.assertTrue(hasattr(host, name))
        self.assertTrue((host.REPO_ROOT / "src" / "kgirl").is_dir())
        self.assertEqual(host.PACKAGE_ROOT.name, "efl_iwme_cascade")

    def test_host_defaults_the_registration_relies_on(self):
        ema, hmm = host.EMACascade(), host.MarkovCascade()
        self.assertEqual((ema.rho, ema.T, ema.bias, ema.h0), (0.4, 0.8, 0.3, 0.0))
        self.assertEqual(hmm.p, (0.7, 0.4))
        self.assertEqual(hmm.P, (((0.9, 0.1), (0.3, 0.7)), ((0.2, 0.8), (0.6, 0.4))))
        self.assertEqual(hmm.pi0, (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
