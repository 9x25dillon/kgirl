"""The code implements the registered designs: constants, configurations, and unedited documents."""

import json
import subprocess
import unittest

import numpy as np

from efl_iwme_cascade import host, models
from efl_iwme_cascade.experiments import layered_coupling as part2
from efl_iwme_cascade.experiments import partial_observability as part1
from efl_iwme_cascade.models import efl

PART1_REGISTERED_AT = ("fbc7239", "research/iwme/experiment_a/PREREGISTRATION.md")
PART2_REGISTERED_AT = ("af758b4", "research/efl_iwme_cascade/PREREGISTRATION_PART2.md")
AMENDMENTS_AT = ("af758b4", "research/efl_iwme_cascade/AMENDMENTS.md")


def _git(*args):
    try:
        done = subprocess.run(["git", "-C", str(host.REPO_ROOT), *args], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


class Part1Design(unittest.TestCase):
    def test_ground_truths(self):
        g = part1.ground_truths()
        self.assertEqual(list(g), ["G0", "G1", "G2", "G3", "G4"])
        self.assertEqual(g["G0"].p, 0.7)
        self.assertEqual(g["G1"].cascade, host.MarkovCascade())
        self.assertEqual((g["G2"].rho, round(g["G2"].T, 12), round(g["G2"].bias, 12)), (0.4, 0.8, 0.3))
        self.assertEqual((g["G3"].rho, round(g["G3"].T, 12), g["G3"].bias), (0.15, 0.4, 0.0))
        self.assertEqual(g["G4"].order, 3)
        np.testing.assert_array_equal(g["G4"].tables[3], [0.80, 0.35, 0.60, 0.25, 0.70, 0.45, 0.30, 0.65])
        np.testing.assert_array_equal(g["G4"].tables[2], [0.70, 0.50, 0.45, 0.35])
        np.testing.assert_array_equal(g["G4"].tables[1], [0.60, 0.40])
        np.testing.assert_array_equal(g["G4"].tables[0], [0.50])
        self.assertTrue(all(m.coupling == 1.0 for m in (g["G2"], g["G3"])))

    def test_own_families_and_parameter_counts(self):
        self.assertEqual(part1.OWN_FAMILY, {"G0": "B", "G1": "HMM2", "G2": "AM", "G3": "AM", "G4": "MC3"})
        self.assertEqual(models.PARAMETERS, {"B": 1, "MC1": 3, "MC2": 7, "MC3": 15, "HMM2": 7, "AM": 3})

    def test_thresholds(self):
        self.assertEqual((part1.RECOVERY_FACTOR, part1.CLOSURE_INFORMATIVE, part1.CLOSURE_RESTORED,
                          part1.CLOSURE_SHRINK, part1.ZERO_SHOT_FACTOR), (5.0, 0.10, 0.10, 5.0, 0.1))

    def test_registered_configuration(self):
        c = part1.load_config("part1_registered")
        self.assertEqual((c.stream, c.depth, c.sizes, c.replicates, c.post_sizes, c.post_replicates, c.registered, c.need),
                         ("chaos", 16, (50, 100, 200, 400, 800, 1600, 3200), 12, (25, 50, 100, 200, 400, 800), 6, True, 11))

    def test_replication_differs_only_in_its_streams(self):
        a = json.loads((part1.CONFIG_DIR / "part1_registered.json").read_text())
        b = json.loads((part1.CONFIG_DIR / "part1_replication_pcg.json").read_text())
        self.assertEqual({k for k in a if a[k] != b[k]}, {"name", "stream"})
        self.assertEqual(b["stream"], "pcg")

    def test_reduced_configuration_is_flagged(self):
        c = part1.load_config("part1_fast")
        self.assertFalse(c.registered)
        self.assertEqual((c.depth, c.replicates, len(c.sizes)), (10, 2, 2))
        with self.assertRaises(ValueError):
            part1.load_config("part2_registered")

    def test_interventions(self):
        ivs = part1.interventions()
        self.assertEqual([(i.name, i.truth, i.handle_family) for i in ivs],
                         [("I1", "G2", "AM"), ("I2", "G3", "AM"), ("I3", "G4", "MC3"), ("I4", "G2", "AM"), ("I5", "G3", "AM")])
        g = part1.ground_truths()
        self.assertEqual((ivs[0].after.rho, round(ivs[0].after.T, 12), round(ivs[0].after.bias, 12)), (0.4, 0.4, 0.3))
        self.assertEqual((ivs[1].after.rho, round(ivs[1].after.T, 12)), (0.6, 0.4))
        self.assertEqual(float(ivs[2].after.tables[3][0]), 0.20)
        np.testing.assert_array_equal(ivs[2].after.tables[3][1:], g["G4"].tables[3][1:])
        for iv, truth in ((ivs[3], "G2"), (ivs[4], "G3")):
            self.assertEqual(iv.after, g[truth].with_coupling(0.0))
        self.assertEqual(part1.SUBSTRATE_INTERVENTIONS, ("I4", "I5"))

    def test_registered_edits_are_the_registered_handles(self):
        ivs = {i.name: i for i in part1.interventions()}
        m = models.AttentionCascade(rho=0.2, gain=2.0, offset=0.5)
        self.assertAlmostEqual(ivs["I1"].edit(m).T, m.T / 2.0)
        self.assertAlmostEqual(ivs["I2"].edit(m).rho, 0.8)
        self.assertEqual(ivs["I2"].edit(models.AttentionCascade(0.3, 2.0, 0.5)).rho, 0.98)       # capped
        self.assertEqual(ivs["I4"].edit(m).coupling, 0.0)
        edited = ivs["I3"].edit(part1.ground_truths()["G4"])
        self.assertEqual(float(edited.tables[3][0]), 0.20)


class Part2Design(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(efl.N_SCALES, 8)
        self.assertEqual(efl.GAINS, (1.0, 2.0, 0.5, 1.0, 3.0, 0.7, 1.5))
        self.assertEqual(efl.PSI, (0.95, 0.90, 0.80, 0.97, 0.60, 0.85, 0.75))
        self.assertEqual(efl.SKIP, 0.6)
        self.assertEqual([(t.name, t.index) for t in efl.truths().values()], [("ADJ", 0), ("SKIP", 1), ("NL", 2)])
        self.assertEqual((part2.RECOVERY_FACTOR, part2.POPULATION_DEFECT), (5.0, 0.10))

    def test_registered_configuration(self):
        c = part2.load_config("part2_registered")
        self.assertEqual((c.stream, c.sizes, c.replicates, c.surrogates, c.registered, c.need, c.need_loose),
                         ("chaos", (20, 40, 80, 160, 320, 640, 1280, 2560), 12, 200, True, 11, 9))
        a = json.loads((part2.CONFIG_DIR / "part2_registered.json").read_text())
        b = json.loads((part2.CONFIG_DIR / "part2_replication_pcg.json").read_text())
        self.assertEqual({k for k in a if a[k] != b[k]}, {"name", "stream"})
        self.assertFalse(part2.load_config("part2_fast").registered)
        with self.assertRaises(ValueError):
            part2.load_config("part1_registered")

    def test_expected_classification(self):
        from efl_iwme_cascade.metrics import closure as cl
        self.assertEqual(part2.EXPECTED, {"ADJ": cl.ADJACENT_LINEAR, "SKIP": cl.NON_ADJACENT, "NL": cl.NONLINEAR_ADJACENT})


class RegisteredDocuments(unittest.TestCase):
    """The pre-registrations are never edited; amendments are appended."""

    def _blob(self, commit, path):
        blob = _git("show", f"{commit}:{path}")
        if blob is None:
            self.skipTest("git history with the registering commits is not available")
        return blob

    def test_part1_preregistration_is_the_registered_text(self):
        self.assertEqual((host.PACKAGE_ROOT / "PREREGISTRATION.md").read_bytes(), self._blob(*PART1_REGISTERED_AT))

    def test_part2_preregistration_is_the_registered_text(self):
        self.assertEqual((host.PACKAGE_ROOT / "PREREGISTRATION_PART2.md").read_bytes(), self._blob(*PART2_REGISTERED_AT))

    def test_amendments_are_append_only(self):
        self.assertTrue((host.PACKAGE_ROOT / "AMENDMENTS.md").read_bytes().startswith(self._blob(*AMENDMENTS_AT)))

    def test_registrations_precede_the_registered_results(self):
        order = _git("log", "--format=%h", "--reverse", "--", "research/efl_iwme_cascade", "research/iwme")
        if order is None:
            self.skipTest("git history is not available")
        commits = order.decode().split()
        results = _git("log", "--format=%h", "--reverse", "--", "research/efl_iwme_cascade/results")
        if not results or not results.decode().split():
            self.skipTest("no committed results yet")
        first_result = results.decode().split()[0]
        for registered in (PART1_REGISTERED_AT[0], PART2_REGISTERED_AT[0]):
            self.assertLess(commits.index(registered), commits.index(first_result))


if __name__ == "__main__":
    unittest.main()
