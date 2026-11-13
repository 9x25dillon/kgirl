"""End-to-end runs of reduced configurations: structure, determinism, and independence from job count and cache.

A reduced configuration supports no claim (PREREGISTRATION.md section 10). These tests check the machinery."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from efl_iwme_cascade import models
from efl_iwme_cascade.experiments import layered_coupling as part2
from efl_iwme_cascade.experiments import partial_observability as part1

TINY = part1.Config("unit_tiny", "pcg", 6, (40, 80), 2, (30,), 1, False)


class Part1Machinery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = part1.run(TINY)

    def test_structure(self):
        r = self.res
        self.assertEqual(r["part"], 1)
        self.assertEqual(list(r["truths"]), ["G0", "G1", "G2", "G3", "G4"])
        self.assertEqual(list(r["interventions"]), ["I1", "I2", "I3", "I4", "I5"])
        self.assertEqual(sorted(r["verdicts"]), [f"A{i}" for i in range(1, 8)])
        for g in r["truths"]:
            for f in models.FAMILIES:
                self.assertEqual(len(r["regret"][g][f][80]), 2)
                self.assertGreaterEqual(min(r["regret"][g][f][80]), -1e-12)
                self.assertGreaterEqual(r["floor"][g][f], -1e-9)
            self.assertLess(r["floor"][g][part1.OWN_FAMILY[g]], 1e-6)
        self.assertEqual(r["provenance"]["kind"], "pcg")
        self.assertFalse(r["config"]["registered"])

    def test_changing_the_memory_coupling_changes_the_law(self):
        for name in part1.SUBSTRATE_INTERVENTIONS:
            self.assertGreater(self.res["interventions"][name]["before_to_after_bits"], 1e-3)

    def test_report_is_labelled_and_has_no_table_rules(self):
        text = part1.render(self.res)
        self.assertIn("REDUCED CONFIGURATION: supports no claim", text)
        self.assertIn("VERDICTS", text)
        self.assertNotIn(chr(124), text)

    def test_results_are_json_ready(self):
        import json
        back = json.loads(json.dumps(part1.plain(self.res)))
        self.assertEqual(back["verdicts"]["A1"]["met"], bool(self.res["verdicts"]["A1"]["met"]))
        self.assertEqual(sorted(back["regret"]["G2"]["AM"]), ["40", "80"])

    def test_same_numbers_again_with_two_processes_and_through_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(part1, "RESULTS_DIR", Path(tmp)):
            first = part1.run(TINY, jobs=2, use_cache=True)
            self.assertTrue((Path(tmp) / ".cache" / "unit_tiny" / "stage1_G2.pkl").exists())
            second = part1.run(TINY, jobs=1, use_cache=True)                     # served from the cache
        for other in (first, second):
            self.assertEqual(other["regret"], self.res["regret"])
            self.assertEqual(other["floor"], self.res["floor"])
            self.assertEqual(other["closure"], self.res["closure"])
            self.assertEqual({k: v["zero_shot"] for k, v in other["interventions"].items()},
                             {k: v["zero_shot"] for k, v in self.res["interventions"].items()})

    def test_cache_is_invalidated_by_a_change_of_configuration(self):
        other = part1.Config("unit_tiny", "pcg", 6, (40, 80), 2, (31,), 1, False)
        self.assertNotEqual(part1._fingerprint(TINY), part1._fingerprint(other))
        self.assertEqual(part1._fingerprint(TINY), part1._fingerprint(TINY))

    def test_save_writes_json_and_text(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(part1, "RESULTS_DIR", Path(tmp)):
            js, txt = part1.save(self.res)
            self.assertEqual((js.name, txt.name), ("unit_tiny.json", "unit_tiny.txt"))
            self.assertTrue(txt.read_text(encoding="utf-8").startswith("PART 1"))

    def test_streams_differ_between_generators_and_designs_do_not(self):
        chaos = part1.run(part1.Config("unit_tiny_chaos", "chaos", 6, (40,), 1, (30,), 1, False))
        self.assertEqual(chaos["floor"]["G2"]["MC1"], self.res["floor"]["G2"]["MC1"])        # exact law: no stream
        self.assertNotEqual(chaos["regret"]["G2"]["AM"][40][0], self.res["regret"]["G2"]["AM"][40][0])


class Part2Machinery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = part2.load_config("part2_fast")
        cls.res = part2.run(cls.config)

    def test_structure(self):
        r = self.res
        self.assertEqual(list(r["truths"]), ["ADJ", "SKIP", "NL"])
        self.assertEqual(sorted(r["verdicts"]), ["B1", "B2", "B3", "B4"])
        for name, tr in r["truths"].items():
            self.assertEqual(len(tr["classified"]), self.config.replicates)
            self.assertEqual(len(tr["population"]["adjacent_coherence"]), 7)
            self.assertAlmostEqual(tr["population"]["floor"]["full"], 0.0, places=9)
        self.assertEqual(len(r["welch"]["links"]), 7)

    def test_population_values_are_the_design(self):
        pop = {name: tr["population"] for name, tr in self.res["truths"].items()}
        self.assertAlmostEqual(pop["ADJ"]["closure_ratio_raw"], 1.0, places=12)
        self.assertAlmostEqual(pop["ADJ"]["floor"]["chain"], 0.0, places=9)
        self.assertGreater(pop["SKIP"]["floor"]["chain"], 1.0)
        self.assertGreater(pop["SKIP"]["defect_normal_score"], 0.10)
        self.assertGreater(pop["NL"]["defect_raw"], 0.10)
        self.assertLess(pop["NL"]["defect_normal_score"], 1e-9)

    def test_deterministic(self):
        again = part2.run(self.config)
        for name in self.res["truths"]:
            self.assertEqual(again["truths"][name]["regret"], self.res["truths"][name]["regret"])
            self.assertEqual(again["truths"][name]["band"], self.res["truths"][name]["band"])

    def test_report_is_labelled_and_has_no_table_rules(self):
        text = part2.render(self.res)
        self.assertIn("REDUCED CONFIGURATION: supports no claim", text)
        self.assertIn("PER-FREQUENCY AGAINST BROADBAND", text)
        self.assertNotIn(chr(124), text)


class CommandLine(unittest.TestCase):
    def test_part2_command_prints_the_report_and_does_not_save_when_asked(self):
        from efl_iwme_cascade.__main__ import main
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["part2", "--config", "part2_fast", "--no-save"])
        self.assertIn(code, (0, 1))
        self.assertIn("PART 2  LAYERED COUPLING", out.getvalue())
        self.assertFalse((part2.RESULTS_DIR / "part2_fast.json").exists())


if __name__ == "__main__":
    unittest.main()
