"""The measured-data entry points, exercised on stand-in data with a known answer.

The stand-ins come from generators that are not registered ground truths. Passing these tests says the
pipeline reads data correctly. It says nothing about any measured system."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from efl_iwme_cascade import checks, models
from efl_iwme_cascade.experiments import real_data
from efl_iwme_cascade.metrics import closure as cl
from efl_iwme_cascade.models import iwme


def stand_in_points(count: int, depth: int = 10) -> np.ndarray:
    rng = checks.CheckStream(700)
    strings = iwme.sequential_sample(checks.AM, rng.uniform((count, depth)))
    return models.points_from_strings(strings) + rng.uniform(count) * 2.0 ** -depth       # anywhere in the cell


def stand_in_layers(count: int, skip: float) -> np.ndarray:
    rows = cl.layered_rows([1.0, 1.4, 0.8, 1.2, 0.9], [0.4, 0.5, 0.3, 0.6, 0.4], skip=skip)
    return checks.CheckStream(701 + int(skip > 0)).normal((count, rows.shape[0])) @ rows.T


class Points(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = real_data.analyze_points(stand_in_points(20000), depth=10, surrogates=40, kind="pcg")

    def test_split_and_fields(self):
        self.assertEqual((self.rep["train"], self.rep["test"], self.rep["depth"]), (16000, 4000, 10))
        self.assertEqual(list(self.rep["families"]), list(models.FAMILIES))
        for fam, row in self.rep["families"].items():
            self.assertEqual(row["parameters"], models.PARAMETERS[fam])
            self.assertGreater(row["held_out_bits"], 0.0)
            self.assertEqual(row["closure_restored"], row["closure_defect"] <= row["surrogate_95"])

    def test_the_generating_family_has_the_shortest_held_out_code_at_equal_size(self):
        fam = self.rep["families"]
        self.assertLess(fam["AM"]["held_out_bits"], fam["MC1"]["held_out_bits"] - 0.01)
        self.assertLess(fam["AM"]["held_out_bits"], fam["B"]["held_out_bits"] - 0.05)

    def test_closure_separates_no_memory_from_the_right_memory(self):
        fam = self.rep["families"]
        self.assertFalse(fam["B"]["closure_restored"])
        self.assertFalse(fam["MC1"]["closure_restored"])
        self.assertLess(fam["AM"]["closure_defect"], fam["B"]["closure_defect"] / 3.0)

    def test_the_closure_test_has_no_power_on_a_few_hundred_test_strings(self):
        """Measured on this stand-in: with 400 test strings the 95 % band of the defect is wider than the
        defect of a model with no memory. The closure test needs test strings in the thousands."""
        small = real_data.analyze_points(stand_in_points(2000), depth=10, surrogates=40, kind="pcg")
        self.assertGreater(small["families"]["B"]["surrogate_95"], 2.0 * self.rep["families"]["B"]["surrogate_95"])
        self.assertGreater(small["families"]["B"]["surrogate_95"], 0.4)

    def test_report(self):
        text = real_data.render(self.rep)
        self.assertIn("MEASURED POINTS", text)
        self.assertIn("lowest held-out code length", text)
        self.assertNotIn(chr(124), text)

    def test_too_few_points_are_refused(self):
        with self.assertRaises(ValueError):
            real_data.analyze_points(np.array([0.1, 0.2, 0.3]), depth=4)
        with self.assertRaises(ValueError):
            real_data.analyze_strings(np.zeros((4, 5), dtype=np.uint8), np.zeros((4, 6), dtype=np.uint8))


class Layers(unittest.TestCase):
    def test_a_skip_coupling_in_stand_in_data_is_classified(self):
        rows = cl.layered_rows([1.0, 1.4, 0.8, 1.2, 0.9], [0.4, 0.5, 0.3, 0.6, 0.4], skip=0.8)
        self.assertGreater(cl.closure_defect(rows @ rows.T), 0.3)                 # the stand-in has a clear defect
        rep = real_data.analyze_layers(stand_in_layers(4096, 0.8), surrogates=60, kind="pcg")
        self.assertEqual(rep["classified"], cl.NON_ADJACENT)
        self.assertLess(rep["held_out_bits"]["full"], rep["held_out_bits"]["chain"])
        self.assertEqual(rep["segments"], 16)
        self.assertIn("per_frequency_mean", rep["links"][0])

    def test_adjacent_stand_in_data(self):
        rep = real_data.analyze_layers(stand_in_layers(4096, 0.0), surrogates=60, kind="pcg")
        self.assertLess(rep["closure_defect"]["raw"], 3.0 * rep["surrogate_95"]["raw"])
        self.assertLess(rep["held_out_bits"]["chain"], rep["held_out_bits"]["diagonal"])
        self.assertLess(abs(rep["held_out_bits"]["chain"] - rep["held_out_bits"]["full"]), 0.05)
        for row in rep["links"]:
            self.assertAlmostEqual(row["broadband"], row["normal_score"], delta=0.05)
            self.assertAlmostEqual(row["broadband"], row["per_frequency_mean"], delta=0.08)
        text = real_data.render(rep)
        self.assertIn("MEASURED LAYERS", text)
        self.assertNotIn(chr(124), text)

    def test_short_records_skip_the_per_frequency_estimate(self):
        rep = real_data.analyze_layers(stand_in_layers(300, 0.0), surrogates=20, kind="pcg")
        self.assertEqual(rep["segments"], 1)
        self.assertNotIn("per_frequency_mean", rep["links"][0])
        self.assertIn("-", real_data.render(rep))

    def test_shape_is_checked(self):
        with self.assertRaises(ValueError):
            real_data.analyze_layers(np.zeros((100, 2)))


class CommandLine(unittest.TestCase):
    def test_data_and_layers_commands_read_text_files(self):
        from efl_iwme_cascade.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            pts, lay = Path(tmp) / "points.txt", Path(tmp) / "layers.txt"
            np.savetxt(pts, stand_in_points(500, depth=8))
            np.savetxt(lay, stand_in_layers(600, 0.0))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["data", str(pts), "--depth", "8", "--json"]), 0)
            rep = json.loads(out.getvalue())
            self.assertEqual((rep["kind"], rep["train"], rep["test"]), ("points", 400, 100))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["layers", str(lay)]), 0)
            self.assertIn("classified:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
