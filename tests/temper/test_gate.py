"""The model-swap gate (kgirl.temper.gate): planning, statistics, leakage, verdicts, CLI.

    python -m unittest discover -s tests/temper -t .
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import tests.temper  # noqa: F401  (sys.path)

from kgirl.harness.util import jaccard, shingles
from kgirl.temper.__main__ import main
from kgirl.temper.gate import (GateConfig, binom_upper_tail, evaluate, leaks, minimum_detectable, required_items)

SHA = "a" * 64


def suite(n):
    return {f"t{i}": f"task {i}: make test_calc_{i} pass where mean_{i} divides by len plus one" for i in range(n)}


def runs(n=300, better=30, worse=6, invalid=1, seeds=(None,), family_hit=None, artifact=SHA):
    """Base succeeds on items 0..n/2; tuned flips `better` failures to success and `worse` successes to failure."""
    out = []
    for i in range(n):
        base_ok = i < n // 2
        tuned_ok = base_ok
        if n // 2 <= i < n // 2 + better:
            tuned_ok = True
        if i < worse:
            tuned_ok = False
        fam = f"f{i % 3}"
        if family_hit and fam == family_hit and base_ok:
            tuned_ok = False
        for k, seed in enumerate(seeds):
            temp = (0.1, 0.5, 0.9)[k % 3]
            out.append({"item": f"t{i}", "family": fam, "arm": "base", "seed": seed, "temperature": temp,
                        "valid": True, "success": base_ok, "artifact": "base-sha"})
            out.append({"item": f"t{i}", "family": fam, "arm": "tuned", "seed": seed, "temperature": temp,
                        "valid": i >= invalid, "success": tuned_ok, "artifact": artifact})
    return out


def shadow(n=100, changed=10):
    return [{"item": f"s{i}", "base": f"OPEN {i}", "tuned": f"OPEN {i}" if i >= changed else f"EDIT {i}"}
            for i in range(n)]


def train_rows():
    return [f"GOAL: unrelated goal number {j} about parsing config files and fixing imports" for j in range(50)]


class Planning(unittest.TestCase):
    def test_sample_sizes_match_the_closed_form(self):
        self.assertTrue(480 <= required_items(0.05, 0.2) <= 500)            # ≈ 490 for a 5-point gain
        self.assertAlmostEqual(minimum_detectable(200, 0.2), 0.079, delta=0.006)
        self.assertAlmostEqual(minimum_detectable(47, 0.2), 0.16, delta=0.01)
        self.assertLessEqual(required_items(minimum_detectable(200, 0.2), 0.2), 200)
        with self.assertRaises(ValueError):
            required_items(0.5, 0.2)

    def test_binomial_tail(self):
        self.assertEqual(binom_upper_tail(0, 10), 1.0)
        self.assertAlmostEqual(binom_upper_tail(10, 10), 2 ** -10)
        self.assertAlmostEqual(binom_upper_tail(3, 5), 0.5)
        self.assertTrue(0 < binom_upper_tail(900, 1000) < 1e-100)          # log space: no overflow


class Verdicts(unittest.TestCase):
    def gate(self, rows, **kw):
        kw.setdefault("artifact", SHA)
        kw.setdefault("suite", suite(300))
        kw.setdefault("train", train_rows())
        kw.setdefault("shadow", shadow())
        return evaluate(rows, GateConfig(), **kw)

    def test_a_real_improvement_is_accepted(self):
        rep = self.gate(runs())
        self.assertEqual(rep["verdict"], "ACCEPT", json.dumps(rep["gates"], indent=1))
        g2 = rep["gates"]["G2"]
        self.assertEqual((g2["method"], g2["tuned_only"], g2["base_only"]), ("exact McNemar", 30, 6))
        self.assertAlmostEqual(g2["delta"], 24 / 300)
        self.assertLess(g2["p"], 1e-4)
        self.assertGreater(rep["gates"]["G1"]["lower_bound"], -0.02)

    def test_seeds_are_repeated_measures_of_an_item(self):
        rep = self.gate(runs(seeds=(0.4123, 0.2718, 0.5772)))
        self.assertEqual(rep["verdict"], "ACCEPT")
        self.assertEqual(rep["gates"]["G0"]["paired_items"], 300)          # 300 items, not 900 trials
        self.assertEqual(rep["gates"]["G2"]["method"], "paired test on per-item mean success")

    def test_each_gate_can_reject(self):
        cases = {
            "G0 leakage": (runs(), {"train": train_rows() + [
                "GOAL: task 7: make test_calc_7 pass where mean_7 divides by len plus one\nTABLE: [1] file calc.py"]}),
            "G0 two artifacts": (runs()[:-2] + [dict(runs()[-1], artifact="b" * 64)], {}),
            "G0 not the served artifact": (runs(), {"artifact": "c" * 64}),
            "G0 too few items": (runs(n=47, better=8, worse=1), {"suite": suite(47)}),
            "G1 validity loss": (runs(invalid=25), {}),
            "G2 no real gain": (runs(better=8, worse=6), {}),
            "G3 a family regresses": (runs(better=60, family_hit="f1"), {}),
            "G4 behaviour moved": (runs(), {"shadow": shadow(changed=40)}),
        }
        for name, (rows, kw) in cases.items():
            with self.subTest(name):
                rep = self.gate(rows, **kw)
                self.assertEqual(rep["verdict"], "REJECT", name)
                self.assertFalse(rep["gates"][name.split()[0]]["pass"], json.dumps(rep["gates"], indent=1))

    def test_unpaired_runs_are_refused(self):
        rows = runs(seeds=(0.4123, 0.2718))
        rows = [r for r in rows if not (r["arm"] == "tuned" and r["item"] == "t5" and r["seed"] == 0.2718)]
        rep = self.gate(rows)
        self.assertFalse(rep["gates"]["G0"]["pass"])
        self.assertIn("different (seed, temperature) runs", rep["gates"]["G0"]["problems"][0])

    def test_missing_measurements_make_it_incomplete_never_accept(self):
        self.assertEqual(self.gate(runs(), shadow=None)["verdict"], "INCOMPLETE")
        self.assertEqual(self.gate(runs(), suite=None)["verdict"], "INCOMPLETE")
        rep = self.gate(runs(), suite=None, train=None, trained=False)       # a swap with no fine-tuning
        self.assertEqual(rep["verdict"], "ACCEPT")
        bad = self.gate(runs(invalid=25), shadow=None)                      # a failure still wins over missing
        self.assertEqual(bad["verdict"], "REJECT")


class Leakage(unittest.TestCase):
    def test_containment_catches_what_jaccard_misses(self):
        item = "make test_parse pass where parse_date drops the timezone offset"
        prompt = ("GOAL: " + item + "\n\nTABLE:\n" + "\n".join(f"[{k}] file src/mod_{k}.py" for k in range(60)))
        self.assertLess(jaccard(shingles(item), shingles(prompt)), 0.55)    # symmetric overlap is diluted
        (hit,) = leaks({"x": item}, [prompt])
        self.assertEqual((hit["item"], hit["containment"]), ("x", 1.0))
        self.assertEqual(leaks({"x": item}, ["GOAL: fix the logger rotation in app/log.py"]), [])
        self.assertEqual(len(leaks({"x": "Fix   THE bug!"}, ["fix the bug"])), 1)    # normalized equality


class Cli(unittest.TestCase):
    def test_gate_exit_codes_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)

            def jsonl(name, rows):
                (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
                return str(d / name)

            ev = jsonl("runs.jsonl", runs())
            su = jsonl("suite.jsonl", [{"item": k, "family": "f", "text": v} for k, v in suite(300).items()])
            tr = jsonl("sft.jsonl", [{"system": "s", "prompt": t, "completion": "OPEN 1"} for t in train_rows()])
            sh = jsonl("shadow.jsonl", shadow())
            args = ["gate", "--eval", ev, "--suite", su, "--train", tr, "--shadow", sh, "--artifact", SHA]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(args + ["--out", str(d / "r" / "gate.json")]), 0)
            self.assertIn("verdict: ACCEPT", out.getvalue())
            self.assertEqual(json.loads((d / "r" / "gate.json").read_text())["verdict"], "ACCEPT")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["gate", "--eval", jsonl("bad.jsonl", runs(invalid=25)), "--suite", su,
                                       "--train", tr, "--shadow", sh]), 1)
                self.assertEqual(main(["gate", "--eval", ev, "--suite", su, "--train", tr]), 2)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["plan", "--items", "200"]), 0)
            self.assertIn("200 paired items detect a gain of 0.0", out.getvalue())


if __name__ == "__main__":
    unittest.main()
