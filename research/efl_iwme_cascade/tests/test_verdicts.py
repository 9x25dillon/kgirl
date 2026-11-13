"""The decision rules, exercised on synthetic results: every rule can be met and every rule can fail."""

import copy
import unittest

from efl_iwme_cascade import models
from efl_iwme_cascade.experiments import layered_coupling as part2
from efl_iwme_cascade.experiments import partial_observability as part1
from efl_iwme_cascade.metrics import closure as cl

SIZES = (50, 100, 200, 400, 800, 1600, 3200)
CONFIG1 = part1.Config("synthetic", "pcg", 16, SIZES, 12, (25,), 1, False)
CONFIG2 = part2.Config("synthetic", "pcg", (20, 80, 2560), 12, 10, 4, False)
GOOD, BAD = 1e-5, 1e-2


def perfect_part1():
    regret = {g: {f: {n: [GOOD if f == own else BAD] * 12 for n in SIZES} for f in models.FAMILIES}
              for g, own in part1.OWN_FAMILY.items()}
    closure = {"G0": {"raw_defect": 0.0, "fitted_defect": {f: 0.0 for f in models.FAMILIES}}}
    for g in part1.MEMORY_TRUTHS:
        fitted = {f: 0.5 for f in models.FAMILIES}
        fitted[part1.OWN_FAMILY[g]] = 0.01
        closure[g] = {"raw_defect": 0.5, "fitted_defect": fitted}
    ivs = {}
    for iv in part1.interventions():
        ivs[iv.name] = {"truth": iv.truth, "handle_family": iv.handle_family,
                        "zero_shot": {f: [GOOD if f == iv.handle_family else BAD] * 12 for f in models.FAMILIES}}
    return {"regret": regret, "closure": closure, "interventions": ivs}


def verdicts1(res):
    return {k: v["met"] for k, v in part1._verdicts(res, CONFIG1).items()}


class Part1Rules(unittest.TestCase):
    def setUp(self):
        self.res = perfect_part1()

    def test_all_met_on_a_perfect_result(self):
        self.assertEqual(verdicts1(self.res), {f"A{i}": True for i in range(1, 8)})

    def test_a1_fails_when_the_own_family_misses_its_bound(self):
        self.res["regret"]["G0"]["B"][3200] = [1e-3] * 12            # bound for one parameter is 7.0e-5
        got = verdicts1(self.res)
        self.assertFalse(got["A1"])
        self.assertTrue(got["A2"])                                    # still the lowest, still wins every pair

    def test_a2_needs_eleven_paired_wins(self):
        self.res["regret"]["G1"]["MC3"][3200] = [GOOD / 2] + [BAD] * 11
        self.assertTrue(verdicts1(self.res)["A2"])                    # 11 of 12
        self.res["regret"]["G1"]["MC3"][3200] = [GOOD / 2] * 2 + [BAD] * 10
        self.assertFalse(verdicts1(self.res)["A2"])                   # 10 of 12

    def test_a2_needs_the_lowest_mean(self):
        self.res["regret"]["G4"]["MC2"][3200] = [0.0] + [GOOD * 1.05] * 11      # loses 11 pairs, lower mean
        self.assertFalse(verdicts1(self.res)["A2"])

    def test_a3_matched_size_on_attention_truths(self):
        self.res["regret"]["G2"]["MC1"][100] = [GOOD / 2] * 2 + [BAD] * 10
        got = verdicts1(self.res)
        self.assertFalse(got["A3"])
        self.assertTrue(got["A2"])                                    # A2 looks at N = 3200 only

    def test_a3_does_not_look_below_n_100(self):
        self.res["regret"]["G2"]["MC1"][50] = [GOOD / 2] * 12
        self.assertTrue(verdicts1(self.res)["A3"])

    def test_a3_requires_the_attention_class_to_lose_where_it_should(self):
        self.res["regret"]["G4"]["AM"][3200] = [GOOD / 2] * 2 + [BAD] * 10
        self.assertFalse(verdicts1(self.res)["A3"])

    def test_a4_is_about_means_at_every_size(self):
        self.res["regret"]["G3"]["HMM2"][50] = [GOOD / 2] * 12
        got = verdicts1(self.res)
        self.assertFalse(got["A4"])
        self.assertTrue(got["A3"])

    def test_a5_binomial_walk_must_close(self):
        self.res["closure"]["G0"]["raw_defect"] = 1e-6
        self.assertFalse(verdicts1(self.res)["A5"])

    def test_a5_own_family_must_restore_closure(self):
        self.res["closure"]["G1"]["fitted_defect"]["HMM2"] = 0.2      # above 0.10
        self.assertFalse(verdicts1(self.res)["A5"])
        self.res = perfect_part1()
        self.res["closure"]["G4"]["raw_defect"] = 0.3
        self.res["closure"]["G4"]["fitted_defect"]["MC3"] = 0.08      # below 0.10 yet above a fifth of 0.3
        self.assertFalse(verdicts1(self.res)["A5"])

    def test_a5_small_raw_defect_is_uninformative_and_does_not_fail(self):
        self.res["closure"]["G1"]["raw_defect"] = 0.05
        self.res["closure"]["G1"]["fitted_defect"]["HMM2"] = 0.2
        out = part1._verdicts(self.res, CONFIG1)["A5"]
        self.assertTrue(out["met"])
        self.assertNotIn("G1", out["informative"])
        self.assertTrue(any("uninformative" in line for line in out["detail"]))

    def test_a5_smaller_memories_must_fail_by_a_factor_of_five(self):
        self.res["closure"]["G2"]["fitted_defect"]["MC1"] = 0.04      # four times the attention-memory value
        self.assertFalse(verdicts1(self.res)["A5"])

    def test_a6_counts_replicates_and_judges_i3_against_the_attention_family_only(self):
        self.res["interventions"]["I1"]["zero_shot"]["AM"][0] = BAD
        self.assertTrue(verdicts1(self.res)["A6"])                    # 11 of 12
        self.res["interventions"]["I1"]["zero_shot"]["AM"][1] = BAD
        got = verdicts1(self.res)
        self.assertFalse(got["A6"])
        self.assertTrue(got["A7"])
        self.res = perfect_part1()
        self.res["interventions"]["I3"]["zero_shot"]["B"] = [0.0] * 12
        self.assertTrue(verdicts1(self.res)["A6"])
        self.res["interventions"]["I3"]["zero_shot"]["AM"] = [GOOD * 5] * 12     # edited MC3 is not ten times better
        self.assertFalse(verdicts1(self.res)["A6"])

    def test_a7_is_the_substrate_interventions_only(self):
        self.res["interventions"]["I5"]["zero_shot"]["MC1"] = [GOOD * 5] * 12
        got = verdicts1(self.res)
        self.assertFalse(got["A7"])
        self.assertTrue(got["A6"])

    def test_a_tie_is_not_a_win(self):
        self.assertEqual(part1._wins([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]), 1)


def perfect_part2():
    def truth(chain, full, diagonal, raw, score, label):
        return {"regret": {"chain": {n: [chain] * 12 for n in CONFIG2.sizes}, "full": {n: [full] * 12 for n in CONFIG2.sizes},
                           "diagonal": {n: [diagonal] * 12 for n in CONFIG2.sizes}},
                "population": {"defect_raw": raw, "defect_normal_score": score}, "classified": [label] * 12}
    return {"truths": {"ADJ": truth(1e-3, 3e-3, 5.0, 0.0, 0.0, cl.ADJACENT_LINEAR),
                       "SKIP": truth(1.5, 1e-2, 9.0, 0.2, 0.2, cl.NON_ADJACENT),
                       "NL": truth(2.0, 7e-2, 5.0, 4.0, 0.0, cl.NONLINEAR_ADJACENT)}}


def verdicts2(res):
    return {k: v["met"] for k, v in part2._verdicts(res, CONFIG2).items()}


class Part2Rules(unittest.TestCase):
    def setUp(self):
        self.res = perfect_part2()

    def test_all_met_on_a_perfect_result(self):
        self.assertEqual(verdicts2(self.res), {"B1": True, "B2": True, "B3": True, "B4": True})

    def test_b1_chain_must_win_at_every_size_and_reach_its_bound(self):
        res = copy.deepcopy(self.res)
        res["truths"]["ADJ"]["regret"]["chain"][20] = [4e-3] * 12
        self.assertFalse(verdicts2(res)["B1"])
        res = copy.deepcopy(self.res)
        res["truths"]["ADJ"]["regret"]["full"][80] = [0.5e-3] * 2 + [3e-3] * 10
        self.assertFalse(verdicts2(res)["B1"])                        # 10 of 12 paired wins at N = 80
        res = copy.deepcopy(self.res)
        res["truths"]["ADJ"]["regret"]["chain"][2560] = [2.5e-2] * 12  # bound 5 x 15 / (2 x 2560 ln 2) = 2.1e-2
        res["truths"]["ADJ"]["regret"]["full"][2560] = [9e-2] * 12
        self.assertFalse(verdicts2(res)["B1"])

    def test_b2_needs_the_defect_the_classification_and_the_regret_order(self):
        for edit in (lambda t: t["population"].update(defect_normal_score=0.05),
                     lambda t: t.update(classified=[cl.NON_ADJACENT] * 10 + [cl.ADJACENT_LINEAR] * 2),
                     lambda t: t["regret"]["full"].update({2560: [2.0] * 12})):
            res = copy.deepcopy(self.res)
            edit(res["truths"]["SKIP"])
            self.assertFalse(verdicts2(res)["B2"])
        res = copy.deepcopy(self.res)
        res["truths"]["SKIP"]["classified"] = [cl.NON_ADJACENT] * 11 + [cl.NONLINEAR_ADJACENT]
        self.assertTrue(verdicts2(res)["B2"])

    def test_b3_allows_three_misses_and_requires_exact_normal_score_closure(self):
        res = copy.deepcopy(self.res)
        res["truths"]["NL"]["classified"] = [cl.NONLINEAR_ADJACENT] * 9 + [cl.ADJACENT_LINEAR] * 3
        self.assertTrue(verdicts2(res)["B3"])
        res["truths"]["NL"]["classified"] = [cl.NONLINEAR_ADJACENT] * 8 + [cl.NON_ADJACENT] * 4
        self.assertFalse(verdicts2(res)["B3"])
        res = copy.deepcopy(self.res)
        res["truths"]["NL"]["population"]["defect_normal_score"] = 1e-6
        self.assertFalse(verdicts2(res)["B3"])

    def test_b4_allows_three_false_alarms(self):
        res = copy.deepcopy(self.res)
        res["truths"]["ADJ"]["classified"] = [cl.ADJACENT_LINEAR] * 9 + [cl.NONLINEAR_ADJACENT] * 3
        self.assertTrue(verdicts2(res)["B4"])
        res["truths"]["ADJ"]["classified"] = [cl.ADJACENT_LINEAR] * 8 + [cl.NON_ADJACENT] * 4
        self.assertFalse(verdicts2(res)["B4"])


if __name__ == "__main__":
    unittest.main()
