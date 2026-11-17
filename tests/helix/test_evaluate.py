"""The transcript audit, ported from helix_kernel.py: its T8-T15, as unit tests.

T1-T7 of the kernel test its own JSONL store, validator and token counter; the package's equivalents are
covered in test_state.py and test_ledger.py.
"""

import unittest

import tests.helix  # noqa: F401  (sys.path)
from kgirl.harness.helix.evaluate import RESTEER, evaluate, propose_resteer, reconcile
from kgirl.harness.helix.state import MAX_DELTA

CLEAN = """⟨HEADER | experts: ENGINEER+CRITIC | B=4000⟩
⟨SEED:L0 | goal: harden harness with production patterns | constraints: evidence per claim | success: executed tests | budget: 4000 tok⟩
⟨ROUTE: ENGINEER + CRITIC | why: build then attack⟩
⟨SEED:L1 | goal: durable state and external accounting | constraints: reuse interfaces | success: tests observed | budget: 1800 tok⟩
⟨W:gain=0.6, temp=0.4, aperture=0.5, rigor=0.6, horizon=0.5, compression=0.6 | intent: defaults⟩
Δ: "gain 0.6 → single recommendation with named alternative"
⟨TUNE | probe: ledger is self-reported⟩
⟨GRAD: rigor+0.1 | because: "ledger lacks external reconciliation"⟩
⟨W:rigor=0.7 | intent: measurement per number⟩
Δ: "rigor 0.7 → every number states its method"
⟨ROUTE: CRITIC | why: adversarial pass⟩
⟨SEED:L1 | goal: attack state drift and theater | constraints: repair per objection | success: quoted defects | budget: 1200 tok⟩
⟨W:gain=0.6, temp=0.4, aperture=0.5, rigor=0.7, horizon=0.5, compression=0.6 | intent: restate current vector⟩
⟨TUNE | probe: commit path untested under write failure⟩
⟨GRAD: gain+0.1 | because: "T2 shows cursor holds under injected failure"⟩
⟨W:gain=0.7 | intent: commit to findings⟩
Δ: "gain 0.7 → no option menus, only findings"
⟨LEDGER:L1 | alloc: 3000 | spent: 2900 | reserve_used: 120 | pruned: 0⟩
⟨LEDGER:L0 | alloc: 4000 | spent: 3600 | reserve_used: 200 | pruned: 0⟩"""

BAD = """⟨SEED:L0 | goal: this goal statement is deliberately padded past the twelve word capsule limit to trigger bloat | constraints: none | success: none | budget: 4000 tok⟩
⟨SEED:L4 | goal: illegal depth | constraints: none | success: none | budget: 100 tok⟩
⟨W:gain=0.6, rigor=0.6 | intent: defaults⟩
⟨W:rigor=0.95 | intent: preference not evidence⟩
⟨LEDGER:L0 | alloc: 4000 | spent: 600 | reserve_used: 0 | pruned: 0⟩"""

TRIVIAL = ("⟨SEED:L0 | goal: lookup one fact | constraints: none | success: answer | budget: 100 tok⟩\n"
           "⟨W:gain=0.6 | intent: x⟩")


def led(level: int, spent: int, alloc: int = 4000) -> str:
    return f"⟨LEDGER:L{level} | alloc: {alloc} | spent: {spent} | reserve_used: 0 | pruned: 0⟩"


def theater(report: dict) -> list[str]:
    return [f["evidence"] for f in report["findings"] if f["detector"] == "budget_theater"]


def detectors(report: dict) -> list[str]:
    return sorted({f["detector"] for f in report["findings"]})


class Reconcile(unittest.TestCase):
    def test_tolerance_is_fifteen_percent_of_the_external_count(self):
        self.assertTrue(reconcile(3600, 3600))
        self.assertTrue(reconcile(3060, 3600))            # exactly 15% under
        self.assertFalse(reconcile(2900, 3600))           # 19.4%: the L1 line that used to fail T8
        self.assertFalse(reconcile(600, 1200))
        self.assertFalse(reconcile(5, 0))


class Audit(unittest.TestCase):
    def test_t8_a_clean_log_passes(self):
        r = evaluate(CLEAN, external_tokens=3600)
        self.assertEqual((r["verdict"], r["findings"]), ("PASS", []))
        self.assertEqual(r["metrics"], {"evidence_anchor_rate": 1.0, "delta_predicate_rate": 1.0})

    def test_t9_the_taxonomy_catches_every_injected_defect(self):
        r = evaluate(BAD, external_tokens=1200)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(detectors(r), ["budget_theater", "capsule_bloat", "fractal_fractal", "steering_drift"])
        self.assertTrue(all(f["evidence"] for f in r["findings"]))

    def test_t11_ceremony_on_a_trivial_task(self):
        self.assertEqual(detectors(evaluate(TRIVIAL, external_tokens=120)), ["ceremony_on_trivial", "missing_ledger"])
        self.assertNotIn("ceremony_on_trivial", detectors(evaluate(TRIVIAL, external_tokens=300)))
        self.assertNotIn("ceremony_on_trivial", detectors(evaluate(TRIVIAL + "\n⟨BYPASS: one fact⟩", 120)))

    def test_t12_a_child_that_outspends_the_root(self):
        r = evaluate("\n".join([led(1, 5000), led(0, 3600)]), external_tokens=3600)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(theater(r), [led(1, 5000)])

    def test_t13_levels_that_disagree_while_both_sit_under_the_root(self):
        r = evaluate("\n".join([led(2, 3000), led(1, 2900), led(0, 3600)]), external_tokens=3600)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(theater(r), [led(2, 3000)])

    def test_t14_emission_order_does_not_change_the_verdict(self):
        lines = CLEAN.splitlines()
        rest = [x for x in lines if "⟨LEDGER:" not in x]
        l0, l1 = (next(x for x in lines if f"⟨LEDGER:L{n}" in x) for n in (0, 1))
        r = evaluate("\n".join(rest + [l0, l1]), external_tokens=3600)
        self.assertEqual((r["verdict"], r["findings"]), ("PASS", []))

    def test_t15_sub_ledgers_with_no_root_cannot_be_reconciled(self):
        r = evaluate(led(1, 2900), external_tokens=3600)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(theater(r), [led(1, 2900)])

    def test_a_sub_ledger_is_never_compared_with_the_external_count(self):
        r = evaluate("\n".join([led(1, 400), led(0, 3600)]), external_tokens=3600)      # L1 is 89% off the total
        self.assertEqual(theater(r), [])

    def test_a_level_is_the_sum_of_its_ledger_lines(self):
        ok = evaluate("\n".join([led(1, 1500), led(1, 1400), led(0, 3600)]), external_tokens=3600)
        over = evaluate("\n".join([led(1, 2000), led(1, 1700), led(0, 3600)]), external_tokens=3600)
        self.assertEqual(theater(ok), [])
        self.assertEqual(theater(over), [led(1, 1700)])

    def test_without_an_external_count_only_the_internal_checks_run(self):
        self.assertEqual(evaluate(led(0, 600)).get("findings"), [])
        self.assertEqual(evaluate(led(1, 2900))["findings"], [])
        self.assertEqual(theater(evaluate("\n".join([led(1, 5000), led(0, 3600)]))), [led(1, 5000)])

    def test_a_large_jump_needs_a_justify_and_any_change_needs_a_grad(self):
        base = "⟨W:rigor=0.5 | intent: start⟩\n"
        grad = '⟨GRAD: rigor+0.4 | because: "three figures had no method"⟩\n'
        jump = "⟨W:rigor=0.9 | intent: more checks⟩\n" + led(0, 100)
        self.assertEqual(detectors(evaluate(base + jump)), ["steering_drift"])
        self.assertEqual(len(evaluate(base + jump)["findings"]), 2)                 # no GRAD, and over the cap
        self.assertEqual(len(evaluate(base + grad + jump)["findings"]), 1)          # over the cap, no JUSTIFY
        self.assertEqual(evaluate(base + grad + "⟨JUSTIFY: audit run⟩\n" + jump)["findings"], [])

    def test_an_empty_log_passes_with_no_rates(self):
        r = evaluate("")
        self.assertEqual(r, {"findings": [], "verdict": "PASS",
                             "metrics": {"evidence_anchor_rate": None, "delta_predicate_rate": None}})


class Resteer(unittest.TestCase):
    def test_t10_two_failures_in_a_row_give_proposals_that_need_approval(self):
        bad = evaluate(BAD, external_tokens=1200)
        props = propose_resteer([bad, bad])
        self.assertEqual([(p["axis"], p["delta"]) for p in props],
                         [("rigor", 0.15), ("rigor", 0.10), ("compression", 0.10)])
        self.assertTrue(all(p["requires_approval"] for p in props))
        self.assertEqual(props[0]["because"], "budget_theater in runs 0,1")

    def test_one_failure_or_failures_apart_propose_nothing(self):
        bad, clean = evaluate(BAD, external_tokens=1200), evaluate(CLEAN, external_tokens=3600)
        self.assertEqual(propose_resteer([bad]), [])
        self.assertEqual(propose_resteer([bad, clean, bad]), [])
        self.assertEqual(propose_resteer([]), [])

    def test_no_proposal_exceeds_one_tuning_pass(self):
        self.assertTrue(all(0 < d <= MAX_DELTA for _, d in RESTEER.values()))


if __name__ == "__main__":
    unittest.main()
