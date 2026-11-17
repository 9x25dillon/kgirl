"""Ledger: the L1 split, provider counts over estimates, overrun compression, the BYPASS predicate."""

import unittest

import tests.helix  # noqa: F401  (sys.path)
from kgirl.harness.helix import ledger as L
from kgirl.harness.llm import Completion


def done(n_out: int, n_in: int = 100, stop: str = "stop") -> Completion:
    return Completion("x", "m", "fake", n_in, n_out, stop)


TEXT = "word " * 100                 # 100 words, 500 characters


class Pools(unittest.TestCase):
    def test_reserve_is_split_once_and_everything_sums_to_B(self):
        led = L.Ledger(4000, {"ENGINEER": 0.75, "CRITIC": 0.25})
        self.assertEqual((led.steer_pool, led.synth_floor, led.content_pool), (200, 400, 3400))
        self.assertEqual(led.steer_pool + led.synth_floor + led.content_pool, 4000)
        self.assertEqual(led.alloc, {"ENGINEER": 2550, "CRITIC": 850})
        self.assertEqual(sum(led.alloc.values()), led.content_pool)       # no second reserve below L1

    def test_weights_over_one_and_a_non_positive_budget_are_refused(self):
        with self.assertRaises(ValueError):
            L.Ledger(4000, {"A": 0.8, "B": 0.3})
        with self.assertRaises(ValueError):
            L.Ledger(0)


class Charging(unittest.TestCase):
    def test_the_provider_count_is_the_truth_and_the_estimate_is_kept_beside_it(self):
        led = L.Ledger(4000, {"ENGINEER": 0.75, "CRITIC": 0.25})
        e = led.charge("ENGINEER", L.EXPERT, TEXT, done(120, 900), 2550)
        self.assertEqual((e.ext, e.in_ext, e.spent), (120, 900, 120))
        self.assertEqual(e.est, L.estimate(TEXT))
        r = led.report()
        self.assertEqual((r["spent_ext"], r["input_ext"], r["uncounted_calls"]), (120, 900, 0))
        self.assertEqual(r["est_over_ext"], round(L.estimate(TEXT) / 120, 3))

    def test_a_backend_that_does_not_count_leaves_ext_null(self):
        led = L.Ledger(4000, {"ENGINEER": 0.75, "CRITIC": 0.25})
        e = led.charge("ENGINEER", L.EXPERT, TEXT, done(0, 0), 2550)
        self.assertIsNone(e.ext)
        self.assertEqual(e.spent, e.est)
        r = led.report()
        self.assertIsNone(r["spent_ext"])
        self.assertIsNone(r["input_ext"])
        self.assertIsNone(r["scaffold_ext"])
        self.assertEqual(r["uncounted_calls"], 1)
        self.assertEqual(r["spent_est"], e.est)

    def test_trailers_are_charged_to_the_steering_pool(self):
        led = L.Ledger(4000, {"ENGINEER": 0.75, "CRITIC": 0.25})
        e = led.charge("ENGINEER", L.EXPERT, TEXT, done(200), 2550, trailer_chars=len(TEXT) // 4)
        self.assertEqual(e.steer_ext, 50)
        self.assertEqual((led.steer_spent(), led.steer_left()), (50, 150))
        self.assertEqual(led.synth_target(), 400 + 3400 - 150)

    def test_a_steering_call_is_charged_whole(self):
        led = L.Ledger(4000, {"ENGINEER": 0.75, "CRITIC": 0.25})
        led.charge("ORACLE", L.STEER, TEXT, done(60), 200)
        self.assertEqual(led.steer_left(), 140)
        self.assertEqual(led.ratio(), 1.0)


class Compression(unittest.TestCase):
    def test_an_overrun_shrinks_the_next_target_by_its_ratio(self):
        led = L.Ledger(4000, {"A": 0.375, "B": 0.375, "CRITIC": 0.25})
        self.assertEqual(led.target("B"), 1275)
        led.charge("A", L.EXPERT, TEXT, done(2550), led.target("A"))        # twice its 1275
        self.assertEqual(led.ratio(), 0.5)
        self.assertEqual(led.target("B"), 638)
        self.assertEqual(led.target("CRITIC"), 425)

    def test_within_budget_nothing_is_compressed(self):
        led = L.Ledger(4000, {"A": 0.75, "CRITIC": 0.25})
        led.charge("A", L.EXPERT, TEXT, done(900), 2550)
        self.assertEqual((led.ratio(), led.target("CRITIC")), (1.0, 850))

    def test_compaction_is_spend_where_it_happens(self):
        led = L.Ledger(4000, {"A": 0.75, "CRITIC": 0.25})
        led.charge("A", L.EXPERT, TEXT, done(2550), 2550)                   # exactly on budget
        self.assertEqual(led.ratio(), 1.0)
        e = led.charge("LOG", L.COMPACTION, TEXT, done(450), 500)
        self.assertEqual(e.alloc, 0)
        self.assertEqual(led.ratio(), 0.85)                                 # 2550 allocated / 3000 spent
        self.assertEqual(led.target("CRITIC"), 722)
        self.assertEqual(led.report()["spent_ext"], 3000)

    def test_the_synthesis_floor_survives_any_overrun(self):
        led = L.Ledger(4000, {"A": 0.75, "CRITIC": 0.25})
        led.charge("A", L.EXPERT, TEXT, done(9000), 2550)
        led.charge("ORACLE", L.STEER, TEXT, done(5000), 200)
        self.assertLess(led.steer_left(), 0)
        self.assertEqual(led.synth_target(), led.synth_floor)
        self.assertTrue(led.report()["overrun"])


class Proxy(unittest.TestCase):
    def test_the_stated_conversion(self):
        self.assertEqual(L.estimate(TEXT), 143)                             # max(100 x 1.35, 500 / 3.5)
        self.assertEqual(L.estimate("a" * 700), 200)                        # one long word: characters win
        self.assertEqual(L.estimate(""), 0)
        self.assertEqual(L.words(135), 100)


class Bypass(unittest.TestCase):
    def test_the_threshold_follows_from_the_floor(self):
        self.assertIsNotNone(L.bypass(2499, floor=500))
        self.assertIsNone(L.bypass(2500, floor=500))
        self.assertIsNotNone(L.bypass(1500, floor=500))
        self.assertIsNone(L.bypass(1500, floor=250))                        # a smaller measured floor moves it
        self.assertIsNotNone(L.bypass(1200, floor=250))

    def test_a_small_expected_answer_bypasses_at_any_budget(self):
        self.assertIsNotNone(L.bypass(100_000, expect=299))
        self.assertIsNone(L.bypass(100_000, expect=300))

    def test_the_reason_is_a_sentence_with_the_numbers(self):
        self.assertIn("33%", L.bypass(1500, floor=500))


if __name__ == "__main__":
    unittest.main()
