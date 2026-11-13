"""Every Level 0 claim in checks.py, one test per claim."""

import unittest

from efl_iwme_cascade import checks


class Level0Claims(unittest.TestCase):
    def test_every_check_passes(self):
        results = checks.run_all()
        self.assertGreaterEqual(len(results), 30)
        for r in results:
            with self.subTest(check=r.id, claim=r.claim):
                self.assertTrue(r.passed, r.detail)

    def test_ids_are_unique_and_ordered(self):
        ids = checks.ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))

    def test_a_check_that_raises_is_a_failure(self):
        checks._REGISTRY.append(("C99", "raises", lambda: 1 / 0))
        try:
            r = checks.run_one("C99")
        finally:
            checks._REGISTRY.pop()
        self.assertFalse(r.passed)
        self.assertIn("ZeroDivisionError", r.detail)

    def test_render_reports_the_count_without_table_rules(self):
        text = checks.render([checks.Check("C00", "claim", True, "detail")])
        self.assertIn("1 of 1 checks pass", text)
        self.assertNotIn(chr(124), text)

    def test_unknown_check(self):
        with self.assertRaises(KeyError):
            checks.run_one("nope")

    def test_check_stream_is_outside_every_experiment_purpose(self):
        from efl_iwme_cascade import streams
        self.assertNotIn(checks.CheckStream.PURPOSE, streams.PURPOSES.values())
        a, b = checks.CheckStream(3).uniform(8), checks.CheckStream(3).uniform(8)
        self.assertTrue((a == b).all())
        self.assertFalse((a == checks.CheckStream(4).uniform(8)).all())


if __name__ == "__main__":
    unittest.main()
