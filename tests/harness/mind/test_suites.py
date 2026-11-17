"""Offline tests for mind.suites: pruning generated suites to the agreed tests, merging, retargeting."""

import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.suites import merge_suites, prune, retarget

SUITE = '''import unittest
from solution import mean


class TestMean(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_single(self):
        self.assertEqual(mean([5]), 5)


class TestOther(unittest.TestCase):
    def test_other(self):
        self.assertTrue(True)
'''


class PruneTests(unittest.TestCase):
    def test_keeps_only_named_tests(self):
        out = prune(SUITE, "solution", {"solution:TestMean.test_basic"})
        self.assertIn("def test_basic", out)
        self.assertNotIn("def test_single", out)
        self.assertNotIn("class TestOther", out)
        compile(out, "solution.py", "exec")

    def test_none_kept_returns_empty(self):
        self.assertEqual(prune(SUITE, "solution", {"solution:TestMean.nope"}), "")

    def test_all_kept_returns_unchanged(self):
        keep = {"solution:TestMean.test_basic", "solution:TestMean.test_single", "solution:TestOther.test_other"}
        self.assertEqual(prune(SUITE, "solution", keep), SUITE)

    def test_unparseable_suite_returned_unchanged(self):
        bad = "def broken(:\n  pass\n"
        self.assertEqual(prune(bad, "solution", {"solution:X.test_y"}), bad)

    def test_helper_class_left_alone(self):
        suite = ("class Helper:\n    def assist(self):\n        return 1\n\n"
                 "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n")
        out = prune(suite, "solution", {"solution:T.test_a"})
        self.assertIn("class Helper", out)
        self.assertIn("def test_a", out)


class MergeSuitesTests(unittest.TestCase):
    def test_banners_and_blank_dropped(self):
        out = merge_suites(["a = 1\n", "  ", "b = 2\n"])
        self.assertIn("generated suite 1", out)
        self.assertIn("generated suite 2", out)
        self.assertIn("a = 1", out)
        self.assertIn("b = 2", out)
        self.assertNotIn("generated suite 3", out)


class RetargetTests(unittest.TestCase):
    def test_solution_is_noop(self):
        self.assertEqual(retarget(SUITE, "solution"), SUITE)

    def test_from_import(self):
        self.assertIn("from lru import mean", retarget(SUITE, "lru"))

    def test_plain_import(self):
        self.assertIn("import lru as solution", retarget("import solution\n", "lru"))

    def test_aliased_import(self):
        self.assertIn("import lru as sol", retarget("import solution as sol\n", "lru"))

    def test_indented_import(self):
        self.assertIn("    from lru import mean", retarget("    from solution import mean\n", "lru"))


if __name__ == "__main__":
    unittest.main()
