"""The claim registry behind the paper: every fast check passes, and the outputs are well formed."""

import json
import unittest

import tests.iwme  # noqa: F401

from kgirl.iwme import verify


class Registry(unittest.TestCase):
    def test_ids_unique_and_ordered(self):
        ids = [row[0] for row in verify.REGISTRY]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))

    def test_fast_checks_pass(self):
        res = verify.run_all(fast=True)
        failed = [(r.cid, r.observed) for r in res if not r.passed]
        self.assertEqual(failed, [])

    def test_outputs(self):
        res = verify.run_all(only=["V15", "V43"])
        self.assertEqual(len(json.loads(verify.to_json(res))), 2)
        tex = verify.to_latex(res)
        self.assertEqual(tex.count("\\\\"), 2)
        self.assertNotIn("_", tex.replace("\\_", ""))
        self.assertIn("2/2 checks pass", verify.render(res))


if __name__ == "__main__":
    unittest.main()
