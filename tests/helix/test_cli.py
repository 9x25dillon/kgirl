"""The `kgirl helix` verb: forwards to the helix CLI verbatim, without building an Assistant."""

import contextlib
import io
import json
import unittest

import tests.helix  # noqa: F401  (sys.path)
from tests.helix._fixtures import HelixCase
from kgirl.harness.cli import _helix_argv, main


class HelixVerb(HelixCase):
    def call(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_floor_forwarded_with_its_own_home(self):
        rc, out, _ = self.call(["helix", "--home", str(self.base), "floor"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["source"], "estimate")

    def test_leading_helix_options_are_forwarded(self):
        rc, out, _ = self.call(["helix", "--session", "s2", "--home", str(self.base), "floor"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["runs"], 0)

    def test_the_harness_home_prefix_is_bridged(self):
        rc, out, _ = self.call(["--home", str(self.base), "helix", "floor"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["tokens"], 500)

    def test_state_of_an_empty_store_is_an_error_not_a_crash(self):
        rc, _, err = self.call(["helix", "--home", str(self.base), "state"])
        self.assertEqual(rc, 1)
        self.assertIn("no such state", err)

    def test_other_verbs_are_not_intercepted(self):
        self.assertIsNone(_helix_argv(["ask", "helix"]))
        self.assertIsNone(_helix_argv(["search", "helix"]))
        self.assertIsNone(_helix_argv(["--home", "/tmp/x", "chat"]))
        self.assertEqual(_helix_argv(["helix", "run", "q", "-B", "4000"]), ["run", "q", "-B", "4000"])


if __name__ == "__main__":
    unittest.main()
