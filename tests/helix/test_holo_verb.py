"""The `kgirl holo` verb: subparsers in cli.py over the public holo API, without building an Assistant.

Offline: the store is seeded through HoloStore's public API; no chunk index, no embedding, no model.
Written, not run.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.helix  # noqa: F401  (sys.path)
from kgirl.harness.cli import _holo_argv, main


def seed_store(home: Path) -> None:
    """A built store: one chunk vector owned by two entities, and a root build event."""
    import numpy as np

    from kgirl.harness.holo import HoloStore
    from kgirl.harness.holo.store import E_BUILT, ROOT
    with HoloStore(home) as s:
        s.ensure(ROOT)
        s.ensure("repo:alpha")
        s.ensure("kind:function")
        v = np.ones(4, dtype=np.int8)
        s.record("repo:alpha", "chunk", v, label="1")
        s.record("kind:function", "chunk", v, label="1")
        s.db.append(ROOT, E_BUILT, {"records": 2})


class HoloVerb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ)                    # the user's own home and index must not leak in
        env.start()
        self.addCleanup(env.stop)
        for key in ("KGIRL_HOME", "KGIRL_CHUNKS_DB", "KGIRL_MATRIX"):
            os.environ.pop(key, None)
        self.home = Path(self.tmp.name) / "home"

    def call(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main(argv)
            except SystemExit as exc:                        # argparse errors (missing subcommand, query XOR)
                rc = exc.code
        return rc, out.getvalue(), err.getvalue()

    def test_other_verbs_are_not_intercepted(self):
        self.assertIsNone(_holo_argv(["ask", "holo"]))
        self.assertIsNone(_holo_argv(["--home", "/tmp/x", "helix", "floor"]))
        self.assertEqual(_holo_argv(["holo", "status"]), ["status"])
        self.assertEqual(_holo_argv(["--home", "/tmp/x", "holo", "status"]), ["--home", "/tmp/x", "status"])
        self.assertEqual(_holo_argv(["holo", "--home", "/tmp/y", "status"]), ["--home", "/tmp/y", "status"])

    def test_status_of_an_empty_home_creates_nothing(self):
        rc, out, _ = self.call(["--home", str(self.home), "holo", "status"])
        self.assertEqual(rc, 0)
        d = json.loads(out)
        self.assertIsNone(d["built"])
        self.assertEqual(d["entities"], 0)
        self.assertFalse((self.home / "holo.db").exists())

    def test_reads_refuse_an_unbuilt_store_without_creating_it(self):
        for cmd in (["verify"], ["overlaps"], ["owners", "--digest", "d"], ["query", "--text", "q"]):
            rc, _, err = self.call(["holo", "--home", str(self.home), *cmd])
            self.assertEqual(rc, 1, cmd)
            self.assertIn("has not been built", err)
            self.assertFalse((self.home / "holo.db").exists(), cmd)

    def test_build_without_an_index_is_an_error_not_a_crash(self):
        rc, _, err = self.call(["holo", "--home", str(self.home), "build"])
        self.assertEqual(rc, 1)
        self.assertIn("no chunk index", err)

    def test_verify_and_overlaps_on_a_seeded_store(self):
        seed_store(self.home)
        rc, out, _ = self.call(["holo", "--home", str(self.home), "verify"])
        self.assertEqual(rc, 0)
        checks = json.loads(out)
        self.assertTrue(checks["repo:alpha"]["ok"])
        self.assertEqual(checks["repo:alpha"]["unproven"], [])
        rc, out, _ = self.call(["holo", "--home", str(self.home), "overlaps"])
        self.assertEqual(rc, 0)
        overlaps = json.loads(out)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(sorted(next(iter(overlaps.values()))), ["kind:function", "repo:alpha"])

    def test_owners_of_a_shared_digest_and_of_an_unknown_one(self):
        seed_store(self.home)
        rc, out, _ = self.call(["holo", "--home", str(self.home), "overlaps"])
        digest = next(iter(json.loads(out)))
        rc, out, _ = self.call(["holo", "--home", str(self.home), "owners", "--digest", digest])
        self.assertEqual(rc, 0)
        d = json.loads(out)
        self.assertEqual(d["digest"], digest)
        self.assertEqual(sorted(d["owners"]), ["kind:function", "repo:alpha"])
        rc, out, _ = self.call(["holo", "--home", str(self.home), "owners", "--digest", "no-such-digest"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["owners"], [])

    def test_query_needs_exactly_one_of_text_and_vector(self):
        seed_store(self.home)
        rc, _, err = self.call(["holo", "--home", str(self.home), "query"])
        self.assertEqual(rc, 2)
        self.assertIn("exactly one of --text and --vector", err)
        rc, _, err = self.call(["holo", "--home", str(self.home), "query", "--text", "q", "--vector", "v.npy"])
        self.assertEqual(rc, 2)

    def test_missing_subcommand_is_an_argparse_error(self):
        rc, _, err = self.call(["holo", "--home", str(self.home)])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
