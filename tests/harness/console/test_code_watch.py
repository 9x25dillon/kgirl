"""The restart hint's contract: a startup fingerprint of KGIRL_CODE, re-checked at most once a minute.

Builds a real throwaway git repo, then pins: clean tree -> no hint; an edit or a new commit drifts the
fingerprint; within the minute the cached answer stands; a non-git root reports unavailable; and the
only git commands ever run are reads (rev-parse, diff, ls-files). Written, not run.
"""

import http.client
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.server import CodeWatch, Handler, _code_fingerprint

GIT_ID = ["-c", "user.name=test", "-c", "user.email=test@example.com"]


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)


class CodeWatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        git(self.root, "init", "-q", "-b", "main")
        (self.root / "src").mkdir()
        (self.root / "src" / "mod.py").write_text("x = 1\n")
        git(self.root, "add", "src/mod.py")
        git(self.root, *GIT_ID, "commit", "-q", "-m", "init")

    def expire(self, watch):
        watch._checked -= watch.INTERVAL_S + 1     # force the next status() to re-read the disk

    def test_clean_tree_matches_startup(self):
        watch = CodeWatch(self.root)
        self.assertIsNotNone(watch.startup)
        d = watch.status()
        self.assertTrue(d["available"])
        self.assertFalse(d["changed"])
        self.assertEqual(d["startup"], watch.startup)
        self.assertEqual(d["current"], watch.startup)

    def test_edit_drifts_but_only_after_the_minute(self):
        watch = CodeWatch(self.root)
        watch.status()
        (self.root / "src" / "mod.py").write_text("x = 2\n")
        self.assertFalse(watch.status()["changed"])    # cached: checked less than a minute ago
        self.expire(watch)
        d = watch.status()
        self.assertTrue(d["changed"])
        self.assertEqual(d["startup"], watch.startup)
        self.assertNotEqual(d["current"], d["startup"])

    def test_new_commit_drifts_the_head(self):
        watch = CodeWatch(self.root)
        (self.root / "src" / "mod.py").write_text("x = 2\n")
        git(self.root, "add", "src/mod.py")
        git(self.root, *GIT_ID, "commit", "-q", "-m", "change")
        self.expire(watch)
        self.assertTrue(watch.status()["changed"])

    def test_untracked_src_file_counts_as_dirty(self):
        before = _code_fingerprint(self.root)
        (self.root / "src" / "new.py").write_text("y = 1\n")
        self.assertNotEqual(_code_fingerprint(self.root), before)

    def test_untracked_outside_src_does_not_drift(self):
        before = _code_fingerprint(self.root)
        (self.root / "README.md").write_text("notes\n")
        self.assertEqual(_code_fingerprint(self.root), before)

    def test_tracked_edit_outside_src_drifts(self):
        (self.root / "README.md").write_text("notes\n")
        git(self.root, "add", "README.md")
        git(self.root, *GIT_ID, "commit", "-q", "-m", "readme")
        watch = CodeWatch(self.root)
        (self.root / "README.md").write_text("changed\n")
        self.expire(watch)
        self.assertTrue(watch.status()["changed"])

    def test_non_git_root_is_unavailable(self):
        with tempfile.TemporaryDirectory() as plain:
            watch = CodeWatch(Path(plain))
            d = watch.status()
        self.assertFalse(d["available"])
        self.assertFalse(d["changed"])

    def test_only_git_reads_ever_run(self):
        seen = []
        real_run = subprocess.run

        def recorder(cmd, *a, **kw):
            seen.append(cmd)
            return real_run(cmd, *a, **kw)

        with mock.patch("kgirl.harness.console.server.subprocess.run", side_effect=recorder):
            watch = CodeWatch(self.root)
            self.expire(watch)
            watch.status()
        self.assertTrue(seen)
        for cmd in seen:
            self.assertEqual(cmd[:2], ["git", "-C"])
            self.assertIn(cmd[3], ("rev-parse", "diff", "ls-files"))

    def test_concurrent_status_shares_one_reread_per_minute(self):
        seen = []
        real_run = subprocess.run

        def recorder(cmd, *a, **kw):
            seen.append(cmd)
            return real_run(cmd, *a, **kw)

        with mock.patch("kgirl.harness.console.server.subprocess.run", side_effect=recorder):
            watch = CodeWatch(self.root)
            watch.status()
            baseline = len(seen)
            self.expire(watch)
            barrier = threading.Barrier(8)
            results = []

            def caller():
                barrier.wait(timeout=10)
                results.append(watch.status())

            threads = [threading.Thread(target=caller) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(len(results), 8)
        self.assertEqual(len(seen) - baseline, 3)      # one fingerprint: rev-parse + diff + ls-files
        self.assertTrue(all(r == results[0] for r in results))


class CodeEndpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        git(self.root, "init", "-q", "-b", "main")
        (self.root / "f.txt").write_text("hello\n")
        git(self.root, "add", "f.txt")
        git(self.root, *GIT_ID, "commit", "-q", "-m", "init")
        Handler.console = SimpleNamespace(token="t")
        self._old_watch = Handler.code_watch
        self.addCleanup(setattr, Handler, "code_watch", self._old_watch)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def get(self, path, token=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path, headers={"X-Kgirl-Token": "t"} if token else {})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def test_endpoint_reports_the_watch(self):
        with mock.patch.dict(os.environ, {"KGIRL_CODE": str(self.root)}):
            Handler.code_watch = CodeWatch()
        status, d = self.get("/api/code")
        self.assertEqual(status, 200)
        self.assertTrue(d["available"])
        self.assertEqual(d["root"], str(self.root))
        self.assertFalse(d["changed"])

    def test_no_watch_is_unavailable(self):
        Handler.code_watch = None
        status, d = self.get("/api/code")
        self.assertEqual(status, 200)
        self.assertFalse(d["available"])
        self.assertFalse(d["changed"])

    def test_code_endpoint_needs_the_token(self):
        self.assertEqual(self.get("/api/code", token=False)[0], 403)


if __name__ == "__main__":
    unittest.main()
