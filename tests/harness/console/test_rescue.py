"""The Rescue panel's read-only contract: the console only ever runs the rescue CLI's `scan --json`.

A stub rescue script (KGIRL_RESCUE_SCRIPT) logs its argv, so the test proves record and commit are never
invoked; feature detection, JSON passthrough and error surfacing are pinned too. Written, not run.
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
from kgirl.harness.console.server import Handler

ROW = {"platform": "qwen", "session_id": "sess-1", "worktree": "/tmp/repo", "branch": "feat/x",
       "uncommitted_count": 2, "eligible_count": 2, "skipped_count": 0, "last_activity": 1700000000.0,
       "last_task": "fix the tray", "is_live": False, "is_stale": True, "state": "STALE",
       "history_count": 3, "history_sessions": ["qwen/old-1", "qwen/old-2", "claude/old-3"],
       "has_unmerged_branch": True, "unmerged_commits": 2}

STUB_SCAN = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "with Path(__file__).with_name('argv.jsonl').open('a') as f:\n"
    "    f.write(json.dumps(sys.argv[1:]) + chr(10))\n"
    "print(json.dumps([" + repr(ROW) + "]))\n"   # repr: Python literals (True/False), not JSON's true/false
)
STUB_FAIL = "import sys\nprint('rescue: boom', file=sys.stderr)\nsys.exit(2)\n"
STUB_BAD_JSON = "print('this is not json')\n"
STUB_SHAPE = "print('{\"not\": \"a list\"}')\n"


class RescueContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.script = Path(self.tmp.name) / "rescue.py"
        env = mock.patch.dict(os.environ, {"KGIRL_RESCUE_SCRIPT": str(self.script)})
        env.start()
        self.addCleanup(env.stop)
        Handler.console = SimpleNamespace(token="t")
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

    def argv_log(self):
        path = Path(self.tmp.name) / "argv.jsonl"
        return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []

    def test_missing_script_is_feature_detected(self):
        status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        self.assertFalse(d["available"])
        self.assertEqual(d["script"], str(self.script))
        self.assertEqual(self.argv_log(), [])            # nothing ran

    def test_only_scan_json_ever_runs_and_rows_pass_through(self):
        self.script.write_text(STUB_SCAN)
        status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        self.assertTrue(d["available"])
        self.assertEqual(d["sessions"], [ROW])
        self.get("/api/rescue")                          # a second scan
        self.assertEqual(self.argv_log(), [["scan", "--json"], ["scan", "--json"]])

    def test_newer_core_fields_pass_through(self):
        self.script.write_text(STUB_SCAN)
        status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        row = d["sessions"][0]
        for key in ("history_count", "history_sessions", "has_unmerged_branch", "unmerged_commits",
                    "eligible_count", "skipped_count", "last_task"):
            self.assertEqual(row[key], ROW[key])

    def test_nonzero_exit_surfaces_the_error(self):
        self.script.write_text(STUB_FAIL)
        status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        self.assertTrue(d["available"])
        self.assertIn("boom", d["error"])
        self.assertNotIn("sessions", d)

    def test_non_json_output_surfaces_an_error(self):
        self.script.write_text(STUB_BAD_JSON)
        status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        self.assertIn("JSON", d["error"])

    def test_unexpected_shape_surfaces_an_error(self):
        self.script.write_text(STUB_SHAPE)
        status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        self.assertIn("shape", d["error"])

    def test_timeout_surfaces_an_error(self):
        self.script.write_text(STUB_SCAN)
        with mock.patch("kgirl.harness.console.server.subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="rescue", timeout=30)):
            status, d = self.get("/api/rescue")
        self.assertEqual(status, 200)
        self.assertIn("timed out", d["error"])

    def test_rescue_needs_the_token(self):
        self.assertEqual(self.get("/api/rescue", token=False)[0], 403)


if __name__ == "__main__":
    unittest.main()
