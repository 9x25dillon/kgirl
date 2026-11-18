"""The read-only Lanes window onto the porter ledger: sessions, claims, events, and liveness.

Offline: a temp sqlite DB with the porter schema, AURIC_PORTER_DB pointed at it, a stub console. No auric
import, no writes to the real ledger. Written, not run.
"""

import http.client
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.server import Handler

SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY, agent TEXT, project TEXT, task TEXT, host TEXT, pid INTEGER,
  started TEXT, last_seen TEXT, ended TEXT, summary TEXT, cursor INTEGER, seen_directive INTEGER);
CREATE TABLE claims (session TEXT, project TEXT, path TEXT, intent TEXT, created TEXT);
CREATE TABLE events (id INTEGER PRIMARY KEY, project TEXT, actor TEXT, kind TEXT, message TEXT, paths TEXT,
  recipient TEXT, ref TEXT, created TEXT, delivered_to TEXT);
"""


def iso(dt):
    return dt.isoformat()


class Lanes(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "porter.sqlite"
        con = sqlite3.connect(self.db)
        con.executescript(SCHEMA)
        now = datetime.now(timezone.utc)
        host = socket.gethostname()
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        rows = [
            ("live", "qwen-code", "/hub", "building", host, os.getpid(), iso(now), iso(now), None, None),
            ("ended", "claude-code", "/hub", "done", host, os.getpid(), iso(now), iso(now), iso(now), "finished"),
            ("stale", "codex", "/hub", "old", host, os.getpid(), iso(now), iso(now - timedelta(hours=13)), None, None),
            ("deadpid", "agy", "/hub", "gone", host, dead.pid, iso(now), iso(now), None, None),
            ("other", "qwen-code", "/elsewhere", "other project", host, os.getpid(), iso(now), iso(now), None, None),
        ]
        con.executemany("INSERT INTO sessions(id,agent,project,task,host,pid,started,last_seen,ended,summary) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        con.execute("INSERT INTO claims VALUES ('live','/hub','src/kgirl/harness/helix','lane','now')")
        con.execute("INSERT INTO events(id,project,actor,kind,message,created) VALUES (1,'/hub','live','progress','hi','now')")
        con.execute("INSERT INTO events(id,project,actor,kind,message,created) VALUES (2,'/hub','live','decision','yo','now')")
        con.commit()
        con.close()
        self.stack = Path(self.tmp.name) / "kgirl-stack"
        self.lane = Path(self.tmp.name) / "kgirl-lane"
        for repo in (self.stack, self.lane):
            repo.mkdir()
            git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run([*git, "init", "-q"], check=True)
            (repo / "f.txt").write_text("base\n")
            subprocess.run([*git, "add", "f.txt"], check=True)
            subprocess.run([*git, "commit", "-qm", "init"], check=True)
        (self.stack / "f.txt").write_text("changed\n")         # one modified path
        (self.stack / "new.txt").write_text("untracked\n")     # one untracked path
        env = mock.patch.dict(os.environ, {"AURIC_PORTER_DB": str(self.db),
                                           "KGIRL_LANDING_STACK": str(self.stack),
                                           "KGIRL_LANDING_LANES": str(self.lane)})
        env.start()
        self.addCleanup(env.stop)
        Handler.console = SimpleNamespace(token="t", jobs=SimpleNamespace(list=lambda: []))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path, headers={"X-Kgirl-Token": "t"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def test_lanes_needs_the_token(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/api/lanes")
        self.assertEqual(conn.getresponse().status, 403)
        conn.close()

    def test_active_sessions_with_liveness_and_claims(self):
        status, d = self.get("/api/lanes")
        self.assertEqual(status, 200)
        self.assertTrue(d["available"])
        by_id = {s["id"]: s for s in d["sessions"]}
        self.assertNotIn("ended", by_id)                       # ended sessions are filtered out
        self.assertTrue(by_id["live"]["alive"])
        self.assertFalse(by_id["stale"]["alive"])              # past the TTL
        self.assertFalse(by_id["deadpid"]["alive"])            # pid is gone
        self.assertEqual(by_id["live"]["claims"][0]["path"], "src/kgirl/harness/helix")

    def test_project_filter(self):
        status, d = self.get("/api/lanes?project=/hub")
        self.assertEqual(status, 200)
        self.assertEqual({s["id"] for s in d["sessions"]}, {"live", "stale", "deadpid"})
        self.assertEqual([e["id"] for e in d["events"]], [2, 1])   # newest first

    def test_missing_ledger_is_a_state_not_a_crash(self):
        os.environ["AURIC_PORTER_DB"] = str(Path(self.tmp.name) / "absent.sqlite")
        status, d = self.get("/api/lanes")
        self.assertEqual(status, 200)
        self.assertFalse(d["available"])
        self.assertEqual(d["sessions"], [])

    def test_landing_counts_uncommitted_paths_and_stays_read_only(self):
        status, d = self.get("/api/lanes")
        self.assertEqual(status, 200)
        lg = d["landing"]
        self.assertEqual(lg["command"],
                         "python3 scripts/smoke/land_lanes.py --stack ~/work/kgirl-stack "
                         "--lane ~/work/kgirl-eng --lane ~/work/kgirl-helix-ui --lane ~/work/kgirl-builder")
        by_name = {w["name"]: w for w in lg["worktrees"]}
        self.assertEqual(by_name["kgirl-stack"]["role"], "stack")
        self.assertEqual(by_name["kgirl-stack"]["uncommitted"], 2)
        self.assertTrue(by_name["kgirl-stack"]["branch"])
        self.assertEqual(by_name["kgirl-lane"]["role"], "lane")
        self.assertEqual(by_name["kgirl-lane"]["uncommitted"], 0)
        self.assertEqual((self.stack / "f.txt").read_text(), "changed\n")   # the read changed nothing
        self.assertTrue((self.stack / "new.txt").exists())

    def test_landing_missing_worktree_is_a_state_not_a_crash(self):
        os.environ["KGIRL_LANDING_LANES"] = str(Path(self.tmp.name) / "absent")
        status, d = self.get("/api/lanes")
        self.assertEqual(status, 200)
        absent = [w for w in d["landing"]["worktrees"] if w["name"] == "absent"][0]
        self.assertIn("error", absent)
        self.assertNotIn("uncommitted", absent)

    def test_landing_present_without_a_ledger(self):
        os.environ["AURIC_PORTER_DB"] = str(Path(self.tmp.name) / "absent.sqlite")
        status, d = self.get("/api/lanes")
        self.assertEqual(status, 200)
        self.assertFalse(d["available"])
        self.assertEqual(len(d["landing"]["worktrees"]), 2)


if __name__ == "__main__":
    unittest.main()
