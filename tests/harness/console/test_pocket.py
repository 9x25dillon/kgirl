"""The read-only Pocket view: local kgirl-pocket state files and a local port check, never adb.

Offline: a temp state dir patched over POCKET_STATE, the port check mocked, a stub console. Written, not run.
"""

import http.client
import json
import socket
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console import server
from kgirl.harness.console.server import Handler, _pocket_listening


class Pocket(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        (self.state / "backup.json").write_text(json.dumps(
            {"name": "vault-x.tar.zst.gpg", "snapshot": "x", "sha256": "ab" * 32, "bytes": 2048,
             "encrypted": True, "written": "2026-11-17T10:00:00", "pack_s": 3, "send_s": 9}))
        (self.state / "drill.json").write_text(json.dumps({"name": "vault-x", "passed": True, "when": "now"}))
        (self.state / "bench.json").write_text(json.dumps({"model": "llama3.2:3b", "gen_tok_s": 8.0}))
        patcher = mock.patch.object(server, "POCKET_STATE", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        listen = mock.patch.object(server, "_pocket_listening", return_value=True)
        listen.start()
        self.addCleanup(listen.stop)
        Handler.console = SimpleNamespace(token="t", jobs=SimpleNamespace(list=lambda: []))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def get(self, path, token="t"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"X-Kgirl-Token": token} if token else {}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_pocket_needs_the_token(self):
        self.assertEqual(self.get("/api/pocket", token=None)[0], 403)

    def test_reads_the_local_state_files(self):
        status, data = self.get("/api/pocket")
        self.assertEqual(status, 200)
        d = json.loads(data)
        self.assertEqual(d["backup"]["name"], "vault-x.tar.zst.gpg")
        self.assertTrue(d["drill"]["passed"])
        self.assertEqual(d["bench"]["model"], "llama3.2:3b")
        self.assertTrue(d["node"]["listening"])
        self.assertEqual(d["node"]["port"], 11435)

    def test_missing_files_are_null_not_a_crash(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with mock.patch.object(server, "POCKET_STATE", empty):
            status, data = self.get("/api/pocket")
        self.assertEqual(status, 200)
        d = json.loads(data)
        self.assertIsNone(d["backup"])
        self.assertIsNone(d["drill"])
        self.assertIsNone(d["bench"])


class PortCheck(unittest.TestCase):
    def test_listening_true_when_connect_succeeds(self):
        with mock.patch.object(socket, "create_connection", return_value=mock.MagicMock()):
            self.assertTrue(_pocket_listening("127.0.0.1", 11435))

    def test_listening_false_when_connect_fails(self):
        with mock.patch.object(socket, "create_connection", side_effect=OSError("refused")):
            self.assertFalse(_pocket_listening("127.0.0.1", 11435))


if __name__ == "__main__":
    unittest.main()
