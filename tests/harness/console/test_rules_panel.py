"""The Memory view's read-only Rules panel: feature-detect mind.rules.is_rule, list only rule prefs.

On this base the mind-rules lane is absent, so /api/rules must report available=False (panel hidden).
With a stub is_rule injected into sys.modules it must list exactly the active preferences the stub
marks, as id/text/scope, and ask soup only for active preferences. Written, not run.
"""

import http.client
import json
import sys
import tempfile
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.server import Handler
from kgirl.harness.soup import Soup

RULES_MODULE = "kgirl.harness.mind.rules"


class FakeSoup:
    def __init__(self, frags):
        self.frags = frags
        self.calls = []

    def all(self, **kw):
        self.calls.append(kw)
        return [f for f in self.frags
                if f.status == kw.get("status", f.status) and f.kind == kw.get("kind", f.kind)]


def frag(i, text, kind="preference", status="active", scope="*"):
    return SimpleNamespace(id=i, text=text, kind=kind, status=status, scope=scope)


class RulesPanel(unittest.TestCase):
    def setUp(self):
        self.soup = FakeSoup([
            frag(1, "Never push unless I name it"),
            frag(2, "likes short answers"),
            frag(3, "Always run compileall before done", scope="kgirl"),
            frag(4, "Never commit", status="staged"),
            frag(5, "Never share secrets", kind="fact"),
        ])
        Handler.console = SimpleNamespace(token="t", a=SimpleNamespace(soup=self.soup))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path, headers={"X-Kgirl-Token": "t"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def test_missing_lane_hides_the_panel(self):
        with mock.patch.dict(sys.modules, {RULES_MODULE: None}):
            status, d = self.get("/api/rules")
        self.assertEqual(status, 200)
        self.assertFalse(d["available"])
        self.assertEqual(d["rules"], [])
        self.assertEqual(self.soup.calls, [])          # nothing listed when the lane is absent

    def test_stub_is_rule_lists_only_active_rule_preferences(self):
        stub = types.ModuleType(RULES_MODULE)
        stub.is_rule = lambda text: str(text).lower().startswith(("never", "always"))
        with mock.patch.dict(sys.modules, {RULES_MODULE: stub}):
            status, d = self.get("/api/rules")
        self.assertEqual(status, 200)
        self.assertTrue(d["available"])
        self.assertEqual(d["rules"], [
            {"id": 1, "text": "Never push unless I name it", "scope": "*"},
            {"id": 3, "text": "Always run compileall before done", "scope": "kgirl"},
        ])
        self.assertEqual(self.soup.calls, [{"status": "active", "kind": "preference"}])

    def test_rules_endpoint_needs_the_token(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/api/rules")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 403)


class RealSoupRulesPanel(unittest.TestCase):
    """The same contract against the real Soup store, not a fake: the endpoint's call shape and
    filtering must work with the actual soup.all(status=, kind=) API and real fragment ids."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.soup = Soup(Path(self.tmp.name) / "soup.db")
        self.addCleanup(self.soup.close)
        self.rule_id = self.soup.add("Never push unless I name it", kind="preference")
        self.soup.add("likes short answers", kind="preference")
        self.soup.add("Never commit", kind="preference", status="staged")
        self.soup.add("Never share secrets", kind="fact")
        Handler.console = SimpleNamespace(token="t", a=SimpleNamespace(soup=self.soup))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def test_real_soup_lists_only_the_active_rule_preference(self):
        stub = types.ModuleType(RULES_MODULE)
        stub.is_rule = lambda text: str(text).lower().startswith(("never", "always"))
        with mock.patch.dict(sys.modules, {RULES_MODULE: stub}):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            conn.request("GET", "/api/rules", headers={"X-Kgirl-Token": "t"})
            resp = conn.getresponse()
            d = json.loads(resp.read())
            conn.close()
        self.assertEqual(resp.status, 200)
        self.assertTrue(d["available"])
        self.assertEqual(d["rules"], [{"id": self.rule_id, "text": "Never push unless I name it", "scope": "*"}])


if __name__ == "__main__":
    unittest.main()
