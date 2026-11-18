"""The rules-check meta a stored assistant turn carries must reach the browser untouched.

The mind-rules lane writes rules_hits / rules_left / rules_revised into the answer's meta; the Chat
view renders chips from those fields. This pins the API passthrough (and that plain turns stay plain)
without needing that lane or a model. Written, not run.
"""

import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.server import Handler
from kgirl.harness.mind.conversations import ConversationStore

REVISED = {"rules_hits": ["push"], "rules_left": [], "rules_revised": True}
STILL_BREAKS = {"rules_hits": ["push", "pull"], "rules_left": ["push"], "rules_revised": True}


class RulesMetaPassthrough(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ConversationStore(Path(self.tmp.name) / "conversations.db")
        self.addCleanup(self.store.close)
        self.cid = self.store.new("rules check")
        self.store.append(self.cid, "user", "write me a release script")
        self.tid_clean = self.store.append(self.cid, "assistant", "here is a safe script", meta=REVISED)
        self.store.append(self.cid, "user", "now add the publish step")
        self.tid_left = self.store.append(self.cid, "assistant", "```sh\ngit push\n```", meta=STILL_BREAKS)
        self.tid_plain = self.store.append(self.cid, "assistant", "no rules meta on this one")
        Handler.console = SimpleNamespace(token="t", store=self.store)
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

    def test_rules_meta_reaches_the_api_unchanged(self):
        status, d = self.get(f"/api/conversations/{self.cid}")
        self.assertEqual(status, 200)
        turns = {t["id"]: t for t in d["turns"]}
        self.assertEqual(turns[self.tid_clean]["meta"], REVISED)
        self.assertEqual(turns[self.tid_left]["meta"], STILL_BREAKS)

    def test_plain_turns_carry_no_rules_fields(self):
        status, d = self.get(f"/api/conversations/{self.cid}")
        self.assertEqual(status, 200)
        plain = next(t for t in d["turns"] if t["id"] == self.tid_plain)
        self.assertEqual(plain["meta"], {})

    def test_feedback_merge_keeps_rules_meta(self):
        self.store.set_meta(self.tid_left, feedback={"verdict": "negative"})
        status, d = self.get(f"/api/conversations/{self.cid}")
        self.assertEqual(status, 200)
        t = next(t for t in d["turns"] if t["id"] == self.tid_left)
        self.assertEqual(t["meta"]["rules_hits"], STILL_BREAKS["rules_hits"])
        self.assertEqual(t["meta"]["rules_left"], STILL_BREAKS["rules_left"])
        self.assertEqual(t["meta"]["rules_revised"], STILL_BREAKS["rules_revised"])
        self.assertEqual(t["meta"]["feedback"], {"verdict": "negative"})

    def test_turns_endpoint_needs_the_token(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", f"/api/conversations/{self.cid}")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 403)


if __name__ == "__main__":
    unittest.main()
