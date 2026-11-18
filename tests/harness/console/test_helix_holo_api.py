"""The console's helix/holo read endpoints over a temp harness home (stub console, no Assistant, no model).

Seeds STATE through the public StateStore API, then reads it back through the HTTP surface. Written, not run.
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
from kgirl.harness.helix import default_state
from kgirl.harness.helix.store import StateStore


class HelixHoloRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        Handler.console = SimpleNamespace(token="t", a=SimpleNamespace(home=self.home),
                                          jobs=SimpleNamespace(list=lambda: []))
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

    def test_empty_helix_state(self):
        status, d = self.get("/api/helix/state?session=main")
        self.assertEqual(status, 200)
        self.assertEqual(d["sessions"], [])
        self.assertIsNone(d["state"])
        self.assertEqual(d["floor"]["source"], "estimate")

    def test_state_history_and_verify_after_a_commit(self):
        with StateStore(self.home) as store:
            store.commit("main", "task", task_seed="a b c")
            store.commit("main", "tune", vector={"gain": 0.8, "temp": 0.4, "aperture": 0.5,
                                                 "rigor": 0.6, "horizon": 0.5, "compression": 0.6})
        status, d = self.get("/api/helix/state?session=main")
        self.assertEqual(status, 200)
        self.assertEqual(d["sessions"], ["main"])
        self.assertEqual(d["state"]["v"], 2)
        self.assertEqual(d["state"]["vector"]["gain"], 0.8)
        status, h = self.get("/api/helix/history?session=main")
        self.assertEqual([x["v"] for x in h], [1, 2])
        self.assertEqual(h[1]["origin"], "tune")
        status, v = self.get("/api/helix/verify?session=main")
        self.assertTrue(v["main"]["ok"])
        self.assertTrue(v["main"]["sequential"])

    def test_empty_holo_status(self):
        status, d = self.get("/api/holo/status")
        self.assertEqual(status, 200)
        self.assertIsNone(d["built"])
        self.assertEqual(d["entities"], 0)
        self.assertTrue(d["chains_ok"])

    def test_empty_holo_overlaps(self):
        status, d = self.get("/api/holo/overlaps")
        self.assertEqual(status, 200)
        self.assertEqual(d["count"], 0)
        self.assertEqual(d["overlaps"], [])

    def test_empty_holo_verify_creates_nothing(self):
        status, d = self.get("/api/holo/verify")
        self.assertEqual(status, 200)
        self.assertEqual(d, {})
        self.assertFalse((self.home / "holo.db").exists())

    def test_holo_verify_of_a_seeded_store(self):
        import numpy as np

        from kgirl.harness.holo import HoloStore
        from kgirl.harness.holo.store import E_BUILT, ROOT
        with HoloStore(self.home) as s:
            s.ensure(ROOT)
            s.ensure("repo:alpha")
            s.record("repo:alpha", "chunk", np.ones(4, dtype=np.int8), label="1")
            s.db.append(ROOT, E_BUILT, {"records": 1})
        status, d = self.get("/api/holo/verify")
        self.assertEqual(status, 200)
        self.assertTrue(d["repo:alpha"]["ok"])
        self.assertEqual(d["repo:alpha"]["unproven"], [])
        self.assertEqual(d["repo:alpha"]["length"], 1)        # the holo.record event
        self.assertTrue(d[ROOT]["ok"])

    def test_helix_endpoints_need_the_token(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/api/helix/state")
        self.assertEqual(conn.getresponse().status, 403)
        conn.close()

    def post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=json.dumps(body),
                     headers={"X-Kgirl-Token": "t", "Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def test_diff_between_two_versions(self):
        with StateStore(self.home) as store:
            store.commit("main", "task", task_seed="a b c")
            store.commit("main", "tune", vector={"gain": 0.8, "temp": 0.4, "aperture": 0.5,
                                                 "rigor": 0.6, "horizon": 0.5, "compression": 0.6})
        status, d = self.get("/api/helix/diff?session=main&a=1&b=2")
        self.assertEqual(status, 200)
        self.assertEqual(d["vector"]["gain"], [0.6, 0.8])

    def test_diff_missing_version_is_a_400(self):
        with StateStore(self.home) as store:
            store.commit("main", "task", task_seed="a b c")
        self.assertEqual(self.get("/api/helix/diff?session=main&a=1&b=99")[0], 400)

    def test_ingest_adopted_into_an_empty_store(self):
        status, d = self.post("/api/helix/ingest",
                              {"session": "main", "state": default_state("a b c").to_dict(), "source": "peer"})
        self.assertEqual(status, 200)
        self.assertTrue(d["accepted"])
        self.assertEqual(d["state"]["v"], 1)
        self.assertEqual(self.get("/api/helix/history?session=main")[1][0]["origin"], "ingest:peer")

    def test_ingest_invalid_is_rejected_fail_closed(self):
        status, d = self.post("/api/helix/ingest", {"session": "main", "state": {"bogus": True}})
        self.assertEqual(status, 200)
        self.assertFalse(d["accepted"])
        self.assertTrue(d["violations"])
        self.assertEqual(self.get("/api/helix/history?session=main")[1], [])   # nothing adopted

    def test_audit_needs_a_transcript(self):
        self.assertEqual(self.post("/api/helix/audit", {})[0], 400)

    def test_audit_of_a_plain_log_passes(self):
        status, d = self.post("/api/helix/audit", {"text": "a plain answer with no ledger ceremony"})
        self.assertEqual(status, 200)
        self.assertEqual(d["verdict"], "PASS")
        self.assertEqual(d["findings"], [])


if __name__ == "__main__":
    unittest.main()
