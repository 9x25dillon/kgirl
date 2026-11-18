"""The read-only GPU panel endpoint: nvidia-smi + Ollama /api/ps + KGIRL_MIND_CTX, with warnings.

Offline: subprocess.run and urllib.request.urlopen are mocked; a stub console. Written, not run.
"""

import http.client
import json
import os
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console import server
from kgirl.harness.console.server import Handler


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def urlopen_factory(ps_payload, show_payload=None):
    def fake(url, timeout=None):
        u = url.full_url if isinstance(url, urllib.request.Request) else url
        if u.endswith("/api/ps"):
            return FakeResp(ps_payload)
        if u.endswith("/api/show"):
            return FakeResp(show_payload or {"model_info": {"llama.context_length": 4096}})
        raise OSError("unexpected url " + u)
    return fake


class Gpu(unittest.TestCase):
    def setUp(self):
        Handler.console = SimpleNamespace(token="t", jobs=SimpleNamespace(list=lambda: []))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        env = mock.patch.dict(os.environ, {"KGIRL_MIND_CTX": "24576"})
        env.start()
        self.addCleanup(env.stop)
        self.smi = mock.patch.object(server.subprocess, "run",
                                     return_value=SimpleNamespace(returncode=0, stdout="279, 8192, 7, 28.50\n"))
        self.smi_mock = self.smi.start()
        self.addCleanup(self.smi.stop)
        self.web = mock.patch.object(server.urllib.request, "urlopen",
                                     side_effect=urlopen_factory({"models": []}))
        self.web_mock = self.web.start()
        self.addCleanup(self.web.stop)

    def get(self, path, token="t"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"X-Kgirl-Token": token} if token else {}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_gpu_needs_the_token(self):
        self.assertEqual(self.get("/api/gpu", token=None)[0], 403)

    def test_parses_nvidia_smi_and_ctx(self):
        status, data = self.get("/api/gpu")
        self.assertEqual(status, 200)
        d = json.loads(data)
        self.assertEqual(d["gpu"]["memory_used"], 279)
        self.assertEqual(d["gpu"]["memory_total"], 8192)
        self.assertEqual(d["gpu"]["temperature"], 28.5)
        self.assertEqual(d["mind_ctx"], 24576)
        self.assertEqual(d["warnings"], [])
        self.assertEqual(d["ollama"]["models"], [])

    def test_heat_alert_at_95(self):
        self.smi_mock.return_value = SimpleNamespace(returncode=0, stdout="279, 8192, 7, 96.0\n")
        d = json.loads(self.get("/api/gpu")[1])
        self.assertTrue(any("heat alert" in w for w in d["warnings"]))

    def test_model_not_fully_gpu_warns_and_reports_ctx(self):
        model = {"name": "llama3.2:3b", "size": 1000, "size_vram": 500,
                 "details": {"parameter_size": "3.2B", "quantization_level": "Q8_0"}}
        self.web_mock.side_effect = urlopen_factory({"models": [model]})
        d = json.loads(self.get("/api/gpu")[1])
        m = d["ollama"]["models"][0]
        self.assertEqual(m["gpu_pct"], 50.0)
        self.assertEqual(m["context_length"], 4096)
        self.assertTrue(any("not 100%" in w for w in d["warnings"]))

    def test_full_gpu_model_no_warning(self):
        model = {"name": "x", "size": 1000, "size_vram": 1000, "details": {}}
        self.web_mock.side_effect = urlopen_factory({"models": [model]})
        d = json.loads(self.get("/api/gpu")[1])
        self.assertEqual(d["ollama"]["models"][0]["gpu_pct"], 100.0)
        self.assertEqual(d["warnings"], [])

    def test_nvidia_smi_absent_is_a_state_not_a_crash(self):
        self.smi_mock.side_effect = FileNotFoundError("nvidia-smi")
        status, data = self.get("/api/gpu")
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(data)["gpu"])


if __name__ == "__main__":
    unittest.main()
