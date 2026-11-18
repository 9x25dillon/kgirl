"""The console's CSRF guards and the critic_host allowlist.

Offline: a real ThreadingHTTPServer on an ephemeral port over a stub console (no Assistant, no model).
A cross-site page must not be able to drive the console: it cannot read the per-launch token (same-origin
only), a simple Content-Type POST is refused, and a foreign Origin or Host is refused. Written, not run.
"""

import http.client
import json
import os
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.server import Handler, _critic_host_allowed


class Guards(unittest.TestCase):
    def setUp(self):
        self.token = "test-launch-token-0123456789"
        Handler.console = SimpleNamespace(token=self.token, jobs=SimpleNamespace(list=lambda: []))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def request(self, method, path, body=None, token=None, origin=None, ctype=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if host is not None:
            headers["Host"] = host
        if token is not None:
            headers["X-Kgirl-Token"] = token
        if origin is not None:
            headers["Origin"] = origin
        if ctype is not None:
            headers["Content-Type"] = ctype
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_get_without_token_is_refused(self):
        self.assertEqual(self.request("GET", "/api/jobs")[0], 403)

    def test_get_with_wrong_token_is_refused(self):
        self.assertEqual(self.request("GET", "/api/jobs", token="not-the-token")[0], 403)

    def test_get_with_token_is_served(self):
        status, data = self.request("GET", "/api/jobs", token=self.token)
        self.assertEqual((status, json.loads(data)), (200, []))

    def test_the_sse_stream_needs_the_token_too(self):
        self.assertEqual(self.request("GET", "/api/events")[0], 403)

    def test_post_without_token_is_refused(self):
        self.assertEqual(self.request("POST", "/api/helix/run", body={"task": "x"})[0], 403)

    def test_a_simple_content_type_post_is_refused(self):
        status, _ = self.request("POST", "/api/helix/run", body={"task": "x"}, token=self.token,
                                 ctype="text/plain")
        self.assertEqual(status, 415)

    def test_a_foreign_origin_is_refused(self):
        for method, path, kw in (("GET", "/api/jobs", {}),
                                 ("POST", "/api/helix/run", {"body": {"task": "x"}})):
            with self.subTest(method=method):
                status, _ = self.request(method, path, token=self.token, origin="http://evil.example", **kw)
                self.assertEqual(status, 403)

    def test_the_consoles_own_origin_is_served(self):
        for origin in (f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"):
            with self.subTest(origin=origin):
                self.assertEqual(self.request("GET", "/api/jobs", token=self.token, origin=origin)[0], 200)

    def test_a_foreign_host_is_refused(self):
        self.assertEqual(self.request("GET", "/api/jobs", token=self.token, host="evil.example")[0], 403)

    def test_the_token_bearing_index_is_behind_the_host_guard(self):
        # DNS rebinding: a foreign Host must not even fetch index.html (which carries the launch token).
        self.assertEqual(self.request("GET", "/", host="evil.example")[0], 403)
        self.assertEqual(self.request("GET", "/static/app.js", host="evil.example")[0], 403)

    def test_a_guarded_post_reaches_the_endpoint(self):
        status, data = self.request("POST", "/api/helix/run", body={}, token=self.token)
        self.assertEqual(status, 400)                       # past the guards: "helix run needs a task"
        self.assertIn("task", json.loads(data)["error"])

    def test_delete_is_guarded(self):
        self.assertEqual(self.request("DELETE", "/api/conversations/1")[0], 403)

    def test_builder_run_needs_a_spec(self):
        status, data = self.request("POST", "/api/builder/run", body={}, token=self.token)
        self.assertEqual(status, 400)
        self.assertIn("spec", json.loads(data)["error"])

    def test_helix_run_refuses_a_foreign_critic_host(self):
        status, data = self.request("POST", "/api/helix/run", token=self.token,
                                    body={"task": "x", "critic_host": "http://evil.example:11434"})
        self.assertEqual(status, 400)
        self.assertIn("critic_host", json.loads(data)["error"])

    def test_index_html_carries_the_launch_token(self):
        status, data = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(self.token.encode(), data)
        self.assertNotIn(b"__KGIRL_TOKEN__", data)

    def test_the_token_is_not_in_the_other_static_files(self):
        status, data = self.request("GET", "/static/app.js")
        self.assertEqual(status, 200)
        self.assertNotIn(self.token.encode(), data)


class CriticHostAllowlist(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("KGIRL_CRITIC_HOSTS", None)

    def test_loopback_is_allowed(self):
        for host in ("http://127.0.0.1:11435", "http://localhost:11434", "http://[::1]:11434"):
            with self.subTest(host=host):
                self.assertTrue(_critic_host_allowed(host))

    def test_a_foreign_host_is_refused(self):
        self.assertFalse(_critic_host_allowed("http://evil.example:11434"))

    def test_the_operator_allowlist_is_honored(self):
        os.environ["KGIRL_CRITIC_HOSTS"] = "http://pocket.local:11434, trusted.host"
        self.assertTrue(_critic_host_allowed("http://pocket.local:11434"))
        self.assertTrue(_critic_host_allowed("http://trusted.host:11434"))
        self.assertFalse(_critic_host_allowed("http://evil.example:11434"))


if __name__ == "__main__":
    unittest.main()
