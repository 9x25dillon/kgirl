import json
import math
import re
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import tests.tauls  # noqa: F401  (sys.path)

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False

JL = Path(__file__).resolve().parents[2] / "src" / "julia" / "tauls_bridge.jl"


def fake_julia_server():
    """An in-process server speaking the bridge protocol (backed by the numpy reference)."""
    from kgirl.tauls.julia_bridge import LOCAL
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if self.path == "/health":
                out = {"ok": True, "service": "tauls_bridge"}
            else:
                calls.append(body["function"])
                res = LOCAL[body["function"]](np.array(body["args"][0], dtype=float))
                out = {k: (1e308 if isinstance(v, float) and math.isinf(v) else v) for k, v in res.items()}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ReferenceTests(unittest.TestCase):
    def test_metrics_on_known_matrices(self):
        from kgirl.tauls.julia_bridge import analyze_local
        q = np.linalg.qr(np.vander(np.linspace(0.1, 1, 8), 8))[0]       # orthogonal: perfectly conditioned
        r = analyze_local(q)
        self.assertAlmostEqual(r["condition"], 1.0, places=6)
        self.assertAlmostEqual(r["stable_rank"], 8.0, places=6)
        self.assertAlmostEqual(r["spectral_entropy"], 1.0, places=6)
        self.assertEqual(r["stability_class"], "stable")
        rank1 = np.outer(np.arange(1, 7.0), np.arange(1, 5.0))
        r = analyze_local(rank1)
        self.assertGreater(r["condition"], 1e6)
        self.assertEqual(r["stability_class"], "marginal" if r["overall_stability"] > 0.4 else "unstable")
        self.assertLess(r["overall_stability"], 0.5)
        self.assertAlmostEqual(r["stable_rank"], 1.0, places=6)
        self.assertAlmostEqual(analyze_local(np.ones((3, 3)))["weight_entropy"], 1.0, places=9)
        self.assertEqual(analyze_local(np.ones((3, 3)))["kfp_fluctuation"], 0.0)


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ClientTests(unittest.TestCase):
    def test_protocol_round_trip_and_fallback(self):
        from kgirl.tauls.julia_bridge import JuliaBridge, analyze_local
        srv, calls = fake_julia_server()
        try:
            w = np.vander(np.linspace(0.5, 1.5, 6), 4)
            b = JuliaBridge(f"http://127.0.0.1:{srv.server_port}")
            self.assertTrue(b.available())
            r = b.analyze(w)
            self.assertEqual(r["backend"], "julia")
            self.assertEqual(sorted(calls), ["entropy_profile", "kfp_profile", "stability_analysis"])
            local = analyze_local(w)
            for k, v in local.items():
                if isinstance(v, float):
                    self.assertAlmostEqual(r[k], v, places=9)
            sing = b.analyze(np.zeros((3, 3)))                        # ∞ condition survives JSON as 1e308
            self.assertTrue(math.isinf(sing["condition"]))
            big = JuliaBridge(b.url, max_elems=10).analyze(w)          # too large to ship: local
            self.assertEqual(big["backend"], "numpy")
        finally:
            srv.shutdown()
            srv.server_close()
        down = JuliaBridge("http://127.0.0.1:9", timeout=0.5)
        self.assertFalse(down.available())
        self.assertEqual(down.analyze(np.eye(3))["backend"], "numpy")

    def test_julia_server_matches_the_reference_surface(self):
        """Static parity: the .jl server exposes the same functions and result keys as the numpy reference."""
        from kgirl.tauls.julia_bridge import LOCAL, analyze_local
        src = JL.read_text(encoding="utf-8")
        exposed = set(re.findall(r'"(\w+)" => (\w+)', src.split("const FUNCTIONS")[1].split("\n\n")[0]))
        self.assertEqual({a for a, b in exposed}, set(LOCAL))
        jl_keys = set(re.findall(r'"(\w+)" =>', src))
        self.assertLessEqual(set(analyze_local(np.eye(3))), jl_keys)
        self.assertIn("corrected = false", src)
        self.assertNotIn("randn", src)


if __name__ == "__main__":
    unittest.main()
