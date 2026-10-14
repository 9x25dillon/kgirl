"""NuRea /optimize backend: real operators, all three NuRea protocols, client fallback."""
import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

import tests.integration  # noqa: F401  (sys.path)

from kgirl import nurea_backend as nb


def _mat():
    x = np.linspace(-1, 1, 33)
    return np.stack([np.cos(3 * np.arccos(x)) + 0.5 * x, np.exp(x), np.abs(x)])


class OperatorTests(unittest.TestCase):
    def test_chebyshev_matches_numpy_chebfit_and_is_exact_on_polynomials(self):
        m = _mat()
        out, diag = nb.chebyshev_projection(m, degree=6)
        x = np.linspace(-1, 1, m.shape[1])
        for i in range(m.shape[0]):
            ref = np.polynomial.chebyshev.chebfit(x, m[i], 6)
            self.assertTrue(np.allclose(diag["coefficients"][i], ref, atol=1e-10))
        self.assertLess(np.abs(out[0] - m[0]).max(), 1e-10)           # T3 + x/2 is degree 3
        self.assertGreater(np.abs(out[2] - m[2]).max(), 1e-3)        # |x| is not a polynomial

    def test_chebyshev_sparsity_zeroes_smallest_coefficients(self):
        _, d0 = nb.chebyshev_projection(_mat(), degree=8)
        _, d1 = nb.chebyshev_projection(_mat(), degree=8, sparsity=0.5)
        self.assertEqual(d1["coefficients_kept"], d1["coefficients_total"] - round(0.5 * d1["coefficients_total"]))
        self.assertGreaterEqual(d1["residual_rms"], d0["residual_rms"])

    def test_soft_threshold_is_the_l1_prox(self):
        m = np.array([[3.0, -0.5, 1.2], [-2.0, 0.9, 0.0]])
        out, _ = nb.soft_threshold(m, 1.0)
        self.assertTrue(np.allclose(out, [[2.0, 0.0, 0.2], [-1.0, 0.0, 0.0]]))

    def test_singular_value_threshold_reduces_rank(self):
        u = np.linalg.qr(np.vander(np.linspace(0, 1, 6), 3))[0]           # 6×3, orthonormal columns
        v = np.linalg.qr(np.vander(np.linspace(-1, 2, 3), 3))[0]          # 3×3 orthogonal
        m = u @ np.diag([5.0, 2.0, 0.5]) @ v.T
        out, d = nb.singular_value_threshold(m, 1.0)
        self.assertEqual((d["rank_before"], d["rank_after"]), (3, 2))
        self.assertTrue(np.allclose(np.linalg.svd(out, compute_uv=False)[:2], [4.0, 1.0]))

    def test_missing_or_bad_matrix_is_an_error_not_random_data(self):
        for bad in (None, [], [[1.0, float("nan")]]):
            with self.assertRaises(ValueError):
                nb.optimize(bad)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = nb.make_server("127.0.0.1", 0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def test_health_endpoints(self):
        with urllib.request.urlopen(self.url + "/healthz", timeout=10) as r:
            self.assertEqual(r.read(), b"ok")
        self.assertTrue(nb.NuReaBackendClient(self.url).health()["reachable"])

    def test_three_protocols_agree(self):
        m = _mat().tolist()
        a = self._post("/optimize", {"matrix": m, "method": "chebyshev_projection", "params": {"sparsity": 0.25}})
        b = self._post("/optimize", {"matrix": m, "sparsity": 0.25})                       # hrm_julia_backend
        c = self._post("/", {"function": "chebyshev_projection", "args": [m, {"sparsity": 0.25}]})
        d = self._post("/", {"function": "optimize_matrix", "args": [m, "chebyshev_projection", {"sparsity": 0.25}]})
        for r in (b, c, d):
            self.assertTrue(np.allclose(r["optimized_matrix"], a["optimized_matrix"]))
        self.assertEqual((a["rows"], a["cols"]), (3, 33))

    def test_bad_requests_are_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/optimize", {"method": "rank"})
        self.assertEqual(ctx.exception.code, 400)

    def test_client_uses_server_then_falls_back(self):
        r = nb.NuReaBackendClient(self.url).optimize(_mat(), "rank", {"tau": 0.5})
        self.assertEqual(r["source"], self.url)
        off = nb.NuReaBackendClient("http://127.0.0.1:9", timeout=0.5).optimize(_mat(), "rank", {"tau": 0.5})
        self.assertEqual(off["source"], "local-reference")
        self.assertTrue(np.allclose(off["optimized_matrix"], r["optimized_matrix"]))


if __name__ == "__main__":
    unittest.main()


class OrchestratorCompatibilityTests(unittest.TestCase):
    """Fields and endpoints NuRea_sim/matrix_orchestrator.py's JuliaBackend relies on."""

    def test_optimize_reply_parses_as_orchestrator_response(self):
        r = nb.handle("/optimize", {"matrix": _mat().tolist(), "method": "sparsity", "params": {"lambda": 0.1},
                                    "adjacency": None})
        for key in ("objective", "matrix_opt", "iterations", "meta"):
            self.assertIn(key, r)
        m, x = _mat(), np.array(r["matrix_opt"])
        self.assertAlmostEqual(r["objective"], 0.5 * ((x - m) ** 2).sum() + 0.1 * np.abs(x).sum())

    def test_dispatch_functions_and_bases(self):
        m = _mat().tolist()
        self.assertEqual(nb.handle("/", {"function": "ping"}), {"ok": True})
        self.assertTrue(nb.handle("/health", {})["ok"])
        e = nb.handle("/", {"function": "entropy", "args": [{"data": m}]})
        self.assertTrue(e["shannon"] >= 0 and e["spectral"] >= 0 and e["compression_ratio"] >= 1)
        p = nb.handle("/", {"function": "chebyshev_project", "args": [m, 5, "legendre"]})
        x = np.linspace(-1, 1, 33)
        ref = np.polynomial.legendre.legval(x, np.polynomial.legendre.legfit(x, _mat()[1], 5))
        self.assertTrue(np.allclose(p["matrix"][1], ref, atol=1e-10))
        mono = nb.handle("/chebyshev_project", {"matrix": m, "degree": 5, "basis": "monomial"})
        self.assertTrue(np.allclose(mono["matrix"], p["matrix"], atol=1e-8))   # same span, same projection


def _load_nurea_orchestrator():
    import importlib.util
    import sys
    from kgirl.paths import find_repo
    path = find_repo("NuRea_sim") / "matrix_orchestrator.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("nurea_matrix_orchestrator", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:                         # httpx / pydantic missing
        return None


class NuReaClientEndToEndTests(unittest.TestCase):
    """NuRea_sim's own JuliaBackend client against the kgirl server: every primary endpoint, no fallbacks."""

    @classmethod
    def setUpClass(cls):
        cls.mo = _load_nurea_orchestrator()
        if cls.mo is None:
            raise unittest.SkipTest("NuRea_sim/matrix_orchestrator.py not importable")
        cls.srv = nb.make_server("127.0.0.1", 0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_orchestrator_client_round_trip(self):
        import asyncio
        mo = self.mo

        async def run():
            b = mo.JuliaBackend(self.url)
            try:
                health = await b.health_check()
                chunk = mo.MatrixChunk(id="c0", data=_mat().tolist())
                ent = await b.analyze_entropy(chunk)
                proj = await b.project_chebyshev(chunk.data, mo.PolySpec(degree=3, basis="chebyshev"))
                opt = await b.optimize(mo.OptimizeRequest(matrix=chunk.data, method="rank", params={"tau": 0.5}))
                return health, ent, proj, opt
            finally:
                await b.aclose()

        health, ent, proj, opt = asyncio.run(run())
        self.assertTrue(health["ok"])
        self.assertNotIn("fallback", health)
        self.assertEqual(ent, mo.EntropyReport(chunk_id="c0", **nb.entropy_report(_mat())))
        self.assertLess(np.abs(np.array(proj[0]) - _mat()[0]).max(), 1e-10)    # degree-3 row is exact
        self.assertEqual(opt.iterations, 1)
        self.assertEqual(opt.meta["rank_after"], nb.singular_value_threshold(_mat(), 0.5)[1]["rank_after"])


class LimpsFunctionTests(unittest.TestCase):
    """LIMPS functions (LiMp / bigLIMp protocol) with measured outputs instead of fixed scores."""

    def test_structure_fits_additive_matrices_exactly(self):
        m = np.add.outer(np.arange(5.0), 10 * np.arange(7.0)) + 3.0
        out, d = nb.additive_structure(m)
        self.assertTrue(np.allclose(out, m))
        self.assertAlmostEqual(d["r2"], 1.0)
        self.assertLess(nb.additive_structure(_mat())[1]["r2"], 0.99)

    def test_auto_selects_the_generating_structure(self):
        x = np.linspace(-1, 1, 40)
        rank2 = np.outer(np.cos(np.arange(30.0)), np.sin(3 * x)) + np.outer(np.arange(30.0) / 30, x ** 2)
        additive = np.add.outer(np.cos(np.arange(30.0)), np.sin(5 * x))
        self.assertEqual(nb.optimize(additive.tolist(), "auto")["diagnostics"]["selected"], "structure")
        sel = nb.optimize(rank2.tolist(), "auto")["diagnostics"]
        self.assertEqual(sel["selected"], "rank")
        self.assertEqual(sel["selected_setting"], {"rank": 2})

    def test_text_structure_information_measures(self):
        t = "the cat saw the dog and the dog saw the cat " * 5
        a = nb.analyze_text_structure(t)
        self.assertEqual(a["unique_words"], 5)
        self.assertAlmostEqual(a["entropy_bits"], a["text_entropy"] / np.log(2))
        self.assertTrue(0.0 < a["normalised_entropy"] <= 1.0)
        self.assertEqual(a["complexity_score"], a["normalised_entropy"])
        self.assertIsNotNone(a["zipf_exponent"])

    def test_polynomials_round_trip(self):
        p = nb.create_polynomials([[1.0, -2.0, 0.0], [0.5, 0.0, 4.0]], ["a", "b", "c"])
        self.assertEqual(p["P1"]["string"], "1*a - 2*b")
        self.assertEqual(p["P2"]["terms"], ["a", "c"])
        a = nb.analyze_polynomials(p)
        self.assertEqual((a["total_polynomials"], a["max_degree"]), (2, 1))
        self.assertTrue(0.0 <= a["complexity_score"] <= 1.0)


def _load_limp_client():
    import importlib.util
    from kgirl.paths import find_repo
    path = find_repo("LiMp") / "julia_client.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("limp_julia_client", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:                         # requests missing
        return None


class LimpClientEndToEndTests(unittest.TestCase):
    """LiMp's own JuliaClient / LIMPSJuliaIntegration against the kgirl server (no fallbacks exist there)."""

    @classmethod
    def setUpClass(cls):
        cls.jc = _load_limp_client()
        if cls.jc is None:
            raise unittest.SkipTest("LiMp/julia_client.py not importable (LIMP_PATH, <kgirl>/LiMp, ../LiMp)")
        cls.srv = nb.make_server("127.0.0.1", 0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_limps_client_round_trip(self):
        client = self.jc.JuliaClient(self.url)
        self.assertTrue(client.test_connection())
        integ = self.jc.LIMPSJuliaIntegration(client)
        m = np.add.outer(np.arange(6.0), np.arange(8.0))
        ent = integ.process_entropy_matrix(m)
        self.assertNotIn("error", ent)
        self.assertIn(ent["optimization_method"], ("rank", "structure", "sparsity"))
        lang = integ.analyze_natural_language("kgirl serves the limps protocol and the limps client reads it")
        self.assertNotIn("error", lang)
        self.assertEqual(lang["polynomial_features"]["P1"]["term_count"], 5)
        opt = integ.optimize_limps_matrix(m, target_compression=0.5)
        self.assertNotIn("error", opt)
        self.assertTrue(0.0 <= opt["achieved_compression"] <= 1.0)


class JuliaParityTests(unittest.TestCase):
    """Julia is not runnable in CI here; at least the .jl must serve every name the Python server does."""

    def test_jl_serves_the_same_functions_and_methods(self):
        from pathlib import Path
        jl = (Path(__file__).resolve().parents[2] / "src" / "julia" / "nurea_backend.jl").read_text()
        for name in ("optimize_matrix", "matrix_to_polynomials", "analyze_matrix_structure", "create_polynomials",
                     "analyze_polynomials", "analyze_text_structure", "chebyshev_project", "health_check", "ping",
                     '"/entropy"', '"/chebyshev_project"', '"/optimize"', '"/health"', '"/healthz"'):
            self.assertIn(name, jl)
        for method in nb.METHODS:
            self.assertIn(f'"{method}"', jl)
