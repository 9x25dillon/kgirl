"""NuRea matrix backend: the `/optimize` service NuRea_sim's orchestrator calls, with real operators.

NuRea_sim ships two Julia servers for this (backends/julia/server.jl, hrm_julia_backend/server.jl). Both
were placeholders:

  * `chebyshev_projection` multiplied the matrix by 0.95,
  * `rank` zeroed small entries elementwise (no SVD),
  * a missing matrix was replaced by `randn` data.

This module is the numpy reference for src/julia/nurea_backend.jl (same formulas, term for term), a
stdlib HTTP server speaking the same protocol (`python -m kgirl.nurea_backend serve`, no Julia needed),
and a client that falls back to the reference when no server answers.

Protocols (all three NuRea clients):

    GET  /healthz                                   → "ok"
    GET  /health                                    → {"status": "healthy", "backend": ..., "methods": [...]}
    POST /optimize {"matrix", "method", "params"}   → {"optimized_matrix", "rows", "cols", "method", "params",
                                                       "diagnostics"}               (backends/julia)
    POST /optimize {"matrix", "sparsity"}           → same, method chebyshev_projection  (hrm_julia_backend)
    POST /         {"function": name, "args": [...]}                                (matrix_orchestrator
                                                       JuliaBackend._dispatch: a method name with
                                                       [matrix, params?], "optimize_matrix" with
                                                       [matrix, method, params], "optimize" with
                                                       [OptimizeRequest], "entropy" with [{"data"}],
                                                       "chebyshev_project" with [matrix, degree, basis],
                                                       "ping")
    POST /health   {}                               → {"ok": true, ...}           (matrix_orchestrator)
    POST /entropy  {"data"}                         → {"shannon", "spectral", "compression_ratio"}
    POST /chebyshev_project {"matrix","degree","basis"} → {"matrix", ...}

/optimize replies carry both field sets: NuRea's backends/julia names (optimized_matrix, rows, cols) and
matrix_orchestrator's OptimizeResponse (objective, matrix_opt, iterations, meta). The NuRea servers sent
only the first, so the orchestrator could not parse their replies. Method "poly" is an alias of
chebyshev_projection (params degree, basis).

Fixes relative to matrix_orchestrator's local fallbacks:
    entropy_local      used np.histogram(density=True) values as probabilities (densities, so the
                       "entropy" could be negative); here p = counts / total
    chebyshev_local    averaged T_k(z) of each entry on its own (no projection); here a least-squares fit

Operators on an m×n matrix M (rows are signals sampled on n equispaced points of [−1, 1]):

    chebyshev_projection  least-squares fit of T₀…T_d per row (d = min(degree, n−1), default 8); the
                          `sparsity` fraction of smallest |coefficients| (all rows together) is zeroed;
                          returns the reconstruction V·C. Diagnostics: residual RMS, coefficients kept.
    sparsity              proximal operator of λ‖M‖₁: sign(x)·max(|x| − λ, 0)        (soft threshold)
    rank                  proximal operator of τ‖M‖_*: U·diag(max(σ − τ, 0))·Vᵀ     (singular value threshold)
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import numpy as np

DEFAULT_URL = os.environ.get("NUREA_JULIA_URL", "http://localhost:9000")
METHODS = ("chebyshev_projection", "sparsity", "rank")


# ---------------------------------------------------------------------------- numpy reference

BASES = ("chebyshev", "legendre", "monomial")


def vandermonde(n: int, degree: int, basis: str = "chebyshev") -> np.ndarray:
    """V[i, k] = φ_k(x_i), x = linspace(−1, 1, n), by recurrence:
    Chebyshev T_k = 2x T_{k−1} − T_{k−2};  Legendre P_k = ((2k−1) x P_{k−1} − (k−1) P_{k−2}) / k;  monomial x^k."""
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}; one of {', '.join(BASES)}")
    x = np.linspace(-1.0, 1.0, n)
    v = np.empty((n, degree + 1))
    v[:, 0] = 1.0
    if degree >= 1:
        v[:, 1] = x
    for k in range(2, degree + 1):
        if basis == "chebyshev":
            v[:, k] = 2.0 * x * v[:, k - 1] - v[:, k - 2]
        elif basis == "legendre":
            v[:, k] = ((2 * k - 1) * x * v[:, k - 1] - (k - 1) * v[:, k - 2]) / k
        else:
            v[:, k] = x * v[:, k - 1]
    return v


def chebyshev_vandermonde(n: int, degree: int) -> np.ndarray:
    return vandermonde(n, degree, "chebyshev")


def chebyshev_projection(m: np.ndarray, degree: int = 8, sparsity: float = 0.0,
                         basis: str = "chebyshev") -> Tuple[np.ndarray, Dict[str, Any]]:
    m = np.asarray(m, dtype=np.float64)
    rows, n = m.shape
    d = int(max(0, min(degree, n - 1)))
    v = vandermonde(n, d, basis)
    coef, *_ = np.linalg.lstsq(v, m.T, rcond=None)                    # (d+1, rows)
    k = int(round(float(sparsity) * coef.size)) if sparsity > 0 else 0
    k = min(k, coef.size)
    if k:
        order = np.argsort(np.abs(coef).ravel(), kind="stable")[:k]   # row-major, stable
        flat = coef.ravel().copy()
        flat[order] = 0.0
        coef = flat.reshape(coef.shape)
    out = (v @ coef).T
    resid = float(np.sqrt(np.mean((out - m) ** 2)))
    return out, {"degree": d, "basis": basis, "coefficients": coef.T.tolist(), "coefficients_kept": int(coef.size - k),
                 "coefficients_total": int(coef.size), "residual_rms": resid}


def entropy_report(data: Any) -> Dict[str, float]:
    """shannon: entropy (bits) of a 64-bin histogram of the values (p = counts / total);
    spectral: entropy (bits) of σ / Σσ;  compression_ratio: size / numerical rank."""
    a = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == 0:
        return {"shannon": 0.0, "spectral": 0.0, "compression_ratio": 1.0}
    a = np.atleast_2d(a)
    counts, _ = np.histogram(a, bins=64)
    p = counts[counts > 0] / counts.sum()
    shannon = float(-(p * np.log2(p)).sum())
    s = np.linalg.svd(a, compute_uv=False)
    ps = s[s > 0] / s.sum() if s.sum() > 0 else np.array([])
    spectral = float(-(ps * np.log2(ps)).sum()) if ps.size else 0.0
    tol = max(a.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    rank = int((s > tol).sum())
    return {"shannon": shannon, "spectral": spectral, "compression_ratio": float(a.size / max(rank, 1))}


def soft_threshold(m: np.ndarray, lam: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    m = np.asarray(m, dtype=np.float64)
    out = np.sign(m) * np.maximum(np.abs(m) - float(lam), 0.0)
    return out, {"lambda": float(lam), "nonzero_fraction": float(np.mean(out != 0.0)),
                 "l1_before": float(np.abs(m).sum()), "l1_after": float(np.abs(out).sum())}


def singular_value_threshold(m: np.ndarray, tau: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    m = np.asarray(m, dtype=np.float64)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    s2 = np.maximum(s - float(tau), 0.0)
    out = (u * s2) @ vt
    tol = max(m.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    return out, {"tau": float(tau), "rank_before": int((s > tol).sum()), "rank_after": int((s2 > tol).sum()),
                 "nuclear_before": float(s.sum()), "nuclear_after": float(s2.sum())}


def optimize(matrix: Any, method: str = "chebyshev_projection", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The /optimize response for `matrix` (list of rows). Raises ValueError on bad input.

    objective (what the operator minimises, evaluated at the result X):
      chebyshev_projection  ½‖X − M‖²_F            sparsity  ½‖X − M‖²_F + λ‖X‖₁      rank  ½‖X − M‖²_F + τ‖X‖_*
    All three are closed-form, so iterations = 1.
    """
    if matrix is None:
        raise ValueError("matrix is required")          # the NuRea servers substituted randn data here
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.size == 0:
        raise ValueError(f"matrix must be a non-empty list of equal-length rows, got shape {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("matrix contains non-finite values")
    p = dict(params or {})
    if method == "poly":
        method = "chebyshev_projection"
    if method == "chebyshev_projection":
        out, diag = chebyshev_projection(m, int(p.get("degree", 8)), float(p.get("sparsity", 0.0)),
                                         str(p.get("basis", "chebyshev")))
        objective = 0.5 * float(((out - m) ** 2).sum())
    elif method == "sparsity":
        lam = float(p.get("lambda", 1.0))
        out, diag = soft_threshold(m, lam)
        objective = 0.5 * float(((out - m) ** 2).sum()) + lam * float(np.abs(out).sum())
    elif method == "rank":
        tau = float(p.get("tau", 1.0))
        out, diag = singular_value_threshold(m, tau)
        objective = 0.5 * float(((out - m) ** 2).sum()) + tau * diag["nuclear_after"]
    else:
        raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS + ('poly',))}")
    rows = out.tolist()
    return {"optimized_matrix": rows, "rows": int(out.shape[0]), "cols": int(out.shape[1]),
            "method": method, "params": p, "diagnostics": diag,
            "objective": objective, "matrix_opt": rows, "iterations": 1, "meta": {"method": method, **diag}}


def chebyshev_project(matrix: Any, degree: int = 3, basis: str = "chebyshev") -> Dict[str, Any]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("matrix must be a non-empty list of equal-length rows")
    out, diag = chebyshev_projection(m, int(degree), 0.0, basis)
    return {"matrix": out.tolist(), "degree": diag["degree"], "basis": basis, "residual_rms": diag["residual_rms"]}


def handle(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Route a POST body exactly as the Julia server does (shared by the stdlib server and the tests)."""
    if path == "/health":
        return {"ok": True, "status": "healthy", "backend": "python-reference", "methods": list(METHODS)}
    if path == "/optimize":
        if "method" in body or "params" in body:
            return optimize(body.get("matrix"), body.get("method", "chebyshev_projection"), body.get("params"))
        return optimize(body.get("matrix"), "chebyshev_projection", {"sparsity": body.get("sparsity", 0.0)})
    if path == "/entropy":
        return entropy_report(body.get("data", []))
    if path == "/chebyshev_project":
        return chebyshev_project(body.get("matrix"), body.get("degree", 3), body.get("basis", "chebyshev"))
    if path == "/":
        name, args = body.get("function"), list(body.get("args", []))
        if name == "ping":
            return {"ok": True}
        if not args:
            raise ValueError("args are required")
        if name == "optimize":
            req = args[0]
            return optimize(req.get("matrix"), req.get("method", "chebyshev_projection"), req.get("params"))
        if name == "entropy":
            return entropy_report(args[0].get("data", []) if isinstance(args[0], dict) else args[0])
        if name == "chebyshev_project":
            return chebyshev_project(args[0], args[1] if len(args) > 1 else 3, args[2] if len(args) > 2 else "chebyshev")
        if name == "optimize_matrix":
            return optimize(args[0], args[1] if len(args) > 1 else "chebyshev_projection",
                            args[2] if len(args) > 2 else None)
        if name in METHODS + ("poly",):
            return optimize(args[0], name, args[1] if len(args) > 1 else None)
        raise ValueError(f"unknown function {name!r}")
    raise LookupError(path)


# ---------------------------------------------------------------------------- stdlib server

class _Handler(BaseHTTPRequestHandler):
    server_version = "kgirl-nurea/1"

    def _send(self, status: int, payload: Any, ctype: str = "application/json") -> None:
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            return self._send(200, "ok", "text/plain")
        if self.path == "/health":
            return self._send(200, {"status": "healthy", "backend": "python-reference", "methods": list(METHODS)})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 64 * 1024 * 1024:
                return self._send(413, {"error": "body too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, handle(self.path, body))
        except LookupError:
            self._send(404, {"error": "not found"})
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self._send(400, {"error": str(e)})

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def make_server(host: str = "127.0.0.1", port: int = 9000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


# ---------------------------------------------------------------------------- client

class NuReaBackendClient:
    """POSTs to a NuRea backend (Julia or the stdlib server); computes locally when none answers."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 10.0, fallback: bool = True):
        self.url, self.timeout, self.fallback = url.rstrip("/"), timeout, fallback

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def health(self) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=self.timeout) as r:
                return {"reachable": True, **json.loads(r.read())}
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {"reachable": False, "error": str(e)}

    def optimize(self, matrix: Any, method: str = "chebyshev_projection",
                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = {"matrix": np.asarray(matrix, dtype=float).tolist(), "method": method, "params": params or {}}
        try:
            res = self._post("/optimize", body)
            res["source"] = self.url
            return res
        except urllib.error.HTTPError as e:              # the server answered: a real error, do not mask it
            raise ValueError(e.read().decode(errors="replace")) from e
        except (urllib.error.URLError, OSError):
            if not self.fallback:
                raise
            res = optimize(body["matrix"], method, body["params"])
            res["source"] = "local-reference"
            return res


def _main(argv: list) -> int:
    if len(argv) >= 1 and argv[0] == "serve":
        port = int(argv[1]) if len(argv) > 1 else int(os.environ.get("JULIA_PORT", "9000"))
        host = os.environ.get("JULIA_HOST", "127.0.0.1")
        print(f"kgirl NuRea backend (python reference) on {host}:{port}: GET /healthz /health, POST /optimize /")
        make_server(host, port).serve_forever()
        return 0
    print("usage: python -m kgirl.nurea_backend serve [port]")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
