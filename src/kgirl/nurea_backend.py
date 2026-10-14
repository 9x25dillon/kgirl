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
METHODS = ("chebyshev_projection", "sparsity", "rank", "structure", "auto")


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


def additive_structure(m: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Least-squares fit M ≈ μ + r_i + c_j (two-way additive model; LiMp's "structure" method).

    r_i = row mean − μ, c_j = column mean − μ. LiMp reported a constant complexity_reduction = 0.3;
    here the fit quality is measured: R² = 1 − ‖M − X‖² / ‖M − μ‖².
    """
    m = np.asarray(m, dtype=np.float64)
    mu = float(m.mean())
    out = m.mean(axis=1, keepdims=True) + m.mean(axis=0, keepdims=True) - mu
    sst = float(((m - mu) ** 2).sum())
    sse = float(((m - out) ** 2).sum())
    return out, {"grand_mean": mu, "r2": 1.0 - sse / sst if sst > 0 else 1.0,
                 "row_effect_var": float(m.mean(axis=1).var()), "col_effect_var": float(m.mean(axis=0).var()),
                 "interaction_rms": float(np.sqrt(sse / m.size)), "parameters": int(m.shape[0] + m.shape[1] - 1)}


def truncated_svd(m: np.ndarray, rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(np.asarray(m, dtype=np.float64), full_matrices=False)
    r = int(max(0, min(rank, len(s))))
    return (u[:, :r] * s[:r]) @ vt[:r]


def _bic(m: np.ndarray, x: np.ndarray, k: int) -> float:
    """Gaussian-residual BIC: N·ln(RSS/N) + k·ln N (RSS floored at 1e−12 of the total sum of squares)."""
    n = m.size
    tss = float(((m - m.mean()) ** 2).sum()) or 1.0
    rss = max(float(((m - x) ** 2).sum()), 1e-12 * tss)
    return n * math.log(rss / n) + k * math.log(n)


def auto_select(m: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pick the representation with the lowest BIC (a description-length criterion).

    LiMp chose a method from fixed thresholds on a complexity score. Here every candidate is fitted
    and scored, and its parameter count is charged:
        structure                 k = m + n − 1
        rank-r truncated SVD      k = r (m + n − r)               r = 1 … min(m, n) − 1, best r
        chebyshev degree d        k = m (d + 1)                   d = 0 … min(n − 1, 12), best d
        sparsity, top-s entries   k = 2s (value + position)       s on a log grid, best s
    """
    m = np.asarray(m, dtype=np.float64)
    rows, cols = m.shape
    cands: Dict[str, Tuple[float, np.ndarray, Dict[str, Any]]] = {}
    xs, ds = additive_structure(m)
    cands["structure"] = (_bic(m, xs, ds["parameters"]), xs, {"r2": ds["r2"]})
    u, sv, vt = np.linalg.svd(m, full_matrices=False)
    best = None
    top = max(1, min(rows, cols) - 1)
    for r in sorted({int(round(v)) for v in np.geomspace(1, top, num=min(24, top))}):   # ≤ 24 candidates
        x = (u[:, :r] * sv[:r]) @ vt[:r]
        b = _bic(m, x, r * (rows + cols - r))
        if best is None or b < best[0]:
            best = (b, x, {"rank": r})
    cands["rank"] = best
    best = None
    for d in range(0, min(cols - 1, 12) + 1):
        x, dd = chebyshev_projection(m, d)
        b = _bic(m, x, rows * (d + 1))
        if best is None or b < best[0]:
            best = (b, x, {"degree": d})
    cands["chebyshev_projection"] = best
    flat = np.abs(m).ravel()
    order = np.argsort(-flat, kind="stable")
    best = None
    for sz in sorted({max(1, int(round(m.size * f))) for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)}):
        x = np.zeros(m.size)
        x[order[:sz]] = m.ravel()[order[:sz]]
        x = x.reshape(m.shape)
        b = _bic(m, x, 2 * sz)
        if best is None or b < best[0]:
            best = (b, x, {"kept": sz})
    cands["sparsity"] = best
    choice = min(cands, key=lambda k: cands[k][0])
    return cands[choice][1], {"selected": choice, "selected_setting": cands[choice][2],
                              "bic": {k: v[0] for k, v in cands.items()},
                              "settings": {k: v[2] for k, v in cands.items()}}


def matrix_structure(m: np.ndarray) -> Dict[str, Any]:
    """LIMPS analyze_matrix_structure / matrix_to_polynomials metrics, measured (no fixed scores).

    complexity_score ∈ [0, 1] is the normalised spectral entropy of σ²: 0 = rank one, 1 = all
    directions equally important (LiMp summed fixed weights on threshold tests).
    """
    m = np.asarray(m, dtype=np.float64)
    rows, cols = m.shape
    s = np.linalg.svd(m, compute_uv=False)
    tol = max(m.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    s2 = s ** 2
    p = s2[s2 > 0] / s2.sum() if s2.sum() > 0 else np.array([])
    ent = float(-(p * np.log(p)).sum() / math.log(len(s))) if len(s) > 1 and p.size else 0.0
    return {"shape": [rows, cols], "matrix_shape": [rows, cols], "polynomial_terms": int(rows * cols),
            "representation": "hybrid_polynomial", "rank": int((s > tol).sum()),
            "stable_rank": float(s2.sum() / s2[0]) if s.size and s[0] > 0 else 0.0,
            "condition_number": float(s[0] / s[-1]) if s.size and s[-1] > 0 else 1e308,   # JSON has no Infinity
            "sparsity": float(np.mean(m == 0.0)),
            "near_zero_fraction": float(np.mean(np.abs(m) < 0.01 * (np.abs(m).max() or 1.0))),
            "additive_r2": additive_structure(m)[1]["r2"], "complexity_score": ent,
            "coeff_matrix": m.tolist()}


def create_polynomials(data: Any, variables: list) -> Dict[str, Any]:
    """Row i of data → the linear form Σ_j a_ij·x_j over `variables` (LIMPS create_polynomials)."""
    m = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if m.shape[1] != len(variables):
        raise ValueError(f"{m.shape[1]} columns but {len(variables)} variables")
    out = {}
    for i, row in enumerate(m, start=1):
        terms = [f"{a:g}*{v}" for a, v in zip(row, variables) if a != 0.0]
        out[f"P{i}"] = {"string": " + ".join(terms).replace("+ -", "- ") or "0", "coeffs": [float(a) for a in row if a != 0.0],
                        "terms": [t.split("*", 1)[1] for t in terms], "degree": 1 if terms else 0, "term_count": len(terms)}
    return out


def analyze_polynomials(polys: Dict[str, Any]) -> Dict[str, Any]:
    """Counts and a bounded complexity score: mean over polynomials of the normalised entropy of |coeffs|.

    Accepts create_polynomials output, or a matrix_to_polynomials result (LiMp's process_entropy_matrix passes
    that): its rows are then read as linear forms.
    """
    if isinstance(polys, dict) and "coeff_matrix" in polys:
        m = np.asarray(polys["coeff_matrix"], dtype=np.float64)
        polys = create_polynomials(m, [f"x{j + 1}" for j in range(m.shape[1])])
    polys = {k: v for k, v in (polys or {}).items() if isinstance(v, dict)}
    if not polys:
        return {"total_polynomials": 0, "complexity_score": 0.0}
    degrees = [int(p.get("degree", 0)) for p in polys.values()]
    counts = [int(p.get("term_count", len(p.get("terms", [])))) for p in polys.values()]
    ents = []
    for p in polys.values():
        c = np.abs(np.asarray(p.get("coeffs", []), dtype=np.float64))
        if c.size > 1 and c.sum() > 0:
            q = c[c > 0] / c.sum()
            ents.append(float(-(q * np.log(q)).sum() / math.log(c.size)))
        else:
            ents.append(0.0)
    return {"total_polynomials": len(polys), "average_degree": float(np.mean(degrees)), "max_degree": int(max(degrees)),
            "average_terms": float(np.mean(counts)), "max_terms": int(max(counts)),
            "complexity_score": float(np.mean(ents))}


def analyze_text_structure(text: str, top: int = 50) -> Dict[str, Any]:
    """Word statistics with information measures (LIMPS analyze_text_structure, corrected).

    text_entropy is in nats (LiMp's unit), entropy_bits in bits, normalised_entropy = H / ln V ∈ [0, 1]
    (LiMp's complexity_score was text_length / 100, unbounded). zipf_exponent: slope s of
    log f(rank) ≈ −s·log rank + c over words seen at least twice (natural language: s ≈ 1).
    """
    words = text.split()
    n = len(words)
    if not n:
        return {"text_length": len(text), "word_count": 0, "unique_words": 0, "text_entropy": 0.0,
                "entropy_bits": 0.0, "normalised_entropy": 0.0, "complexity_score": 0.0}
    vocab, counts = np.unique(np.array(words, dtype=object), return_counts=True)
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    order = np.argsort(-counts, kind="stable")
    f = counts[order].astype(np.float64)
    zipf = None
    rep = f[f >= 2]
    if rep.size >= 3:
        slope, _ = np.polyfit(np.log(np.arange(1, rep.size + 1)), np.log(rep), 1)
        zipf = float(-slope)
    norm = h / math.log(len(vocab)) if len(vocab) > 1 else 0.0
    return {"text_length": len(text), "word_count": n, "unique_words": int(len(vocab)),
            "average_word_length": float(np.mean([len(w) for w in words])),
            "text_entropy": h, "entropy_bits": h / math.log(2), "normalised_entropy": norm,
            "vocabulary_richness": len(vocab) / n, "zipf_exponent": zipf, "complexity_score": norm,
            "word_frequency": {str(vocab[i]): int(counts[i]) for i in order[:top]}}


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
        kept = diag["coefficients_kept"]
    elif method == "sparsity":
        lam = float(p["lambda"]) if "lambda" in p else 0.1 * float(np.abs(m).max())   # default: LiMp's 10 % of max
        out, diag = soft_threshold(m, lam)
        objective = 0.5 * float(((out - m) ** 2).sum()) + lam * float(np.abs(out).sum())
        kept = int(np.count_nonzero(out))
    elif method == "rank":
        if "tau" in p:
            tau = float(p["tau"])
            out, diag = singular_value_threshold(m, tau)
            objective = 0.5 * float(((out - m) ** 2).sum()) + tau * diag["nuclear_after"]
        else:                                            # no τ: rank chosen by BIC, hard truncation
            sel = auto_select(m)[1]["settings"]["rank"]["rank"]
            out = truncated_svd(m, sel)
            diag = {"rank_selected_by_bic": sel}
            objective = 0.5 * float(((out - m) ** 2).sum())
        r = diag.get("rank_after", diag.get("rank_selected_by_bic", 0))
        kept = int(r * (m.shape[0] + m.shape[1] - r))
    elif method == "structure":
        out, diag = additive_structure(m)
        objective = 0.5 * float(((out - m) ** 2).sum())
        kept = diag["parameters"]
    elif method == "auto":
        out, diag = auto_select(m)
        objective = 0.5 * float(((out - m) ** 2).sum())
        kept = None
    else:
        raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS + ('poly',))}")
    rows = out.tolist()
    if kept is not None:
        diag["compression_ratio"] = float(max(0.0, 1.0 - kept / m.size))   # LiMp field: share of numbers saved
    return {"optimized_matrix": rows, "rows": int(out.shape[0]), "cols": int(out.shape[1]),
            "method": method, "params": p, "diagnostics": diag, "compression_ratio": diag.get("compression_ratio"),
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
        if name in ("ping", "health_check"):
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
        if name in ("matrix_to_polynomials", "analyze_matrix_structure"):
            return matrix_structure(np.asarray(args[0], dtype=np.float64))
        if name == "create_polynomials":
            return create_polynomials(args[0], list(args[1]) if len(args) > 1 else
                                      [f"x{j + 1}" for j in range(np.atleast_2d(args[0]).shape[1])])
        if name == "analyze_polynomials":
            return analyze_polynomials(args[0])
        if name == "analyze_text_structure":
            return analyze_text_structure(str(args[0]))
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
        if self.path == "/":                                    # LiMp JuliaClient.test_connection
            return self._send(200, handle("/health", {}))
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
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:                                  # never drop the connection without a reply
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

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
