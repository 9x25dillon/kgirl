"""TA-ULS ↔ Julia bridge: spectral stability, KFP fluctuation and entropy profiles of weight matrices.

Protocol (the one `src/julia/julia_server_script.jl` and Fractal_cascade_simulation's JuliaBackend use):

    POST <url>/         {"function": "<name>", "args": [<matrix as row lists>, ...]}  →  JSON object
    POST <url>/health   {}                                                             →  {"ok": true, ...}

Server: `julia src/julia/tauls_bridge.jl` (port 9000; packages HTTP, JSON3). Every function here has a
numpy reference implementation with the same formulas, used when no server answers or a matrix is
larger than `max_elems`, so the report never depends on Julia being up. Analysis only: nothing here
modifies weights (the v1 server's `kfp_optimize_matrix` rewrote them and injected `randn` noise).

Metrics of an m×n matrix W with singular values s₁ ≥ … ≥ s_k (k = min(m, n)); variances are population
variances (Julia `var(x; corrected=false)`):

    condition          s₁ / s_k                         stable_rank   Σ sᵢ² / s₁²
    control_stability  1 / (1 + condition / 100)        learning_stability  1 / (1 + var(s²) / mean(s²))
    overall            mean of the two; class stable > 0.7, marginal > 0.4, else unstable   (v1 thresholds)
    spectral_entropy   −Σ pᵢ log pᵢ / log k,  pᵢ = sᵢ² / Σ s²
    kfp_fluctuation    mean row variance + mean column variance          (v1 KFP fluctuation intensity)
    kfp_hot_fraction   share of (i, j) with (rowvar_i + colvar_j) / 2 above it
    weight_entropy     normalised entropy of |W| / Σ|W|                   (v1 entropy_regularization)
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

DEFAULT_URL = os.environ.get("TAULS_JULIA_URL", "http://localhost:9000")


# ---------------------------------------------------------------------------- numpy reference

def stability_analysis(w: np.ndarray) -> dict[str, Any]:
    w = np.asarray(w, dtype=np.float64)
    s = np.linalg.svd(w, compute_uv=False)
    s2 = s * s
    k = len(s)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    control = 1.0 / (1.0 + cond / 100.0) if math.isfinite(cond) else 0.0
    learning = 1.0 / (1.0 + float(s2.var() / max(s2.mean(), 1e-30)))
    overall = (control + learning) / 2
    p = s2 / max(s2.sum(), 1e-30)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(k)) if k > 1 else 0.0
    return {"sigma_max": float(s[0]), "sigma_min": float(s[-1]), "condition": cond,
            "stable_rank": float(s2.sum() / max(s2[0], 1e-30)), "frobenius": float(math.sqrt(s2.sum())),
            "control_stability": control, "learning_stability": learning, "overall_stability": overall,
            "stability_class": "stable" if overall > 0.7 else "marginal" if overall > 0.4 else "unstable",
            "spectral_entropy": ent}


def kfp_profile(w: np.ndarray) -> dict[str, Any]:
    w = np.asarray(w, dtype=np.float64)
    rv = w.var(axis=1)
    cv = w.var(axis=0)
    total = float(rv.mean() + cv.mean())
    hot = float(((rv[:, None] + cv[None, :]) / 2 > total).mean())
    return {"kfp_fluctuation": total, "kfp_hot_fraction": hot, "row_var_max": float(rv.max()),
            "col_var_max": float(cv.max())}


def entropy_profile(w: np.ndarray) -> dict[str, Any]:
    a = np.abs(np.asarray(w, dtype=np.float64)).ravel()
    p = a / max(a.sum(), 1e-30)
    nz = p[p > 0]
    return {"weight_entropy": float(-(nz * np.log(nz)).sum() / math.log(len(p))) if len(p) > 1 else 0.0}


LOCAL = {"stability_analysis": stability_analysis, "kfp_profile": kfp_profile, "entropy_profile": entropy_profile}


def analyze_local(w: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in LOCAL.values():
        out.update(f(w))
    return out


# ---------------------------------------------------------------------------- client

class JuliaBridge:
    """Client for the Julia server; falls back to the numpy reference per call."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 30.0, max_elems: int = 1_000_000):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_elems = max_elems
        self._up: bool | None = None

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:      # noqa: S310 (configured local URL)
            return json.loads(r.read().decode("utf-8"))

    def available(self) -> bool:
        if self._up is None:
            try:
                self._up = bool(self._post("/health", {}).get("ok"))
            except (OSError, ValueError, urllib.error.URLError):
                self._up = False
        return self._up

    def call(self, function: str, *args: Any) -> dict:
        res = self._post("/", {"function": function, "args": [a.tolist() if isinstance(a, np.ndarray) else a
                                                                for a in args]})
        if "error" in res:
            raise RuntimeError(f"julia {function}: {res['error']}")
        return {k: math.inf if isinstance(v, float) and v >= 1e300 else v for k, v in res.items()}  # JSON has no ∞

    def analyze(self, w: np.ndarray) -> dict[str, Any]:
        """All three profiles; from Julia when it is up and the matrix is small enough, else numpy."""
        if w.size <= self.max_elems and self.available():
            out: dict[str, Any] = {}
            try:
                for name in LOCAL:
                    out.update(self.call(name, np.asarray(w, dtype=np.float64)))
                out["backend"] = "julia"
                return out
            except (OSError, ValueError, RuntimeError, urllib.error.URLError):
                self._up = False
        out = analyze_local(w)
        out["backend"] = "numpy"
        return out


# ---------------------------------------------------------------------------- model report

def weight_report(state_dict: dict, bridge: JuliaBridge | None = None, include_embedding: bool = True) -> list[dict]:
    """One row per 2-D weight matrix of a TA-ULS state dict."""
    rows = []
    for name, t in state_dict.items():
        if getattr(t, "ndim", 0) != 2 or (not include_embedding and name.startswith("embed")):
            continue
        w = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        r = bridge.analyze(w) if bridge else {**analyze_local(w), "backend": "numpy"}
        rows.append({"name": name, "shape": list(w.shape), **r})
    return rows


def summarize(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {}
    classes = [r["stability_class"] for r in rows]
    finite = [r["condition"] for r in rows if math.isfinite(r["condition"])]
    worst = min(rows, key=lambda r: r["overall_stability"])
    return {"matrices": len(rows), "mean_overall_stability": float(np.mean([r["overall_stability"] for r in rows])),
            "max_condition": max(finite) if finite else math.inf,
            "mean_stable_rank": float(np.mean([r["stable_rank"] for r in rows])),
            "unstable": classes.count("unstable"), "marginal": classes.count("marginal"),
            "worst": worst["name"], "backend": rows[0].get("backend", "numpy")}
