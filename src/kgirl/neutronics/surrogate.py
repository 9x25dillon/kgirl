"""Polynomial ridge surrogate: 39 rod enrichments → (k_inf, PPPF).

    Φ(x) = [1, z_i, z_i z_j (i ≤ j)],  z = (x − μ) / σ  (train statistics)     degree 2: 1 + 39 + 780 terms
    B    = argmin ‖Φ B − Y‖² + λ ‖B_{1:}‖²  (intercept unpenalised)            closed form, both targets at once
    λ    chosen by 5-fold cross-validation on the training split (folds = row index mod 5; deterministic)

Metrics on the held-out split: RMSE, MAE, R², max |error| per target, the same for a mean predictor
(baseline), and the k_inf error in pcm (10⁵ Δk/k, the reactor-physics unit).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import LatticeSplit

TARGETS = ("k_inf", "pppf")
LAMBDAS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def poly_terms(d: int, degree: int) -> List[tuple]:
    return [()] + [c for k in range(1, degree + 1) for c in itertools.combinations_with_replacement(range(d), k)]


def design(z: np.ndarray, terms: Sequence[tuple]) -> np.ndarray:
    phi = np.empty((z.shape[0], len(terms)))
    for j, t in enumerate(terms):
        phi[:, j] = np.prod(z[:, list(t)], axis=1) if t else 1.0
    return phi


def _ridge(phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    g = phi.T @ phi
    reg = lam * np.eye(g.shape[0])
    reg[0, 0] = 0.0                                  # intercept is not shrunk
    return np.linalg.solve(g + reg, phi.T @ y)


def metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for j, name in enumerate(TARGETS):
        e = pred[:, j] - y[:, j]
        ss_res = float((e ** 2).sum())
        ss_tot = float(((y[:, j] - y[:, j].mean()) ** 2).sum())
        out[name] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                     "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, "max_abs_err": float(np.abs(e).max())}
    k = out["k_inf"]
    k["rmse_pcm"] = 1e5 * k["rmse"] / float(y[:, 0].mean())
    k["max_abs_err_pcm"] = 1e5 * k["max_abs_err"] / float(y[:, 0].mean())
    return out


@dataclass
class Surrogate:
    degree: int = 2
    lam: float = 0.0
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    coef: Optional[np.ndarray] = None               # (n_terms, 2)
    terms: List[tuple] = field(default_factory=list)
    cv: Dict[str, float] = field(default_factory=dict)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def fit(self, train: LatticeSplit, lambdas: Sequence[float] = LAMBDAS, folds: int = 5) -> "Surrogate":
        x, y = train.enrichment, train.targets
        self.mu, self.sigma = x.mean(axis=0), x.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0
        self.terms = poly_terms(x.shape[1], self.degree)
        phi = design(self._z(x), self.terms)
        fold = np.arange(len(x)) % folds
        for lam in lambdas:                         # CV error: mean over targets of RMSE / target std
            errs = []
            for f in range(folds):
                tr, va = fold != f, fold == f
                b = _ridge(phi[tr], y[tr], lam)
                r = phi[va] @ b - y[va]
                errs.append(np.mean(np.sqrt((r ** 2).mean(axis=0)) / y.std(axis=0)))
            self.cv[f"{lam:g}"] = float(np.mean(errs))
        self.lam = float(min(lambdas, key=lambda l: self.cv[f"{l:g}"]))
        self.coef = _ridge(phi, y, self.lam)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return design(self._z(np.atleast_2d(x)), self.terms) @ self.coef

    def save(self, path: Path) -> None:
        np.savez(path, degree=self.degree, lam=self.lam, mu=self.mu, sigma=self.sigma, coef=self.coef,
                 terms=np.array([";".join(map(str, t)) for t in self.terms]))

    def limps_payload(self, x_preview: np.ndarray, test_metrics: dict) -> dict:
        """Eopiez/LIMPS payload shape (build_payload in refactored/neutronics_surrogate.py)."""
        names = ["1"] + ["*".join(f"x{i + 1}" for i in t) for t in self.terms[1:]]
        return {"matrix": x_preview.tolist(), "variables": [f"x{i + 1}" for i in range(x_preview.shape[1])],
                "degree_limit": self.degree, "min_rank": None, "structure": "polynomial_ridge",
                "coeff_threshold": 1e-8, "chebyshev": False,
                "terms": names, "lambda": self.lam,
                "targets_preview": {"n_targets_used": 2,
                                    "rmse_first10": [test_metrics[t]["rmse"] for t in TARGETS]}}


def run(train: LatticeSplit, test: LatticeSplit, degrees: Sequence[int] = (1, 2),
        out: Optional[Path] = None) -> dict:
    """Fit each degree on train, score on test; returns (and writes) the telemetry record."""
    record = {"dataset": {"train_rows": train.n, "test_rows": test.n, "rods": train.enrichment.shape[1]},
              "baseline_mean_predictor": metrics(np.tile(train.targets.mean(axis=0), (test.n, 1)), test.targets),
              "models": {}}
    best = None
    for d in degrees:
        t0 = time.perf_counter()
        s = Surrogate(degree=d).fit(train)
        fit_s = time.perf_counter() - t0
        m = metrics(s.predict(test.enrichment), test.targets)
        record["models"][f"degree{d}"] = {"terms": len(s.terms), "lambda": s.lam, "cv_error": s.cv,
                                         "fit_seconds": round(fit_s, 3), "test": m}
        if best is None or m["k_inf"]["rmse"] < best[1]["k_inf"]["rmse"]:
            best = (s, m)
    record["best"] = f"degree{best[0].degree}"
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        best[0].save(out / "surrogate.npz")
        (out / "limps_payload.json").write_text(json.dumps(best[0].limps_payload(test.enrichment[:16], best[1])))
        (out / "telemetry.json").write_text(json.dumps(record, indent=2))
    return record
