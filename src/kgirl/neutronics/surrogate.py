"""Polynomial ridge surrogate: 39 rod enrichments → (k_inf, PPPF).

    Φ(x) = [1, z_i, z_i z_j (i ≤ j)],  z = (x − μ) / σ  (train statistics)     degree 2: 1 + 39 + 780 terms
    features="maxtype" appends five peak-sensitive summaries of x, standardized with train statistics:
           max_i x_i, mean of the three largest, std_i x_i, ½·log Σ_i exp(2 x_i) (a smooth max), max / mean
    B    = argmin ‖Φ B − Y‖² + λ ‖B_{1:}‖²  (intercept unpenalised)            closed form, both targets at once
    λ    chosen by 5-fold cross-validation on the training split (folds = row index mod 5; deterministic)

Why the extra features: PPPF is a maximum over pins, set by the hottest rod and its neighbours. A global
quadratic captures that poorly (held-out R² 0.74); the max-type summaries lift it to about 0.75–0.78 and also
lower the k_inf error (see docs/neutronics/README.md and `python -m kgirl.neutronics cv`).

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
FEATURE_SETS = ("poly", "maxtype")
MAXTYPE_NAMES = ("max", "top3_mean", "std", "lse2", "max_over_mean")
CV_SEEDS = (0.4123, 0.2718, 0.5772)


def poly_terms(d: int, degree: int) -> List[tuple]:
    return [()] + [c for k in range(1, degree + 1) for c in itertools.combinations_with_replacement(range(d), k)]


def design(z: np.ndarray, terms: Sequence[tuple]) -> np.ndarray:
    phi = np.empty((z.shape[0], len(terms)))
    for j, t in enumerate(terms):
        phi[:, j] = np.prod(z[:, list(t)], axis=1) if t else 1.0
    return phi


def max_type_features(x: np.ndarray) -> np.ndarray:
    """(n, d) designs → (n, 5): max, mean of the top 3, std, ½·log Σ exp(2x) (stable), max / mean."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    s = np.sort(x, axis=1)
    top = s[:, -1]
    lse = top + 0.5 * np.log(np.exp(2.0 * (x - top[:, None])).sum(axis=1))
    return np.stack([top, s[:, -3:].mean(axis=1), x.std(axis=1), lse, top / x.mean(axis=1)], axis=1)


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


def model_name(degree: int, features: str) -> str:
    return f"degree{degree}" + ("" if features == "poly" else f"+{features}")


@dataclass
class Surrogate:
    degree: int = 2
    features: str = "poly"
    lam: float = 0.0
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    fmu: Optional[np.ndarray] = None                # max-type feature statistics (features="maxtype")
    fsd: Optional[np.ndarray] = None
    coef: Optional[np.ndarray] = None               # (n_features, 2)
    terms: List[tuple] = field(default_factory=list)
    cv: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features not in FEATURE_SETS:
            raise ValueError(f"features must be one of {FEATURE_SETS}, got {self.features!r}")

    @property
    def n_features(self) -> int:
        return len(self.terms) + (len(MAXTYPE_NAMES) if self.features == "maxtype" else 0)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def _phi(self, x: np.ndarray) -> np.ndarray:
        phi = design(self._z(x), self.terms)
        if self.features == "maxtype":
            phi = np.hstack([phi, (max_type_features(x) - self.fmu) / self.fsd])
        return phi

    def fit(self, train: LatticeSplit, lambdas: Sequence[float] = LAMBDAS, folds: int = 5) -> "Surrogate":
        x, y = train.enrichment, train.targets
        self.mu, self.sigma = x.mean(axis=0), x.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0
        if self.features == "maxtype":
            f = max_type_features(x)
            self.fmu, self.fsd = f.mean(axis=0), f.std(axis=0)
            self.fsd[self.fsd == 0] = 1.0
        self.terms = poly_terms(x.shape[1], self.degree)
        phi = self._phi(x)
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
        return self._phi(np.atleast_2d(x)) @ self.coef

    def save(self, path: Path) -> None:
        empty = np.zeros(0)
        np.savez(path, degree=self.degree, features=self.features, lam=self.lam, mu=self.mu, sigma=self.sigma,
                 fmu=self.fmu if self.fmu is not None else empty, fsd=self.fsd if self.fsd is not None else empty,
                 coef=self.coef, terms=np.array([";".join(map(str, t)) for t in self.terms]))

    def limps_payload(self, x_preview: np.ndarray, test_metrics: dict) -> dict:
        """Eopiez/LIMPS payload shape (build_payload in refactored/neutronics_surrogate.py)."""
        names = ["1"] + ["*".join(f"x{i + 1}" for i in t) for t in self.terms[1:]]
        if self.features == "maxtype":
            names += list(MAXTYPE_NAMES)
        return {"matrix": x_preview.tolist(), "variables": [f"x{i + 1}" for i in range(x_preview.shape[1])],
                "degree_limit": self.degree, "min_rank": None,
                "structure": "polynomial_ridge" + ("" if self.features == "poly" else f"+{self.features}"),
                "coeff_threshold": 1e-8, "chebyshev": False,
                "terms": names, "lambda": self.lam,
                "targets_preview": {"n_targets_used": 2,
                                    "rmse_first10": [test_metrics[t]["rmse"] for t in TARGETS]}}


def _skill(m: dict, base: dict) -> float:
    """Mean over targets of RMSE relative to the mean predictor (lower is better)."""
    return float(np.mean([m[t]["rmse"] / base[t]["rmse"] for t in TARGETS]))


def run(train: LatticeSplit, test: LatticeSplit, degrees: Sequence[int] = (1, 2),
        out: Optional[Path] = None, features: Sequence[str] = ("poly", "maxtype")) -> dict:
    """Fit each (degree, feature set) on train, score on test; returns (and writes) the telemetry record.

    `best` minimises the mean over both targets of RMSE relative to the mean predictor.
    """
    base = metrics(np.tile(train.targets.mean(axis=0), (test.n, 1)), test.targets)
    record = {"dataset": {"train_rows": train.n, "test_rows": test.n, "rods": train.enrichment.shape[1]},
              "baseline_mean_predictor": base, "models": {}}
    best = None
    for d in degrees:
        for f in features:
            t0 = time.perf_counter()
            s = Surrogate(degree=d, features=f).fit(train)
            fit_s = time.perf_counter() - t0
            m = metrics(s.predict(test.enrichment), test.targets)
            record["models"][model_name(d, f)] = {"degree": d, "features": f, "terms": s.n_features,
                                                 "lambda": s.lam, "cv_error": s.cv,
                                                 "fit_seconds": round(fit_s, 3), "test": m}
            if best is None or _skill(m, base) < best[2]:
                best = (s, m, _skill(m, base))
    record["best"] = model_name(best[0].degree, best[0].features)
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        best[0].save(out / "surrogate.npz")
        (out / "limps_payload.json").write_text(json.dumps(best[0].limps_payload(test.enrichment[:16], best[1])))
        (out / "telemetry.json").write_text(json.dumps(record, indent=2))
    return record


def cross_validate(train: LatticeSplit, degree: int = 2, seeds: Sequence[float] = CV_SEEDS, folds: int = 5,
                   lambdas: Sequence[float] = LAMBDAS, bootstrap: int = 1000) -> dict:
    """poly vs maxtype under k-fold CV repeated over chaos seeds, plus a paired bootstrap of ΔR²(PPPF).

    Folds come from ChaosGenerator permutations (no PRNG). λ is chosen inside every outer fold by CV over the
    remaining folds; Gram matrices add over folds, so each inner fit is an exact solve, not a refit. Inputs are
    standardized once with whole-split statistics (inputs only; the targets never leak). The bootstrap
    resamples out-of-fold rows of the first seed with ChaosGenerator indices.
    """
    from ..evolve.chaos import ChaosGenerator

    x, y = train.enrichment, train.targets
    n = len(x)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    f = max_type_features(x)
    fsd = f.std(axis=0)
    fsd[fsd == 0] = 1.0
    terms = poly_terms(x.shape[1], degree)
    a = np.hstack([design((x - mu) / sd, terms), (f - f.mean(axis=0)) / fsd])
    cols = {"poly": len(terms), "maxtype": a.shape[1]}       # poly uses the leading columns
    ystd = y.std(axis=0)

    def solve(g, c, lam, p):
        reg = lam * np.eye(p)
        reg[0, 0] = 0.0
        return np.linalg.solve(g[:p, :p] + reg, c[:p])

    per_fold = {k: [] for k in cols}
    oof_first = {}
    for seed in seeds:
        fold = np.empty(n, dtype=int)
        fold[ChaosGenerator(seed).permutation(n)] = np.arange(n) % folds
        gf = [a[fold == k].T @ a[fold == k] for k in range(folds)]
        cf = [a[fold == k].T @ y[fold == k] for k in range(folds)]
        gt, ct = sum(gf), sum(cf)
        for name, p in cols.items():
            oof = np.empty_like(y)
            for k in range(folds):
                best = None
                for lam in lambdas:
                    err = 0.0
                    for j in range(folds):
                        if j != k:
                            b = solve(gt - gf[k] - gf[j], ct - cf[k] - cf[j], lam, p)
                            r = a[fold == j][:, :p] @ b - y[fold == j]
                            err += float(np.mean(np.sqrt((r ** 2).mean(axis=0)) / ystd))
                    if best is None or err < best[0]:
                        best = (err, lam)
                b = solve(gt - gf[k], ct - cf[k], best[1], p)
                oof[fold == k] = a[fold == k][:, :p] @ b
                m = metrics(oof[fold == k], y[fold == k])
                per_fold[name].append({"seed": seed, "fold": k, "lambda": best[1],
                                       "pppf_r2": m["pppf"]["r2"], "pppf_rmse": m["pppf"]["rmse"],
                                       "k_inf_r2": m["k_inf"]["r2"], "k_inf_rmse_pcm": m["k_inf"]["rmse_pcm"]})
            if seed == seeds[0]:
                oof_first[name] = oof[:, 1]

    def summary(rows):
        out = {}
        for key in ("pppf_r2", "pppf_rmse", "k_inf_r2", "k_inf_rmse_pcm"):
            v = np.array([r[key] for r in rows])
            out[key] = {"mean": float(v.mean()), "sd": float(v.std())}
        out["lambdas"] = sorted({r["lambda"] for r in rows})
        return out

    delta = np.array([q["pppf_r2"] - p["pppf_r2"] for p, q in zip(per_fold["poly"], per_fold["maxtype"])])
    yp = y[:, 1]
    e0, e1 = oof_first["poly"] - yp, oof_first["maxtype"] - yp
    gen, boot = ChaosGenerator(0.9137), []
    for start in range(0, bootstrap, 50):
        idx = gen.integers(n, (min(50, bootstrap - start), n))
        ys = yp[idx]
        sst = ((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        boot.extend(((e0[idx] ** 2).sum(axis=1) - (e1[idx] ** 2).sum(axis=1)) / sst)
    boot = np.sort(np.array(boot))
    lo, hi = int(0.025 * len(boot)), int(0.975 * len(boot)) - 1
    return {"rows": n, "degree": degree, "folds": folds, "seeds": list(seeds),
            "poly": summary(per_fold["poly"]), "maxtype": summary(per_fold["maxtype"]),
            "delta_pppf_r2": {"mean": float(delta.mean()), "min": float(delta.min()), "max": float(delta.max()),
                              "positive": int((delta > 0).sum()), "of": int(delta.size)},
            "bootstrap_delta_pppf_r2": {"point": float(((e0 ** 2).sum() - (e1 ** 2).sum()) / ((yp - yp.mean()) ** 2).sum()),
                                        "ci95": [float(boot[lo]), float(boot[hi])], "resamples": len(boot)},
            "per_fold": per_fold}
