"""EFL closure across scales.

The EFL recursion Φ_{n+1} = Φ_n ψ_{n+1}, with ψ the squared coherence between adjacent scales, read as an
identity, says that end-to-end coherence is the product of the adjacent coherences. For scale signals
Y_1 … Y_n with covariance matrix C:

    ψ_k     = Corr²(Y_{k−1}, Y_k)                        adjacent-scale coherence
    R(a, b) = Corr²(Y_a, Y_b) / Π_{k=a+1}^{b} ψ_k         closure ratio
    D       = abs(log2 R)                                   closure defect

Level 0 facts (recomputed in checks.py):

  * A Gaussian chain that is Markov in scale has R = 1: correlations multiply along the chain.
  * A walk with uncorrelated increments has R = 1: Cov(Y_a, Y_b) = Var(Y_a) and the product telescopes.
    The residuals of a model whose conditional mean is correct are uncorrelated, so R = 1 on a model's
    residual walk is a second-order test that the model's memory is sufficient.
  * Persistent increments give R < 1 and anti-persistent ones R > 1.
  * A coupling between non-adjacent scales moves R off 1. So does a nonlinear layer, with no such coupling,
    when coherence is linear; normal scores remove that case (`classify`).
  * A sample estimate of D is noisy when the end-to-end coherence is small. It is never read without the
    band of the same statistic under a fitted model (`surrogate_band`).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .coherence import normal_scores

Array = np.ndarray

ADJACENT_LINEAR = "adjacent, linear"
NONLINEAR_ADJACENT = "nonlinear layers, adjacent coupling"
NON_ADJACENT = "non-adjacent coupling"


def second_moments(signals: Array, weights: Optional[Array] = None) -> Array:
    """Second-moment matrix about zero of the columns of `signals` (rows weighted, default equal)."""
    x = np.asarray(signals, dtype=float)
    w = np.full(x.shape[0], 1.0 / x.shape[0]) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    return x.T @ (w[:, None] * x)


def walk_covariance(increments: Array, weights: Optional[Array] = None) -> Array:
    """Covariance matrix of the cumulative sums Y_k = e_1 + … + e_k over rows of `increments`
    (rows weighted by `weights`, default equal). Entry [a−1, b−1] is Cov(Y_a, Y_b)."""
    walk = np.cumsum(np.asarray(increments, dtype=float), axis=1)
    w = np.full(walk.shape[0], 1.0 / walk.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    centred = walk - w @ walk
    return centred.T @ (w[:, None] * centred)


def coherence(cov: Array, a: int, b: int) -> float:
    """Corr²(Y_a, Y_b) from a covariance matrix; scales are numbered from 1."""
    return float(cov[a - 1, b - 1] ** 2 / (cov[a - 1, a - 1] * cov[b - 1, b - 1]))


def adjacent_coherences(cov: Array) -> list:
    return [coherence(cov, k - 1, k) for k in range(2, cov.shape[0] + 1)]


def closure_ratio(cov: Array, a: int = 1, b: Optional[int] = None) -> float:
    """R(a, b): end-to-end coherence over the product of the adjacent coherences between scales a and b."""
    b = cov.shape[0] if b is None else b
    if not 1 <= a < b <= cov.shape[0]:
        raise ValueError("need 1 <= a < b <= number of scales")
    adjacent = math.prod(coherence(cov, k - 1, k) for k in range(a + 1, b + 1))
    return coherence(cov, a, b) / adjacent


def closure_defect(cov: Array, a: int = 1, b: Optional[int] = None) -> float:
    """D = abs(log2 R(a, b)); 0 means the EFL recursion closes between scales a and b."""
    return abs(math.log2(closure_ratio(cov, a, b)))


def local_profile(cov: Array) -> Array:
    """abs(log2 R(k − 1, k + 1)) for k = 2 … n − 1: where along the scales closure fails."""
    return np.array([closure_defect(cov, k - 1, k + 1) for k in range(2, cov.shape[0])])


def ar1_walk_covariance(phi: float, n: int) -> Array:
    """Exact covariance of a walk whose unit-variance increments have correlation φ^lag (calibration)."""
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    lower = np.tril(np.ones((n, n)))
    return lower @ (float(phi) ** lag) @ lower.T


def sample_defects(signals: Array) -> Dict[str, float]:
    """Closure defect of measured scale signals (columns), raw and after normal scores."""
    return {"raw": closure_defect(second_moments(signals)),
            "normal_score": closure_defect(second_moments(normal_scores(signals)))}


def surrogate_band(draw: Callable[[], Array], surrogates: int = 200, level: float = 0.95) -> Dict[str, float]:
    """The `level` point of both sample defects over surrogate data sets returned by `draw()`."""
    raw, scored = [], []
    for _ in range(surrogates):
        d = sample_defects(draw())
        raw.append(d["raw"])
        scored.append(d["normal_score"])
    return {"raw": float(np.quantile(raw, level)), "normal_score": float(np.quantile(scored, level))}


def classify(defects: Dict[str, float], band: Dict[str, float]) -> str:
    """The registered rule (PREREGISTRATION_PART2.md §4)."""
    if defects["normal_score"] > band["normal_score"]:
        return NON_ADJACENT
    if defects["raw"] > band["raw"]:
        return NONLINEAR_ADJACENT
    return ADJACENT_LINEAR


def layered_rows(gains: Sequence[float], injected: Sequence[float], skip: float = 0.0) -> Array:
    """Coefficient rows of a linear layered chain X_k = g_k X_{k−1} + skip · X_{k−2} + E_k over independent
    unit sources (X_0 and the E_k), with Var(X_0) = 1 and Var(E_k) = injected[k−1]."""
    n = len(gains)
    rows = np.zeros((n + 1, n + 1))
    rows[0, 0] = 1.0
    for k in range(1, n + 1):
        rows[k] = gains[k - 1] * rows[k - 1] + (skip * rows[k - 2] if k >= 2 else 0.0)
        rows[k, k] += math.sqrt(injected[k - 1])
    return rows


def layered_closure(gains: Sequence[float], injected: Sequence[float], skip: float = 0.0) -> Dict[str, object]:
    """Exact closure of a white linear layered chain. skip = 0 is adjacent coupling: R = 1."""
    rows = layered_rows(gains, injected, skip)
    cov = rows @ rows.T
    return {"closure_ratio": closure_ratio(cov), "end_to_end": coherence(cov, 1, cov.shape[0]),
            "adjacent": adjacent_coherences(cov)}
