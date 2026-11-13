"""The EFL model on fully observed scale signals, and the layered ground truths of Part 2.

Models (zero-mean Gaussian laws on the scale signals X_0 … X_7):

    chain      Markov in scale: the correlation between scales i < j is the product of the adjacent
               correlations between them. This is the EFL recursion as a model.   k = 15
    full       unstructured covariance                                              k = 36
    diagonal   no cross-scale structure                                             k = 8

Ground truths (PREREGISTRATION_PART2.md §1):

    ADJ    X_k = g_k X_{k−1} + E_k                              adjacent coupling: the chain is correct
    SKIP   X_k = g_k X_{k−1} + 0.6 X_{k−2} + E_k                a coupling between non-adjacent scales
    NL     the ADJ chain, unit variance per layer, odd layers cubed: adjacent coupling, nonlinear layers

All covariances are exact: linear algebra for ADJ and SKIP, Gaussian moments for NL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..metrics.closure import layered_rows, second_moments

Array = np.ndarray

N_SCALES = 8
GAINS: Tuple[float, ...] = (1.0, 2.0, 0.5, 1.0, 3.0, 0.7, 1.5)
PSI: Tuple[float, ...] = (0.95, 0.90, 0.80, 0.97, 0.60, 0.85, 0.75)
SKIP = 0.6
MODELS: Tuple[str, ...] = ("chain", "full", "diagonal")
PARAMETERS: Dict[str, int] = {"chain": 15, "full": 36, "diagonal": 8}


def injected_variances() -> Tuple[float, ...]:
    """Var(E_k) = g_k² Var(X_{k−1}) (1/ψ_k − 1) along the ADJ chain, so that its adjacent coherences are PSI."""
    out, var = [], 1.0
    for gain, psi in zip(GAINS, PSI):
        added = gain * gain * var * (1.0 / psi - 1.0)
        out.append(added)
        var = gain * gain * var + added
    return tuple(out)


def _correlation(cov: Array) -> Array:
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


@dataclass(frozen=True)
class LayeredTruth:
    name: str
    index: int

    def rows(self) -> Array:
        """Coefficient rows of the Gaussian chain underneath, over 8 independent unit sources."""
        return layered_rows(GAINS, injected_variances(), SKIP if self.name == "SKIP" else 0.0)

    def latent_correlation(self) -> Array:
        """Correlation matrix of the Gaussian chain underneath. Its closure is the population value of the
        normal-score statistic."""
        rows = self.rows()
        return _correlation(rows @ rows.T)

    def covariance(self) -> Array:
        """Exact second moments of the observed signals."""
        rows = self.rows()
        if self.name != "NL":
            return rows @ rows.T
        c = self.latent_correlation()
        odd = (np.arange(N_SCALES) % 2 == 1)
        both, one = np.outer(odd, odd), np.logical_xor.outer(odd, odd)
        out = np.where(both, 9.0 * c + 6.0 * c ** 3, np.where(one, 3.0 * c, c))
        out[np.diag_indices(N_SCALES)] = np.where(odd, 15.0, 1.0)
        return out

    def sample(self, normals: Array) -> Array:
        """Observed signals from rows of 8 independent standard normals."""
        rows = self.rows()
        x = np.asarray(normals, dtype=float) @ rows.T
        if self.name != "NL":
            return x
        z = x / np.sqrt(np.diag(rows @ rows.T))
        odd = (np.arange(N_SCALES) % 2 == 1)
        return np.where(odd[None, :], z ** 3, z)


def truths() -> Dict[str, LayeredTruth]:
    return {name: LayeredTruth(name, i) for i, name in enumerate(("ADJ", "SKIP", "NL"))}


def chain_covariance(variances: Array, adjacent: Array) -> Array:
    """Covariance of the scale-Markov Gaussian with the given variances and adjacent correlations."""
    n = len(variances)
    log_abs = np.concatenate([[0.0], np.cumsum(np.log(np.maximum(np.abs(adjacent), 1e-300)))])
    sign = np.concatenate([[1.0], np.cumprod(np.sign(adjacent))])
    corr = np.exp(-np.abs(np.subtract.outer(log_abs, log_abs))) * np.outer(sign, sign)
    sd = np.sqrt(np.asarray(variances, dtype=float))
    return corr[:n, :n] * np.outer(sd, sd)


def fit(name: str, signals: Array) -> Array:
    """Maximum-likelihood covariance of one model from second moments about zero."""
    s = second_moments(signals)
    if name == "full":
        return s
    if name == "diagonal":
        return np.diag(np.diag(s))
    if name == "chain":
        var = np.diag(s)
        adjacent = np.array([s[k - 1, k] / math.sqrt(var[k - 1] * var[k]) for k in range(1, s.shape[0])])
        return chain_covariance(var, adjacent)
    raise KeyError(f"unknown model {name!r}; choose from {MODELS}")


def draw(cov: Array, normals: Array) -> Array:
    """Samples of the zero-mean Gaussian with covariance `cov` from rows of standard normals."""
    return np.asarray(normals, dtype=float) @ np.linalg.cholesky(cov).T
