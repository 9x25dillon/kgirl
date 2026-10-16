"""Checks on the quantitative content of Sources B and D that IWME Draft 1.0 cites.

Source D (Edwards & McConnell, *The Convergence Point*):
  wht / reported_tier1      Tier 1 reports Walsh coefficients [4, 0, 0, −4, 0, 4, 4, 0] for the input
                            [1, 1, 1, −1, 1, 1, 1, 1]. The transform of that input is [6, 2, 2, −2, −2, 2, 2, −2]
                            (natural order; no ordering or normalization turns it into the reported vector).
                            Both vectors carry energy 64 = N·E_in, so the reported Parseval PASS cannot detect
                            the wrong coefficients: Parseval holds for every orthogonal transform of every signal.
  z9_z3_homomorphism        "φ(a^k) = φ(a)^k" for φ = reduction mod 3 holds for all a ∈ Z9 and all k: a check that
                            cannot fail carries no information. No unital ring map Z4 → Z3 exists.
  low_order_concentration   Prediction 5 ("energy concentrated in low-order Walsh coefficients") is maximized by a
                            constant signal: it measures smoothness, the opposite of the complexity it is meant to index.
Source B (§4.8, eq. 4.10):
  pc_max_weight             PC(E) = (max_i w_i − 1/M)/(1 − 1/M) scores M identical answers 0 and M mutually orthogonal
                            answers 1 (whenever the eigensolver returns a basis vector), and is undefined for antipodal
                            answers (the Perron vector changes sign). kgirl.main.spectral_coherence already replaces it
                            with the rescaled principal share (λ_max/M − 1/M)/(1 − 1/M) on its fallback path.

numpy only.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

Array = np.ndarray

# ----------------------------------------------------------------------------------------------
# Walsh–Hadamard
# ----------------------------------------------------------------------------------------------

SOURCE_D_INPUT = np.array([1, 1, 1, -1, 1, 1, 1, 1], dtype=float)
SOURCE_D_REPORTED = np.array([4, 0, 0, -4, 0, 4, 4, 0], dtype=float)


def wht(f: Sequence[float]) -> Array:
    """Unnormalized Walsh–Hadamard transform in natural (Hadamard) order: Wf(a) = Σ_i (−1)^{popcount(a & i)} f(i)."""
    a = np.array(f, dtype=float)
    n = len(a)
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            x, y = a[i:i + h].copy(), a[i + h:i + 2 * h].copy()
            a[i:i + h], a[i + h:i + 2 * h] = x + y, x - y
        h *= 2
    return a


def iwht(F: Sequence[float]) -> Array:
    return wht(F) / len(F)


def sequency_order(F: Sequence[float]) -> Array:
    """Reorder natural-order coefficients by sequency (number of sign changes of the Walsh function)."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    H = np.array([[(-1) ** bin(a & i).count("1") for i in range(n)] for a in range(n)])
    changes = (np.diff(H, axis=1) != 0).sum(axis=1)
    return F[np.argsort(changes, kind="stable")]


def parseval_ok(f: Sequence[float], F: Sequence[float], tol: float = 1e-9) -> bool:
    f, F = np.asarray(f, float), np.asarray(F, float)
    return abs(float((F * F).sum()) - len(f) * float((f * f).sum())) < tol


def reported_tier1() -> Dict[str, object]:
    true = wht(SOURCE_D_INPUT)
    rep = SOURCE_D_REPORTED
    any_match = any(np.allclose(np.sort(np.abs(c * true)), np.sort(np.abs(rep))) for c in (1.0, 1 / 8, 1 / math.sqrt(8)))
    return {
        "input": SOURCE_D_INPUT.tolist(),
        "reported": rep.tolist(),
        "computed_natural": true.tolist(),
        "computed_sequency": sequency_order(true).tolist(),
        "reported_matches_any_ordering_or_scale": bool(any_match),
        "parseval_reported": parseval_ok(SOURCE_D_INPUT, rep),
        "parseval_computed": parseval_ok(SOURCE_D_INPUT, true),
        "signal_whose_transform_is_reported": iwht(rep).tolist(),
    }


def low_order_concentration(f: Sequence[float], k: int = 2) -> float:
    """Share of WHT energy in the k lowest-sequency coefficients."""
    F = sequency_order(wht(f))
    e = F * F
    return float(e[:k].sum() / e.sum())


# ----------------------------------------------------------------------------------------------
# Multi-field "verification"
# ----------------------------------------------------------------------------------------------

def z9_z3_homomorphism(k_max: int = 30) -> Dict[str, object]:
    """Exhaustive test of φ(a^k mod 9) = φ(a)^k mod 3 for φ(a) = a mod 3."""
    fails = [(a, k) for a in range(9) for k in range(k_max + 1) if (pow(a, k, 9) % 3) != pow(a % 3, k, 3)]
    # a unital ring homomorphism Z4 → Z3 sends 1 ↦ 1, so it would send 4·1 = 0 to 4·1 = 1 ≠ 0 in Z3
    z4_z3 = (4 % 3) == 0
    return {"checks": 9 * (k_max + 1), "failures": len(fails), "can_fail": False, "z4_to_z3_unital_hom_exists": z4_z3}


# ----------------------------------------------------------------------------------------------
# Phase coherence of model outputs (Source B eq. 4.10)
# ----------------------------------------------------------------------------------------------

def _rows_normalized(E: Array) -> Array:
    E = np.atleast_2d(np.asarray(E, dtype=float))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def pc_max_weight(E: Array) -> float:
    """Source B (4.10) as written: principal eigenvector of K = EEᵀ, normalized to Σw = 1, then
    (max w − 1/M)/(1 − 1/M). Returns nan when Σw = 0 (sign-changing Perron vector)."""
    E = _rows_normalized(E)
    M = E.shape[0]
    vals, vecs = np.linalg.eigh(E @ E.T)
    w = vecs[:, -1]
    s = w.sum()
    if abs(s) < 1e-12:
        return float("nan")
    w = w / s
    return float((w.max() - 1.0 / M) / (1.0 - 1.0 / M))


def pc_spectral_share(E: Array) -> float:
    """The corrected measure (as in kgirl.main.spectral_coherence): (λ_max/M − 1/M)/(1 − 1/M) ∈ [0, 1]."""
    E = _rows_normalized(E)
    M = E.shape[0]
    lam = float(np.linalg.eigvalsh(E @ E.T)[-1])
    return float(np.clip((lam / M - 1.0 / M) / (1.0 - 1.0 / M), 0.0, 1.0))


def pc_resultant(E: Array) -> float:
    """Sign-aware alternative: rescaled squared mean resultant length (the Kuramoto order parameter of unit
    vectors), (‖ē‖² − 1/M)/(1 − 1/M) clipped to [0, 1]. Identical → 1, orthogonal → 0, antipodal pair → 0.
    The spectral share is sign-blind (it scores an antipodal pair 1)."""
    E = _rows_normalized(E)
    M = E.shape[0]
    r2 = float(np.linalg.norm(E.mean(axis=0)) ** 2)
    return float(np.clip((r2 - 1.0 / M) / (1.0 - 1.0 / M), 0.0, 1.0))


def pc_cases(M: int = 4, d: int = 8) -> Dict[str, Dict[str, float]]:
    e = np.eye(d)
    cases = {
        "identical": np.tile(e[0], (M, 1)),
        "orthogonal": e[:M],
        "antipodal_pair": np.array([e[0], -e[0]]),
        "two_clusters": np.array([e[0], e[0], e[1], e[1]]),
    }
    return {k: {"pc_4_10": pc_max_weight(E), "pc_spectral": pc_spectral_share(E), "pc_resultant": pc_resultant(E)}
            for k, E in cases.items()}
