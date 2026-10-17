"""Representation and weight geometry: the measurable invariants a vessel's identity is made of.

Every quantity here is a function of a probe activation matrix X (n probes × d features) or of a weight
matrix W, and every one declares the transformation group it is invariant under. Invariance is what makes a
quantity an *identity* rather than a coordinate: a quantity that changes when the hidden units are merely
permuted describes the substrate's bookkeeping, not what the substrate holds.

    centered Gram       G(X) = H X Xᵀ H / ‖H X Xᵀ H‖_F         H = I − 11ᵀ/n
                        invariant under X → c·X·Q + 1bᵀ  (Q orthogonal, c > 0, any width d — Q may be d × d′
                        with orthonormal columns), so two networks of different width share one G-space
    linear CKA          ⟨G(X), G(Y)⟩_F                          Kornblith et al. 2019 (Level 0)
    ∂CKA/∂X             2 · M · X,  M = (A − ⟨G,A⟩·G) / ‖H X Xᵀ H‖_F      for a fixed normalized anchor A
    sample subspace     top-k left singular vectors of H X (n × k): lives in probe space, so it too is
                        width- and rotation-free; compared by principal angles (Björck & Golub 1973)
    effective rank      exp(entropy of normalized singular values)  (Roy & Vetterli 2007)
    tail exponent α     power-law exponent of the eigenvalue tail of WᵀW (Clauset–Shalizi–Newman MLE with
                        KS-chosen x_min). Heavy-tailed self-regularization (Martin & Mahoney 2021) reports
                        α ≈ 2–4 in well-trained layers and larger α in random-like ones. Reading α as a health
                        signal for small models is Level 2.

Complexity: Gram O(n²d), CKA O(n²), gradient O(n²d), subspace O(n·d·min(n,d)), α O(m²) for m eigenvalues.
Pure numpy; every function is deterministic.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------------------------- Gram / CKA

def center_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def centered_gram(X: np.ndarray) -> np.ndarray:
    """H X Xᵀ H, computed as (HX)(HX)ᵀ (one centering pass, symmetric by construction)."""
    Xc = center_rows(X)
    K = Xc @ Xc.T
    return (K + K.T) / 2.0


def normalized_gram(X: np.ndarray) -> np.ndarray:
    """G(X): centered Gram with unit Frobenius norm. A zero matrix stays zero (a collapsed representation)."""
    K = centered_gram(X)
    s = float(np.linalg.norm(K))
    return K / s if s > EPS else np.zeros_like(K)


def gram_alignment(A: np.ndarray, B: np.ndarray) -> float:
    """⟨A, B⟩_F for two normalized Grams: the CKA when both came from `normalized_gram`."""
    return float(np.clip(np.sum(A * B), -1.0, 1.0))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices over the same n probes (feature widths may differ)."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA compares the same probes: X and Y need equal row counts")
    return gram_alignment(normalized_gram(X), normalized_gram(Y))


def cka_to_anchor_grad(X: np.ndarray, anchor: np.ndarray) -> tuple[float, np.ndarray]:
    """CKA(X, anchor) and its gradient with respect to X, for a fixed normalized anchor Gram.

    f = ⟨K, A⟩ / ‖K‖ with K = H X Xᵀ H.  ∂f/∂K = (A − f·K/‖K‖) / ‖K‖ =: M, which is symmetric and centered
    (A and K are), so ∂f/∂X = (M + Mᵀ) H X = 2 M X.
    """
    X = np.asarray(X, dtype=np.float64)
    K = centered_gram(X)
    s = float(np.linalg.norm(K))
    if s <= EPS:
        return 0.0, np.zeros_like(X)
    G = K / s
    f = float(np.sum(G * anchor))
    M = (anchor - f * G) / s
    return f, 2.0 * M @ X


# ----------------------------------------------------------------------------------------------- subspaces

def sample_subspace(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k left singular vectors of the centered activations (n × k) and all singular values."""
    Xc = center_rows(X)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = max(1, min(k, U.shape[1]))
    return U[:, :k], S


def principal_cosines(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between span(P) and span(Q) (orthonormal columns), descending."""
    s = np.linalg.svd(P.T @ Q, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def subspace_overlap(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean squared cosine of the principal angles: 1 for identical spans, k/n expected for random ones."""
    c = principal_cosines(P, Q)
    return float(np.mean(c ** 2)) if c.size else 0.0


# ----------------------------------------------------------------------------------------------- spectra

def effective_rank(singular_values: np.ndarray) -> float:
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > EPS]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(math.exp(-np.sum(p * np.log(p))))


def stable_rank(W: np.ndarray) -> float:
    """‖W‖_F² / ‖W‖₂²: a noise-robust lower bound on rank."""
    s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    return float(np.sum(s ** 2) / max(s[0] ** 2, EPS)) if s.size else 0.0


def tail_exponent(W: np.ndarray, min_tail: int = 8) -> dict:
    """Power-law exponent α of the eigenvalue tail of the correlation matrix WᵀW / rows.

    Continuous power-law MLE  α = 1 + n / Σ ln(λ_i / λ_min)  over λ_i ≥ λ_min, with λ_min chosen to minimise the
    Kolmogorov–Smirnov distance between the empirical and fitted tails (Clauset, Shalizi & Newman 2009).
    Returns α, λ_min, tail size and KS distance; α is NaN when the spectrum has fewer than `min_tail` usable
    eigenvalues (a too-small matrix has no tail to speak of).
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("tail_exponent needs a 2-D weight matrix")
    rows = W.shape[0]
    lam = np.sort(np.linalg.svd(W, compute_uv=False) ** 2 / max(rows, 1))
    lam = lam[lam > EPS]
    best = {"alpha": float("nan"), "lambda_min": float("nan"), "tail": 0, "ks": float("inf")}
    if lam.size < min_tail:
        return best
    logs = np.log(lam)
    # suffix sums of ln λ let every candidate x_min be scored in O(1) for α and O(tail) for KS
    suffix = np.cumsum(logs[::-1])[::-1]
    for i in range(0, lam.size - min_tail + 1):
        n = lam.size - i
        denom = suffix[i] - n * logs[i]
        if denom <= EPS:
            continue
        alpha = 1.0 + n / denom
        tail = lam[i:]
        empirical = np.arange(n) / n
        fitted = 1.0 - (tail / lam[i]) ** (1.0 - alpha)
        ks = float(np.max(np.abs(empirical - fitted)))
        if ks < best["ks"]:
            best = {"alpha": float(alpha), "lambda_min": float(lam[i]), "tail": int(n), "ks": ks}
    return best


def gram_entropy(G: np.ndarray) -> float:
    """Von Neumann entropy of a normalized centered Gram read as a density (eigenvalues renormalized to sum 1)."""
    w = np.linalg.eigvalsh((G + G.T) / 2.0)
    w = w[w > EPS]
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    return float(-np.sum(p * np.log(p)))
