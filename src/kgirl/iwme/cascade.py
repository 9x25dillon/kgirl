"""Multifractal cascades: the IWME draft's "entropy as a fractal language" made computable.

Three generators on the dyadic tree (level n has 2^n cells of length ε = 2^{−n}):

  binomial      i.i.d. split p : (1 − p). Exact: Σ μ_i^q = (p^q + (1 − p)^q)^n, so
                D_q = log2(p^q + (1 − p)^q) / (1 − q), D_1 = h2(p), D_0 = 1, D_{±∞} = −log2 max/min(p, 1 − p).
  markov        the split depends on a hidden state s ∈ {0, 1} that follows the branch taken:
                τ(q) = (q − 1) D_q = −log2 ρ(M_q),  M_q(s, s′) = Σ_c w_c(s)^q P(s′ | s, c).
  ema           a continuous memory h carried down the tree, h′ = (1 − ρ) h + ρ (±1), with an attention-like
                split p(h) = σ((h + β)/T). Its spectrum is the leading eigenvalue of a transfer operator; a
                discretized operator reproduces the exact enumeration (pre-registered expectation E-A:
                memory changes the spectrum and stays inside the thermodynamic formalism).

Also the finite-dimensional model of the EFL fixed point: the Hutchinson operator
    (Φμ)(A) = p μ(2A) + (1 − p) μ(2A − 1)
is a contraction with factor 1/2 in the Wasserstein-1 metric; its unique fixed point is the binomial measure,
which realizes "E ≃ Φ_λ E" concretely (Hutchinson 1981).

numpy only. All enumerations are exact (no sampling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

Array = np.ndarray
LOG2 = math.log(2.0)


# ----------------------------------------------------------------------------------------------
# Binomial cascade: closed forms
# ----------------------------------------------------------------------------------------------

def h2(p: float) -> float:
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def binomial_Dq(p: float, q: float) -> float:
    """Rényi dimension of the binomial measure (exact at every level n)."""
    if q == 1.0:
        return h2(p)
    if math.isinf(q):
        return -math.log2(max(p, 1 - p) if q > 0 else min(p, 1 - p))
    return math.log2(p ** q + (1 - p) ** q) / (1.0 - q)


def binomial_tau(p: float, q: float) -> float:
    return -math.log2(p ** q + (1 - p) ** q)


def binomial_measure(p: float, n: int) -> Array:
    """Exact cell masses at level n (length 2^n), left child gets p."""
    mu = np.array([1.0])
    w = np.array([p, 1.0 - p])
    for _ in range(n):
        mu = np.kron(mu, w)
    return mu


def partition_function(mu: Array, q: float) -> float:
    m = mu[mu > 0]
    if q == 1.0:
        return float(-(m * np.log2(m)).sum())      # Shannon entropy H_n in bits (for D_1)
    return float((m ** q).sum())


def coarse_grain(mu: Array) -> Array:
    """Pushforward to the next coarser dyadic level: sum sibling pairs."""
    return mu.reshape(-1, 2).sum(axis=1)


def estimate_Dq(mu_finest: Array, q: float, levels: Sequence[int]) -> float:
    """D_q from the slope of log2 Z_n(q) against n across the given levels (box counting, ε = 2^{−n})."""
    n_max = int(round(math.log2(len(mu_finest))))
    mus = {n_max: mu_finest}
    m = mu_finest
    for n in range(n_max - 1, min(levels) - 1, -1):
        m = coarse_grain(m)
        mus[n] = m
    xs = np.array(list(levels), dtype=float)
    if q == 1.0:
        ys = np.array([partition_function(mus[n], 1.0) for n in levels])      # H_n = n D_1
        return float(np.polyfit(xs, ys, 1)[0])
    ys = np.array([math.log2(partition_function(mus[n], q)) for n in levels])  # log2 Z = −n τ(q)
    tau = -np.polyfit(xs, ys, 1)[0]
    return float(tau / (q - 1.0))


def shannon_entropy_levels(p: float, n_max: int) -> Array:
    """H_n for n = 1..n_max, which the closed form says equals n h2(p)."""
    return np.array([partition_function(binomial_measure(p, n), 1.0) for n in range(1, n_max + 1)])


# ----------------------------------------------------------------------------------------------
# Markov-modulated cascade: transfer matrix
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkovCascade:
    """Two hidden states. In state s the left child receives w_L(s) = p[s]; the child inherits the state
    s′ with probability P[s][c][s′] (c = 0 left, 1 right). The initial state distribution is pi0."""

    p: Tuple[float, float] = (0.7, 0.4)
    P: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]] = (
        ((0.9, 0.1), (0.3, 0.7)),
        ((0.2, 0.8), (0.6, 0.4)),
    )
    pi0: Tuple[float, float] = (0.5, 0.5)

    def weights(self, s: int) -> Tuple[float, float]:
        return (self.p[s], 1.0 - self.p[s])

    def Mq(self, q: float) -> Array:
        M = np.zeros((2, 2))
        for s in range(2):
            w = self.weights(s)
            for c in range(2):
                for s2 in range(2):
                    M[s, s2] += (w[c] ** q) * self.P[s][c][s2]
        return M

    def tau(self, q: float) -> float:
        return float(-math.log2(max(abs(np.linalg.eigvals(self.Mq(q))))))

    def Dq(self, q: float) -> float:
        if q == 1.0:
            # D_1 = −τ′(1)/… : derivative of τ at q = 1 (τ(1) = 0)
            hstep = 1e-6
            return float((self.tau(1 + hstep) - self.tau(1 - hstep)) / (2 * hstep))
        return self.tau(q) / (q - 1.0)

    def Zn(self, q: float, n: int) -> float:
        """Annealed partition function E_s[Σ_i μ_i^q] over the hidden-state process, computed exactly by
        enumerating states along the tree: Z_n = π0ᵀ M_q^n 1."""
        v = np.array(self.pi0, dtype=float)
        M = self.Mq(q)
        for _ in range(n):
            v = v @ M
        return float(v.sum())

    def enumerate_Zn(self, q: float, n: int) -> float:
        """Brute-force check of Zn: carry (mass^q-weighted probability) per state cell by cell."""
        # state vector per cell: weight[s] = Σ over histories P(history) μ^q, cells merged since Z sums them
        cells = [np.array(self.pi0, dtype=float)]
        for _ in range(n):
            nxt = []
            for wv in cells:
                for c in range(2):
                    child = np.zeros(2)
                    for s in range(2):
                        if wv[s] == 0.0:
                            continue
                        wc = self.weights(s)[c] ** q
                        for s2 in range(2):
                            child[s2] += wv[s] * wc * self.P[s][c][s2]
                    nxt.append(child)
            cells = nxt
        return float(sum(c.sum() for c in cells))


# ----------------------------------------------------------------------------------------------
# EMA-memory cascade: continuous hidden state, transfer operator
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EMACascade:
    """Deterministic cascade with memory: h starts at h0; at each split the left child gets
    p(h) = σ((h + bias)/T) and the memory updates h_L = (1 − ρ)h + ρ, h_R = (1 − ρ)h − ρ."""

    rho: float = 0.4
    T: float = 0.8
    bias: float = 0.3
    h0: float = 0.0

    def p_left(self, h: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-(np.asarray(h) + self.bias) / self.T))

    def enumerate_Zn(self, q: float, n: int) -> float:
        """Exact Σ_i μ_i^q at level n by enumerating all 2^n cells."""
        h = np.array([self.h0])
        lq = np.array([0.0])                      # log of μ^q per cell
        for _ in range(n):
            p = self.p_left(h)
            lq = np.concatenate([lq + q * np.log(p), lq + q * np.log(1 - p)])
            h = np.concatenate([(1 - self.rho) * h + self.rho, (1 - self.rho) * h - self.rho])
        mx = lq.max()
        return float(np.exp(mx) * np.exp(lq - mx).sum())

    def tau_enumeration(self, q: float, n1: int = 10, n2: int = 18) -> float:
        z1, z2 = self.enumerate_Zn(q, n1), self.enumerate_Zn(q, n2)
        return float(-(math.log2(z2) - math.log2(z1)) / (n2 - n1))

    def transfer_operator(self, q: float, bins: int = 801) -> Tuple[Array, Array]:
        """(L_q f)(h) = Σ_c p_c(h)^q f(h_c) on a grid over the invariant interval [−1, 1] (linear interp)."""
        grid = np.linspace(-1.0, 1.0, bins)
        L = np.zeros((bins, bins))
        p = self.p_left(grid)
        for c, (wc, hc) in enumerate(((p, (1 - self.rho) * grid + self.rho), (1 - p, (1 - self.rho) * grid - self.rho))):
            pos = (hc + 1.0) / 2.0 * (bins - 1)
            i0 = np.clip(np.floor(pos).astype(int), 0, bins - 2)
            frac = pos - i0
            rows = np.arange(bins)
            np.add.at(L, (rows, i0), (wc ** q) * (1 - frac))
            np.add.at(L, (rows, i0 + 1), (wc ** q) * frac)
        return grid, L

    def tau_transfer(self, q: float, bins: int = 801) -> float:
        _, L = self.transfer_operator(q, bins)
        lam = max(abs(np.linalg.eigvals(L)))
        return float(-math.log2(lam))


def ema_D1(cascade: EMACascade, bins: int = 801, h: float = 1e-4) -> float:
    """Information dimension D_1 = τ′(1) from the transfer operator (central difference in q)."""
    return (cascade.tau_transfer(1.0 + h, bins) - cascade.tau_transfer(1.0 - h, bins)) / (2.0 * h)


def memoryless_match(cascade: EMACascade, q: float = 1.0, bins: int = 801) -> float:
    """p ≥ ½ of the binomial cascade with the same D_1 as the EMA cascade (transfer-operator value)."""
    target = ema_D1(cascade, bins)
    lo, hi = 0.5, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h2(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------------
# EFL fixed point as the Hutchinson operator on dyadic measures
# ----------------------------------------------------------------------------------------------

def hutchinson_step(mu: Array, p: float) -> Array:
    """Φμ at one level finer: left half gets p·μ (rescaled into [0, 1/2]), right half (1 − p)·μ."""
    return np.concatenate([p * mu, (1 - p) * mu])


def wasserstein1_dyadic(mu: Array, nu: Array) -> float:
    """Exact W1 = ∫_0^1 |F − G| between two measures with uniform density inside the cells of a dyadic
    grid (masses mu, nu). F − G is piecewise linear, so each cell integrates in closed form."""
    if len(mu) != len(nu):
        n = max(len(mu), len(nu))
        mu = np.repeat(mu, n // len(mu)) / (n // len(mu))
        nu = np.repeat(nu, n // len(nu)) / (n // len(nu))
    d = np.concatenate([[0.0], np.cumsum(np.asarray(mu) - np.asarray(nu))])
    d0, d1 = d[:-1], d[1:]
    w = 1.0 / len(mu)
    same = d0 * d1 >= 0
    a0, a1 = np.abs(d0), np.abs(d1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = np.where(a0 + a1 > 0, (d0 * d0 + d1 * d1) / (2 * (a0 + a1)), 0.0)
    return float(w * np.where(same, 0.5 * (a0 + a1), cross).sum())


def hutchinson_contraction(p: float, n_iter: int = 12, level0: int = 3) -> Dict[str, list]:
    """Iterate Φ from the uniform measure and from the binomial fixed point (both at level0). Because the two
    maps x ↦ x/2, x ↦ (x + 1)/2 do not overlap, W1(Φμ, Φν) = W1(μ, ν)/2 exactly; Φ(binomial_n) = binomial_{n+1}."""
    mu = np.full(2 ** level0, 1.0 / 2 ** level0)
    fixed = binomial_measure(p, level0)
    dists, ratios, fixed_err = [], [], []
    for k in range(n_iter):
        d = wasserstein1_dyadic(mu, fixed)
        dists.append(d)
        if k:
            ratios.append(d / dists[-2])
        fixed_err.append(float(np.abs(fixed - binomial_measure(p, level0 + k)).max()))
        mu, fixed = hutchinson_step(mu, p), hutchinson_step(fixed, p)
    return {"W1": dists, "ratios": ratios, "fixed_point_error": fixed_err}


def efl_effective_dimension(lam: float, r_max: int = 40) -> Array:
    """Source B's D_eff = limsup log N(r) / log λ^r with N(r) = λ^r: identically 1 (the scale factor cancels)."""
    r = np.arange(1, r_max + 1, dtype=float)
    N = lam ** r
    return np.log(N) / np.log(lam ** r)


def box_dimension_from_branching(b: float, lam: float) -> float:
    """What a fractal dimension needs: b branches per level at linear scale 1/λ gives D = log b / log λ."""
    return math.log(b) / math.log(lam)
