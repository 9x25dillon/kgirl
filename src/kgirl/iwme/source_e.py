"""Source E (*From Triangle to Form*: E8 holography, the proton radius) — the claims made computable.

The Phase 0–3 paper left Source E outside IWME's dependency graph (scope note in §Source E). A later proposal
asks to wire it into the core as the fix for Finding F1, Source B's wake defects (B1–B7) and checkpoint Q5.
This module recomputes every claim of that proposal that can be computed, so the registry (V48–) can say which
hold, which hold only by construction, and which fail. Nothing here is evidence for or against E8 physics; it
tests the specific mathematical and numerical statements offered as fixes.

    roots        the 240 E8 roots; |α|² normalization; subsystem counts (closed under negation ⇒ even)
    octonions    Cayley–Dickson octonions; Hermitian octonionic matrices; the Jordan identity for n = 2, 3, 4
    vortex rings Thomson's ring of N point vortices: linear stability vs N; the chordal log energy and its Hessian
    pde          ∂t M = α∇²M − β|M|²M + γ M × (∇ × M) [− μM]: energy and max principle (2D periodic), the radial
                 steady state with a source (far-field decay, gradient energy vs domain size and core size)
    form factor  dipole r_p, ⟨r⁴⟩, the angle that fits a datum, shape discrimination at low Q²
    modular      Tomita–Takesaki flow of a Gibbs state = Heisenberg evolution under βH

Deterministic: variates from kgirl.evolve.chaos.ChaosRNG, never a PRNG.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

HBARC_GEV_FM = 0.1973269804
M_PROTON_GEV = 0.93827208816

# Reference radii used by the proposal (fm, 1σ). Further data enter through `compare_bins`.
RADIUS_DATA = {
    "CODATA 2018": (0.8414, 0.0019),
    "CREMA 2013 (muonic H)": (0.84087, 0.00039),
}


def _rng(seed: float = 0.4123):
    from ..evolve.chaos import ChaosRNG
    return ChaosRNG(seed)


# =============================================================================================== roots


def e8_roots() -> np.ndarray:
    """The 240 roots of E8 in the even coordinate system: 112 of type ±e_i ± e_j, 128 of type (±½)^8, even # of −."""
    roots = []
    for i, j in itertools.combinations(range(8), 2):
        for si, sj in itertools.product((1.0, -1.0), repeat=2):
            v = np.zeros(8)
            v[i], v[j] = si, sj
            roots.append(v)
    for signs in itertools.product((0.5, -0.5), repeat=8):
        if sum(s < 0 for s in signs) % 2 == 0:
            roots.append(np.array(signs))
    return np.array(roots)


def positive_roots(R: np.ndarray, generic: np.ndarray | None = None) -> np.ndarray:
    """Positive system cut by a generic hyperplane (no root orthogonal to `generic`)."""
    g = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.1]) if generic is None else generic
    s = R @ g
    if np.any(np.abs(s) < 1e-12):
        raise ValueError("generic vector is orthogonal to a root")
    return R[s > 0]


def root_facts(scale: float = 1.0) -> dict:
    R = e8_roots() * scale
    P = positive_roots(R)
    norms = np.einsum("ij,ij->i", R, R)
    G = np.round(R @ R.T / scale ** 2, 9)
    integer_type = int(np.sum(np.all(np.abs(R / scale - np.round(R / scale)) < 1e-12, axis=1)))
    a = R[0]
    orth = int(np.sum(np.abs(R @ a) < 1e-12))
    return {
        "n_roots": len(R), "n_positive": len(P), "norm2_values": sorted(set(np.round(norms, 12))),
        "inner_products": sorted(set(G.ravel())),
        "sum_norm2_over_120": float(np.sum(np.einsum("ij,ij->i", P, P)) / 120.0),
        "D8_subsystem": integer_type, "E7_subsystem_orthogonal_to_a_root": orth,
    }


def negation_closed_counts_are_even(R: np.ndarray) -> bool:
    """Any root subsystem Φ' satisfies Φ' = −Φ' and 0 ∉ Φ', so it pairs off: |Φ'| is even."""
    keys = {tuple(np.round(r, 9)) for r in R}
    return all(tuple(np.round(-r, 9)) in keys and not np.allclose(r, 0) for r in R)


# =============================================================================================== octonions


def omul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cayley–Dickson product on R^(2^k): (p, q)(r, s) = (pr − s̄q, sp + q r̄)."""
    n = len(a)
    if n == 1:
        return a * b
    h = n // 2
    p, q, r, s = a[:h], a[h:], b[:h], b[h:]
    return np.concatenate([omul(p, r) - omul(conj(s), q), omul(s, p) + omul(q, conj(r))])


def conj(a: np.ndarray) -> np.ndarray:
    c = -a.copy()
    c[0] = a[0]
    return c


def hermitian_octonion_matrix(n: int, rng) -> np.ndarray:
    """n×n Hermitian matrix with octonion entries: shape (n, n, 8); real diagonal, X_ji = conj(X_ij)."""
    X = np.zeros((n, n, 8))
    for i in range(n):
        X[i, i, 0] = rng.normal()
        for j in range(i + 1, n):
            X[i, j] = rng.normal(8)
            X[j, i] = conj(X[i, j])
    return X


def mat_mul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Z = np.zeros_like(X)
    for i in range(n):
        for k in range(n):
            acc = np.zeros(8)
            for j in range(n):
                acc += omul(X[i, j], Y[j, k])
            Z[i, k] = acc
    return Z


def jordan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return 0.5 * (mat_mul(X, Y) + mat_mul(Y, X))


def jordan_identity_residual(n: int, trials: int = 3, seed: float = 0.4123) -> float:
    """max over trials of ‖(x∘y)∘(x∘x) − x∘(y∘(x∘x))‖ / (‖x‖³‖y‖) for Hermitian octonionic n×n matrices."""
    rng = _rng(seed)
    worst = 0.0
    for _ in range(trials):
        x, y = hermitian_octonion_matrix(n, rng), hermitian_octonion_matrix(n, rng)
        x2 = jordan(x, x)
        lhs, rhs = jordan(jordan(x, y), x2), jordan(x, jordan(y, x2))
        scale = np.linalg.norm(x) ** 3 * np.linalg.norm(y)
        worst = max(worst, float(np.linalg.norm(lhs - rhs) / scale))
    return worst


# =============================================================================================== vortex rings


def ring_velocity(z: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Velocity of N point vortices of equal circulation in the plane: conj(dz_k/dt) = Γ/(2πi) Σ_j 1/(z_k − z_j)."""
    d = z[:, None] - z[None, :]
    np.fill_diagonal(d, np.inf)
    w = (gamma / (2j * np.pi)) * np.sum(1.0 / d, axis=1)
    return np.conj(w)


def ring_max_growth(N: int, gamma: float = 1.0, eps: float = 1e-6) -> float:
    """Largest real part of the linearized dynamics of a rotating N-ring, in the co-rotating frame.

    Thomson (1883) / Havelock (1931): linearly stable for N ≤ 6, neutral at N = 7, unstable for N ≥ 8.
    """
    z0 = np.exp(2j * np.pi * np.arange(N) / N)
    omega = gamma * (N - 1) / (4 * np.pi)
    def f(x):
        z = x[:N] + 1j * x[N:]
        v = ring_velocity(z, gamma) - 1j * omega * z      # co-rotating frame
        return np.concatenate([v.real, v.imag])
    x0 = np.concatenate([z0.real, z0.imag])
    J = np.zeros((2 * N, 2 * N))
    for k in range(2 * N):
        e = np.zeros(2 * N)
        e[k] = eps
        J[:, k] = (f(x0 + e) - f(x0 - e)) / (2 * eps)
    return float(np.max(np.linalg.eigvals(J).real))


def chordal_energy(theta: np.ndarray, V0: float = 1.0, E_self: float = 0.0) -> float:
    N = len(theta)
    e = N * E_self
    for j, k in itertools.combinations(range(N), 2):
        e -= V0 * math.log(2 * abs(math.sin((theta[j] - theta[k]) / 2)))
    return e


def chordal_hessian_reduced(N: int, V0: float = 1.0, h: float = 1e-4) -> np.ndarray:
    """Hessian of E at equal spacing in the N−1 gap coordinates (rigid rotation removed)."""
    base = 2 * np.pi / N

    def E(gaps):
        th = np.concatenate([[0.0], np.cumsum(gaps)])
        return chordal_energy(th, V0)
    g0 = np.full(N - 1, base)
    H = np.zeros((N - 1, N - 1))
    for a in range(N - 1):
        for b in range(N - 1):
            ea, eb = np.eye(N - 1)[a] * h, np.eye(N - 1)[b] * h
            H[a, b] = (E(g0 + ea + eb) - E(g0 + ea - eb) - E(g0 - ea + eb) + E(g0 - ea - eb)) / (4 * h * h)
    return H


# =============================================================================================== PDE


@dataclass(frozen=True)
class PDEParams:
    alpha: float = 1.0
    beta: float = 2.0       # the proposal's "E8-locked" value
    gamma: float = 1.0
    mu: float = 0.0         # linear restoring term; 0 = the proposal's equation


def _grad(f, h):
    return ((np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * h), (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * h))


def _lap(f, h):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f) / (h * h)


def lamb_term(Mx, My, h):
    """Planar M = (Mx, My, 0): ∇×M = ω ẑ with ω = ∂x My − ∂y Mx; M × ωẑ = (My ω, −Mx ω)."""
    dMy_dx, _ = _grad(My, h)
    _, dMx_dy = _grad(Mx, h)
    w = dMy_dx - dMx_dy
    return My * w, -Mx * w


def evolve_2d(Mx, My, p: PDEParams, h: float, dt: float, steps: int):
    """Explicit Euler on a periodic grid; returns final field and the histories of ∫|M|² and max|M|."""
    l2, mx = [], []
    for _ in range(steps):
        l2.append(float(np.sum(Mx ** 2 + My ** 2) * h * h))
        mx.append(float(np.sqrt(np.max(Mx ** 2 + My ** 2))))
        n2 = Mx ** 2 + My ** 2
        Lx, Ly = lamb_term(Mx, My, h)
        Mx, My = (Mx + dt * (p.alpha * _lap(Mx, h) - p.beta * n2 * Mx + p.gamma * Lx - p.mu * Mx),
                  My + dt * (p.alpha * _lap(My, h) - p.beta * n2 * My + p.gamma * Ly - p.mu * My))
    return Mx, My, np.array(l2), np.array(mx)


def initial_field(n: int, amplitude: float = 0.95, seed: float = 0.4123):
    """Smooth rotational field with max |M| = amplitude, built from chaos-weighted Fourier modes."""
    rng = _rng(seed)
    x = np.arange(n) * 2 * np.pi / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    psi = np.zeros((n, n))
    for kx in range(1, 4):
        for ky in range(1, 4):
            a, b = rng.normal(2)
            psi += (a * np.sin(kx * X + ky * Y) + b * np.cos(kx * X - ky * Y)) / (kx * kx + ky * ky)
    h = x[1] - x[0]
    gx, gy = _grad(psi, h)
    Mx, My = gy, -gx                                    # divergence-free, rotational
    s = amplitude / np.sqrt(np.max(Mx ** 2 + My ** 2))
    return Mx * s, My * s, h


def radial_steady(R: float, n: int, p: PDEParams, q: float = 1.0, core: float = 0.5, iters: int = 60) -> tuple:
    """Steady radial amplitude u(r): α(1/r)(r u')' − β u³ − μ u + f = 0, f = q/(π core²) on r < core.

    u'(0) = 0, u(R) = 0. Finite volumes with cell centres r_i = (i+½)h, Newton iterations.
    Returns (r, u, gradient energy ∫|u'|² 2πr dr).
    """
    h = R / n
    r = (np.arange(n) + 0.5) * h
    rf = np.arange(n + 1) * h                          # faces
    f = np.where(r < core, q / (np.pi * core * core), 0.0)
    u = np.zeros(n)
    for _ in range(iters):
        # residual F_i = α/(r_i h²) [r_{i+½}(u_{i+1}−u_i) − r_{i−½}(u_i−u_{i−1})] − βu³ − μu + f
        up = np.append(u[1:], -u[-1])                  # ghost: u(R) = 0 between last cell and ghost
        um = np.insert(u[:-1], 0, u[0])                # u'(0) = 0
        F = p.alpha / (r * h * h) * (rf[1:] * (up - u) - rf[:-1] * (u - um)) - p.beta * u ** 3 - p.mu * u + f
        lo = p.alpha * rf[:-1] / (r * h * h)
        hi = p.alpha * rf[1:] / (r * h * h)
        diag = -(lo + hi) - 3 * p.beta * u ** 2 - p.mu
        diag[0] += lo[0]                               # Neumann at the axis
        diag[-1] -= hi[-1]                             # Dirichlet through the ghost
        du = _thomas(lo[1:], diag, hi[:-1], -F)
        u = u + du
        if np.max(np.abs(du)) < 1e-12 * max(1.0, np.max(np.abs(u))):
            break
    du_dr = np.diff(np.concatenate([u, [-u[-1]]])) / h  # at faces r_{i+1}
    energy = float(np.sum(du_dr ** 2 * 2 * np.pi * rf[1:] * h))
    return r, u, energy


def _thomas(a, b, c, d):
    n = len(b)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def decay_rate(r: np.ndarray, u: np.ndarray, r1: float, r2: float) -> float:
    """Mean logarithmic decay rate −Δln u / Δr between r1 and r2 (exponential ⇒ constant, algebraic ⇒ → 0)."""
    i1, i2 = np.searchsorted(r, r1), np.searchsorted(r, r2)
    return float(-(math.log(u[i2]) - math.log(u[i1])) / (r[i2] - r[i1]))


# =============================================================================================== form factor


def dipole_lambda(angle_deg: float = 30.0) -> float:
    return M_PROTON_GEV * math.cos(math.radians(angle_deg))


def dipole_radius_fm(Lambda_gev: float) -> float:
    return math.sqrt(12.0) / Lambda_gev * HBARC_GEV_FM


def angle_for_radius(r_fm: float) -> float:
    return math.degrees(math.acos(math.sqrt(12.0) * HBARC_GEV_FM / (r_fm * M_PROTON_GEV)))


def shapes(r_fm: float) -> dict:
    """Charge form factors with the same rms radius (Q² in GeV²)."""
    r2 = (r_fm / HBARC_GEV_FM) ** 2
    return {
        "dipole (exponential density)": lambda Q2: (1 + Q2 * r2 / 12) ** -2,
        "monopole (Yukawa density)": lambda Q2: (1 + Q2 * r2 / 6) ** -1,
        "Gaussian density": lambda Q2: np.exp(-Q2 * r2 / 6),
    }


def r4_over_r2sq(shape: str) -> float:
    """⟨r⁴⟩/⟨r²⟩² of the charge density; G = 1 − ⟨r²⟩Q²/6 + ⟨r⁴⟩Q⁴/120 − …, so ⟨r⁴⟩ = 60 G″(0).

    exponential e^{−Λr}: ⟨r^n⟩ = (n+2)!/(2Λ^n) → 360/144 = 5/2;  Yukawa e^{−mr}/r: (n+1)!/m^n → 120/36 = 10/3;
    Gaussian: 15s⁴/(3s²)² = 5/3.
    """
    return {"dipole (exponential density)": 5.0 / 2.0, "monopole (Yukawa density)": 10.0 / 3.0,
            "Gaussian density": 5.0 / 3.0}[shape]


def max_shape_spread(r_fm: float, q2_lo: float = 2.1e-4, q2_hi: float = 0.06) -> float:
    """Largest relative difference between equal-radius shapes over a Q² window (PRad's range by default)."""
    Q2 = np.linspace(q2_lo, q2_hi, 400)
    vals = np.array([g(Q2) for g in shapes(r_fm).values()])
    return float(np.max((vals.max(0) - vals.min(0)) / vals.min(0)))


def compare_bins(path: str, r_fm: float | None = None) -> list[dict]:
    """Compare the dipole prediction with G_E bins from a CSV (columns: Q2_GeV2, GE, sigma)."""
    r_fm = r_fm or dipole_radius_fm(dipole_lambda())
    g = shapes(r_fm)["dipole (exponential density)"]
    out = []
    data = np.genfromtxt(path, delimiter=",", names=True)
    for row in np.atleast_1d(data):
        pred = float(g(row["Q2_GeV2"]))
        out.append({"Q2": float(row["Q2_GeV2"]), "GE": float(row["GE"]), "pred": pred,
                    "rel_dev": (pred - row["GE"]) / row["GE"], "pull": (pred - row["GE"]) / row["sigma"]})
    return out


# =============================================================================================== modular flow


def modular_vs_heisenberg(n: int = 5, beta: float = 0.7, tau: float = 0.37, seed: float = 0.4123) -> float:
    """‖ρ^{iτ} O ρ^{−iτ} − e^{−iβτH} O e^{iβτH}‖ for ρ = e^{−βH}/Z: the modular flow is H-dynamics at time −βτ."""
    rng = _rng(seed)
    A = rng.normal((n, n)) + 1j * rng.normal((n, n))
    H = (A + A.conj().T) / 2
    O = rng.normal((n, n)) + 1j * rng.normal((n, n))
    w, V = np.linalg.eigh(H)
    logrho = -beta * w - np.log(np.sum(np.exp(-beta * w)))

    def U(phases):
        return V @ np.diag(np.exp(1j * phases)) @ V.conj().T
    mod = U(tau * logrho) @ O @ U(-tau * logrho)
    heis = U(-beta * tau * w) @ O @ U(beta * tau * w)
    return float(np.linalg.norm(mod - heis))
