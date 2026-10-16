"""IRT geometry: the Inverse Resonant Toroid of *The Geometry of Observation* (Source C), recomputed.

Source C, Definition 2.1, embeds three parameters (θ, φ, ψ) in four coordinates (x, y, z, w):

    x = ρ cos θ,  y = ρ sin θ,  ρ(φ, ψ) = (R − r cos φ) e^{−αψ}
    z = r sin φ sinh(βψ)
    w = −∫_0^ψ κ(τ) dτ                      (the "shadow" coordinate)

with R > r > 0, α, β > 0, ψ ≥ 0. The image is a parametrized 3-manifold in R⁴ and the induced
metric is g_ij = ∂_i X · ∂_j X. Results (Level 0, verified three independent ways in this module):

  * g_θθ = u² e^{−2αψ}                                  u := R − r cos φ  (> 0)
    g_φφ = r² (sin²φ e^{−2αψ} + cos²φ sinh²βψ)
    g_ψψ = α² u² e^{−2αψ} + β² r² sin²φ cosh²βψ + κ²
    g_φψ = r sin φ (β r cos φ sinh βψ cosh βψ − α u e^{−2αψ})        (nonzero in general)
    g_θφ = g_θψ = 0                                     (θ is a Killing direction)
  * det g = u² e^{−2αψ} [ r² e^{−2αψ} D² + κ² g_φφ ],   D := β r sin²φ cosh βψ + α u cos φ sinh βψ
    so the Jacobian has rank 3 except on  {ψ = 0, sin φ = 0}  ∪  {κ(ψ) = 0, D(φ, ψ) = 0}.
  * M is the warped product Σ ×_ρ S¹ of the profile surface Σ = {(ρ, z, w)(φ, ψ)} ⊂ R³, hence
        Scal = 2 K_Σ − 2 (Δ_Σ ρ) / ρ ,
    and κ enters the curvature only through κ² and κκ′ (never κ″).

Source C's Appendix A.1 lists g_φφ = r² e^{−2αψ} cosh²βψ, a sinh² in g_ψψ and g_φψ = 0; all three are
wrong. The curvature routes here (Gauss equation, intrinsic Riemann tensor, warped product) agree to
round-off and supersede A.2–A.3.

numpy only; `symbolic_metric()` and `symbolic_scalar_curvature_check()` import sympy lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ----------------------------------------------------------------------------------------------
# Parameters and resonance kernels
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class IRTParams:
    """Shape parameters of the IRT. Source C requires R > r > 0 and α, β > 0."""

    R: float = 3.0
    r: float = 1.0
    alpha: float = 0.3
    beta: float = 0.5

    def __post_init__(self) -> None:
        if not (self.R > self.r > 0.0):
            raise ValueError("IRT needs R > r > 0")
        if not (self.alpha > 0.0 and self.beta > 0.0):
            raise ValueError("IRT needs alpha, beta > 0")


@dataclass(frozen=True)
class Kernel:
    """Absorption kernel κ(ψ) of the form Σ_n A_n e^{−λ_n ψ} cos(ω_n ψ + φ_n) (Source C, eq. 15).

    A constant kernel is the single mode (A, λ=0, ω=0, φ=0). Every quantity is closed form:
    κ, κ′, κ″ and the shadow coordinate w(ψ) = −∫_0^ψ κ.
    """

    A: Tuple[float, ...] = (1.0,)
    lam: Tuple[float, ...] = (0.0,)
    omega: Tuple[float, ...] = (0.0,)
    phase: Tuple[float, ...] = (0.0,)
    name: str = "kernel"

    def __post_init__(self) -> None:
        n = len(self.A)
        if not (len(self.lam) == len(self.omega) == len(self.phase) == n):
            raise ValueError("kernel mode arrays must have equal length")

    def _modes(self):
        return zip(self.A, self.lam, self.omega, self.phase)

    def k(self, psi) -> Array:
        psi = np.asarray(psi, dtype=float)
        out = np.zeros_like(psi)
        for A, lam, om, ph in self._modes():
            out = out + A * np.exp(-lam * psi) * np.cos(om * psi + ph)
        return out

    def k1(self, psi) -> Array:
        """κ′(ψ)."""
        psi = np.asarray(psi, dtype=float)
        out = np.zeros_like(psi)
        for A, lam, om, ph in self._modes():
            e = np.exp(-lam * psi)
            out = out + A * e * (-lam * np.cos(om * psi + ph) - om * np.sin(om * psi + ph))
        return out

    def k2(self, psi) -> Array:
        """κ″(ψ)."""
        psi = np.asarray(psi, dtype=float)
        out = np.zeros_like(psi)
        for A, lam, om, ph in self._modes():
            e = np.exp(-lam * psi)
            c, s = np.cos(om * psi + ph), np.sin(om * psi + ph)
            out = out + A * e * ((lam * lam - om * om) * c + 2.0 * lam * om * s)
        return out

    def w(self, psi) -> Array:
        """Shadow coordinate w(ψ) = −∫_0^ψ κ(τ) dτ, exact antiderivative of every mode."""
        psi = np.asarray(psi, dtype=float)
        out = np.zeros_like(psi)
        for A, lam, om, ph in self._modes():
            out = out - A * _int_damped_cos(psi, lam, om, ph)
        return out

    def w_source_c_eq16(self, psi) -> Array:
        """Source C eq. (16) as printed: −Σ (A/λ) e^{−λψ} + (A/ω) sin(ωψ + φ). Kept to audit it."""
        psi = np.asarray(psi, dtype=float)
        out = np.zeros_like(psi)
        for A, lam, om, ph in self._modes():
            out = out - (A / lam) * np.exp(-lam * psi) + (A / om) * np.sin(om * psi + ph)
        return out

    def zeros(self, psi_max: float, n: int = 20001) -> Array:
        """Zeros of κ on (0, psi_max], located by sign change and refined by bisection."""
        grid = np.linspace(0.0, psi_max, n)[1:]
        vals = self.k(grid)
        idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        roots = [_bisect(lambda s: float(self.k(s)), grid[i], grid[i + 1]) for i in idx]
        roots += [float(grid[i]) for i in np.nonzero(vals == 0.0)[0]]
        return np.array(sorted(roots))


def _int_damped_cos(psi: Array, lam: float, om: float, ph: float) -> Array:
    """∫_0^ψ e^{−λτ} cos(ωτ + φ) dτ in closed form (λ = ω = 0 handled)."""
    d = lam * lam + om * om
    if d == 0.0:
        return psi * math.cos(ph)
    F = lambda t: np.exp(-lam * t) * (om * np.sin(om * t + ph) - lam * np.cos(om * t + ph)) / d  # noqa: E731
    return F(psi) - F(np.zeros_like(psi))


def _bisect(f: Callable[[float], float], a: float, b: float, tol: float = 1e-14, it: int = 200) -> float:
    fa = f(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5 * (a + b)


CONSTANT_KERNEL = Kernel(A=(1.0,), lam=(0.0,), omega=(0.0,), phase=(0.0,), name="constant κ=1")
DAMPED_KERNEL = Kernel(A=(1.0,), lam=(0.2,), omega=(2.0,), phase=(0.0,), name="e^{-0.2ψ} cos 2ψ")


def constant_kernel(k0: float) -> Kernel:
    return Kernel(A=(float(k0),), lam=(0.0,), omega=(0.0,), phase=(0.0,), name=f"constant κ={k0:g}")


@dataclass(frozen=True)
class QuadraticKernel:
    """κ(ψ) = c0 + c1 (ψ − ψ0) + c2 (ψ − ψ0)²: fixes κ, κ′ at ψ0 while κ″ = 2 c2 varies freely.

    Used to show that the curvature at ψ0 is blind to κ″(ψ0) (Gauss equation: only X and its second
    derivatives enter, and X_ψψ carries κ′, never κ″).
    """

    c0: float = 1.0
    c1: float = 0.3
    c2: float = 0.0
    psi0: float = 1.0
    name: str = "quadratic"

    def k(self, psi):
        s = np.asarray(psi, dtype=float) - self.psi0
        return self.c0 + self.c1 * s + self.c2 * s * s

    def k1(self, psi):
        s = np.asarray(psi, dtype=float) - self.psi0
        return self.c1 + 2.0 * self.c2 * s

    def k2(self, psi):
        return 2.0 * self.c2 + 0.0 * np.asarray(psi, dtype=float)

    def w(self, psi):
        psi = np.asarray(psi, dtype=float)
        s, s0 = psi - self.psi0, -self.psi0
        return -(self.c0 * psi + self.c1 * (s * s - s0 * s0) / 2.0 + self.c2 * (s ** 3 - s0 ** 3) / 3.0)


# ----------------------------------------------------------------------------------------------
# Embedding and its derivatives (closed form)
# ----------------------------------------------------------------------------------------------

def _profile(phi, psi, p: IRTParams):
    """ρ, z and their first and second derivatives in (φ, ψ)."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    R, r, a, b = p.R, p.r, p.alpha, p.beta
    e = np.exp(-a * psi)
    sp_, cp = np.sin(phi), np.cos(phi)
    sh, ch = np.sinh(b * psi), np.cosh(b * psi)
    u = R - r * cp
    rho = u * e
    d = {
        "u": u, "rho": rho,
        "rho_f": r * sp_ * e, "rho_p": -a * u * e,
        "rho_ff": r * cp * e, "rho_fp": -a * r * sp_ * e, "rho_pp": a * a * u * e,
        "z": r * sp_ * sh,
        "z_f": r * cp * sh, "z_p": b * r * sp_ * ch,
        "z_ff": -r * sp_ * sh, "z_fp": b * r * cp * ch, "z_pp": b * b * r * sp_ * sh,
    }
    return d


def embedding(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """X(θ, φ, ψ) ∈ R⁴, shape (..., 4)."""
    theta = np.asarray(theta, dtype=float)
    d = _profile(phi, psi, p)
    return np.stack(np.broadcast_arrays(d["rho"] * np.cos(theta), d["rho"] * np.sin(theta), d["z"],
                                        kern.w(psi)), axis=-1)


def tangents(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """Jacobian columns X_θ, X_φ, X_ψ as an array of shape (..., 4, 3)."""
    theta = np.asarray(theta, dtype=float)
    d = _profile(phi, psi, p)
    c, s = np.cos(theta), np.sin(theta)
    zero = np.zeros(np.broadcast(theta, d["rho"]).shape)
    Xt = np.stack(np.broadcast_arrays(-d["rho"] * s, d["rho"] * c, zero, zero), axis=-1)
    Xf = np.stack(np.broadcast_arrays(d["rho_f"] * c, d["rho_f"] * s, d["z_f"], zero), axis=-1)
    Xp = np.stack(np.broadcast_arrays(d["rho_p"] * c, d["rho_p"] * s, d["z_p"], -kern.k(psi) + zero), axis=-1)
    return np.stack([Xt, Xf, Xp], axis=-1)


def second_derivatives(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """X_ij for i, j ∈ (θ, φ, ψ), shape (..., 3, 3, 4)."""
    theta = np.asarray(theta, dtype=float)
    d = _profile(phi, psi, p)
    c, s = np.cos(theta), np.sin(theta)
    zero = np.zeros(np.broadcast(theta, d["rho"]).shape)

    def vec(a0, a1, a2, a3):
        return np.stack(np.broadcast_arrays(a0, a1, a2, a3), axis=-1)

    Xtt = vec(-d["rho"] * c, -d["rho"] * s, zero, zero)
    Xtf = vec(-d["rho_f"] * s, d["rho_f"] * c, zero, zero)
    Xtp = vec(-d["rho_p"] * s, d["rho_p"] * c, zero, zero)
    Xff = vec(d["rho_ff"] * c, d["rho_ff"] * s, d["z_ff"], zero)
    Xfp = vec(d["rho_fp"] * c, d["rho_fp"] * s, d["z_fp"], zero)
    Xpp = vec(d["rho_pp"] * c, d["rho_pp"] * s, d["z_pp"], -kern.k1(psi) + zero)
    rows = [[Xtt, Xtf, Xtp], [Xtf, Xff, Xfp], [Xtp, Xfp, Xpp]]
    return np.stack([np.stack(rw, axis=-2) for rw in rows], axis=-3)


# ----------------------------------------------------------------------------------------------
# Metric: closed form, from the Jacobian, and as published
# ----------------------------------------------------------------------------------------------

def metric(phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """Induced metric in closed form (θ-independent), shape (..., 3, 3), order (θ, φ, ψ)."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    R, r, a, b = p.R, p.r, p.alpha, p.beta
    sp_, cp = np.sin(phi), np.cos(phi)
    sh, ch = np.sinh(b * psi), np.cosh(b * psi)
    e2 = np.exp(-2.0 * a * psi)
    u = R - r * cp
    k = kern.k(psi)
    g_tt = u * u * e2
    g_ff = r * r * (sp_ * sp_ * e2 + cp * cp * sh * sh)
    g_pp = a * a * u * u * e2 + b * b * r * r * sp_ * sp_ * ch * ch + k * k
    g_fp = r * sp_ * (b * r * cp * sh * ch - a * u * e2)
    z = np.zeros(np.broadcast(phi, psi).shape)
    g_tt, g_ff, g_pp, g_fp = np.broadcast_arrays(g_tt, g_ff, g_pp, g_fp)
    return np.stack([np.stack([g_tt, z, z], -1), np.stack([z, g_ff, g_fp], -1), np.stack([z, g_fp, g_pp], -1)], -2)


def metric_from_jacobian(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """g = Jᵀ J computed from the tangent vectors (independent of the closed form)."""
    J = tangents(theta, phi, psi, p, kern)
    return np.einsum("...ai,...aj->...ij", J, J)


def published_metric(phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """Source C, Appendix A.1 (eq. 27), transcribed as printed: diagonal, cosh² in g_φφ, sinh² in g_ψψ."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    R, r, a, b = p.R, p.r, p.alpha, p.beta
    e2 = np.exp(-2.0 * a * psi)
    u = R - r * np.cos(phi)
    g_tt = u * u * e2
    g_ff = r * r * e2 * np.cosh(b * psi) ** 2
    g_pp = a * a * u * u * e2 + b * b * r * r * np.sin(phi) ** 2 * np.sinh(b * psi) ** 2 + kern.k(psi) ** 2
    z = np.zeros(np.broadcast(phi, psi).shape)
    g_tt, g_ff, g_pp = np.broadcast_arrays(g_tt, g_ff, g_pp)
    return np.stack([np.stack([g_tt, z, z], -1), np.stack([z, g_ff, z], -1), np.stack([z, z, g_pp], -1)], -2)


def D_factor(phi, psi, p: IRTParams = IRTParams()) -> Array:
    """D = β r sin²φ cosh βψ + α u cos φ sinh βψ: the (ρ, z) minor of the (φ, ψ) Jacobian is r e^{−αψ} D."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    u = p.R - p.r * np.cos(phi)
    return p.beta * p.r * np.sin(phi) ** 2 * np.cosh(p.beta * psi) + p.alpha * u * np.cos(phi) * np.sinh(p.beta * psi)


def det_metric(phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """Closed form det g = u² e^{−2αψ} [ r² e^{−2αψ} D² + κ² g_φφ ] (Cauchy–Binet on the 4×2 block)."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    r, a = p.r, p.alpha
    e2 = np.exp(-2.0 * a * psi)
    u = p.R - r * np.cos(phi)
    g_ff = r * r * (np.sin(phi) ** 2 * e2 + np.cos(phi) ** 2 * np.sinh(p.beta * psi) ** 2)
    return u * u * e2 * (r * r * e2 * D_factor(phi, psi, p) ** 2 + kern.k(psi) ** 2 * g_ff)


def jacobian_rank(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL,
                  rtol: float = 1e-10) -> int:
    """Numerical rank of the 4×3 Jacobian at one point (singular values relative to the largest)."""
    J = tangents(theta, phi, psi, p, kern)
    s = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    return int(np.sum(s > rtol * s[0]))


def singular_locus(p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL, psi_max: float = 6.0,
                   include_boundary: bool = True, n_phi: int = 4001) -> list:
    """Every (φ, ψ) with det g = 0, ψ ∈ [0, psi_max]. Each entry is a circle in θ.

    Boundary circles: ψ = 0, φ ∈ {0, π} (the φ-direction folds because z ≡ 0 at ψ = 0).
    Interior circles: zeros ψ_k of κ, and on each the roots in φ of D(φ, ψ_k) = 0.
    """
    out = []
    if include_boundary:
        out += [{"phi": 0.0, "psi": 0.0, "kind": "boundary fold"}, {"phi": math.pi, "psi": 0.0, "kind": "boundary fold"}]
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi)
    for pk in kern.zeros(psi_max):
        vals = D_factor(phis, pk, p)
        idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        for i in idx:
            f0 = _bisect(lambda f: float(D_factor(f, pk, p)), phis[i], phis[i + 1])
            out.append({"phi": f0, "psi": float(pk), "kind": "kernel zero with D = 0"})
    return out


# ----------------------------------------------------------------------------------------------
# Curvature, route A: Gauss equation (hypersurface in R⁴), exact derivatives
# ----------------------------------------------------------------------------------------------

def _normal(J: Array) -> Array:
    """Generalized cross product of the three columns of J (..., 4, 3); |n|² = det(JᵀJ)."""
    n = np.empty(J.shape[:-2] + (4,))
    for i in range(4):
        rows = [k for k in range(4) if k != i]
        n[..., i] = ((-1) ** i) * np.linalg.det(J[..., rows, :])
    return n


def second_fundamental_form(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """h_ij = X_ij · N with N the unit normal, shape (..., 3, 3)."""
    J = tangents(theta, phi, psi, p, kern)
    n = _normal(J)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    X2 = second_derivatives(theta, phi, psi, p, kern)
    return np.einsum("...ija,...a->...ij", X2, n)


def scalar_curvature_gauss(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """Scal = (tr S)² − tr(S²), S = g⁻¹ h (Gauss equation for a hypersurface of flat R⁴)."""
    g = metric_from_jacobian(theta, phi, psi, p, kern)
    h = second_fundamental_form(theta, phi, psi, p, kern)
    S = np.linalg.solve(g, h)
    trS = np.trace(S, axis1=-2, axis2=-1)
    trS2 = np.trace(S @ S, axis1=-2, axis2=-1)
    return trS * trS - trS2


def principal_curvatures(theta, phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """Eigenvalues of the shape operator at one point (sorted)."""
    g = metric_from_jacobian(theta, phi, psi, p, kern)
    h = second_fundamental_form(theta, phi, psi, p, kern)
    return np.sort(np.real(np.linalg.eigvals(np.linalg.solve(g, h))))


# ----------------------------------------------------------------------------------------------
# Curvature, route B: intrinsic Riemann tensor from the metric field alone (finite differences)
# ----------------------------------------------------------------------------------------------

def _metric_field(x: Sequence[float], p: IRTParams, kern: Kernel) -> Array:
    return metric(x[1], x[2], p, kern)


def christoffel_numeric(x: Sequence[float], p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL,
                        h: float = 1e-5) -> Array:
    """Γ^k_ij at x = (θ, φ, ψ) from central differences of the closed-form metric."""
    x = np.asarray(x, dtype=float)
    g = _metric_field(x, p, kern)
    dg = np.zeros((3, 3, 3))  # dg[l, i, j] = ∂_l g_ij
    for l in range(3):
        e = np.zeros(3)
        e[l] = h
        dg[l] = (_metric_field(x + e, p, kern) - _metric_field(x - e, p, kern)) / (2 * h)
    ginv = np.linalg.inv(g)
    first = 0.5 * (np.einsum("ijl->lij", dg) + np.einsum("jil->lij", dg) - dg)  # Γ_{l,ij}
    return np.einsum("kl,lij->kij", ginv, first)


def riemann_numeric(x: Sequence[float], p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL,
                    h: float = 1e-4) -> Array:
    """R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}."""
    x = np.asarray(x, dtype=float)
    G = christoffel_numeric(x, p, kern)
    dG = np.zeros((3, 3, 3, 3))  # dG[m, k, i, j] = ∂_m Γ^k_ij
    for m in range(3):
        e = np.zeros(3)
        e[m] = h
        dG[m] = (christoffel_numeric(x + e, p, kern) - christoffel_numeric(x - e, p, kern)) / (2 * h)
    Rm = (np.einsum("mrns->rsmn", dG) - np.einsum("nrms->rsmn", dG)
          + np.einsum("rml,lns->rsmn", G, G) - np.einsum("rnl,lms->rsmn", G, G))
    return Rm


def scalar_curvature_intrinsic(x: Sequence[float], p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL,
                               h: float = 1e-4) -> float:
    Rm = riemann_numeric(x, p, kern, h)
    ric = np.einsum("rsrn->sn", Rm)
    ginv = np.linalg.inv(_metric_field(np.asarray(x, dtype=float), p, kern))
    return float(np.einsum("sn,sn->", ginv, ric))


# ----------------------------------------------------------------------------------------------
# Curvature, route C: warped product M = Σ ×_ρ S¹  ⇒  Scal = 2 K_Σ − 2 Δ_Σ ρ / ρ
# ----------------------------------------------------------------------------------------------

def profile_gaussian_curvature(phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> Array:
    """K_Σ of the profile surface Σ(φ, ψ) = (ρ, z, w) ⊂ R³, exact (LN − M²)/(EG − F²)."""
    d = _profile(phi, psi, p)
    k, k1 = kern.k(psi), kern.k1(psi)
    Sf = np.stack(np.broadcast_arrays(d["rho_f"], d["z_f"], 0.0 * d["z_f"]), -1)
    Sp = np.stack(np.broadcast_arrays(d["rho_p"], d["z_p"], -k + 0.0 * d["z_p"]), -1)
    Sff = np.stack(np.broadcast_arrays(d["rho_ff"], d["z_ff"], 0.0 * d["z_ff"]), -1)
    Sfp = np.stack(np.broadcast_arrays(d["rho_fp"], d["z_fp"], 0.0 * d["z_fp"]), -1)
    Spp = np.stack(np.broadcast_arrays(d["rho_pp"], d["z_pp"], -k1 + 0.0 * d["z_pp"]), -1)
    nrm = np.cross(Sf, Sp)
    E, F, G = (Sf * Sf).sum(-1), (Sf * Sp).sum(-1), (Sp * Sp).sum(-1)
    W2 = E * G - F * F
    L, M, N = (Sff * nrm).sum(-1), (Sfp * nrm).sum(-1), (Spp * nrm).sum(-1)
    return (L * N - M * M) / (W2 * W2)  # unnormalized normal: |nrm|² = W2 enters twice


def laplacian_rho_on_profile(phi: float, psi: float, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL,
                             h: float = 1e-4) -> float:
    """Δ_Σ ρ = (1/√|g_Σ|) ∂_a (√|g_Σ| g_Σ^{ab} ∂_b ρ), central differences on the closed-form block."""

    def flux(f, s):
        g = metric(f, s, p, kern)[1:, 1:]
        gi = np.linalg.inv(g)
        sq = math.sqrt(np.linalg.det(g))
        d = _profile(f, s, p)
        grad = np.array([float(d["rho_f"]), float(d["rho_p"])])
        return sq * gi @ grad

    dF = (flux(phi + h, psi)[0] - flux(phi - h, psi)[0]) / (2 * h)
    dP = (flux(phi, psi + h)[1] - flux(phi, psi - h)[1]) / (2 * h)
    sq = math.sqrt(np.linalg.det(metric(phi, psi, p, kern)[1:, 1:]))
    return float((dF + dP) / sq)


def scalar_curvature_warped(phi: float, psi: float, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> float:
    rho = float(_profile(phi, psi, p)["rho"])
    return float(2.0 * profile_gaussian_curvature(phi, psi, p, kern)
                 - 2.0 * laplacian_rho_on_profile(phi, psi, p, kern) / rho)


def slice_gaussian_curvature(phi, psi, p: IRTParams = IRTParams()) -> Array:
    """Gaussian curvature of the ψ-slice T_ψ = {X(θ, φ, ψ)}: a torus of revolution in R³ whose profile is
    the ellipse (ρ, z)(φ) = (e^{−αψ}(R − r cos φ), r sinh(βψ) sin φ).

    K = z′ (ρ′ z″ − ρ″ z′) / (ρ (ρ′² + z′²)²)  (′ = d/dφ). By Gauss–Bonnet ∫ K dA = 2πχ(T²) = 0, so K
    takes both signs on every slice with ψ > 0; it changes sign where cos φ = 0.
    """
    d = _profile(phi, psi, p)
    rf, zf, rff, zff, rho = d["rho_f"], d["z_f"], d["rho_ff"], d["z_ff"], d["rho"]
    return zf * (rf * zff - rff * zf) / (rho * (rf * rf + zf * zf) ** 2)


# ----------------------------------------------------------------------------------------------
# Source C's published curvature claims, transcribed
# ----------------------------------------------------------------------------------------------

def published_scalar_curvature(phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL,
                               reading: str = "sum") -> Array:
    """Source C eq. (29) without its O(r/R) remainder.

    The PDF typesets −2α² over a single denominator. reading="sum" takes the evident intent
    −2α²/r² − 3β²/cosh²βψ + κ″/κ; reading="typeset" takes −2α² / (r² − 3β²/cosh²βψ + κ″/κ).
    """
    psi = np.asarray(psi, dtype=float)
    a, b, r = p.alpha, p.beta, p.r
    kk = kern.k2(psi) / kern.k(psi)
    if reading == "sum":
        return -2 * a * a / (r * r) - 3 * b * b / np.cosh(b * psi) ** 2 + kk + 0.0 * np.asarray(phi)
    if reading == "typeset":
        return -2 * a * a / (r * r - 3 * b * b / np.cosh(b * psi) ** 2 + kk) + 0.0 * np.asarray(phi)
    raise ValueError("reading must be 'sum' or 'typeset'")


def published_christoffels(phi, psi, p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL) -> dict:
    """Source C eq. (28) as printed."""
    a, b, r = p.alpha, p.beta, p.r
    e2 = math.exp(-2 * a * psi)
    u = p.R - r * math.cos(phi)
    gpp = float(published_metric(phi, psi, p, kern)[2, 2])
    return {
        ("theta", "theta", "psi"): -a,
        ("phi", "phi", "psi"): -a + b * math.tanh(b * psi),
        ("psi", "theta", "theta"): -a * u * u * e2 / gpp,
        ("psi", "phi", "phi"): (-a * r * r * e2 + b * r * r * math.sinh(b * psi) * math.cosh(b * psi)) / gpp,
    }


def christoffel_of(metric_fn: Callable[[float, float], Array], x: Sequence[float], h: float = 1e-6) -> Array:
    """Γ^k_ij of any θ-independent metric field metric_fn(φ, ψ) (used to test eq. 28 against eq. 27)."""
    x = np.asarray(x, dtype=float)
    g = metric_fn(x[1], x[2])
    dg = np.zeros((3, 3, 3))
    for l in (1, 2):
        e = np.zeros(3)
        e[l] = h
        dg[l] = (metric_fn(*(x + e)[1:]) - metric_fn(*(x - e)[1:])) / (2 * h)
    first = 0.5 * (np.einsum("ijl->lij", dg) + np.einsum("jil->lij", dg) - dg)
    return np.einsum("kl,lij->kij", np.linalg.inv(g), first)


# ----------------------------------------------------------------------------------------------
# Survey: sign of the curvature over the (φ, ψ) chart
# ----------------------------------------------------------------------------------------------

@dataclass
class CurvatureSurvey:
    params: IRTParams
    kernel: str
    phi: Array = field(repr=False)
    psi: Array = field(repr=False)
    scal: Array = field(repr=False)
    frac_negative: float = 0.0
    frac_positive: float = 0.0
    min: float = 0.0
    max: float = 0.0

    def summary(self) -> dict:
        return {"kernel": self.kernel, "R": self.params.R, "r": self.params.r, "alpha": self.params.alpha,
                "beta": self.params.beta, "frac_negative": self.frac_negative,
                "frac_positive": self.frac_positive, "min": self.min, "max": self.max}


def curvature_survey(p: IRTParams = IRTParams(), kern: Kernel = DAMPED_KERNEL, n_phi: int = 181,
                     n_psi: int = 161, psi_range: Tuple[float, float] = (0.05, 4.0), clip_det: float = 1e-10) -> CurvatureSurvey:
    """Scalar curvature (Gauss route) on a grid; points with det g below clip_det are dropped."""
    phi = np.linspace(0.0, 2 * math.pi, n_phi)
    psi = np.linspace(*psi_range, n_psi)
    PH, PS = np.meshgrid(phi, psi, indexing="ij")
    scal = scalar_curvature_gauss(0.0, PH, PS, p, kern)
    ok = det_metric(PH, PS, p, kern) > clip_det
    vals = scal[ok]
    return CurvatureSurvey(params=p, kernel=kern.name, phi=phi, psi=psi, scal=np.where(ok, scal, np.nan),
                           frac_negative=float(np.mean(vals < 0)), frac_positive=float(np.mean(vals > 0)),
                           min=float(vals.min()), max=float(vals.max()))


# ----------------------------------------------------------------------------------------------
# Symbolic layer (sympy, optional)
# ----------------------------------------------------------------------------------------------

def symbolic_metric():
    """Return (coords, g_jacobian, g_closed, kappa) as sympy objects; g_jacobian = JᵀJ unsimplified."""
    import sympy as sp

    th, ph, ps = sp.symbols("theta phi psi", real=True)
    R, r, a, b = sp.symbols("R r alpha beta", positive=True)
    kap = sp.Function("kappa")(ps)
    u = R - r * sp.cos(ph)
    rho = u * sp.exp(-a * ps)
    X = [rho * sp.cos(th), rho * sp.sin(th), r * sp.sin(ph) * sp.sinh(b * ps)]
    coords = (th, ph, ps)
    J = sp.Matrix([[sp.diff(c, q) for q in coords] for c in X] + [[0, 0, -kap]])
    gJ = (J.T * J)
    e2 = sp.exp(-2 * a * ps)
    g = sp.zeros(3, 3)
    g[0, 0] = u ** 2 * e2
    g[1, 1] = r ** 2 * (sp.sin(ph) ** 2 * e2 + sp.cos(ph) ** 2 * sp.sinh(b * ps) ** 2)
    g[2, 2] = a ** 2 * u ** 2 * e2 + b ** 2 * r ** 2 * sp.sin(ph) ** 2 * sp.cosh(b * ps) ** 2 + kap ** 2
    g[1, 2] = g[2, 1] = r * sp.sin(ph) * (b * r * sp.cos(ph) * sp.sinh(b * ps) * sp.cosh(b * ps) - a * u * e2)
    return coords, gJ, g, kap


def symbolic_metric_check() -> dict:
    """Prove JᵀJ = closed form componentwise (sympy simplify of each difference) and that
    det g matches the Cauchy–Binet closed form (simplify of the difference after rewriting in exp)."""
    import sympy as sp

    (th, ph, ps), gJ, g, kap = symbolic_metric()
    comp = {}
    names = ("theta", "phi", "psi")
    for i in range(3):
        for j in range(i, 3):
            diff = sp.simplify(sp.expand_trig((gJ[i, j] - g[i, j]).rewrite(sp.exp)))
            comp[f"g_{names[i]}{names[j]}"] = diff == 0
    syms = {s.name: s for s in g.free_symbols}
    R, r, a, b = syms["R"], syms["r"], syms["alpha"], syms["beta"]
    u = R - r * sp.cos(ph)
    e2 = sp.exp(-2 * a * ps)
    D = b * r * sp.sin(ph) ** 2 * sp.cosh(b * ps) + a * u * sp.cos(ph) * sp.sinh(b * ps)
    closed = u ** 2 * e2 * (r ** 2 * e2 * D ** 2 + kap ** 2 * g[1, 1])
    det_diff = sp.simplify(sp.expand((g.det() - closed).rewrite(sp.exp)))
    comp["det"] = det_diff == 0
    return comp
