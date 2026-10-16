"""Wake-field dynamics (draft E15–E16, Source B §4): solver, energy bookkeeping, and the Source-B audit.

    W_tt + γ W_t − v² ΔW = f(x, t)             (the draft's λ is renamed γ to free λ for scale, F7)

Schemes on a periodic grid (2d + 1-point Laplacian, d = 1 or 2):

  centered  (1 + γΔt/2) W^{n+1} = 2W^n − (1 − γΔt/2) W^{n−1} + Δt² (v² Δ_h W^n + f^n)
            second order; stable iff v Δt/Δx < 1/√d, whatever γ; exact discrete energy identity
            E^{n+½} − E^{n−½} = −γ Δt ‖(W^{n+1} − W^{n−1})/(2Δt)‖² + Δt ⟨f^n, (W^{n+1} − W^{n−1})/(2Δt)⟩,
            E^{n+½} = ½‖(W^{n+1} − W^n)/Δt‖² + ½ v² ⟨∇_h W^{n+1}, ∇_h W^n⟩   (grid sums × cell volume).
  backward  Source B eq. (4.13): W^{n+1} = (2 − γΔt) W^n − (1 − γΔt) W^{n−1} + Δt² (v² Δ_h W^n + f^n)
            first order in the damping term; stable iff v Δt/Δx < √((1 − γΔt/2)/d) and γΔt < 2, so damping
            tightens the CFL limit that Source B states as "v Δt/Δx ≤ c_d".

HybridWake implements the F27 repair of E16: the discrete-time memory H_t enters the PDE through a
zero-order hold, f(x, t) = η Σ_j (H_t)_j φ_j(x) on [tΔ, (t + 1)Δ).

Source-B checks (each returns numbers; kgirl.iwme.verify turns them into PASS/FAIL claims):
  defect_identity     the winding number is carried by curl ∇θ (plaquette circulation 2πm at the core);
                      the discrete divergence of ∇θ vanishes, so "Δ arg ψ = 2π Σ n_k δ" (Prop. 4.2) is false
  coherence_equation  ∂_t c + μ c = −γ_c |∇θ|² drives c below 0 wherever ∇θ ≠ 0: c ∈ [0, 1] is not invariant
  static_source_energy  with a stationary nonzero source the energy (4.6) grows from rest ("decays
                      monotonically if S, c stationary", Prop. 4.1, needs zero forcing)
  steady_point_source  without a restoring term (4.2a) has no localized steady wake; (4.9) needs γ_m W
  decoherence_pulse   uniform mode W_tt + γW_t = βσe^{−σt}: W(∞) = β/γ, independent of σ
  euler_lagrange_4_4  (sympy) the Lagrangian (4.4) yields ξ(1 − c) and v²U′(W), not β∂_t(1 − c)

numpy only (sympy for euler_lagrange_4_4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ----------------------------------------------------------------------------------------------
# Grid operators (periodic)
# ----------------------------------------------------------------------------------------------

def laplacian(W: Array, dx: float) -> Array:
    out = -2.0 * W.ndim * W
    for ax in range(W.ndim):
        out = out + np.roll(W, 1, ax) + np.roll(W, -1, ax)
    return out / (dx * dx)


def grad_dot(U: Array, W: Array, dx: float) -> float:
    """⟨∇_h U, ∇_h W⟩ with forward differences, summed × cell volume (consistent with `laplacian`)."""
    s = 0.0
    for ax in range(U.ndim):
        s += float(((np.roll(U, -1, ax) - U) * (np.roll(W, -1, ax) - W)).sum())
    return s / (dx * dx) * dx ** U.ndim


# ----------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------

@dataclass
class WakeRun:
    W: Array
    energy: Array            # E^{n+½}
    work: Array              # cumulative forcing work Δt Σ ⟨f, W_t⟩
    dissipated: Array        # cumulative damping loss γ Δt Σ ‖W_t‖²
    max_abs: Array
    scheme: str
    stable: bool


def run_wake(W0: Array, V0: Array, *, v: float = 1.0, gamma: float = 0.1, dx: float = 1.0, dt: float = 0.5,
             steps: int = 200, scheme: str = "centered",
             force: Optional[Callable[[int, float], Array]] = None, blowup: float = 1e12) -> WakeRun:
    """Integrate W_tt + γW_t − v²ΔW = f with the chosen scheme; W^1 from a Taylor start."""
    if scheme not in ("centered", "backward"):
        raise ValueError("scheme must be 'centered' or 'backward'")
    vol = dx ** W0.ndim
    f0 = np.zeros_like(W0) if force is None else force(0, 0.0)
    Wm = np.asarray(W0, dtype=float)
    W = Wm + dt * V0 + 0.5 * dt * dt * (v * v * laplacian(Wm, dx) - gamma * V0 + f0)
    a = 0.5 * gamma * dt
    E, work, diss, mx = [], [0.0], [0.0], []

    def energy(Wnew, Wold):
        Dp = (Wnew - Wold) / dt
        return 0.5 * float((Dp * Dp).sum()) * vol + 0.5 * v * v * grad_dot(Wnew, Wold, dx)

    E.append(energy(W, Wm))
    mx.append(float(np.abs(W).max()))
    stable = True
    for n in range(1, steps):
        f = np.zeros_like(W) if force is None else force(n, n * dt)
        L = laplacian(W, dx)
        if scheme == "centered":
            Wp = (2.0 * W - (1.0 - a) * Wm + dt * dt * (v * v * L + f)) / (1.0 + a)
        else:
            Wp = (2.0 - gamma * dt) * W - (1.0 - gamma * dt) * Wm + dt * dt * (v * v * L + f)
        Wt = (Wp - Wm) / (2.0 * dt)
        work.append(work[-1] + dt * float((f * Wt).sum()) * vol)
        diss.append(diss[-1] + gamma * dt * float((Wt * Wt).sum()) * vol)
        Wm, W = W, Wp
        E.append(energy(W, Wm))
        m = float(np.abs(W).max())
        mx.append(m)
        if not np.isfinite(m) or m > blowup:
            stable = False
            break
    return WakeRun(W=W, energy=np.array(E), work=np.array(work), dissipated=np.array(diss),
                   max_abs=np.array(mx), scheme=scheme, stable=stable)


# ----------------------------------------------------------------------------------------------
# von Neumann analysis
# ----------------------------------------------------------------------------------------------

def amplification(scheme: str, c2: float, gamma_dt: float) -> float:
    """Largest |root| of the scheme's characteristic polynomial for one Fourier mode, c² = (vΔt k̃)²."""
    if scheme == "centered":
        a = 0.5 * gamma_dt
        coeffs = [1.0 + a, -(2.0 - c2), 1.0 - a]
    elif scheme == "backward":
        coeffs = [1.0, -(2.0 - gamma_dt - c2), 1.0 - gamma_dt]
    else:
        raise ValueError(scheme)
    return float(max(abs(np.roots(coeffs))))


def cfl_limit(scheme: str, gamma_dt: float, d: int) -> float:
    """Largest v Δt/Δx with all modes non-growing (closed form from the Jury conditions)."""
    if scheme == "centered":
        return 1.0 / math.sqrt(d)
    if scheme == "backward":
        if not 0.0 <= gamma_dt < 2.0:
            return 0.0                  # |1 − γΔt| ≥ 1: the product of the roots leaves the unit disk
        return math.sqrt((1.0 - 0.5 * gamma_dt) / d)   # γΔt = 0 is plain leapfrog (marginal, |ξ| = 1)
    raise ValueError(scheme)


def cfl_limit_numeric(scheme: str, gamma_dt: float, d: int, tol: float = 1e-10) -> float:
    """Bisection on the Courant number using `amplification` over the full discrete spectrum k̃² ∈ [0, 4d/Δx²]."""
    def ok(cour):
        for c2 in np.linspace(1e-6, 4.0 * d * cour * cour, 400):
            if amplification(scheme, c2, gamma_dt) > 1.0 + 1e-12:
                return False
        return True
    lo, hi = 0.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        if hi - lo < tol:
            break
    return lo


def temporal_order(scheme: str, omega: float = 1.0, gamma: float = 0.5, T_end: float = 4.0,
                   dts: Sequence[float] = (0.04, 0.02, 0.01, 0.005)) -> float:
    """Observed order of the scheme on the mode ODE W'' + γW' + ω²W = 0 (W(0) = 1, W'(0) = 0)."""
    disc = gamma * gamma / 4.0 - omega * omega
    if disc >= 0:
        raise ValueError("choose an underdamped mode")
    wd = math.sqrt(-disc)
    exact = math.exp(-gamma * T_end / 2) * (math.cos(wd * T_end) + gamma / (2 * wd) * math.sin(wd * T_end))
    errs = []
    for dt in dts:
        n = int(round(T_end / dt))
        Wm = 1.0
        # Taylor start to O(dt³): W'' = −ω², W''' = γω²
        W = 1.0 - 0.5 * omega * omega * dt * dt + gamma * omega * omega * dt ** 3 / 6.0
        for _ in range(1, n):
            if scheme == "centered":
                a = 0.5 * gamma * dt
                Wp = (2 * W - (1 - a) * Wm - dt * dt * omega * omega * W) / (1 + a)
            else:
                Wp = (2 - gamma * dt) * W - (1 - gamma * dt) * Wm - dt * dt * omega * omega * W
            Wm, W = W, Wp
        errs.append(abs(W - exact))
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    return float(slope)


# ----------------------------------------------------------------------------------------------
# F27 repair: memory → wake through a zero-order hold
# ----------------------------------------------------------------------------------------------

@dataclass
class HybridWake:
    """Couples a discrete memory sequence H_t ∈ R^J to the wake via f = η Σ_j (H_t)_j φ_j(x), held for
    `hold` wave steps per memory step (Δ = hold · dt)."""

    basis: Array                     # (J, *grid) spatial modes φ_j
    eta: float = 1.0
    hold: int = 10

    def forcing(self, H_seq: Array) -> Callable[[int, float], Array]:
        H_seq = np.atleast_2d(np.asarray(H_seq, dtype=float))

        def f(n: int, t: float) -> Array:
            k = min(n // self.hold, len(H_seq) - 1)
            return self.eta * np.tensordot(H_seq[k], self.basis, axes=(0, 0))
        return f


# ----------------------------------------------------------------------------------------------
# Source B audit: phase defects
# ----------------------------------------------------------------------------------------------

def wrap(a: Array) -> Array:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def vortex_phase(n: int = 64, m: int = 1, center: Tuple[float, float] = (0.5, 0.5)) -> Tuple[Array, float]:
    """θ = m·atan2(y − y0, x − x0) on an n×n grid of [0, 1]² (core placed inside a plaquette)."""
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    return m * np.arctan2(Y - center[1], X - center[0]), h


def defect_identity(n: int = 64, m: int = 1, loops: Sequence[int] = (1, 2, 4, 8, 16, 31)) -> dict:
    """Discrete check of the distributional identities for an ideal vortex of charge m.

    For square loops of half-width k nodes around the core (which sits inside the central plaquette):
      circulation  Σ of wrapped phase differences along the loop       → ∮ ∇θ · dℓ = 2πm   (curl ∇θ = 2πm δ)
      flux         Σ of wrapped normal differences across the loop      → ∮ ∇θ · n ds = 0    (div ∇θ = Δθ = 0)
    Source B (4.8) asserts Δ arg ψ = 2π Σ n_k δ, i.e. flux 2πm through every loop around the core.
    The flux equals the sum of the node-wise discrete Laplacians inside the loop (discrete divergence theorem).
    """
    th, h = vortex_phase(n, m)
    dx = wrap(np.diff(th, axis=0))          # dx[i, j] = θ(i+1, j) − θ(i, j)
    dy = wrap(np.diff(th, axis=1))          # dy[i, j] = θ(i, j+1) − θ(i, j)
    c = n // 2 - 1                          # the core lies between nodes c and c + 1 in each direction
    circ_out, flux_out = [], []
    for k in loops:
        lo, hi = c - k + 1, c + k           # nodes lo..hi bound the loop
        if lo < 1 or hi > n - 2:
            continue
        # counter-clockwise: bottom (j = lo) left→right, right (i = hi) up, top (j = hi) right→left, left down
        circ = (dx[lo:hi, lo].sum() + dy[hi, lo:hi].sum() - dx[lo:hi, hi].sum() - dy[lo, lo:hi].sum())
        # outward normal differences across the four sides, one cell outside the loop
        flux = (dx[hi, lo:hi + 1].sum() - dx[lo - 1, lo:hi + 1].sum()
                + dy[lo:hi + 1, hi].sum() - dy[lo:hi + 1, lo - 1].sum())
        circ_out.append(float(circ))
        flux_out.append(float(flux))
    # per-node discrete Laplacian (interior) and its sum, which telescopes to the outer flux
    lap = (dx[1:, 1:-1] - dx[:-1, 1:-1] + dy[1:-1, 1:] - dy[1:-1, :-1])
    return {
        "m": m,
        "loop_half_widths": [k for k in loops if c - k + 1 >= 1 and c + k <= n - 2],
        "circulation": circ_out,
        "flux": flux_out,
        "max_abs_flux": float(np.max(np.abs(flux_out))),
        "sum_node_laplacians": float(lap.sum()),
        "source_b_prediction_flux": 2.0 * np.pi * m,
    }


def coherence_equation(c0: float = 1.0, mu: float = 1.0, gamma_c: float = 0.5, grad2: float = 1.0) -> dict:
    """Closed form of (4.2c) for constant |∇θ|² = G: c(t) = c0 e^{−μt} − (γ_c G/μ)(1 − e^{−μt})."""
    c_inf = -gamma_c * grad2 / mu
    t_zero = math.log(1.0 + mu * c0 / (gamma_c * grad2)) / mu if grad2 > 0 else math.inf
    return {"c_infinity": c_inf, "t_cross_zero": t_zero, "leaves_unit_interval": c_inf < 0.0}


def coherence_near_vortex(r: Array, m: int = 1, mu: float = 1.0, gamma_c: float = 0.5) -> Array:
    """Steady state of (4.2c) around a charge-m vortex: c* = −γ_c m²/(μ r²) → −∞ at the core."""
    return -gamma_c * m * m / (mu * np.asarray(r, dtype=float) ** 2)


# ----------------------------------------------------------------------------------------------
# Source B audit: energy, steady states, pulse
# ----------------------------------------------------------------------------------------------

def static_source_energy(n: int = 128, v: float = 1.0, gamma: float = 0.2, steps: int = 400) -> dict:
    """1D periodic box, W(0) = W_t(0) = 0, stationary forcing f = α ΔS with S a Gaussian bump."""
    L = 1.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    S = np.exp(-((x - 0.5) / 0.05) ** 2)
    f = laplacian(S, dx) * 1e-3
    dt = 0.5 * dx / v
    run = run_wake(np.zeros(n), np.zeros(n), v=v, gamma=gamma, dx=dx, dt=dt, steps=steps,
                   force=lambda k, t: f)
    E = run.energy
    return {"E_start": float(E[0]), "E_max": float(E.max()), "E_end": float(E[-1]),
            "first_increase_step": int(np.argmax(np.diff(E) > 0)) if np.any(np.diff(E) > 0) else -1,
            "monotone_decay": bool(np.all(np.diff(E) <= 1e-15))}


def steady_point_source(L: float, v: float = 1.0, mass: float = 0.0, A: float = 1.0, n: int = 4001) -> Tuple[Array, Array]:
    """Solve −v² W″ + mass·W = A δ(x) on [−L, L], W(±L) = 0, by finite differences."""
    x = np.linspace(-L, L, n)
    h = x[1] - x[0]
    N = n - 2
    main = np.full(N, 2 * v * v / (h * h) + mass)
    off = np.full(N - 1, -v * v / (h * h))
    rhs = np.zeros(N)
    rhs[(n - 1) // 2 - 1] = A / h
    W = np.zeros(n)
    W[1:-1] = solve_tridiagonal(off, main, off, rhs)
    return x, W


def solve_tridiagonal(lower: Array, diag: Array, upper: Array, rhs: Array) -> Array:
    """Thomas algorithm (diagonally dominant systems), O(n)."""
    n = len(diag)
    c = np.zeros(n - 1)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / den
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / den
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def eq_4_9(x: Array, v: float, mass: float, A: float) -> Array:
    """Source B (4.9) with αA → A: W = A/(2v√γ) exp(−√γ |x|/v)."""
    return A / (2 * v * math.sqrt(mass)) * np.exp(-math.sqrt(mass) * np.abs(x) / v)


def decoherence_pulse(beta: float = 1.0, sigma: float = 2.0, gamma: float = 0.5, T_end: float = 80.0,
                      dt: float = 1e-3) -> dict:
    """Uniform mode of (4.2a): W″ + γW′ = βσ e^{−σt}, W(0) = W′(0) = 0. Closed form W(∞) = β/γ."""
    n = int(T_end / dt)
    W, V = 0.0, 0.0
    for k in range(n):                      # RK4
        t = k * dt
        def fdot(tt, w, vv):
            return vv, beta * sigma * math.exp(-sigma * tt) - gamma * vv
        k1 = fdot(t, W, V)
        k2 = fdot(t + dt / 2, W + dt / 2 * k1[0], V + dt / 2 * k1[1])
        k3 = fdot(t + dt / 2, W + dt / 2 * k2[0], V + dt / 2 * k2[1])
        k4 = fdot(t + dt, W + dt * k3[0], V + dt * k3[1])
        W += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        V += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return {"W_final": W, "closed_form": beta / gamma, "source_b_scaling": beta * sigma / gamma}


def point_source_energy_2d(h: float) -> float:
    """Dirichlet energy ∫_{h<r<1} |∇W|² of the 2D Green's function W = −ln r/(2π): (1/2π) ln(1/h) → ∞,
    so a point source puts the wake outside H¹ (Theorem 4.1's energy space) for d = 2."""
    return math.log(1.0 / h) / (2.0 * math.pi)


# ----------------------------------------------------------------------------------------------
# Source B audit: Euler–Lagrange of the Lagrangian (4.4)
# ----------------------------------------------------------------------------------------------

def euler_lagrange_4_4() -> dict:
    """Derive the field equation of L = W_t²/(2v²) − |∇W|²/2 − ω²W²/2 + χ W ΔS + ξ W (1 − c) in 1D."""
    import sympy as sp

    t, x = sp.symbols("t x", real=True)
    v, om, chi, xi, al, be = sp.symbols("v omega chi xi alpha beta", positive=True)
    W = sp.Function("W")(t, x)
    S = sp.Function("S")(t, x)
    c = sp.Function("c")(t, x)
    Lag = W.diff(t) ** 2 / (2 * v * v) - W.diff(x) ** 2 / 2 - om * om * W * W / 2 + chi * W * S.diff(x, 2) + xi * W * (1 - c)
    el = sp.euler_equations(Lag, [W], [t, x])[0].lhs
    # normalize to W_tt − v² W_xx + … = …  (multiply by −v²)
    field_eq = sp.expand(-v * v * el)
    target = sp.expand(W.diff(t, 2) - v * v * W.diff(x, 2) - al * S.diff(x, 2) - be * (1 - c).diff(t))
    residual = sp.simplify(field_eq - target)
    return {"euler_lagrange": sp.sstr(sp.Eq(field_eq, 0)),
            "minus_source_b_4_2a_without_damping": sp.sstr(residual),
            "has_restoring_term": residual.has(om),
            "coherence_source_is_time_derivative": sp.expand(field_eq).has(c.diff(t))}
