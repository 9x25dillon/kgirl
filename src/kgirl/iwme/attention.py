"""Attention–memory core of IWME: softmax attention with temperature T feeding an EMA memory.

    scores   s_i = ⟨H + b, k_i⟩ / T            (b: an input-driven bias, default 0)
    weights  a = softmax(s)                     Gibbs distribution with energies −⟨H + b, k_i⟩
    read     m(H) = Σ_i a_i v_i
    memory   Φ(H) = (1 − ρ) H + ρ m(H)          (draft eq. E4 with the loop closed through the scores)

What this module establishes (each function is exercised by kgirl.iwme.verify):

  P2 (open loop)   With an attention sequence fixed in advance, ∂H_t/∂H_s = (1 − ρ)^{t−s} I: retention
                   is set by ρ alone. (`open_loop_sensitivity`)
  P2′ (closed loop) When the scores read the memory, DΦ(H) = (1 − ρ) I + (ρ/T) Cov_a(v, k), so the
                   sensitivity of H_t to H_s is the product of these Jacobians along the trajectory and
                   depends on T and on the key–value geometry. Attention changes retention dynamics in
                   closed loop. (`jacobian`, `closed_loop_sensitivity`, `effective_time_constant`)
  Lemma            ‖Cov_a(v, k)‖ ≤ D_v D_k / 4 for every attention distribution a (Popoviciu), so Φ is a
                   contraction whenever T > D_v D_k / 4. The bound is attained by the two-point example.
  P3               Two-point example v = k = (+1, −1): m(H) = tanh(H/T). T > 1: unique stable fixed point 0.
                   T = 1: still unique, but convergence is algebraic, H_t ≈ √(3 / (2ρt)). T < 1: 0 unstable,
                   ±H*(T) stable, H* ≈ T √(3(1 − T)) near T = 1. (`fixed_points_tanh`, `critical_decay`)
  P3′              At a centered symmetric point the bifurcation type is read from the spectrum c_j of
                   C₀ = Cov_uniform(v, k): real c > 0 crosses +1 at T = c (pitchfork, lock-in); real c < 0
                   crosses −1 at T = ρ|c|/(2 − ρ) (flip, period-2 memory); complex c gives a Neimark–Sacker
                   crossing (quasi-periodic memory). (`linear_thresholds`)
  Hysteresis       Sweeping the input bias b up and down traces a loop of positive area only for T < 1.
                   (`hysteresis_area`)
  Gated forgetting A forget gate ρ_t = ρ(a_t) makes retention attention-dependent even in open loop.
                   (`GatedMemory`)

Levels: everything here is Level 0 (standard dynamical systems; Curie–Weiss / Hopfield mean field). The
IWME-specific content is the identification of T as the control parameter of the program's ablations.
numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ----------------------------------------------------------------------------------------------
# Softmax attention and the memory map
# ----------------------------------------------------------------------------------------------

def softmax(s: Array) -> Array:
    s = np.asarray(s, dtype=float)
    z = s - s.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class AttentionMemory:
    """Keys K (n×d), values V (n×d'), temperature T, write rate ρ. With d' = d the map Φ: R^d → R^d."""

    K: Array
    V: Array
    T: float = 1.0
    rho: float = 0.5

    def __post_init__(self) -> None:
        K = np.atleast_2d(np.asarray(self.K, dtype=float))
        V = np.atleast_2d(np.asarray(self.V, dtype=float))
        if K.shape[0] != V.shape[0]:
            raise ValueError("keys and values need the same number of items")
        if not (self.T > 0.0 and 0.0 < self.rho <= 1.0):
            raise ValueError("need T > 0 and 0 < rho <= 1")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "V", V)

    def weights(self, H: Array, b: Optional[Array] = None) -> Array:
        q = np.asarray(H, dtype=float) + (0.0 if b is None else np.asarray(b, dtype=float))
        return softmax(self.K @ q / self.T)

    def read(self, H: Array, b: Optional[Array] = None) -> Array:
        return self.weights(H, b) @ self.V

    def step(self, H: Array, b: Optional[Array] = None) -> Array:
        return (1.0 - self.rho) * np.asarray(H, dtype=float) + self.rho * self.read(H, b)

    def jacobian(self, H: Array, b: Optional[Array] = None) -> Array:
        """DΦ(H) = (1 − ρ) I + (ρ/T) Σ_i a_i (v_i − m)(k_i − k̄)ᵀ."""
        a = self.weights(H, b)
        m = a @ self.V
        kbar = a @ self.K
        cov = (self.V - m).T @ (a[:, None] * (self.K - kbar))
        return (1.0 - self.rho) * np.eye(self.V.shape[1], self.K.shape[1]) + (self.rho / self.T) * cov

    def run(self, H0: Array, steps: int, b: Optional[Callable[[int], Array]] = None) -> Array:
        H = np.asarray(H0, dtype=float).copy()
        out = np.empty((steps + 1,) + H.shape)
        out[0] = H
        for t in range(steps):
            H = self.step(H, None if b is None else b(t))
            out[t + 1] = H
        return out


def diameter(P: Array) -> float:
    P = np.atleast_2d(np.asarray(P, dtype=float))
    diff = P[:, None, :] - P[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max())


def cov_bound(K: Array, V: Array) -> float:
    """Popoviciu bound: sup_a ‖Cov_a(v, k)‖_op ≤ D_v D_k / 4."""
    return diameter(V) * diameter(K) / 4.0


def contraction_temperature(K: Array, V: Array) -> float:
    """T above which Φ is a global contraction for every ρ ∈ (0, 1]: T > D_v D_k / 4."""
    return cov_bound(K, V)


def sup_cov_norm(mem: AttentionMemory, H_samples: Array) -> float:
    """max over sampled H of ‖Cov_a(H)(v, k)‖_op (to compare with the Popoviciu bound)."""
    best = 0.0
    for H in np.atleast_2d(H_samples):
        a = mem.weights(H)
        m, kbar = a @ mem.V, a @ mem.K
        cov = (mem.V - m).T @ (a[:, None] * (mem.K - kbar))
        best = max(best, float(np.linalg.norm(cov, 2)))
    return best


# ----------------------------------------------------------------------------------------------
# P2 and its closed-loop qualification
# ----------------------------------------------------------------------------------------------

def open_loop_sensitivity(rho: float, lag: int) -> float:
    """∂H_t/∂H_{t−lag} for the EMA with an attention sequence fixed in advance: (1 − ρ)^lag."""
    return (1.0 - rho) ** lag


def closed_loop_sensitivity(mem: AttentionMemory, H0: Array, lag: int,
                            b: Optional[Callable[[int], Array]] = None) -> Array:
    """∂H_lag/∂H_0 along the closed-loop trajectory: the ordered product of DΦ(H_τ)."""
    H = np.asarray(H0, dtype=float).copy()
    S = np.eye(H.shape[0])
    for t in range(lag):
        bt = None if b is None else b(t)
        S = mem.jacobian(H, bt) @ S
        H = mem.step(H, bt)
    return S


def finite_difference_sensitivity(mem: AttentionMemory, H0: Array, lag: int, eps: float = 1e-6) -> Array:
    """Independent check of closed_loop_sensitivity by perturbing H0."""
    H0 = np.asarray(H0, dtype=float)
    d = H0.shape[0]
    S = np.empty((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        S[:, j] = (mem.run(H0 + e, lag)[-1] - mem.run(H0 - e, lag)[-1]) / (2 * eps)
    return S


def effective_time_constant(multiplier: float) -> float:
    """Steps for a perturbation to shrink by e: −1 / ln|μ| (∞ at |μ| = 1)."""
    mu = abs(multiplier)
    if mu >= 1.0:
        return math.inf
    if mu == 0.0:
        return 0.0
    return -1.0 / math.log(mu)


# ----------------------------------------------------------------------------------------------
# P3: the exactly solvable two-point example
# ----------------------------------------------------------------------------------------------

def two_point_memory(T: float, rho: float = 0.5, aligned: bool = True) -> AttentionMemory:
    """k = (+1, −1); v = k (aligned, m = tanh(H/T)) or v = −k (anti-aligned, m = −tanh(H/T))."""
    K = np.array([[1.0], [-1.0]])
    return AttentionMemory(K=K, V=K if aligned else -K, T=T, rho=rho)


def fixed_points_tanh(T: float, tol: float = 1e-15) -> Tuple[float, ...]:
    """Fixed points of H = tanh(H/T): (0,) for T ≥ 1, (−H*, 0, H*) for T < 1."""
    if T >= 1.0:
        return (0.0,)
    lo, hi = 1e-300, 1.0          # g(H) = tanh(H/T) − H: g > 0 just above 0, g(1) < 0
    g = lambda h: math.tanh(h / T) - h  # noqa: E731
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    hs = 0.5 * (lo + hi)
    return (-hs, 0.0, hs)


def multiplier_tanh(H: float, T: float, rho: float, aligned: bool = True) -> float:
    """dΦ/dH for the two-point example: 1 − ρ ± ρ sech²(H/T)/T."""
    s = 1.0 / math.cosh(H / T) ** 2
    return 1.0 - rho + (rho / T) * s * (1.0 if aligned else -1.0)


def critical_decay(rho: float = 0.5, H0: float = 0.5, steps: int = 200000) -> dict:
    """At T = 1 the map H ↦ (1 − ρ)H + ρ tanh H converges algebraically: H_t √t → √(3/(2ρ))."""
    mem = two_point_memory(1.0, rho)
    H = np.array([H0])
    ts = np.unique(np.geomspace(10, steps, 25).astype(int))
    out, t = [], 0
    for target in ts:
        while t < target:
            H = mem.step(H)
            t += 1
        out.append((int(t), float(H[0]), float(H[0]) * math.sqrt(t)))
    return {"rho": rho, "trace": out, "limit": math.sqrt(3.0 / (2.0 * rho)),
            "final_scaled": out[-1][2]}


def decay_exponent(T: float, rho: float = 0.5, H0: float = 0.5, t1: int = 200, t2: int = 2000) -> float:
    """Local log–log slope d ln|H_t| / d ln t between t1 and t2 (≈ −1/2 at T = 1; steepening for T > 1)."""
    traj = two_point_memory(T, rho).run(np.array([H0]), t2)[:, 0]
    return float((math.log(abs(traj[t2])) - math.log(abs(traj[t1]))) / (math.log(t2) - math.log(t1)))


def bifurcation_diagram(Ts: Sequence[float], rho: float = 0.5) -> list:
    rows = []
    for T in Ts:
        fps = fixed_points_tanh(T)
        rows.append({"T": float(T), "fixed_points": fps,
                     "stable": tuple(abs(multiplier_tanh(h, T, rho)) < 1.0 for h in fps)})
    return rows


def flip_threshold(rho: float, c: float = 1.0) -> float:
    """Anti-aligned (negative c) crossing of −1: T_f = ρ|c| / (2 − ρ)."""
    return rho * abs(c) / (2.0 - rho)


def period_two_amplitude(T: float, rho: float = 0.5, H0: float = 0.3, burn: int = 4000) -> float:
    """Amplitude of the attractor of the anti-aligned two-point map (0 when the fixed point is stable)."""
    traj = two_point_memory(T, rho, aligned=False).run(np.array([H0]), burn + 2)[:, 0]
    return float(0.5 * abs(traj[-1] - traj[-2]))


# ----------------------------------------------------------------------------------------------
# P3′: bifurcation type from the key–value cross-covariance
# ----------------------------------------------------------------------------------------------

def uniform_cross_covariance(K: Array, V: Array) -> Array:
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    return (V - V.mean(0)).T @ (K - K.mean(0)) / K.shape[0]


def linear_thresholds(K: Array, V: Array, rho: float) -> list:
    """For centered values (H = 0 a fixed point), the temperatures at which an eigenvalue μ_j(T) of
    (1 − ρ) I + (ρ/T) C₀ leaves the unit disk, with the type of crossing."""
    C0 = uniform_cross_covariance(K, V)
    out = []
    for c in np.linalg.eigvals(C0):
        if abs(c.imag) < 1e-12:
            c = float(c.real)
            if c > 1e-12:
                out.append({"c": c, "T_crit": c, "type": "pitchfork (μ → +1): multistable lock-in"})
            elif c < -1e-12:
                out.append({"c": c, "T_crit": flip_threshold(rho, c), "type": "flip (μ → −1): period-2 memory"})
        else:
            # |1 − ρ + ρ c / T| = 1  ⇔  ρ²|c|²/T² + 2ρ(1 − ρ) Re c / T + (1 − ρ)² − 1 = 0, positive root in 1/T
            A, B, Cc = rho * rho * abs(c) ** 2, 2 * rho * (1 - rho) * c.real, (1 - rho) ** 2 - 1.0
            disc = B * B - 4 * A * Cc
            x = (-B + math.sqrt(disc)) / (2 * A)
            out.append({"c": complex(c), "T_crit": 1.0 / x, "type": "Neimark–Sacker (|μ| → 1, complex): quasi-periodic memory"})
    return out


def spectral_radius_at_zero(K: Array, V: Array, T: float, rho: float) -> float:
    C0 = uniform_cross_covariance(K, V)
    J = (1.0 - rho) * np.eye(C0.shape[0]) + (rho / T) * C0
    return float(np.max(np.abs(np.linalg.eigvals(J))))


# ----------------------------------------------------------------------------------------------
# History dependence: hysteresis under a slow input sweep
# ----------------------------------------------------------------------------------------------

def hysteresis_loop(T: float, rho: float = 0.5, b_max: float = 1.0, n: int = 400, settle: int = 60) -> Tuple[Array, Array, Array]:
    """Sweep b: −b_max → +b_max → −b_max with `settle` memory steps per value; return (b_up, H_up, H_down)."""
    mem = two_point_memory(T, rho)
    bs = np.linspace(-b_max, b_max, n)
    H = np.array([-1.0])
    up, down = np.empty(n), np.empty(n)
    for i, b in enumerate(bs):
        for _ in range(settle):
            H = mem.step(H, np.array([b]))
        up[i] = H[0]
    for i, b in enumerate(bs[::-1]):
        for _ in range(settle):
            H = mem.step(H, np.array([b]))
        down[n - 1 - i] = H[0]
    return bs, up, down


def _trapz(y: Array, x: Array) -> float:
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def hysteresis_area(T: float, rho: float = 0.5, b_max: float = 1.0, n: int = 400, settle: int = 60) -> float:
    """Loop area at a finite sweep rate (`settle` memory steps per input value).

    Near T = 1 relaxation is slow (critical slowing), so a finite-rate sweep lags and shows a small
    dynamic loop even for T ≥ 1; `static_hysteresis_area` gives the adiabatic limit.
    """
    bs, up, down = hysteresis_loop(T, rho, b_max, n, settle)
    return _trapz(down - up, bs)


def _equilibrium_branch(T: float, bs: Array, H_start: float) -> Array:
    """Follow a solution of H = tanh((H + b)/T) along bs by Newton continuation; jump when it vanishes."""
    out = np.empty(len(bs))
    H = H_start
    for i, b in enumerate(bs):
        for _ in range(200):
            th = math.tanh((H + b) / T)
            f = H - th
            df = 1.0 - (1.0 - th * th) / T
            if abs(df) < 1e-12:
                break
            step = f / df
            H_new = H - step
            if abs(H_new) > 1.5 or (df < 0):     # left the stable branch: saddle-node passed
                break
            H = H_new
            if abs(step) < 1e-14:
                break
        if abs(H - math.tanh((H + b) / T)) > 1e-10 or 1.0 - (1.0 - math.tanh((H + b) / T) ** 2) / T <= 0:
            # branch lost: relax to the surviving stable equilibrium by iterating the map
            for _ in range(20000):
                H = math.tanh((H + b) / T)
        out[i] = H
    return out


def static_hysteresis_area(T: float, b_max: float = 1.0, n: int = 2001) -> float:
    """Adiabatic loop area ∮ H db between the up- and down-sweep equilibrium branches.

    For T ≥ 1 the equation H = tanh((H + b)/T) has a unique solution for every b (H − tanh is monotone),
    so the area is exactly 0. For T < 1 the two stable branches coexist on a band of b and the area is > 0.
    """
    bs = np.linspace(-b_max, b_max, n)
    up = _equilibrium_branch(T, bs, -1.0)
    down = _equilibrium_branch(T, bs[::-1], 1.0)[::-1]
    return _trapz(down - up, bs)


# ----------------------------------------------------------------------------------------------
# Gated forgetting: retention controlled by attention in open loop
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GatedMemory:
    """H_{t+1} = (1 − ρ_t) H_t + ρ_t m_t with ρ_t = ρ_min + (ρ_max − ρ_min) · (1 − Ĥ(a_t)).

    Ĥ is the normalized attention entropy: diffuse attention (Ĥ → 1) writes slowly and retains, sharp
    attention (Ĥ → 0) overwrites. Impulse response at lag k: Π (1 − ρ_τ), attention-dependent.
    """

    rho_min: float = 0.05
    rho_max: float = 0.8

    def rate(self, a: Array) -> float:
        a = np.asarray(a, dtype=float)
        n = a.shape[-1]
        if n < 2:
            return self.rho_max
        nz = a[a > 0]
        ent = float(-(nz * np.log(nz)).sum() / math.log(n))
        return self.rho_min + (self.rho_max - self.rho_min) * (1.0 - ent)

    def impulse_response(self, attention_seq: Sequence[Array]) -> Array:
        """∂H_t/∂H_0 for t = 0..len(seq) with the attention sequence fixed (open loop)."""
        out = [1.0]
        for a in attention_seq:
            out.append(out[-1] * (1.0 - self.rate(a)))
        return np.array(out)
