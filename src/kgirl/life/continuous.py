"""Continuous-time counterpart of the scalar subsystem x = (F, O, C, A, R), integrated with RK4.

    Ḟ = η_in E − W − D                                     (projected: F stays ≥ 0)
    Ȯ = a F C (1 − O) − b (D + e²) O
    Ċ = η_C (C_* − C) − η_V (v − v_ref) + γ_C [ 1/(C − C_min) − 1/(C_max − C) ] · h²/4
    Ȧ = η_A (A_target − A) − η_AV v
    Ṙ = η_R (G (1 − R) − L R)

with W = w0 + wA A + wM m + wu c̄ + w_ext X and D = δ0 + δF F + δε e². The exogenous statistics
(E, e², v, m, c̄, X, G, L) are constants or functions of t.

Finding (Level 0): the appendix's confinement term  −γ_C (C − C_min)_+ − γ_C (C_max − C)_+  equals the constant
−γ_C (C_max − C_min) everywhere inside the window, because (C − C_min) + (C_max − C) = C_max − C_min. It is a
uniform downward bias, not a barrier. `paper_confinement` keeps it for comparison. The log-barrier gradient above
is +∞ at C_min and −∞ at C_max, so for bounded forcing C cannot leave (C_min, C_max). The factor h²/4
(h = half-width) makes the barrier vanish to second order at the centre.

Entropy production, all heat released at temperature T into the environment:
    Ṡ_total = [(1 − η_in) E + W + D] / kT ≥ 0          (every term is non-negative).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .core import LifeParams

Forcing = Callable[[float], float] | float


def _at(f: Forcing, t: float) -> float:
    return f(t) if callable(f) else float(f)


def paper_confinement(p: LifeParams, C: float, gamma_C: float) -> float:
    return -gamma_C * max(0.0, C - p.C_min) - gamma_C * max(0.0, p.C_max - C)


def barrier(p: LifeParams, C: float, gamma_C: float) -> float:
    h = p.half_width
    C = min(p.C_max - 1e-6, max(p.C_min + 1e-6, C))     # RK4 stages may probe past the wall; keep the sign
    return gamma_C * (h * h / 4.0) * (1.0 / (C - p.C_min) - 1.0 / (p.C_max - C))


@dataclass
class Exogenous:
    energy: Forcing = 0.5
    e2: Forcing = 0.0
    var: Forcing = 0.0
    model_norm: Forcing = 0.0
    action_cost: Forcing = 1.25
    work: Forcing = 0.0
    G: Forcing = 0.5
    L: Forcing = 0.05


def vector_field(p: LifeParams, x: tuple[float, ...], t: float, ex: Exogenous, gamma_C: float = 0.0,
                 confinement: str = "barrier") -> tuple[float, ...]:
    F, O, C, A, R = x
    E, e2, v = _at(ex.energy, t), _at(ex.e2, t), _at(ex.var, t)
    W = p.w0 + p.wA * A + p.wM * _at(ex.model_norm, t) + p.wu * _at(ex.action_cost, t) + p.w_ext * _at(ex.work, t)
    D = p.delta0 + p.deltaF * F + p.delta_eps * e2
    dF = p.eta_in * E - W - D
    if F <= 0.0 and dF < 0.0:
        dF = 0.0
    dO = p.a * F * C * (1.0 - O) - p.b * (D + e2) * O
    conf = barrier(p, C, gamma_C) if confinement == "barrier" else paper_confinement(p, C, gamma_C)
    dC = p.eta_C * (p.C_star - C) - p.eta_V * (v - p.v_ref) + conf
    dA = p.eta_A * (p.A_target - A) - p.eta_AV * v
    G, L = _at(ex.G, t), _at(ex.L, t)
    dR = p.eta_R * (G * (1.0 - R) - L * R)
    return dF, dO, dC, dA, dR


def entropy_rate(p: LifeParams, x: tuple[float, ...], t: float, ex: Exogenous) -> float:
    F, _, _, A, _ = x
    E, e2 = _at(ex.energy, t), _at(ex.e2, t)
    W = p.w0 + p.wA * A + p.wM * _at(ex.model_norm, t) + p.wu * _at(ex.action_cost, t) + p.w_ext * _at(ex.work, t)
    D = p.delta0 + p.deltaF * F + p.delta_eps * e2
    return ((1.0 - p.eta_in) * E + W + D) / p.kT


@dataclass(frozen=True)
class Trajectory:
    t: tuple[float, ...]
    x: tuple[tuple[float, ...], ...]          # rows (F, O, C, A, R)
    entropy: float                            # ∫ Ṡ_total dt
    min_entropy_rate: float

    def final(self) -> dict[str, float]:
        return dict(zip(("F", "O", "C", "A", "R"), self.x[-1]))


def integrate(p: LifeParams, x0: tuple[float, ...] = (2.0, 0.5, 0.5, 0.5, 0.3), t_end: float = 500.0,
              dt: float = 0.1, ex: Exogenous | None = None, gamma_C: float = 0.0, confinement: str = "barrier",
              every: int = 10) -> Trajectory:
    """Fixed-step RK4 with projection onto the domain (F ≥ 0, O, R, A in their intervals, C inside the window)."""
    ex = ex or Exogenous()
    lo, hi = p.C_min + 1e-6, p.C_max - 1e-6

    def proj(x):
        F, O, C, A, R = x
        if confinement == "barrier" and gamma_C > 0:
            C = min(hi, max(lo, C))
        return (max(0.0, F), min(1.0, max(0.0, O)), C, min(p.A_max, max(0.0, A)), min(1.0, max(0.0, R)))

    def f(x, t):
        return vector_field(p, x, t, ex, gamma_C, confinement)

    x, t, ts, xs = proj(tuple(x0)), 0.0, [0.0], [tuple(x0)]
    s_total, s_min = 0.0, math.inf
    steps = int(round(t_end / dt))
    for i in range(1, steps + 1):
        k1 = f(x, t)
        k2 = f(tuple(a + 0.5 * dt * b for a, b in zip(x, k1)), t + 0.5 * dt)
        k3 = f(tuple(a + 0.5 * dt * b for a, b in zip(x, k2)), t + 0.5 * dt)
        k4 = f(tuple(a + dt * b for a, b in zip(x, k3)), t + dt)
        sr = entropy_rate(p, x, t, ex)
        s_total += sr * dt
        s_min = min(s_min, sr)
        x = proj(tuple(a + dt / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)))
        t = i * dt
        if i % every == 0 or i == steps:
            ts.append(t)
            xs.append(x)
    return Trajectory(tuple(ts), tuple(xs), s_total, s_min)
