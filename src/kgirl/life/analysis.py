"""Closed forms for the life equation and an observable replacement for the "99.9% ceiling".

Steady state under stationary statistics (Ē, e² = E‖ε‖², v = Var ε, m = ‖M‖_F, c̄, X̄), from ẋ = α − βx:

    A*  = clip(A_target − (η_AV/η_A) v, 0, A_max)
    C*  = clip(C_* − (η_V/η_C)(v − v_ref), C_min + margin, C_max − margin)
    F*  = [η_in Ē − (w0 + wA A* + wM m + wu c̄ + w_ext X̄) − δ0 − δε e²]_+ / δF       (τ_F = 1/δF)
    O*  = g / (g + ℓ),   g = a F* C*,   ℓ = b (δ0 + δF F* + (1 + δε) e²)              (τ_O = 1/(g + ℓ))
    R*  = G / (G + L)

Viability (O* ≥ O_min) holds iff F* ≥ F_crit with

    F_crit = b O_min (δ0 + (1 + δε) e²) / (a C* (1 − O_min) − b δF O_min),

and no free energy suffices when the denominator is ≤ 0. Feeding F_crit back through F* gives the critical
inflow E_crit. These are Level 0 algebra on a Level 2 model; tests/life checks them against simulation.

Euler multiplier of the organization map: λ_O = 1 − (g + ℓ). It oscillates for g + ℓ > 1 and diverges for
g + ℓ > 2; the published defaults sit at ≈ 2.6 (core.py, fix 1).

The ceiling. The manuscript asks to "replace the arbitrary 99.9% threshold with an observable criterion, such as
a measured loss of adaptability or rising recovery time". Two standard criteria are implemented:

  * critical slowing down (Scheffer et al. 2009; Dakos et al. 2012): rising lag-1 autocorrelation and variance
    in a rolling window, scored by Kendall's τ against time. This is Level 0 for systems approaching a
    bifurcation. That it signals lost adaptability in the harness is Level 2.
  * recovery time after a perturbation, and its trend across successive perturbations.

`rigidity` combines them. It returns None for a verdict until there are enough samples.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .core import LifeParams


@dataclass(frozen=True)
class SteadyState:
    A: float
    C: float
    F: float
    O: float
    R: float | None
    C_eff: float | None
    g: float                         # organization construction rate a F* C*
    loss: float                      # organization loss rate b (D* + e²)
    tau_F: float
    tau_O: float
    euler_multiplier_O: float        # 1 − (g + ℓ); |·| < 1 needed for the literal Euler map
    starving: bool                   # the inflow cannot cover the stationary work: F* = 0

    def to_dict(self) -> dict:
        return asdict(self)


def steady_state(p: LifeParams, energy: float, e2: float = 0.0, var: float = 0.0, model_norm: float = 0.0,
                 action_cost: float | None = None, work: float = 0.0, G: float | None = None,
                 L: float | None = None) -> SteadyState:
    cbar = sum(p.action_costs) / len(p.action_costs) if action_cost is None else action_cost
    A = min(p.A_max, max(0.0, p.A_target - (p.eta_AV / p.eta_A) * var)) if p.eta_A > 0 else p.A_target
    lo, hi = p.C_min + p.margin, p.C_max - p.margin
    C = min(hi, max(lo, p.C_star - (p.eta_V / p.eta_C) * (var - p.v_ref))) if p.eta_C > 0 else p.C_star
    surplus = (p.eta_in * energy - (p.w0 + p.wA * A + p.wM * model_norm + p.wu * cbar + p.w_ext * work)
               - p.delta0 - p.delta_eps * e2)
    if p.deltaF > 0:
        F = max(0.0, surplus / p.deltaF)
    else:
        F = math.inf if surplus > 0 else 0.0
    g = p.a * F * C
    loss = p.b * (p.delta0 + p.deltaF * F + (1.0 + p.delta_eps) * e2)
    O = g / (g + loss) if g + loss > 0 else 0.0
    R = G / (G + L) if G is not None and L is not None and G + L > 0 else None
    ceff = None
    if R is not None and math.isfinite(F):
        ceff = (A / p.A_max) * C * O * R * F / (F + p.K_F)
    return SteadyState(A, C, F, O, R, ceff, g, loss, 1.0 / p.deltaF if p.deltaF > 0 else math.inf,
                       1.0 / (g + loss) if g + loss > 0 else math.inf, 1.0 - (g + loss), surplus <= 0)


def critical_free_energy(p: LifeParams, C: float, e2: float = 0.0) -> float:
    """Smallest stationary F at which O* ≥ O_min; inf when no F suffices."""
    k1 = p.a * C * (1.0 - p.O_min) - p.b * p.deltaF * p.O_min
    k0 = p.b * p.O_min * (p.delta0 + (1.0 + p.delta_eps) * e2)
    return k0 / k1 if k1 > 0 else math.inf


def critical_energy(p: LifeParams, e2: float = 0.0, var: float = 0.0, model_norm: float = 0.0,
                    action_cost: float | None = None, work: float = 0.0) -> float:
    """Smallest mean inflow Ē that keeps the stationary organization at or above O_min."""
    ss = steady_state(p, 0.0, e2, var, model_norm, action_cost, work)
    F_crit = critical_free_energy(p, ss.C, e2)
    if not math.isfinite(F_crit):
        return math.inf
    cbar = sum(p.action_costs) / len(p.action_costs) if action_cost is None else action_cost
    need = (p.deltaF * F_crit + p.w0 + p.wA * ss.A + p.wM * model_norm + p.wu * cbar + p.w_ext * work
            + p.delta0 + p.delta_eps * e2)
    return need / p.eta_in


# ---------------------------------------------------------------------- early-warning statistics


def lag1(x: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = sum(x) / n
    num = sum((x[i] - m) * (x[i + 1] - m) for i in range(n - 1))
    den = sum((v - m) ** 2 for v in x)
    return num / den if den > 1e-300 else 0.0


def variance(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    return sum((v - m) ** 2 for v in x) / (n - 1)


def kendall_tau(y: Sequence[float]) -> float:
    """Kendall's τ_b of y against its index (trend strength in [−1, 1]); O(n²)."""
    n = len(y)
    if n < 3:
        return 0.0
    conc = disc = ties = 0
    for i in range(n - 1):
        yi = y[i]
        for j in range(i + 1, n):
            dlt = y[j] - yi
            if dlt > 0:
                conc += 1
            elif dlt < 0:
                disc += 1
            else:
                ties += 1
    pairs = n * (n - 1) / 2
    den = math.sqrt(pairs * (pairs - ties))
    return (conc - disc) / den if den > 0 else 0.0


def detrend(x: Sequence[float], span: int) -> list[float]:
    """Residuals around an exponentially weighted mean (removes the slow drift before AR estimation)."""
    a, m, out = 2.0 / (span + 1.0), x[0], []
    for v in x:
        m += a * (v - m)
        out.append(v - m)
    return out


@dataclass(frozen=True)
class EarlyWarning:
    n: int
    window: int
    lag1_tau: float
    var_tau: float
    lag1_last: float
    var_last: float
    slowing_down: bool

    def to_dict(self) -> dict:
        return asdict(self)


def early_warning(x: Sequence[float], window: int | None = None, stride: int | None = None,
                  detrend_span: int | None = None, threshold: float = 0.3) -> EarlyWarning | None:
    """Rolling lag-1 autocorrelation and variance with their Kendall trends. None if too short."""
    n = len(x)
    window = window or max(10, n // 2)
    if n < window + 5:
        return None
    stride = stride or max(1, (n - window) // 60)
    series = detrend(x, detrend_span) if detrend_span else list(x)
    ac, vr = [], []
    for start in range(0, n - window + 1, stride):
        w = series[start:start + window]
        ac.append(lag1(w))
        vr.append(variance(w))
    t_ac, t_var = kendall_tau(ac), kendall_tau(vr)
    return EarlyWarning(n, window, t_ac, t_var, ac[-1], vr[-1], t_ac > threshold and t_var > 0)


def recovery_time(x: Sequence[float], t_shock: int, baseline: int = 50, fraction: float = 0.1,
                  floor: float = 0.0) -> int | None:
    """Steps after the shock until x returns within `fraction` of its peak deviation from the pre-shock mean.

    `floor` is the noise band (for example 2 SD of the pre-shock series): a deviation never larger than it
    is not a perturbation, and recovery to within it counts. None if x never recovers in the series."""
    pre = x[max(0, t_shock - baseline):t_shock]
    if not pre or t_shock >= len(x):
        return None
    m = sum(pre) / len(pre)
    post = x[t_shock:]
    dev = [abs(v - m) for v in post]
    k_peak = max(range(len(dev)), key=dev.__getitem__)
    band = max(fraction * dev[k_peak], floor)
    if dev[k_peak] <= band:
        return 0
    for k in range(k_peak, len(dev)):
        if dev[k] <= band:
            return k
    return None


@dataclass(frozen=True)
class Rigidity:
    early: EarlyWarning | None
    recovery_times: tuple[int | None, ...]
    recovery_tau: float | None
    verdict: str | None              # "adaptive" | "slowing" | "rigidifying" | None (not enough data)

    def to_dict(self) -> dict:
        return {"early": self.early.to_dict() if self.early else None,
                "recovery_times": list(self.recovery_times), "recovery_tau": self.recovery_tau,
                "verdict": self.verdict}


def rigidity(x: Sequence[float], shocks: Sequence[int] = (), threshold: float = 0.3, **kw) -> Rigidity:
    """Observable loss of adaptability: critical slowing down plus a rising recovery time across shocks."""
    ew = early_warning(x, threshold=threshold, **kw)
    rts = tuple(recovery_time(x, t) for t in shocks)
    finite = [r for r in rts if r is not None]
    r_tau = kendall_tau(finite) if len(finite) >= 3 else None
    unrecovered = any(r is None for r in rts)
    slowing = ew is not None and ew.slowing_down
    rising = r_tau is not None and r_tau > threshold
    if ew is None and r_tau is None and not unrecovered:
        verdict = None
    elif unrecovered or (slowing and rising):
        verdict = "rigidifying"
    elif slowing or rising:
        verdict = "slowing"
    else:
        verdict = "adaptive"
    return Rigidity(ew, rts, r_tau, verdict)
