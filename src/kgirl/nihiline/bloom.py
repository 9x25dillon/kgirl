"""Bloom sentinel for the living cells on the c-BPE cathode.

Adapted from `bloom_bridge.py` (Thorne & K1ll, Phase-Curvature Biosentinels,
Eq. 9). The closed-form Fresnel Bloom integral is kept exactly:

    B̃(t) = e^(−i n₀(t)) · (1 + i κ(t) σ²)^(−1/2)
    amplitude |B̃|² = (1 + κ²σ⁴)^(−1/2)      1 − |B̃|² ≈ κ²σ⁴/2   (quadratic in κ)
    phase  arg B̃ = −n₀ − ½ arctan(κσ²)      ≈ −κσ²/2              (linear in κ)

What changed for the patent: the cell states Q / M / A are now the MCF-7 cells
sitting on the BPE cathode during the in-situ read, and the apoptotic onset is
driven by the *electrical stress of the drive pulse* (pulse.py) instead of a
fixed time. The order asymmetry (phase ∝ κ, amplitude ∝ κ²) is why phase
variance breaches before mean intensity collapses: that Δt is an early-warning
window that the drive is hurting the cells — earlier than Calcein-AM/PI end-point
staining (Example 5) can say. The ECL marker signal is multiplied by viability,
so a dying monolayer under-reports CEA/AFP; the sentinel flags those reads.

Confidence: the Bloom algebra and the order asymmetry are exact (Level 0 math);
THz phase-curvature sensing of living cells is the preprint's hypothesis
(Level 2–3); the κ(t) trajectories are illustrative (Level 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SentinelParams:
    sigma_um: float = 2.0          # sentinel waist (matching condition κσ² ~ 1)
    T_s: float = 1800.0
    dt_s: float = 0.05
    window_s: float = 30.0
    baseline_end_s: float = 400.0


def kappa_match(sigma_um: float) -> float:
    return 1.0 / sigma_um ** 2


def _logistic(n: int, seed: float, r: float) -> np.ndarray:
    out = np.empty(n)
    s = seed
    for i in range(n):
        s = r * s * (1 - s)
        out[i] = s
    return out


def kappa_quiescent(t: np.ndarray, km: float, seed: float = 0.4123) -> np.ndarray:
    return 0.05 * km * np.sin(2 * np.pi * t / 800.0) + (_logistic(len(t), seed, 3.78) - 0.5) * 0.08 * km


def kappa_mitotic(t: np.ndarray, km: float, t0: float = 900.0, w: float = 120.0) -> np.ndarray:
    return km * np.exp(-((t - t0) ** 2) / (2 * w ** 2))


def kappa_apoptotic(t: np.ndarray, km: float, t_onset: float = 600.0, seed: float = 0.71) -> np.ndarray:
    envelope = 0.05 + 0.55 / (1.0 + np.exp(-(t - t_onset) / 80.0))
    return 0.02 * km * np.sin(2 * np.pi * t / 1000.0) + (_logistic(len(t), seed, 3.95) - 0.5) * envelope * km * 2.0


def onset_from_stress(stress_rate: float, recovery_rate: float, f_nihil: float, stress_crit: float = 1.0,
                      viability_floor: float = 0.6) -> float:
    """Seconds of pulsed drive until viability exp(−s/s_crit) falls to `viability_floor`.

    Mean-field version of pulse.py's stress ODE: ds/dt ≈ c(1−f) − h·f·s, so
    s → s* = c(1−f)/(h f) with time constant 1/(h f). Returns inf if s* never reaches the floor.
    """
    s_target = -stress_crit * np.log(viability_floor)
    drive, heal = stress_rate * (1 - f_nihil), recovery_rate * f_nihil
    if heal <= 0:
        return s_target / drive if drive > 0 else float("inf")
    s_star = drive / heal
    if s_star <= s_target:
        return float("inf")
    return float(-np.log(1 - s_target / s_star) / heal)


def bloom_integral(kappa: np.ndarray, sigma_um: float, n0: np.ndarray | None = None) -> np.ndarray:
    z = 1.0 + 1j * kappa * sigma_um ** 2
    phase = np.exp(-1j * n0) if n0 is not None else 1.0
    return phase * z ** -0.5


def sliding(x: np.ndarray, w: int, fn: str = "var") -> np.ndarray:
    """Centered sliding mean/variance in O(n) via cumulative sums."""
    half = w // 2
    n = len(x)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = hi - lo
    mean = (c1[hi] - c1[lo]) / cnt
    if fn == "mean":
        return mean
    return np.maximum((c2[hi] - c2[lo]) / cnt - mean ** 2, 0.0)


@dataclass
class SentinelResult:
    t: np.ndarray
    kappa: dict
    phase_variance: np.ndarray
    mean_intensity: np.ndarray
    t_phase_breach: float | None
    t_amp_collapse: float | None
    order_fit: dict
    sigma_sweep: list

    @property
    def delta_t(self) -> float | None:
        if self.t_phase_breach is None or self.t_amp_collapse is None:
            return None
        return self.t_amp_collapse - self.t_phase_breach

    def summary(self) -> dict:
        return {"t_phase_breach_s": self.t_phase_breach, "t_amp_collapse_s": self.t_amp_collapse,
                "delta_t_s": self.delta_t, "order_fit": self.order_fit,
                "sigma_sweep": self.sigma_sweep}


def run(p: SentinelParams | None = None, t_onset: float = 600.0) -> SentinelResult:
    p = p or SentinelParams()
    t = np.arange(0.0, p.T_s, p.dt_s)
    km = kappa_match(p.sigma_um)
    kq, kmit, ka = kappa_quiescent(t, km), kappa_mitotic(t, km), kappa_apoptotic(t, km, t_onset)
    BA = bloom_integral(ka, p.sigma_um)
    w = int(p.window_s / p.dt_s)
    var_phi = sliding(np.unwrap(np.angle(BA)), w, "var")
    mean_amp = sliding(np.abs(BA) ** 2, w, "mean")
    base = t < min(p.baseline_end_s, t_onset - 4 * 80.0)   # sigmoid onset width 80 s: baseline is flat here
    phi_thr = var_phi[base].mean() + 3 * var_phi[base].std()
    amp_thr = mean_amp[base].mean() - 3 * mean_amp[base].std()
    after = t > base.sum() * p.dt_s
    tp = t[after & (var_phi > phi_thr)]
    ta = t[after & (mean_amp < amp_thr)]

    k_test = np.linspace(-0.1 * km, 0.1 * km, 201)          # small-κ regime where the asymptotics hold
    B_test = bloom_integral(k_test, p.sigma_um)
    lin = float(np.polyfit(k_test, np.angle(B_test), 1)[0])
    quad = float(np.polyfit(k_test, 1 - np.abs(B_test) ** 2, 2)[0])
    order = {"phase_linear_coeff": lin, "expected_phase": -p.sigma_um ** 2 / 2,
             "amp_quadratic_coeff": quad, "expected_amp": p.sigma_um ** 4 / 2}

    sweep = []
    for s_um in (0.5, 1.0, 2.0, 4.0, 8.0):
        Bs = bloom_integral(ka, s_um)
        av, pv = float(np.var(np.abs(Bs) ** 2)), float(np.var(np.unwrap(np.angle(Bs))))
        sweep.append({"sigma_um": s_um, "amp_var": av, "phase_var": pv, "phase_over_amp": pv / (av + 1e-12)})
    return SentinelResult(t, {"Q": kq, "M": kmit, "A": ka}, var_phi, mean_amp,
                          float(tp[0]) if len(tp) else None, float(ta[0]) if len(ta) else None, order, sweep)


def gated_marker_signal(intensity: np.ndarray, viability: np.ndarray, floor: float = 0.8) -> dict:
    """ECL reads scale with live-cell surface antigen; reads below the viability floor are flagged."""
    corrected = np.where(viability > 0, intensity / np.maximum(viability, 1e-6), np.nan)
    return {"corrected": corrected, "flagged": viability < floor}
