"""Nihil leverage for the c-BPE drive: how much *off* does the 5 V pulse need?

Adapted from `nihil_leverage.py` and `nihiline_full-1.py`. There, a field ψ was
forced to zero for a fraction f of every cycle (the nihil window) before a
rebirth kick, and the sweep found the dose of nothing that maximized what the
system produced. Here the field is the anode ECL emission of the closed BPE:

    drive d(t) = 1 for (1 − f)·P, then 0 for f·P     (f = nihil fraction, P = pulse period)
    emission   E' = (d·g(C)·(1 − p)·(1 + η·ξ) − E) / τ_ecl      (first-order ECL kinetics)
    passivation p' = a(C)·d·(1 − p) − b·(1 − d)·p              (TPA-product fouling; heals in nihil)
    control    C ∈ [0, 1] maps TPA 20–70 mM (claim 7): g(C) = 0.6 + 0.8C, a(C) ∝ 0.4 + 1.2C
    cell stress s' = c·d − h·(1 − d)·s ;  viability v = exp(−s / s_crit)
    ξ: logistic-map chaos, r = 3.7 + 0.29·(1 − C)  — deterministic, no PRNG (nihiline_full move 1)

Per-pulse photon integrals Q_k give the assay figures of merit:

    yield          mean Q_k (normalized)            what each read is worth
    reproducibility 1 − CV(Q_k)                     the "coherence" of the recursion
    viability      v at the end of the run          the cells must survive the read (Example 5)
    S_n            Shannon entropy of per-pulse features (bits/cycle; nihiline_full move 2)
    product P      yield × reproducibility × viability

All time constants are illustrative (Level 2): this is a design-space explorer
for pulsed c-BPE-ECL, not a fitted model of the patent's chip. Every run is
vectorized across parameter sets, so sweeps cost one time loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patent import CLAIM_RANGES


@dataclass(frozen=True)
class PulseParams:
    period_s: float = 2.0
    tau_ecl_s: float = 0.15
    gain: float = 1.0
    passivation_rate: float = 0.35     # 1/s while driven
    heal_rate: float = 1.2             # 1/s during nihil
    stress_rate: float = 0.02          # 1/s while driven
    recovery_rate: float = 0.25        # 1/s during nihil
    stress_crit: float = 1.0
    chaos_amplitude: float = 0.25
    T_s: float = 60.0
    dt_s: float = 0.002


def control_from_tpa(tpa_mM: float) -> float:
    """Map coreactant concentration inside the claim range (20–70 mM) to control C ∈ [0, 1]."""
    lo, hi, _ = CLAIM_RANGES["tpa_mM"]
    return float(np.clip((tpa_mM - lo) / (hi - lo), 0.0, 1.0))


def simulate(f_nihil, control=0.6, p: PulseParams | None = None, seed: float = 0.4123,
             period_s=None, keep_trace: bool = False) -> dict:
    """Vectorized over any broadcastable (f_nihil, control, period_s) arrays."""
    p = p or PulseParams()
    f, C, P = np.broadcast_arrays(np.atleast_1d(np.asarray(f_nihil, float)),
                                  np.atleast_1d(np.asarray(control, float)),
                                  np.atleast_1d(np.asarray(p.period_s if period_s is None else period_s, float)))
    f, C, P = f.ravel(), C.ravel(), P.ravel()
    n = f.size
    steps = int(p.T_s / p.dt_s)
    E = np.zeros(n)
    pas = np.zeros(n)
    s = np.zeros(n)
    chaos = 0.05 + 0.9 * ((seed + 0.618034 * np.arange(n)) % 1.0)   # distinct deterministic orbits per run
    r = 3.7 + 0.29 * (1.0 - C)
    gain = p.gain * (0.6 + 0.8 * C)                    # more TPA → brighter ECL ...
    foul = p.passivation_rate * (0.4 + 1.2 * C)        # ... and more oxidation products fouling the anode
    n_pulses = int(np.ceil(p.T_s / P.min())) + 1
    Q = np.zeros((n, n_pulses))
    peak = np.zeros((n, n_pulses))
    trace = np.zeros((steps, n)) if keep_trace else None
    rows = np.arange(n)
    for i in range(steps):
        t = i * p.dt_s
        phase = (t % P) / P
        d = (phase < (1.0 - f)).astype(float)
        chaos = r * chaos * (1.0 - chaos)
        drive = d * gain * (1.0 - pas) * (1.0 + p.chaos_amplitude * (chaos - 0.5) * (1.0 - 0.5 * C))
        E += (drive - E) * (p.dt_s / p.tau_ecl_s)
        pas += (foul * d * (1.0 - pas) - p.heal_rate * (1.0 - d) * pas) * p.dt_s
        s += (p.stress_rate * d - p.recovery_rate * (1.0 - d) * s) * p.dt_s
        k = np.minimum((t // P).astype(int), n_pulses - 1)
        Q[rows, k] += E * p.dt_s
        peak[rows, k] = np.maximum(peak[rows, k], E)
        if keep_trace:
            trace[i] = E
    full = np.floor(p.T_s / P).astype(int)
    out = {"f_nihil": f, "control": C, "period_s": P, "viability": np.exp(-s / p.stress_crit)}
    yields, cvs, sn = np.zeros(n), np.zeros(n), np.zeros(n)
    for j in range(n):
        q = Q[j, 1:max(2, full[j])]          # drop the first (transient) pulse and any partial tail
        yields[j] = q.mean() / (P[j] * p.gain)
        cvs[j] = q.std() / (q.mean() + 1e-12)
        sn[j] = _cycle_entropy(np.column_stack([q, peak[j, 1:max(2, full[j])]]))
    out.update(yield_=yields, reproducibility=np.clip(1.0 - cvs, 0.0, 1.0), cycle_entropy_bits=sn)
    out["product"] = out["yield_"] * out["reproducibility"] * out["viability"]
    if keep_trace:
        out["trace"] = trace
        out["t"] = np.arange(steps) * p.dt_s
    return out


def _cycle_entropy(features: np.ndarray, bins: int = 8) -> float:
    """Shannon entropy (bits) of the per-pulse feature distribution."""
    if len(features) < 4:
        return 0.0
    lo, hi = features.min(axis=0), features.max(axis=0)
    norm = (features - lo) / (hi - lo + 1e-12)
    H, _ = np.histogramdd(norm, bins=bins, range=[[0, 1]] * features.shape[1])
    q = H.ravel() / H.sum()
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


def leverage_sweep(control=0.6, f_grid=None, p: PulseParams | None = None) -> dict:
    """nihil_leverage: the master curve P(f) and the optimal dose of nothing."""
    f_grid = np.linspace(0.0, 0.9, 19) if f_grid is None else np.asarray(f_grid)
    r = simulate(f_grid, control, p)
    i = int(np.argmax(r["product"]))
    return {**r, "f_opt": float(f_grid[i]), "product_opt": float(r["product"][i])}


def control_nihil_sweep(C_grid=None, f_grid=None, p: PulseParams | None = None) -> dict:
    """nihiline_full move 3: does high control (more coreactant) need less nothing?"""
    C_grid = np.linspace(0.0, 1.0, 9) if C_grid is None else np.asarray(C_grid)
    f_grid = np.linspace(0.0, 0.9, 13) if f_grid is None else np.asarray(f_grid)
    CC, FF = np.meshgrid(C_grid, f_grid, indexing="ij")
    r = simulate(FF.ravel(), CC.ravel(), p)
    P = r["product"].reshape(CC.shape)
    f_opt = f_grid[np.argmax(P, axis=1)]
    slope = float(np.polyfit(C_grid, f_opt, 1)[0]) if len(C_grid) > 1 else 0.0
    verdict = ("high control needs LESS nothing" if slope < -0.01 else
               "high control needs MORE nothing" if slope > 0.01 else "control does not move the optimal nothing")
    return {"C": C_grid, "f": f_grid, "product": P, "f_opt_by_C": f_opt, "slope": slope, "verdict": verdict}


def ratio_sweep(ratios=None, f_nihil=0.3, control=0.6, p: PulseParams | None = None) -> dict:
    """nihiline_full move 4: pulse rate vs ECL relaxation rate — where reads lock into a pattern.

    ratio = (1/period) / (1/τ_ecl). Pattern = reproducible per-pulse integrals (high 1 − CV).
    """
    p = p or PulseParams()
    ratios = np.linspace(0.02, 1.0, 25) if ratios is None else np.asarray(ratios)
    periods = p.tau_ecl_s / ratios
    r = simulate(f_nihil, control, p, period_s=periods)
    return {"ratio": ratios, "period_s": periods, "pattern": r["reproducibility"], "yield": r["yield_"],
            "cycle_entropy_bits": r["cycle_entropy_bits"], "product": r["product"]}
