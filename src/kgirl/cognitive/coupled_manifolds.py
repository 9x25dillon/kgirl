"""Two coupled manifolds: a "binocular voice sense" between two unified-operator lattices.

Each manifold A, B is a 13 × 50 complex lattice ψ running the unified operator (inner-time Gaussian memory
with curvature phase, seam-twisted angular hopping, axial hopping, cubic saturation, decay, drive).
They couple ONLY through the macroscopic Bloom voice Ψ = Σ ρ ψ (a Gaussian sentinel average):

    κ_A(t) = κ₀ + α⟨|ψ_A|²⟩ + β|Ψ_A|² + γ|Ψ_B| + A_drive cos(ω_A t + φ_A(t))        (and A ↔ B)

Observables that exist only for the coupled pair:

    C_AB = Ψ_A* Ψ_B          cross channel;  intimacy ⟨|C_AB|⟩;  Δφ_AB = arg C_AB
    R_sync = |⟨e^{iΔφ}⟩|      phase-lock order parameter;  drift = slope of unwrapped Δφ (0 when locked)
    coherence                 |⟨C_AB⟩| / √(⟨|Ψ_A|²⟩⟨|Ψ_B|²⟩)
    I_AB                      mutual information (bits) between arg Ψ_A and arg Ψ_B (16 × 16 bins, plug-in)
    effective dimension       participation ratio of the 4-D (Re, Im) joint trajectory

Transmission test (does the sense carry information A → B?). A's drive is keyed per bit:
    psk   φ_A = ±π/3                         (the original script's code)
    ask   A_drive · (1 ± depth)              (amplitude keying: the coupling γ|Ψ_A| is phase-blind)
and the receiver sees **B alone**: per bit window, the mean |Ψ_B| (ask) or mean arg Ψ_B (psk) relative to B's
own pre-signal baseline, thresholded at the median (bits are balanced). Reported: BER, and mutual information
between sent and decoded bits. γ = 0 is the control: at γ = 0, B's trace cannot depend on the bits, so any
decoding above chance is an artefact.

Why receiver = B alone: Δφ_AB = arg Ψ_B − arg Ψ_A contains A's own phase, so decoding it can succeed with no
coupling at all. That would read the transmitter, not the channel.

All randomness (initial phases, bit sequences) is logistic chaos. Levels: the lattice dynamics and every
number here are L0 (computed); calling the coupled observables a "new sense" is an interpretation (L3).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ManifoldParams:
    n_p: int = 13
    n_z: int = 50
    seam_phase: float = math.pi * 0.75
    j_ang: float = 0.22
    j_ax: float = 0.28
    alpha_fb: float = 0.18
    beta_outer: float = 0.25
    kappa_0: float = 0.08
    decay: float = 0.05
    g_cubic: float = 0.10
    tau_inner: int = 6
    sat_max: float = 2.0
    a_drive: float = 0.20
    sentinel_sigma: Tuple[float, float] = (4.0, 14.0)


def sentinel(p: ManifoldParams) -> np.ndarray:
    n = np.arange(p.n_p) - (p.n_p - 1) / 2
    m = np.arange(p.n_z) - (p.n_z - 1) / 2
    rho = np.exp(-n[:, None] ** 2 / (2 * p.sentinel_sigma[0] ** 2)) * np.exp(-m[None, :] ** 2 / (2 * p.sentinel_sigma[1] ** 2))
    return rho / rho.sum()


def init_psi(seed: float, p: ManifoldParams) -> np.ndarray:
    """0.1·e^{2πi s} with s the r = 3.97 logistic orbit from `seed`, row-major (as the original)."""
    s = seed
    phases = np.empty(p.n_p * p.n_z)
    for i in range(phases.size):
        s = 3.97 * s * (1 - s)
        phases[i] = s
    return (0.10 * np.exp(2j * np.pi * phases)).reshape(p.n_p, p.n_z)


def manifold_step(psi: np.ndarray, history: List[np.ndarray], t: int, kappa_prev: float, kappa_ext: float,
                  drive_amp: float, drive_phase: float, omega: float, p: ManifoldParams,
                  rho: np.ndarray) -> Tuple[np.ndarray, float]:
    """One unified-operator step (term for term the original `manifold_step_omega`); returns (ψ', κ_t)."""
    look = min(p.tau_inner, len(history))
    if look > 1:
        k = np.arange(look)
        tw = np.exp(-k ** 2 / (2 * (p.tau_inner / 2) ** 2))
        tw /= tw.sum()
        tp = np.exp(-1j * kappa_prev * k ** 2 / 2)
        b_inner = sum(history[-(i + 1)] * tw[i] * tp[i] for i in range(look))
    else:
        b_inner = psi.copy()
    pl = np.roll(b_inner, 1, axis=0)
    pr = np.roll(b_inner, -1, axis=0)
    pl[0, :] *= np.exp(-1j * p.seam_phase)
    pr[-1, :] *= np.exp(+1j * p.seam_phase)
    b_ang = 0.5 * (pl + pr)
    b_ax = 0.5 * (np.roll(b_inner, 1, axis=1) + np.roll(b_inner, -1, axis=1))
    voice = np.sum(rho * psi)
    kappa_t = (p.kappa_0 + p.alpha_fb * np.mean(np.abs(psi) ** 2) + p.beta_outer * abs(voice) ** 2
               + kappa_ext + drive_amp * math.cos(omega * t + drive_phase))
    new = (psi * np.exp(-1j * kappa_t) * (1.0 - p.decay) + p.j_ang * b_ang + p.j_ax * b_ax
           - p.g_cubic * np.abs(psi) ** 2 * psi)
    a = np.abs(new)
    m = a > p.sat_max
    new[m] = new[m] * p.sat_max / a[m]
    return new, float(kappa_t)


def run_coupled(gamma: float, T: int = 500, T_drop: int = 200, *, phase_fn: Optional[Callable[[int], float]] = None,
                amp_fn: Optional[Callable[[int], float]] = None, seed_a: float = 0.4123, seed_b: float = 0.7891,
                omega_a: float = 1.05, omega_b: float = 1.25,
                p: ManifoldParams = ManifoldParams()) -> Tuple[np.ndarray, np.ndarray]:
    """Voices Ψ_A(t), Ψ_B(t) for t ∈ [T_drop, T). phase_fn / amp_fn key A's drive phase / amplitude factor."""
    rho = sentinel(p)
    psi_a, psi_b = init_psi(seed_a, p), init_psi(seed_b, p)
    hist_a, hist_b = [psi_a.copy()], [psi_b.copy()]
    ka = kb = p.kappa_0
    va, vb = np.zeros(T, complex), np.zeros(T, complex)
    for t in range(T):
        va[t], vb[t] = np.sum(rho * psi_a), np.sum(rho * psi_b)
        phase = phase_fn(t) if phase_fn else 0.0
        amp = p.a_drive * (amp_fn(t) if amp_fn else 1.0)
        psi_a, ka = manifold_step(psi_a, hist_a, t, ka, gamma * abs(vb[t]), amp, phase, omega_a, p, rho)
        psi_b, kb = manifold_step(psi_b, hist_b, t, kb, gamma * abs(va[t]), p.a_drive, 0.0, omega_b, p, rho)
        hist_a.append(psi_a.copy())
        hist_b.append(psi_b.copy())
        if len(hist_a) > p.tau_inner + 2:
            hist_a.pop(0)
            hist_b.pop(0)
    return va[T_drop:], vb[T_drop:]


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    h, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = h / h.sum()
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])).sum())


def observables(va: np.ndarray, vb: np.ndarray) -> Dict[str, float]:
    c = np.conj(va) * vb
    dphi = np.angle(c)
    x = np.column_stack([va.real, va.imag, vb.real, vb.imag])
    x = x - x.mean(0)
    s = np.linalg.svd(x, compute_uv=False) ** 2
    s = s / s.sum() if s.sum() > 0 else s
    return {"intimacy": float(np.abs(c).mean()),
            "R_sync": float(abs(np.mean(np.exp(1j * dphi)))),
            "coherence": float(abs(np.mean(c)) / (math.sqrt(np.mean(abs(va) ** 2) * np.mean(abs(vb) ** 2)) + 1e-12)),
            "drift_rad_per_step": float(np.polyfit(np.arange(len(dphi)), np.unwrap(dphi), 1)[0]),
            "mutual_information_bits": _mutual_information(np.angle(va), np.angle(vb)),
            "effective_dimension": float(1.0 / np.sum(s ** 2)) if s.sum() > 0 else 0.0}


def sweep(gammas: Sequence[float] = (0.0, 0.10, 0.20, 0.35, 0.55, 0.80), T: int = 500, T_drop: int = 200,
          p: ManifoldParams = ManifoldParams()) -> Dict[float, Dict[str, float]]:
    return {float(g): observables(*run_coupled(g, T, T_drop, p=p)) for g in gammas}


def chaos_bits(n: int, seed: float = 0.6180) -> List[int]:
    """Balanced bit sequence: the n/2 largest logistic-orbit values are the ones."""
    s, xs = seed, []
    for _ in range(n):
        s = 3.99 * s * (1 - s)
        xs.append(s)
    order = np.argsort(xs, kind="stable")
    bits = np.zeros(n, int)
    bits[order[n // 2:]] = 1
    return bits.tolist()


@dataclass
class Transmission:
    gamma: float
    scheme: str
    bits: List[int]
    decoded: List[int]
    ber: float
    mutual_information_bits_per_symbol: float
    bits_per_step: float
    separation_dprime: float
    extra: Dict[str, float] = field(default_factory=dict)


def _bit_mi(sent: Sequence[int], got: Sequence[int]) -> float:
    s, g = np.asarray(sent), np.asarray(got)
    mi = 0.0
    for a in (0, 1):
        for b in (0, 1):
            pab = np.mean((s == a) & (g == b))
            if pab > 0:
                mi += pab * math.log2(pab / (np.mean(s == a) * np.mean(g == b)))
    return float(mi)


def transmit(gamma: float, bits: Sequence[int], scheme: str = "ask", bit_duration: int = 60, start: int = 200,
             depth: float = 0.5, p: ManifoldParams = ManifoldParams(), **kw) -> Transmission:
    """Key A's drive with `bits`; decode from B's voice alone (see module docstring)."""
    bits = list(bits)
    T = start + bit_duration * len(bits)

    def sym(t: int) -> Optional[int]:
        if t < start:
            return None
        i = (t - start) // bit_duration
        return bits[i] if i < len(bits) else None

    if scheme == "psk":
        phase_fn = lambda t: 0.0 if sym(t) is None else (math.pi / 3 if sym(t) else -math.pi / 3)  # noqa: E731
        amp_fn = None
    elif scheme == "ask":
        phase_fn = None
        amp_fn = lambda t: 1.0 if sym(t) is None else (1 + depth if sym(t) else 1 - depth)  # noqa: E731
    else:
        raise ValueError("scheme must be 'ask' or 'psk'")
    va, vb = run_coupled(gamma, T, 0, phase_fn=phase_fn, amp_fn=amp_fn, p=p, **kw)
    base = vb[start - 60:start]
    if scheme == "ask":
        ref = np.abs(base).mean()
        stat = np.array([np.abs(vb[start + i * bit_duration: start + (i + 1) * bit_duration]).mean() - ref
                         for i in range(len(bits))])
    else:
        ref = np.angle(np.mean(base))
        stat = np.array([np.angle(np.mean(vb[start + i * bit_duration: start + (i + 1) * bit_duration]
                                          * np.exp(-1j * ref))) for i in range(len(bits))])
    thr = float(np.median(stat))
    decoded = [int(v > thr) for v in stat]
    b = np.asarray(bits)
    s1, s0 = stat[b == 1], stat[b == 0]
    pooled = math.sqrt((s1.var() + s0.var()) / 2) if len(s1) > 1 and len(s0) > 1 else 0.0
    dprime = float((s1.mean() - s0.mean()) / pooled) if pooled > 0 else (0.0 if np.allclose(s1.mean(), s0.mean()) else math.inf)
    ber = float(np.mean(np.asarray(decoded) != b))
    mi = _bit_mi(bits, decoded)
    return Transmission(gamma, scheme, bits, decoded, ber, mi, mi / bit_duration, dprime,
                        {"threshold": thr, "observables": observables(va[start:], vb[start:])})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.cognitive.coupled_manifolds")
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--bit-duration", type=int, default=60)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.35, 0.55, 0.80])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sw = sweep(a.gammas)
    bits = chaos_bits(a.bits)
    tx = {f"{s}@{g:g}": asdict(transmit(g, bits, s, a.bit_duration)) for s in ("ask", "psk") for g in a.gammas}
    for v in tx.values():
        v.pop("bits")
        v.pop("decoded")
    if a.json:
        print(json.dumps({"sweep": sw, "transmission": tx}, indent=2, default=float))
        return 0
    print("γ      R_sync  coherence  drift(rad/step)  I_AB(bits)  eff.dim  intimacy")
    for g, o in sw.items():
        print(f"{g:<5.2f}  {o['R_sync']:.3f}   {o['coherence']:.3f}      {o['drift_rad_per_step']:+.4f}         "
              f"{o['mutual_information_bits']:.3f}       {o['effective_dimension']:.2f}     {o['intimacy']:.4f}")
    print(f"\ntransmission A → B, decoded from Ψ_B alone, {a.bits} balanced chaos bits × {a.bit_duration} steps")
    print("scheme  γ      BER    I(bits;decoded)  bits/step   d'")
    for key, v in tx.items():
        print(f"{v['scheme']:<6}  {v['gamma']:<5.2f}  {v['ber']:.3f}  {v['mutual_information_bits_per_symbol']:.3f}            "
              f"{v['bits_per_step']:.5f}   {v['separation_dprime']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
