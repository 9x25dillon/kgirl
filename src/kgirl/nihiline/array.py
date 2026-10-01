"""The c-BPE array chip as a phase lattice — crosstalk, uniformity, defects, robustness.

Adapted from `phase_topology_lab.py`. The unified operator is kept:

    ψ_{t+1} = P_sat [ L_κ + B_ang + B_ax + N_GL + D_ext + N_noise ] ψ_t
    κ_t = κ_0 + α_fb⟨|ψ|²⟩ + β_outer|Ψ_t|² + A cos(ω t)

Reinterpretation for the patent's chip (claim 10: ITO electrode array under a
PDMS microfluidic chip, CEA and AFP read on one chip):

    ψ[n, m]   complex ECL emission phasor at lane n, position m along the channel
              (|ψ|² ~ local intensity, arg ψ ~ emission timing relative to the drive)
    lanes     first half = CEA lanes, second half = AFP lanes
    seam      the CEA|AFP interface: different probes/aptamers → a phase twist
    J_ang     lane-to-lane crosstalk (shared drive field, optical bleed through PDMS)
    J_ax      along-channel coupling (diffusion of Ru³⁺/TPA• in the anode channel)
    A, σ      drive-pulse modulation depth and chaotic (logistic, no PRNG) noise

Observables mapped to assay quality:

    R            Kuramoto order — array-wide synchrony of emission
    xi           correlation length along the channel, in electrode pitches (crosstalk reach)
    lane_rsd     relative std of lane intensities inside each marker group (uniformity /
                 batch reproducibility claim)
    defects      bulk phase vortices — dark/hot spots in the emission map
    lyapunov     twin-trajectory divergence — sensitivity of a read to a tiny perturbation

Level 2: an exploratory model for chip-layout questions (how far does crosstalk
reach, when do lanes desynchronize), not a fit to the patent's data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ArrayParams:
    lanes: int = 8
    length: int = 40
    J_ang: float = 0.22
    J_ax: float = 0.28
    alpha_fb: float = 0.18
    beta_outer: float = 0.25
    kappa_0: float = 0.08
    decay: float = 0.05
    g_cubic: float = 0.10
    tau_inner: int = 6
    sat_max: float = 2.0
    seam_phase: float = np.pi * 0.75
    omega_d: float = 1.15
    A_drive: float = 0.2
    sigma_noise: float = 0.05
    noise_cap: float = 0.5


def _sentinel(lanes: int, length: int, sn: float = 4.0, sz: float = 14.0) -> np.ndarray:
    n = np.arange(lanes) - (lanes - 1) / 2
    m = np.arange(length) - (length - 1) / 2
    rho = np.exp(-(n[:, None] ** 2) / (2 * sn ** 2)) * np.exp(-(m[None, :] ** 2) / (2 * sz ** 2))
    return rho / rho.sum()


class ChaosField:
    """Per-site logistic maps — deterministic noise, vectorized over the lattice."""

    def __init__(self, shape, seed: float = 0.4123, r: float = 3.978):
        k = np.arange(int(np.prod(shape))).reshape(shape)
        self.s = 0.05 + 0.9 * ((seed + 0.618034 * k) % 1.0)
        self.r = r

    def step(self) -> np.ndarray:
        self.s = self.r * self.s * (1 - self.s)
        return (self.s - 0.5) * np.sqrt(12.0)


def init_psi(p: ArrayParams, seed: float = 0.4123) -> np.ndarray:
    k = np.arange(p.lanes * p.length).reshape(p.lanes, p.length)
    s = 0.05 + 0.9 * ((seed + 0.618034 * k) % 1.0)
    for _ in range(8):
        s = 3.97 * s * (1 - s)
    return 0.10 * np.exp(2j * np.pi * s)


class ArraySim:
    def __init__(self, p: ArrayParams, seed: float = 0.4123):
        self.p = p
        self.psi = init_psi(p, seed)
        self.hist = [self.psi.copy()]
        self.kappa = p.kappa_0
        self.t = 0
        self.rho = _sentinel(p.lanes, p.length)
        self.nR = ChaosField(self.psi.shape, 0.31 + seed / 10)
        self.nI = ChaosField(self.psi.shape, 0.59 + seed / 10)
        look = np.arange(p.tau_inner)
        w = np.exp(-look ** 2 / (2 * (p.tau_inner / 2) ** 2))
        self._tw = w / w.sum()

    def step(self) -> np.ndarray:
        p, psi = self.p, self.psi
        look = min(p.tau_inner, len(self.hist))
        if look > 1:
            tw = self._tw[:look] / self._tw[:look].sum()
            tp = np.exp(-1j * self.kappa * np.arange(look) ** 2 / 2)
            B = sum(self.hist[-(k + 1)] * tw[k] * tp[k] for k in range(look))
        else:
            B = psi.copy()
        pL, pR = np.roll(B, 1, axis=0), np.roll(B, -1, axis=0)
        pL[0, :] *= np.exp(-1j * p.seam_phase)
        pR[-1, :] *= np.exp(1j * p.seam_phase)
        B_ang = 0.5 * (pL + pR)
        B_ax = 0.5 * (np.roll(B, 1, axis=1) + np.roll(B, -1, axis=1))
        Psi = np.sum(self.rho * psi)
        self.kappa = (p.kappa_0 + p.alpha_fb * np.mean(np.abs(psi) ** 2) + p.beta_outer * abs(Psi) ** 2
                      + p.A_drive * np.cos(p.omega_d * self.t))
        new = psi * np.exp(-1j * self.kappa) * (1 - p.decay) + p.J_ang * B_ang + p.J_ax * B_ax \
            - p.g_cubic * np.abs(psi) ** 2 * psi
        if p.sigma_noise > 0:
            noise = p.sigma_noise * (self.nR.step() + 1j * self.nI.step())
            mag = np.abs(noise)
            noise *= np.minimum(mag, p.noise_cap) / (mag + 1e-12)
            new = new + noise
        a = np.abs(new)
        over = a > p.sat_max
        new[over] *= p.sat_max / a[over]
        self.psi = new
        self.hist.append(new.copy())
        if len(self.hist) > p.tau_inner + 2:
            self.hist.pop(0)
        self.t += 1
        return new


# ---------------------------------------------------------------- observables (vectorized)

def kuramoto_R(psi: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * np.angle(psi)))))


def spectral_entropy(psi: np.ndarray) -> float:
    F = np.abs(np.fft.fft2(psi)) ** 2
    P = F.ravel() / (F.sum() + 1e-12)
    P = P[P > 1e-12]
    return float(-(P * np.log(P)).sum() / np.log(P.size)) if P.size > 1 else 0.0


def correlation_length(psi: np.ndarray, max_r: int | None = None) -> float:
    phs = np.angle(psi)
    max_r = max_r or psi.shape[1] - 1
    for r in range(1, max_r):
        if abs(np.mean(np.exp(1j * (phs - np.roll(phs, r, axis=1))))) < np.exp(-1):
            return float(r)
    return float(max_r)


def wrapped_charge(psi: np.ndarray, seam_phase: float) -> np.ndarray:
    """Integer winding per plaquette: wrap EACH edge difference, then sum.

    The original lab summed raw (unwrapped) differences and wrapped the total;
    raw differences around a closed loop telescope to exactly 0, so its bulk
    charge was identically zero. Wrapping per edge is the standard vortex count.
    """
    phs = np.angle(psi)
    n2 = np.roll(phs, -1, axis=0)
    w = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    a, b, c, d = phs[:, :-1], phs[:, 1:], n2[:, :-1], n2[:, 1:]
    seam = np.zeros_like(a)
    seam[-1, :] = seam_phase
    total = w(c - a - seam) + w(d - c) + w(b - d + seam) + w(a - b)
    return np.round(total / (2 * np.pi))


def lane_rsd(psi: np.ndarray) -> dict:
    inten = (np.abs(psi) ** 2).mean(axis=1)
    half = len(inten) // 2
    out = {}
    for name, grp in (("CEA", inten[:half]), ("AFP", inten[half:])):
        out[name] = float(grp.std() / (grp.mean() + 1e-12))
    return out


# ---------------------------------------------------------------- experiments

def characterize(p: ArrayParams | None = None, steps: int = 300, burn: int = 150, every: int = 5,
                 seed: float = 0.4123) -> dict:
    p = p or ArrayParams()
    sim = ArraySim(p, seed)
    acc = {"R": [], "S": [], "xi": [], "defects": []}
    for i in range(steps):
        psi = sim.step()
        if i >= burn and i % every == 0:
            acc["R"].append(kuramoto_R(psi))
            acc["S"].append(spectral_entropy(psi))
            acc["xi"].append(correlation_length(psi))
            q = wrapped_charge(psi, p.seam_phase)
            acc["defects"].append(int(np.count_nonzero(q[:-1])))   # exclude the seam row
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["lane_rsd"] = lane_rsd(sim.psi)
    out["final"] = sim.psi
    return out


def phase_diagram(A_grid=None, sigma_grid=None, p: ArrayParams | None = None, steps: int = 200,
                  burn: int = 100) -> dict:
    p = p or ArrayParams()
    A_grid = np.linspace(0.0, 0.45, 4) if A_grid is None else np.asarray(A_grid)
    sigma_grid = np.linspace(0.0, 0.25, 4) if sigma_grid is None else np.asarray(sigma_grid)
    maps = {k: np.zeros((len(A_grid), len(sigma_grid))) for k in ("R", "S", "xi", "defects")}
    for i, A in enumerate(A_grid):
        for j, s in enumerate(sigma_grid):
            r = characterize(replace(p, A_drive=float(A), sigma_noise=float(s)), steps, burn,
                             seed=0.31 + 0.07 * i + 0.013 * j)
            for k in maps:
                maps[k][i, j] = r[k]
    return {"A": A_grid, "sigma": sigma_grid, **maps}


def lyapunov(p: ArrayParams | None = None, steps: int = 200, perturb: float = 1e-6) -> dict:
    """Twin trajectories with identical chaos sources; slope of log‖ψa − ψb‖ over the growth window."""
    p = p or ArrayParams()
    a, b = ArraySim(p), ArraySim(p)
    b.psi = b.psi.copy()
    b.psi[p.lanes // 2, p.length // 2] += perturb * (1 + 1j) / np.sqrt(2)
    b.hist = [b.psi.copy()]
    div = np.empty(steps)
    for t in range(steps):
        div[t] = np.linalg.norm(a.step() - b.step())
    valid = (div > 1e-12) & (div < 1.0)
    lam = float(np.polyfit(np.where(valid)[0], np.log(div[valid]), 1)[0]) if valid.sum() > 20 else float("nan")
    regime = "ordered" if lam < -0.01 else "critical" if abs(lam) <= 0.01 else "chaotic"
    return {"lambda": lam, "regime": regime, "divergence": div}
