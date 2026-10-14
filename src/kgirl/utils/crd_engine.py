"""CRD — Coherence Recovery Dynamics: band coherence from EEG, and recovery of collapsed bands.

`kgirl.utils.crd.UnifiedCoherenceRecovery` constructed `CRD(sfreq, bands, eta, tau, theta, T0, rho,
kappa_base, noise_std, window_s, hop_s)` from a module that was never committed. This is that engine,
built from what its parameters say.

Measurement (Level 0, standard signal processing):
    For each window of `window_s` seconds, stepped by `hop_s`, and each band (lo, hi) Hz:
      multi-channel  κ_b = phase-locking value = mean_t | mean_c exp(i φ_c(t)) |,  φ from the analytic
                     signal of the FFT band-passed channel (Hilbert transform via the FFT)
      one channel    κ_b = band power / total power in the union of the bands

Recovery (Level 2: a model of the parameters, not a validated physiological law):
    healthy memory  m_b ← ρ_b m_b + (1 − ρ_b) κ_b            while band b is not recovering
    collapse        κ_b < θ − σ_b  (σ_b = noise_std: hysteresis, so noise does not toggle recovery)
    recovering      κ̂_b = κ_b + (1 − e^{−t/τ}) (target_b − κ_b),  target_b = max(κ_base,b, m_b),
                    t = time since collapse; leaves recovery once t ≥ T0 and κ_b ≥ θ + σ_b
    global          κ_global = Σ_b η_b κ̂_b / Σ_b η_b

Deterministic: no noise is ever injected (noise_std only sets the hysteresis band).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


@dataclass
class BandState:
    memory: float
    recovering: bool = False
    t_in_recovery: float = 0.0
    collapses: int = 0


@dataclass
class CRD:
    sfreq: float = 250.0
    bands: Mapping[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    eta: Optional[Mapping[str, float]] = None
    tau: float = 30.0
    theta: float = 0.35
    T0: float = 2.0
    rho: Optional[Mapping[str, float]] = None
    kappa_base: Optional[Mapping[str, float]] = None
    noise_std: Optional[Mapping[str, float]] = None
    window_s: float = 2.0
    hop_s: float = 0.25

    def __post_init__(self):
        names = list(self.bands)
        self.eta = dict(self.eta or {b: 1 / len(names) for b in names})
        self.rho = dict(self.rho or {b: 0.7 for b in names})
        self.kappa_base = dict(self.kappa_base or {b: 0.3 for b in names})
        self.noise_std = dict(self.noise_std or {b: 0.01 for b in names})
        for b, (lo, hi) in self.bands.items():
            if not 0 <= lo < hi <= self.sfreq / 2:
                raise ValueError(f"band {b} ({lo}, {hi}) Hz must lie within 0..Nyquist ({self.sfreq / 2} Hz)")
        self.state = {b: BandState(memory=self.kappa_base[b]) for b in names}
        self.updates = 0

    # ------------------------------------------------------------------ measurement

    def _analytic_band(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Analytic signal of x (channels × samples) restricted to [lo, hi) Hz (one-sided spectrum ×2)."""
        n = x.shape[-1]
        f = np.fft.fftfreq(n, d=1.0 / self.sfreq)
        spec = np.fft.fft(x, axis=-1)
        mask = (f >= lo) & (f < hi)
        return np.fft.ifft(spec * (2.0 * mask), axis=-1)

    def band_kappa(self, eeg: np.ndarray) -> List[Dict[str, float]]:
        """κ per band for each window of the recording (eeg: samples or channels × samples)."""
        x = np.atleast_2d(np.asarray(eeg, dtype=np.float64))
        x = x - x.mean(axis=-1, keepdims=True)
        win = int(round(self.window_s * self.sfreq))
        hop = max(1, int(round(self.hop_s * self.sfreq)))
        if x.shape[-1] < win:
            raise ValueError(f"need at least {win} samples ({self.window_s} s), got {x.shape[-1]}")
        out = []
        for s in range(0, x.shape[-1] - win + 1, hop):
            seg = x[:, s:s + win]
            k: Dict[str, float] = {}
            if seg.shape[0] > 1:
                for b, (lo, hi) in self.bands.items():
                    z = self._analytic_band(seg, lo, hi)
                    ph = np.exp(1j * np.angle(z))
                    k[b] = float(np.abs(ph.mean(axis=0)).mean())
            else:
                f = np.fft.rfftfreq(win, d=1.0 / self.sfreq)
                p = np.abs(np.fft.rfft(seg[0])) ** 2
                inband = {b: float(p[(f >= lo) & (f < hi)].sum()) for b, (lo, hi) in self.bands.items()}
                tot = sum(inband.values()) or 1.0
                k = {b: v / tot for b, v in inband.items()}
            out.append(k)
        return out

    # ------------------------------------------------------------------ recovery dynamics

    def update(self, kappa: Mapping[str, float], dt: Optional[float] = None) -> Dict[str, float]:
        """One step of the recovery dynamics; returns κ̂ per band plus 'global'."""
        dt = self.hop_s if dt is None else dt
        out: Dict[str, float] = {}
        for b in self.bands:
            k = float(np.clip(kappa.get(b, self.kappa_base[b]), 0.0, 1.0))
            st, sig = self.state[b], self.noise_std[b]
            if not st.recovering and k < self.theta - sig:
                st.recovering, st.t_in_recovery = True, 0.0
                st.collapses += 1
            if st.recovering:
                st.t_in_recovery += dt
                target = max(self.kappa_base[b], st.memory)
                k_hat = k + (1.0 - math.exp(-st.t_in_recovery / self.tau)) * max(0.0, target - k)
                if st.t_in_recovery >= self.T0 and k >= self.theta + sig:
                    st.recovering = False
            else:
                st.memory = self.rho[b] * st.memory + (1.0 - self.rho[b]) * k
                k_hat = k
            out[b] = k_hat
        w = sum(self.eta[b] for b in self.bands)
        out["global"] = sum(self.eta[b] * out[b] for b in self.bands) / w if w else 0.0
        self.updates += 1
        return out

    def process_eeg(self, eeg: np.ndarray) -> List[Dict[str, float]]:
        """Measure every window, then run the recovery dynamics over them in order."""
        return [self.update(k) for k in self.band_kappa(eeg)]

    def statistics(self) -> Dict[str, object]:
        return {"updates": self.updates,
                "collapses": {b: s.collapses for b, s in self.state.items()},
                "recovering": [b for b, s in self.state.items() if s.recovering]}
