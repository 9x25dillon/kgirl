"""Second-order digital phase-locked loop with designed gains and an in-phase lock detector.

Replaces RVSE's PhaseLockedLoop. That loop used kp = 2ζB and ki = B² with no units: B was not a frequency, the
detector gain was ignored, and the correction was added in Hz to a phase advanced in radians. It declared
lock when the raw phase error had a small standard deviation, which a loop that never acquired can satisfy.

Loop (input x = A cos θ, NCO phase φ):

    e  = −x · sin φ  = (A/2) sin(θ − φ) − (A/2) sin(θ + φ)        mixer phase detector, K_d = A/2
    ω  = ω₀ + K_p e + K_i ∫e dt                                    PI loop filter → NCO (rad/s)
    φ ← φ + ω Δt

Linearized: H(s) = (K_d K_p s + K_d K_i) / (s² + K_d K_p s + K_d K_i), so for natural frequency ω_n and damping ζ

    K_p = 2ζω_n / K_d,      K_i = ω_n² / K_d                        (Gardner, Phaselock Techniques, ch. 2)

The 2θ term rides on e. Keep ω_n well below the carrier, which the config enforces (bandwidth ≤ f₀/10).

Lock detector: the in-phase product i = x·cos φ ≈ (A/2) cos(θ − φ), low-passed with time constant τ and
divided by A/2. It reads ≈ 1 when locked, ≈ 0 when slipping cycles, and ≈ −1 when locked to the wrong
half-cycle (which a type-2 loop leaves). Locked means quality ≥ threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PLLConfig:
    fs: float                          # sample rate, Hz
    f0: float                          # NCO centre frequency, Hz
    bandwidth: float = 2.0             # natural frequency f_n, Hz
    damping: float = 0.707
    amplitude: float = 1.0             # expected input amplitude A (sets K_d)
    lock_tau: float = 0.1              # lock detector time constant, s
    lock_threshold: float = 0.9

    def __post_init__(self):
        if self.fs <= 0 or self.f0 <= 0 or self.bandwidth <= 0 or self.damping <= 0 or self.amplitude <= 0:
            raise ValueError("fs, f0, bandwidth, damping and amplitude must be positive")
        if self.f0 >= self.fs / 2:
            raise ValueError("f0 must lie below Nyquist")
        if self.bandwidth > self.f0 / 10:
            raise ValueError("loop bandwidth must stay ≤ f0/10 so the 2f0 detector ripple is rejected")


@dataclass(frozen=True)
class PLLTrace:
    phase: np.ndarray
    frequency: np.ndarray              # instantaneous NCO frequency, Hz
    quality: np.ndarray                # lock detector output, ≈ 1 locked
    fs: float
    threshold: float

    @property
    def locked(self) -> bool:
        return bool(self.quality[-1] >= self.threshold)

    def lock_time(self) -> float:
        """Seconds until the detector last rose above threshold and stayed there (NaN if never locked)."""
        above = self.quality >= self.threshold
        if not above[-1]:
            return float("nan")
        idx = np.where(~above)[0]
        return float(((idx[-1] + 1) if idx.size else 0) / self.fs)

    def frequency_estimate(self, window: float = 0.5) -> float:
        """Mean NCO frequency over the last `window` seconds (averages out the 2f ripple)."""
        k = max(1, int(window * self.fs))
        return float(self.frequency[-k:].mean())

    def summary(self) -> dict:
        return {"locked": self.locked, "lock_time_s": self.lock_time(), "frequency_hz": self.frequency_estimate(),
                "quality": float(self.quality[-1])}


class PLL:
    def __init__(self, cfg: PLLConfig):
        self.cfg = cfg
        wn = 2 * math.pi * cfg.bandwidth
        kd = cfg.amplitude / 2
        self.kp, self.ki = 2 * cfg.damping * wn / kd, wn * wn / kd
        self.dt = 1.0 / cfg.fs
        self.lp = min(1.0, self.dt / cfg.lock_tau)
        self.reset()

    def reset(self) -> None:
        self.phase, self.integ, self.i_lp = 0.0, 0.0, 0.0

    def run(self, x: np.ndarray) -> PLLTrace:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        phases, freqs, qual = np.empty(n), np.empty(n), np.empty(n)
        w0, dt, kp, ki, lp = 2 * math.pi * self.cfg.f0, self.dt, self.kp, self.ki, self.lp
        norm = 2.0 / self.cfg.amplitude
        phase, integ, i_lp = self.phase, self.integ, self.i_lp
        for k in range(n):
            s, c = math.sin(phase), math.cos(phase)
            e = -x[k] * s
            integ += ki * e * dt
            w = w0 + kp * e + integ
            i_lp += lp * (x[k] * c * norm - i_lp)
            phases[k], freqs[k], qual[k] = phase, w / (2 * math.pi), i_lp
            phase = (phase + w * dt) % (2 * math.pi)
        self.phase, self.integ, self.i_lp = phase, integ, i_lp
        return PLLTrace(phases, freqs, qual, self.cfg.fs, self.cfg.lock_threshold)


def track(x: np.ndarray, fs: float, f0: float, **kw) -> PLLTrace:
    return PLL(PLLConfig(fs=fs, f0=f0, **kw)).run(x)
