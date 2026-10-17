"""Resonate-and-fire neuron bank: spiking frequency recognition. Replaces RVSE's
NeuromorphicResonanceRecognizer, whose "frequency tuning" was a Gaussian on the raw sample *amplitude* (so
it recognized signal level, not frequency), and whose recurrent weights were np.random.

Each neuron is a damped complex oscillator (Izhikevich 2001, "Resonate-and-fire neurons"):

    ż = (b_k + iω_k) z + 2|b_k|·I(t),      b_k = −2π·f_k / Q < 0      (constant Q, like a cochlear filterbank)

discretized exactly for the linear part. A neuron fires when Im z crosses its threshold upward, and its state
resets to 0. The gain 2|b_k| makes every neuron's resonant amplitude equal the input amplitude A, so low and
high neurons compete fairly. Off-resonance, the amplitude falls as A·|b_k|/|ω_k − ω_in|, so a neuron one
bin away (relative spacing r − 1, default Q = 2/(r − 1)) reaches about A/2. The bank's firing rates are
therefore a spiking spectrum. A higher Q gives sharper tuning but a slower response (time constant 1/|b_k|).

Measured: a constant-bandwidth bank (the first version written here) let a 7.3 Hz tone fire every neuron
from 5 to 10 Hz, because bandwidth is wide relative to spacing at low frequency. Constant Q fixes this.

Input is scaled to unit RMS, so amplitude does not decide which neuron wins. Each neuron's threshold is
constant-false-alarm-rate (CFAR): κ times the standard deviation of its own Im z under unit white noise,
σ_k = g_k / √(2(1 − |d_k|²)), where g_k is the input gain and d_k the per-sample decay. Narrow
low-frequency neurons get low thresholds and broad high ones get high thresholds, so a noise-only input
fires almost never (κ = 5: an upward 5σ crossing occurs about e^{−12.5} per cycle). A tone fires wherever
its amplitude exceeds κσ_k.

Detection and estimation are separate. Spikes (CFAR, with reset) decide *whether* a resonance is present. A
neuron driven off resonance is forced at the input frequency and also fires once per input cycle, so spike
counts cannot locate the tone. A parallel reset-free state, whose gain-normalized energy forms a
constant-Q spectrum, decides *where*.

Measured, for the record: a single threshold relative to total RMS (threshold 1.0) was exact on single tones
but silent on two equal tones and on a tone at SNR 1, because shared power lowers each component's amplitude.
At threshold 0.7 it fired 3 false spikes on pure noise. CFAR passes all three cases.

Complexity: O(N·K) for N samples and K neurons, vectorized over neurons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BankResult:
    frequencies: np.ndarray
    rates: np.ndarray                  # spikes per second per neuron (CFAR detection)
    energy: np.ndarray                 # mean |z|² of the reset-free state: a constant-Q spectrum
    spikes: int

    @property
    def detected(self) -> np.ndarray:
        return self.rates > 0

    @property
    def dominant(self) -> float:
        """Peak of the reset-free energy, refined by a parabola in (log f, log E). NaN if no neuron fired:
        without a spike, nothing was recognized."""
        if not self.spikes:
            return float("nan")
        e, lf = self.energy, np.log(self.frequencies)
        i = int(np.argmax(e))
        if 0 < i < len(e) - 1:
            y0, y1, y2 = np.log(e[i - 1:i + 2])
            den = y0 - 2 * y1 + y2
            if den < 0:
                return float(np.exp(lf[i] + 0.5 * (y0 - y2) / den * (lf[i + 1] - lf[i - 1]) / 2))
        return float(self.frequencies[i])

    def peaks(self, rel: float = 0.25) -> list[float]:
        """Local maxima of the energy spectrum at ≥ `rel` × the largest, among neurons that fired."""
        if not self.spikes:
            return []
        e, d = self.energy, self.detected
        return [float(self.frequencies[i]) for i in range(len(e))
                if d[i] and e[i] >= rel * e.max() and (i == 0 or e[i] >= e[i - 1]) and (i == len(e) - 1 or e[i] > e[i + 1])]

    def entropy(self) -> float:
        """Spectral entropy of the normalized energy (low for a pure tone, ln K for flat); 0 if nothing fired."""
        if not self.spikes:
            return 0.0
        p = self.energy / self.energy.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def summary(self) -> dict:
        return {"dominant_hz": round(self.dominant, 3), "peaks_hz": [round(f, 3) for f in self.peaks()],
                "spikes": self.spikes, "entropy": round(self.entropy(), 4)}


class ResonatorBank:
    def __init__(self, frequencies: np.ndarray, fs: float, q: float = 30.0, kappa: float = 5.0):
        self.f = np.asarray(frequencies, dtype=np.float64)
        if np.any(self.f <= 0) or np.any(self.f >= fs / 2):
            raise ValueError("resonator frequencies must lie in (0, fs/2)")
        if q <= 0 or kappa <= 0:
            raise ValueError("q and kappa must be positive")
        self.fs, self.q, self.kappa = fs, q, kappa
        b = -2 * math.pi * self.f / q
        self.decay = np.exp((b + 2j * math.pi * self.f) / fs)
        self.gain = 2 * np.abs(b) / fs
        self.threshold = kappa * self.gain / np.sqrt(2.0 * (1.0 - np.abs(self.decay) ** 2))

    def run(self, x: np.ndarray) -> BankResult:
        x = np.asarray(x, dtype=np.float64)
        rms = float(np.sqrt(np.mean(x * x)))
        if rms == 0:
            return BankResult(self.f, np.zeros_like(self.f), np.zeros_like(self.f), 0)
        x = x / rms
        z = np.zeros(len(self.f), dtype=np.complex128)
        free = np.zeros(len(self.f), dtype=np.complex128)
        counts = np.zeros(len(self.f))
        energy = np.zeros(len(self.f))
        prev = np.zeros(len(self.f))
        for v in x:
            z = z * self.decay + v * self.gain
            free = free * self.decay + v * self.gain
            im = z.imag
            fire = (im >= self.threshold) & (prev < self.threshold)
            counts += fire
            energy += (free * free.conjugate()).real
            z[fire] = 0
            prev = np.where(fire, 0.0, im)
        dur = len(x) / self.fs
        return BankResult(self.f, counts / dur, energy / len(x), int(counts.sum()))


def recognize(x: np.ndarray, fs: float, lo: float = 1.0, hi: Optional[float] = None, neurons: int = 64,
              q: Optional[float] = None, kappa: float = 5.0) -> BankResult:
    """Log-spaced bank from `lo` to `hi` (default 0.45·fs); default Q = 2/(r − 1) for neighbour ratio r."""
    hi = hi or 0.45 * fs
    f = np.exp(np.linspace(math.log(lo), math.log(hi), neurons))
    r = f[1] / f[0] if neurons > 1 else 2.0
    return ResonatorBank(f, fs, q if q is not None else 2.0 / (r - 1.0), kappa).run(x)
