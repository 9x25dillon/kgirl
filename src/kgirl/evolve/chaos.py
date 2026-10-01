"""ChaosRNG: uniform and normal variates from the r = 4 logistic map. No PRNG anywhere.

Your lab scripts insist that noise come from deterministic chaos, not a random
number generator. The engine keeps that rule, without giving up the statistics
evolution needs:

    x_{n+1} = 4 x_n (1 − x_n)                invariant density ρ(x) = 1 / (π √(x(1 − x)))
    u = (2/π) · arcsin(√x)                   maps ρ exactly onto U(0, 1)   (conjugacy to the tent map)

Practical safeguards (Level 0 numerics):
  * 64 independent lanes advanced together in numpy, seeded by a golden-ratio Weyl sequence
  * decimation (every 3rd iterate) to weaken lag-1 correlation of a single orbit
  * a Weyl rotation added mod 1 (keeps the marginal exactly uniform, breaks residual structure)
  * orbits that collapse onto the fixed points 0 or 3/4 (finite precision) are re-seeded
"""

from __future__ import annotations

import math

import numpy as np

PHI = (math.sqrt(5.0) - 1.0) / 2.0


class ChaosRNG:
    def __init__(self, seed: float = 0.4123, lanes: int = 64, decimate: int = 3):
        k = np.arange(lanes)
        self._x = 0.05 + 0.9 * ((seed + PHI * (k + 1)) % 1.0)
        self._weyl = (seed * 7.0) % 1.0
        self._decimate = decimate
        self._reseeds = 0
        for _ in range(16):                      # burn in transients
            self._advance()

    def _advance(self) -> None:
        x = self._x
        for _ in range(self._decimate):
            x = 4.0 * x * (1.0 - x)
        stuck = (x < 1e-12) | (x > 1 - 1e-12) | (np.abs(x - 0.75) < 1e-12)
        if stuck.any():
            self._reseeds += int(stuck.sum())
            x[stuck] = 0.05 + 0.9 * ((np.arange(stuck.sum()) * PHI + self._weyl + 0.137) % 1.0)
        self._x = x

    def uniform(self, size=None) -> np.ndarray | float:
        n = 1 if size is None else int(np.prod(size))
        out = np.empty(n)
        lanes = len(self._x)
        for i in range(0, n, lanes):
            self._advance()
            u = (2.0 / np.pi) * np.arcsin(np.sqrt(self._x))
            self._weyl = (self._weyl + PHI) % 1.0
            u = (u + self._weyl) % 1.0
            take = min(lanes, n - i)
            out[i:i + take] = u[:take]
        return float(out[0]) if size is None else out.reshape(size)

    def normal(self, size=None) -> np.ndarray | float:
        n = 1 if size is None else int(np.prod(size))
        m = (n + 1) // 2
        u1 = np.clip(self.uniform(m), 1e-12, 1.0)
        u2 = self.uniform(m)
        r = np.sqrt(-2.0 * np.log(u1))
        z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]
        return float(z[0]) if size is None else z.reshape(size)

    def integers(self, high: int, size=None):
        u = self.uniform(size)
        return np.minimum((u * high).astype(int), high - 1) if size is not None else min(int(u * high), high - 1)
