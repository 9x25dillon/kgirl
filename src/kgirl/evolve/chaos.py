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
        return self._u(size)

    def _u(self, size=None) -> np.ndarray | float:
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
        u1 = np.clip(self._u(m), 1e-12, 1.0)
        u2 = self._u(m)
        r = np.sqrt(-2.0 * np.log(u1))
        z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]
        return float(z[0]) if size is None else z.reshape(size)

    def integers(self, high: int, size=None):
        u = self._u(size)
        return np.minimum((u * high).astype(int), high - 1) if size is not None else min(int(u * high), high - 1)


class ChaosGenerator(ChaosRNG):
    """The subset of the numpy.random API the kgirl modules use, served by ChaosRNG.

    Lets a module swap `np.random.<f>(...)` for `_rng.<f>(...)` and become reproducible without
    rewriting its algorithms. Distributions are exact transforms of the U(0,1) stream (inversion,
    Box–Muller); the stream is deterministic chaos, not i.i.d. and not cryptographic (Level 0).
    """

    def __init__(self, seed: float = 0.4123, lanes: int = 1024, decimate: int = 3, block: int = 8192):
        super().__init__(seed=seed, lanes=lanes, decimate=decimate)
        self._block = block
        self._buf = np.empty(0)
        self._pos = 0

    def _u(self, size=None):
        """Serve the U(0,1) stream from a block buffer, so scalar draws do not discard lanes."""
        n = 1 if size is None else int(np.prod(size))
        if self._pos + n > len(self._buf):
            fresh = ChaosRNG._u(self, max(n, self._block))
            self._buf = np.concatenate([self._buf[self._pos:], np.atleast_1d(fresh)])
            self._pos = 0
        out = self._buf[self._pos:self._pos + n]
        self._pos += n
        return float(out[0]) if size is None else out.reshape(size)

    def random(self, size=None):
        return self._u(size)

    def rand(self, *shape):
        return self._u(shape if shape else None)

    def randn(self, *shape):
        return ChaosRNG.normal(self, shape if shape else None)

    def uniform(self, low=0.0, high=1.0, size=None):    # numpy order: (low, high, size)
        return low + (high - low) * self._u(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return loc + scale * ChaosRNG.normal(self, size)

    def choice(self, a, size=None, replace=True, p=None):
        pool = np.arange(a) if isinstance(a, (int, np.integer)) else np.asarray(a)
        n = len(pool)
        if p is not None:
            cdf = np.cumsum(np.asarray(p, dtype=float))
            cdf /= cdf[-1]
            idx = np.searchsorted(cdf, self._u(size), side="right")
            idx = np.minimum(idx, n - 1)
        elif replace or size is None:
            idx = self.integers(n, size)
        else:                                            # without replacement: chaotic shuffle
            k = int(np.prod(size))
            if k > n:
                raise ValueError("cannot take a larger sample than population when replace=False")
            idx = np.argsort(self._u(n))[:k].reshape(size)
        return pool[idx]

    def poisson(self, lam=1.0, size=None):
        """Inversion of the Poisson CDF (exact for any lam; loop length ~ lam + 10√lam)."""
        u = np.atleast_1d(self._u(size if size is not None else 1)).astype(float)
        k = np.zeros(u.shape, dtype=int)
        pmf = np.full(u.shape, math.exp(-lam))
        cdf = pmf.copy()
        for i in range(1, int(lam + 10 * math.sqrt(lam) + 20)):
            more = u > cdf
            if not more.any():
                break
            k += more
            pmf = pmf * lam / i
            cdf = cdf + pmf
        return int(k[0]) if size is None else k.reshape(size)

    def permutation(self, x):
        arr = np.arange(x) if isinstance(x, (int, np.integer)) else np.asarray(x)
        return arr[np.argsort(self._u(len(arr)))]

    def shuffle(self, x) -> None:
        x[:] = self.permutation(x)
