"""Chaos streams for TA-ULS: weight init, data order and sampling with no PRNG anywhere.

kgirl's rule is that noise comes from deterministic chaos (the r = 4 logistic map), never from
`random` or `np.random`. `kgirl.evolve.chaos.ChaosRNG` serves sequential draws; training also
needs *addressable* draws: "the 7th uniform of stream (seed, s)", so that a batch at step n or a
parameter called `blocks.3.attn.qkv.weight` gets the same numbers whatever ran before it. That is
what makes resume bit-exact and lets modules be added without reshuffling every other init.

    lane i of stream s:   x0 = 0.05 + 0.9 · frac(seed + φ·(i+1) + √2·s)          (Weyl start, in (0,1))
                          x_{k+1} = 4 x_k (1 − x_k),  k = 0 … BURN−1              (Lyapunov exponent ln 2)
                          u = (2/π) · arcsin(√x_BURN)                             (exact map onto U(0,1))

After BURN = 40 iterations two starts that differ by δ differ by ~δ·2^40, so neighbouring lanes
(δ ≈ 0.618) and neighbouring streams (δ ≈ 0.414) are decorrelated (Level 0: tested empirically
in tests/tauls/test_chaos.py; not a cryptographic guarantee). Arithmetic is float64 on the CPU:
float32 orbits of the logistic map collapse onto short cycles.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

PHI = (math.sqrt(5.0) - 1.0) / 2.0
SQRT2_FRAC = math.sqrt(2.0) - 1.0
BURN = 40
_CHUNK = 1 << 22


def stream_id(*parts: object) -> int:
    """A stable stream number from any labels (parameter names, split names, ...)."""
    h = hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:6], "big")


def _lanes(seed: float, stream: int, start: int, n: int) -> np.ndarray:
    i = np.arange(start + 1, start + n + 1, dtype=np.float64)
    off = (seed + SQRT2_FRAC * (stream % 1_000_003) + PHI * (stream // 1_000_003 % 1_000_003)) % 1.0
    x = 0.05 + 0.9 * ((off + PHI * i) % 1.0)
    for _ in range(BURN):
        x = 4.0 * x * (1.0 - x)
    bad = (x <= 1e-15) | (x >= 1.0 - 1e-15)
    if bad.any():                                  # finite-precision collapse: restart those lanes off-orbit
        x[bad] = 0.05 + 0.9 * ((off + 0.137 + PHI * PHI * i[bad]) % 1.0)
    return x


def uniform(n: int, seed: float, stream: int = 0, start: int = 0) -> np.ndarray:
    """n addressable U(0,1) draws: element j is lane start+j of `stream`."""
    out = np.empty(n, dtype=np.float64)
    for a in range(0, n, _CHUNK):
        b = min(n, a + _CHUNK)
        x = _lanes(seed, stream, start + a, b - a)
        out[a:b] = (2.0 / np.pi) * np.arcsin(np.sqrt(x))
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def normal(n: int, seed: float, stream: int = 0) -> np.ndarray:
    """n N(0,1) draws of `stream` (Box–Muller over its first 2·⌈n/2⌉ uniforms)."""
    m = (n + 1) // 2
    u1 = uniform(m, seed, stream, 0)
    u2 = uniform(m, seed, stream, m)
    r = np.sqrt(-2.0 * np.log(u1))
    return np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]


def permutation(n: int, seed: float, stream: int = 0) -> np.ndarray:
    """A chaos-ordered permutation of range(n) (argsort of n uniforms; ties impossible in practice)."""
    return np.argsort(uniform(n, seed, stream), kind="stable")


class Sampler:
    """Sequential chaos draws for generation: one call, one fresh block of the stream."""

    def __init__(self, seed: float = 0.4123, stream: int = 0):
        self.seed, self.stream, self.pos = seed, stream, 0

    def uniform(self, n: int = 1) -> np.ndarray:
        u = uniform(n, self.seed, self.stream, self.pos)
        self.pos += n
        return u
