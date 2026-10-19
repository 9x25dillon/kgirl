"""ChaosStream: a single-lane, standard-library twin of `kgirl.evolve.ChaosRNG`.

The repository draws its noise from deterministic chaos, not from a PRNG. `kgirl.life` must run inside the
harness, whose core is standard-library only, so it cannot import the numpy engine. This is the same
construction on one lane:

    x_{n+1} = 4 x_n (1 − x_n)                 invariant density 1 / (π √(x(1 − x)))
    u = (2/π) arcsin √x  (+ Weyl rotation)     exactly U(0, 1) under that density
    z = √(−2 ln u₁) cos 2π u₂                  Box–Muller, for normal variates

Decimation by 3 and the golden-ratio Weyl rotation weaken the lag-1 correlation of a single orbit; orbits
that collapse onto the fixed points 0 or 3/4 in finite precision are re-seeded. Same seed, same stream.
"""

from __future__ import annotations

import math

PHI = (math.sqrt(5.0) - 1.0) / 2.0


class ChaosStream:
    __slots__ = ("_x", "_weyl", "_spare", "reseeds")

    def __init__(self, seed: float = 0.4123):
        self._x = 0.05 + 0.9 * ((seed + PHI) % 1.0)
        self._weyl = (seed * 7.0) % 1.0
        self._spare: float | None = None
        self.reseeds = 0
        for _ in range(16):
            self.uniform()

    def uniform(self) -> float:
        x = self._x
        for _ in range(3):
            x = 4.0 * x * (1.0 - x)
        if x < 1e-12 or x > 1.0 - 1e-12 or abs(x - 0.75) < 1e-12:
            self.reseeds += 1
            x = 0.05 + 0.9 * ((self.reseeds * PHI + self._weyl + 0.137) % 1.0)
        self._x = x
        self._weyl = (self._weyl + PHI) % 1.0
        return ((2.0 / math.pi) * math.asin(math.sqrt(x)) + self._weyl) % 1.0

    def normal(self) -> float:
        if self._spare is not None:
            z, self._spare = self._spare, None
            return z
        u1 = max(self.uniform(), 1e-300)
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)

    def choice(self, probs: tuple[float, ...] | list[float]) -> int:
        u, acc = self.uniform() * sum(probs), 0.0
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                return i
        return len(probs) - 1
