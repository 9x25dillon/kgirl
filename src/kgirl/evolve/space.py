"""Search spaces: every genome lives in the unit cube; decoding applies bounds and log scales."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    log: bool = False
    unit: str = ""

    def decode(self, u: float) -> float:
        u = min(1.0, max(0.0, float(u)))
        if self.log:
            return math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        return self.lo + u * (self.hi - self.lo)

    def encode(self, v: float) -> float:
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)


class Space:
    def __init__(self, params: list[Param]):
        self.params = list(params)
        self.names = [p.name for p in self.params]

    @property
    def dim(self) -> int:
        return len(self.params)

    def decode(self, x: np.ndarray) -> dict[str, float]:
        return {p.name: p.decode(v) for p, v in zip(self.params, x)}

    def decode_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {p.name: np.array([p.decode(v) for v in X[:, j]]) for j, p in enumerate(self.params)}

    def encode(self, values: dict[str, float]) -> np.ndarray:
        return np.array([p.encode(values[p.name]) for p in self.params])
