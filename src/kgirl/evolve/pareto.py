"""Non-dominated sorting (all objectives maximized)."""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def front(F: np.ndarray) -> np.ndarray:
    """Indices of the first Pareto front of an (n, k) objective matrix."""
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominators = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        if dominators.any():
            keep[i] = False
    return np.nonzero(keep)[0]


def ranks(F: np.ndarray) -> np.ndarray:
    """Front number of every point (0 = non-dominated)."""
    remaining = np.arange(len(F))
    out = np.full(len(F), -1)
    r = 0
    while remaining.size:
        f = remaining[front(F[remaining])]
        out[f] = r
        remaining = np.setdiff1d(remaining, f)
        r += 1
    return out
