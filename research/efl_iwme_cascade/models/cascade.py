"""Symbol strings on the dyadic tree: symbolization, exact laws, sampling.

A point of [0, 1) is symbolized by its first n binary digits c_1 … c_n: 0 is the left child at a level,
1 the right child, and s_k = +1 for left, −1 for right. A model is anything with

    cond_left(strings) -> array (strings, n)     probability of "left" at each level given the prefix
    k                                            its number of free parameters

Laws over all 2^n strings are exact, so regrets and closure statistics need no test sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

Array = np.ndarray
PROB_FLOOR = 1e-12          # conditional probabilities are clipped to [floor, 1 − floor] inside logarithms


def all_strings(n: int) -> Array:
    """Every binary string of length n as rows of a (2^n, n) array; the first symbol is the most significant."""
    idx = np.arange(1 << n, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((idx[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def symbolize(points: Array, n: int) -> Array:
    """The pre-registered symbolization: a point of [0, 1) maps to its first n binary digits."""
    x = np.asarray(points, dtype=float)
    if x.size and (x.min() < 0.0 or x.max() >= 1.0):
        raise ValueError("points must lie in [0, 1)")
    out = np.empty((x.size, n), dtype=np.uint8)
    frac = x.copy().ravel()
    for k in range(n):
        frac = frac * 2.0
        bit = frac >= 1.0
        out[:, k] = bit
        frac = frac - bit
    return out


def signs(strings: Array) -> Array:
    """s = +1 for left (0), −1 for right (1)."""
    return 1.0 - 2.0 * np.asarray(strings, dtype=float)


def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))


def points_from_strings(strings: Array) -> Array:
    """The left end of the dyadic cell of each string: the inverse of `symbolize` up to the cell width."""
    strings = np.asarray(strings, dtype=float)
    return strings @ (0.5 ** np.arange(1, strings.shape[1] + 1))


def log2_prob(model, strings: Array) -> Array:
    """log2 of the probability of each string."""
    q = np.clip(model.cond_left(strings), PROB_FLOOR, 1.0 - PROB_FLOOR)
    return np.where(np.asarray(strings) == 0, np.log2(q), np.log2(1.0 - q)).sum(axis=1)


def string_law(model, n: int, strings: Optional[Array] = None) -> Array:
    """Exact probability of every string of length n (order of `all_strings`)."""
    strings = all_strings(n) if strings is None else strings
    return np.exp2(log2_prob(model, strings))


def sample_indices(law: Array, size: int, rng) -> Array:
    """`size` draws from a discrete law by inverse CDF, using rng.uniform (a `streams.Stream`)."""
    cdf = np.cumsum(law)
    u = np.asarray(rng.uniform(size), dtype=float) * cdf[-1]
    return np.minimum(np.searchsorted(cdf, u, side="right"), len(law) - 1)
