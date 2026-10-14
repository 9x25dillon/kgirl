"""Real input data for demos and examples, with a deterministic fallback.

Demos used to feed `np.random` noise into every subsystem. This module serves measured data instead:
the 24 000 NuRea_sim PWR lattice-physics simulations (k_inf, PPPF, 39 rod enrichments; see
kgirl.neutronics). When that dataset is absent, it falls back to logistic-map chaos (kgirl.evolve.chaos)
so the demos still run deterministically. `source()` says which one is in use, and demos print it.

    lattice_matrix(r, c)   enrichments, row-major, scaled to [0, 1]           (r·c values; rows wrap)
    lattice_signal(n)      k_inf (or PPPF) series in file order, z-scored     (n values)
    binary_pattern(g)      g×g enrichment map thresholded at its median        (balanced 0/1)
    spike_input(n, rate)   Bernoulli spikes with per-neuron rate ∝ enrichment (chaos-driven draw)
"""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np

from kgirl.evolve.chaos import ChaosGenerator

E_MIN, E_MAX = 0.7, 5.0                 # wt% U-235 range of the dataset


@functools.lru_cache(maxsize=1)
def _lattice():
    try:
        from kgirl.neutronics import dataset
        return dataset.load("train")
    except Exception:                    # missing repo, LFS pointer, unreadable file
        return None


def source() -> str:
    s = _lattice()
    return f"NuRea_sim lattice physics ({s.n} simulations)" if s is not None else "logistic-map chaos (dataset not found)"


def chaos(seed: float = 0.4123) -> ChaosGenerator:
    return ChaosGenerator(seed=seed)


def lattice_matrix(rows: int, cols: int, offset: int = 0, seed: float = 0.31) -> np.ndarray:
    s = _lattice()
    n = rows * cols
    if s is None:
        return chaos(seed).random((rows, cols))
    flat = s.enrichment.ravel()
    start = (offset * s.enrichment.shape[1]) % flat.size
    idx = (start + np.arange(n)) % flat.size
    return ((flat[idx] - E_MIN) / (E_MAX - E_MIN)).reshape(rows, cols)


def lattice_signal(n: int, target: str = "k_inf", offset: int = 0, seed: float = 0.37) -> np.ndarray:
    s = _lattice()
    if s is None:
        x = chaos(seed).normal(0.0, 1.0, n)
        return np.cumsum(x) / np.sqrt(np.arange(1, n + 1))
    y = getattr(s, target)
    y = y[(offset + np.arange(n)) % y.size]
    return (y - y.mean()) / (y.std() or 1.0)


def binary_pattern(g: int, offset: int = 0) -> np.ndarray:
    m = lattice_matrix(g, g, offset)
    return (m > np.median(m)).astype(int)


def spike_input(n: int, rate: float = 0.1, offset: int = 0, rng: Optional[ChaosGenerator] = None) -> np.ndarray:
    e = lattice_matrix(1, n, offset).ravel()
    p = np.clip(rate * e / max(e.mean(), 1e-9), 0.0, 1.0)
    return ((rng or chaos(0.53)).random(n) < p).astype(int)
