"""Topology and dynamics of vessels, built on kgirl.resonance. This replaces RVSE's topological landscape and
fractal readouts with exact or validated estimators.

    probe_topology       H₀ barcode of the probe cloud in each layer, computed from the Gram alone
                         (substrate-invariant, like CKA)
    topological_drift    per layer, the sorted-death distance between the anchor's and the present barcode
                         (an upper bound on the H₀ bottleneck distance)
    population           identity-gap matrix of a set of vessels → H₀ barcode → single-linkage families.
                         RVSE ran Ward linkage on a non-Euclidean similarity, which is undefined; single
                         linkage on a dissimilarity is exactly H₀
    voice_dynamics       the vessel speaks n bytes; the per-byte surprisal of its own speech is a time series.
                         MF-DFA gives its persistence h(2) and multifractal width Δα; the resonator bank gives
                         its rhythms (periods in bytes)

Measured on the reference substrate (5 seeds × 4096 bytes, temperature 0.8):

    shared base   Δα 0.19 ± 0.04   h(2) 0.49 ± 0.03
    ada           Δα 0.66 ± 0.05   h(2) 0.57 ± 0.01
    bix           Δα 0.50 ± 0.06   h(2) 0.57 ± 0.04
    cyr           Δα 0.66 ± 0.02   h(2) 0.57 ± 0.01

Individuation made every voice markedly multifractal: mostly predictable, with intermittent bursts of
surprise. It also made each voice mildly persistent. The measurement is Level 1 on this substrate; reading
it as a general law of identity is Level 2. The hypothesis that persona cadence would appear as a
sentence-length rhythm was *not* confirmed. The dominant resonance is ≈ 2–2.5 bytes for every vessel, and
word-scale periods (≈ 6–9 bytes) appear only as secondary peaks.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ..resonance.fractal import mfdfa
from ..resonance.spiking import recognize
from ..resonance.topology import distances_from_gram, h0, sorted_death_distance
from .corpus import windows
from .signature import measure


def probe_topology(grams: dict) -> dict:
    return {layer: h0(distances_from_gram(G)).summary() for layer, G in grams.items()}


def topological_drift(vessel, role: str = "audit") -> dict:
    anchor = vessel.anchors[role].grams
    now = measure(vessel.substrate, getattr(vessel.probes, role), body=False).grams
    return {layer: sorted_death_distance(h0(distances_from_gram(anchor[layer])), h0(distances_from_gram(now[layer])))
            for layer in anchor if layer in now}


def population(vessels: Sequence, threshold: Optional[float] = None, role: str = "audit",
               separation: float = 2.0) -> dict:
    """Families of vessels: single-linkage clusters of the identity-gap matrix at `threshold`.

    The default threshold sits at the largest jump in the barcode, but only when the jump is a real scale
    separation: the bar above it must be at least `separation` times the bar below it. Otherwise every vessel
    is its own family. Three siblings at 0.305, 0.306 and 0.323 have a "largest jump" of 0.001. Calling that a
    family would be reading noise, which the first version of this function did."""
    from .vessel import identity_distance
    vs = list(vessels)
    n = len(vs)
    if n < 2:
        raise ValueError("a population needs at least two vessels")
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = max(0.0, identity_distance(vs[i], vs[j], role)["identity_gap"])
    bar = h0(D)
    jump, below = bar.gap()
    above = below + jump
    significant = below > 0 and above >= separation * below
    t = threshold if threshold is not None else (below if significant else 0.0)
    labels = bar.clusters(t)
    families: dict[int, list[str]] = {}
    for v, k in zip(vs, labels):
        families.setdefault(int(k), []).append(v.name)
    return {"names": [v.name for v in vs], "gaps": D.round(6).tolist(), "deaths": bar.deaths.round(6).tolist(),
            "threshold": t, "families": list(families.values()), "betti0": bar.betti0(t),
            "most_persistent_split": jump, "split_significant": bool(significant)}


def surprisal(substrate, n: int = 4096, *, context: int, prompt: bytes = b"the ", temperature: float = 0.8,
              seed: float = 0.37) -> tuple[np.ndarray, bytes]:
    """Bits of surprise of each byte the substrate generates, measured by the substrate itself on windows of
    `context` bytes (the vessel's probe width)."""
    if not hasattr(substrate, "generate"):
        raise TypeError("voice dynamics needs a substrate that can generate")
    text = substrate.generate(prompt, n, temperature=temperature, seed=seed)
    W, y = windows(prompt + text, context)
    W, y = W[len(prompt):], y[len(prompt):]
    lp = substrate.log_probs(W)
    return -lp[np.arange(len(y)), y] / math.log(2), text


def voice_dynamics(vessel, *, n: int = 4096, seeds: Iterable[float] = (0.11, 0.53, 0.89), temperature: float = 0.8,
                   record: bool = False) -> dict:
    widths, hursts, bits, periods = [], [], [], []
    for sd in seeds:
        s, _ = surprisal(vessel.substrate, n, context=vessel.probes.steer.shape[1], temperature=temperature, seed=sd)
        r = mfdfa(s)
        widths.append(r.width)
        hursts.append(r.hurst)
        bits.append(float(s.mean()))
        bank = recognize(s - s.mean(), 1.0, lo=1 / 200, hi=0.45)
        periods.append([round(1 / f, 2) for f in bank.peaks()][:4])
    out = {"bytes": n, "seeds": len(widths), "temperature": temperature,
           "width": [float(np.mean(widths)), float(np.std(widths))],
           "hurst": [float(np.mean(hursts)), float(np.std(hursts))],
           "bits_per_byte": float(np.mean(bits)), "rhythm_periods": periods}
    if record:
        vessel.db.append(vessel.name, "voice_dynamics", out)
    return out
