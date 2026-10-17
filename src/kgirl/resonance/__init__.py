"""kgirl.resonance — the salvageable core of RVSE v2.1 (Resonant Vector Sanctuary), rebuilt and validated.

    fractal    DFA, MF-DFA (h(q), τ, α, f(α), width) and rescaled range; validated against white noise,
               random walks and the exact binomial-cascade spectrum (kgirl.iwme.cascade)
    topology   exact H₀ persistent homology (Prim's MST), β₀(t), persistence entropy, single-linkage families,
               Gram → distance; sorted-death distance (an upper bound on the bottleneck)
    pll        second-order digital PLL with designed gains (ω_n, ζ) and an in-phase lock detector
    spiking    constant-Q resonate-and-fire neuron bank: CFAR spiking detection, reset-free spectral estimate

numpy only; all noise in tests comes from ChaosRNG. The original is kept verbatim in
research/resonance/originals/, and research/resonance/README.md records what was kept, rebuilt or left out.
"""

from .fractal import MFDFA, dfa, mfdfa, rescaled_range, trace_dimension
from .pll import PLL, PLLConfig, PLLTrace, track
from .spiking import BankResult, ResonatorBank, recognize
from .topology import H0, distances_from_gram, h0, pairwise_distances, sorted_death_distance

__all__ = ["MFDFA", "dfa", "mfdfa", "rescaled_range", "trace_dimension", "PLL", "PLLConfig", "PLLTrace", "track",
           "BankResult", "ResonatorBank", "recognize", "H0", "distances_from_gram", "h0", "pairwise_distances",
           "sorted_death_distance"]
