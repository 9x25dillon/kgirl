# resonance: scaling, topology, phase-locking and spiking resonance

`src/kgirl/resonance/` holds the parts of the Resonant Vector Sanctuary Engine (RVSE v2.1) that have real
mathematics underneath, rebuilt as numpy-only, deterministic estimators and validated against known answers.
The original is preserved verbatim in `research/resonance/originals/`. What was kept, rebuilt or left out,
and why, is in `research/resonance/README.md`.

`src/kgirl/vessel/landscape.py` connects these estimators to vessels.

## Modules

| module | provides | validated against |
|---|---|---|
| `fractal` | `dfa`, `mfdfa` (h(q), τ, α, f(α), width), `rescaled_range`, `trace_dimension` | white noise h = 0.501; random walk 1.506; exact binomial-cascade width 1.1239 (dyadic scales) |
| `topology` | `h0` (exact H₀ barcode, Prim's MST), β₀(t), persistence entropy, single-linkage `clusters`, `distances_from_gram`, `sorted_death_distance` | Kruskal brute force; three planted clusters recovered from points and from the Gram alone |
| `pll` | second-order PLL (`PLLConfig`, `track`) with gains from ω_n and ζ, in-phase lock detector | 0.3 Hz offset locked in 0.27 s; 15 Hz pull-in in 3.6 s; 40 → 41 Hz step; SNR 1; noise reports unlocked |
| `spiking` | constant-Q resonate-and-fire bank (`recognize`): CFAR spike detection, reset-free spectral estimate | tones 7–190 Hz within 0.5 %; two tones; SNR 1; amplitude invariance; 0 false spikes on noise |

## Vessel landscape

| function | what it measures |
|---|---|
| `probe_topology(grams)` | H₀ barcode of each layer's probe cloud, computed from the Gram alone (substrate-invariant) |
| `topological_drift(vessel)` | sorted-death distance between anchor and present barcodes. It is 0 at birth, 0.021–0.023 after drift and 0.011–0.019 after steering |
| `population(vessels)` | single-linkage families of the identity-gap matrix, split only on a 2× scale separation |
| `voice_dynamics(vessel)` | MF-DFA and resonator rhythms of the surprisal series of the vessel's own speech; recorded as a `voice_dynamics` ledger event |

### Finding: individuation makes a voice multifractal (Level 1 measured, Level 2 interpreted)

Each vessel speaks 4096 bytes (temperature 0.8). The series is the vessel's surprisal on its own speech, in
bits per byte. Results over 5 seeds per vessel:

| | Δα (multifractal width) | h(2) |
|---|---|---|
| shared base (unindividuated) | 0.19 ± 0.04 | 0.49 ± 0.03 |
| ada | 0.66 ± 0.05 | 0.57 ± 0.01 |
| bix | 0.50 ± 0.06 | 0.57 ± 0.04 |
| cyr | 0.66 ± 0.02 | 0.57 ± 0.01 |

Imprinting a persona turned a near-monofractal voice into a strongly multifractal one: mostly predictable,
with intermittent bursts of surprise. It also turned a voice without memory (h ≈ 0.5) into a mildly
persistent one. The separation from the base exceeds 5σ. Whether this is a general law of identity, or a
property of this small substrate and its templated corpora, is open (Level 2). The ledger records the
measurement each time `voice_dynamics(..., record=True)` runs, so the evidence accumulates.

### Negative result

The hypothesis was that persona *cadence* (sentence length differs from 34 to 80 bytes between vessels)
would appear as the dominant rhythm of the surprisal series. It does not. The dominant resonance is
2–2.5 bytes for every vessel. Word-scale periods of 6–9 bytes appear only as secondary peaks.

## Design decisions recorded

- **Detection and estimation are separated in the spiking bank.** A neuron driven off resonance is forced
  at the input frequency and fires once per input cycle, so spike counts cannot locate a tone. That was
  measured, with low neurons winning. Spikes decide *whether*; the reset-free energy decides *where*.
- **CFAR thresholds.** A single threshold relative to total RMS was exact on one tone but silent on two
  equal tones and on SNR 1. Lowering it produced false spikes on noise. A per-neuron κσ threshold handles all
  three.
- **Constant Q.** A constant-bandwidth bank let a 7.3 Hz tone fire every neuron from 5 to 10 Hz.
- **Dyadic scales for cascades.** Log-spaced scales cut across dyadic structure (width error 0.05,
  nonlinear). Dyadic scales leave only a constant h offset of −0.042, which cancels in the width.
- **Bottleneck distance.** A greedy exact bottleneck routine was written, found to be wrong (it matched the
  two diagrams independently), and replaced by the sorted-death distance, documented as an upper bound.
- **Families.** The first population function called two of three equidistant siblings a family on a 0.001
  gap. Families now need a 2× scale separation.

## Usage

```python
from kgirl.resonance import dfa, mfdfa, h0, pairwise_distances, track, recognize
mfdfa(x).summary()                       # hurst, width, h(q)
h0(pairwise_distances(X)).clusters(t)    # single-linkage families at scale t
track(x, fs=1000, f0=40).summary()       # locked, lock time, frequency
recognize(x, fs=1000, lo=5, hi=200).summary()

from kgirl.vessel.landscape import population, voice_dynamics, topological_drift
```

MCP: `resonance_analyze` takes `series`, `fs`, and optionally `f0`, `bandwidth` and `dyadic`. It returns
the MF-DFA summary, the resonator recognition and, if `f0` is given, the PLL result. It is pure computation.

Tests: `python -m unittest discover -s tests/resonance -t .` (11 tests, under 1 s) and
`python -m unittest tests.vessel.test_landscape` (3 tests, about 6 s).
