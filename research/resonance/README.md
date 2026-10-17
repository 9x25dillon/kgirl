# resonance — the original and what became of it

`originals/quaternary_hyperspatial_rvse_v2_1.py` is the Quaternary Hyperspatial Engine v1.0 with the
Resonant Vector Sanctuary Engine v2.1 integration, as pasted in session (2026-10-17). It is kept verbatim.

It does not run as-is:
- It imports `quaternion` (numpy-quaternion), which it never uses.
- `integrate_discovery` calls `nx.Graph()` without importing networkx.
- `scipy.signal.morlet2` was removed from recent scipy.
- Several base classes are placeholders.

Its noise comes from `np.random`, which kgirl replaces with ChaosRNG.

The working versions live in `src/kgirl/resonance/` (numpy only) and `src/kgirl/vessel/landscape.py`. The
guide is `docs/resonance/README.md`.

## Triage

| original | decision | why |
|---|---|---|
| `PhaseLockedLoop` | **rebuilt** → `resonance/pll.py` | It is a real PI-loop idea, but `kp = 2ζB`, `ki = B²` have no units: B is not a frequency, and the detector gain is ignored. A Hz correction is added to a phase advanced in radians. "Locked" means a small std of the raw error, which an unlocked loop can also satisfy. The rebuild uses Gardner's gains from ω_n and ζ and an in-phase lock detector. It acquires a 0.3 Hz offset in 0.27 s, pulls in 15 Hz in 3.6 s, holds lock at SNR 1, and reports *not locked* on noise. |
| `MultiscaleAnalyzer.compute_hurst_exponent` | **rebuilt** → `fractal.rescaled_range`, plus `dfa` | R/S regressed on unit-spaced scales, which weights the shortest, most biased windows most heavily. Scales are now log-spaced. DFA is added: white noise gives 0.501, a random walk gives 1.506. |
| `compute_multifractal_spectrum` | **replaced** → `fractal.mfdfa` | `log mean|Δx|^q / (q log n)` is not a scaling exponent. MF-DFA (Kantelhardt 2002) recovers the exact binomial-cascade width: 1.1239 vs 1.1239. |
| `wavelet_packet_decomposition` | left out | It convolves with full-length Morlet kernels and labels the outputs approx/detail without any filter-bank structure. `morlet2` no longer exists. The resonator bank covers the time-frequency need. |
| `PersistentHomology`, `_compute_topological_features` | **rebuilt** → `resonance/topology.py` | Births and deaths were `np.random` draws, and the entropy was unnormalized. The Betti numbers were a hard-coded `(1, 1, 1)`. Exact H₀ comes from Prim's MST (deaths equal Kruskal's). There is no claim about H₁ or H₂. |
| `NeuromorphicResonanceRecognizer` | **rebuilt** → `resonance/spiking.py` | Its "frequency tuning" was a Gaussian on sample *amplitude*, so it recognized signal level, not frequency. Its recurrent weights were random. The rebuild is a constant-Q resonate-and-fire bank (Izhikevich 2001) with CFAR spike detection. It recognizes tones from 7 to 190 Hz within 0.5 %, separates two tones, and stays silent on noise. |
| `get_topological_landscape` | **rebuilt** → `vessel/landscape.population` | Ward linkage on `1 − similarity` is undefined for a non-Euclidean similarity. Single linkage on a dissimilarity *is* H₀. Families are called only on a 2× scale separation. |
| `QuantumCircuit` | left out | "Hadamard" is `0.7·x + 0.3·roll(x)`, "Pauli-X" is a roll, and `T` and `CNOT` are no-ops. This is not a quantum simulation. A real state-vector simulator is easy to write but had no consumer here. |
| `QuaternaryHyperspatialEngine` (cytometer, interferometer, transceiver, entropy flow) | left out | The "codex" simulators sample THz chirps (1–2 THz, 8.5 THz) on a 10 kHz grid, so every waveform is aliased to meaningless frequencies. The patent numbers are not modelled. |
| `MuZeroKnot` | left out | T(2,3) is a knot, a single component, so a linking number is undefined. `p(q−1)/2 = 2` is neither its linking number nor its genus (1) or crossing number (3). |
| Octitrice / digital twin / narrative / sanctuary | left out | These are placeholders with no computation behind them. The vessel package already provides the identity container they gesture at, with measured coherence. |

Confidence levels: every rebuilt estimator is Level 0 (established method) and is validated in
`tests/resonance/`. Reading voice multifractality as a law of identity is Level 2 (see `docs/resonance`).
