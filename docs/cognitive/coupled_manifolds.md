# Two coupled manifolds and the semantic-field ontology

## `kgirl.cognitive.coupled_manifolds`: measured

Two 13 × 50 unified-operator lattices (A, B) couple only through their macroscopic Bloom voices: γ|Ψ_other| enters each one's curvature κ. A is driven at ω = 1.05 and B at ω = 1.25.

```sh
PYTHONPATH=src python -m kgirl.cognitive.coupled_manifolds          # sweep + 64-bit transmission test
```

**Coupling sweep.** The original script's numbers are reproduced exactly (L0):

| γ | R_sync | Coherence | Drift (rad/step) | I(arg Ψ_A; arg Ψ_B) (bits) | Effective dimension | Intimacy |
|---|---|---|---|---|---|---|
| 0.00 | 0.398 | 0.392 | −0.0298 | 0.77 | 3.47 | 1.756 |
| 0.10 | 0.831 | 0.831 | −0.0053 | 1.53 | 2.37 | 1.444 |
| 0.20 | 0.750 | 0.743 | −0.0071 | 1.46 | 2.58 | 1.205 |
| 0.35 | 0.626 | 0.621 | −0.0103 | 1.02 | 2.88 | 1.005 |
| 0.55 | 0.855 | 0.842 | −0.0020 | 1.63 | 2.34 | 0.745 |
| 0.80 | 0.010 | 0.002 | +1.3234 | 0.61 | 2.00 | 0.002 (amplitude death) |

**Transmission test.** 64 balanced bits × 60 steps, decoded from **Ψ_B alone**:

| Scheme | γ = 0 (control) | 0.10 | **0.20** | 0.35 | 0.55 | 0.80 |
|---|---|---|---|---|---|---|
| Amplitude keying, BER | 0.53 | 0.50 | **0.28** | 0.34 | 0.44 | 0.53 |
| Amplitude keying, I(bit; decoded) (bits/symbol) | 0.003 | 0.000 | **0.143** | 0.072 | 0.011 | 0.003 |
| Phase keying ±π/3 (original code), BER | 0.47 | 0.50 | 0.50 | 0.53 | 0.56 | 0.56 |

**Reading:**

- **The original decoder read the transmitter, not the channel.** It used Δφ_AB = arg Ψ_B − arg Ψ_A, which contains A's own phase. It also decoded at chance: 4 of 8 errors at γ = 0.55, and 4 of 8 at γ = 0.
- **Information crosses only through amplitude.** The coupling γ|Ψ_A| is phase-blind, and the measurement confirms it.
- **The channel is open only at moderate coupling.** At γ = 0.20, 18 errors in 64 bits has p ≈ 3 × 10⁻⁴ under chance. The strongest lock (γ = 0.55) carries less, and amplitude death (γ = 0.80) carries nothing.
- **In this model, synchronisation and information transfer trade off.**
- **Interpretation:** calling the pair-only observables a "new sense" is an interpretation (L3). The numbers are L0.

## `kgirl.cognitive.nonlocal_consciousness`: conceptual (L3)

This is a translation of Chris H. Hardy's *Nonlocal Consciousness in the Universe* into data structures:

- semantic fields;
- the infinite spiral staircase;
- the CSR hyperdimension;
- sygons;
- psi processes;
- the law of syg-influence.

It is an ontology, not a simulation, and nothing in it is a measurement. The kgirl change records which psi process and which fields each sygon links. The coupled manifolds above are the measurable counterpart of "a link that exists only for the pair".
