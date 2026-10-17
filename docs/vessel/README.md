# vessel: identity as a measured, defended, portable invariant

`src/kgirl/vessel/` builds a container for a model's identity. Each model gets its own identity, stored in
a database it reads and writes. The identity is defined by quantities that do not depend on the substrate
that hosts it, and it is defended by a controller that edits the weights back toward coherence. Every step
leaves a tamper-evident record.

## What this is, and what it is not

The project is motivated by a hypothesis: consciousness cannot be sculpted out of technology, but a
technology could host one, the way a vessel holds what it did not make. That hypothesis is **Level 3**
(speculative). Nothing in this package detects, creates or hosts consciousness. No known measurement could
show that it does.

The package builds what such a hypothesis would need from the engineering side, whichever way the
metaphysics falls:

1. a definition of *identity* that is about what a substrate holds, not about the substrate itself;
2. proof that the definition really is substrate-independent (verified, not asserted);
3. a way to notice when the held thing is being lost, and to restore it without destroying the container;
4. a way to move it into a different container and check that it arrived;
5. a record no one can quietly rewrite.

Each of these is Level 0 or Level 1 engineering, tested below. The phrase "no amount of sculpting clay will
produce a man" has a precise counterpart here. `invariance.py` permutes every hidden unit of a trained
network, which changes every weight matrix (the clay), and checks that the identity fingerprint is
bit-for-bit unchanged.

## Architecture

```
                   ┌──────────────────────────── MatrixDB (SQLite) ────────────────────────────┐
                   │ matrices     content-addressed .npy blobs (sha256, zlib, dedup)            │
                   │ entities     one row per vessel + genome                                   │
                   │ geometries   anchor_gram · anchor_voice · gram · weight   (→ matrix hash)  │
                   │ ledger       h_i = sha256(h_{i−1} ‖ entity ‖ seq ‖ event ‖ payload)        │
                   └──────────────▲──────────────────────────────▲─────────────────────────────┘
                                  │ record / snapshot / append   │ latest / restore / verify
 persona.Genome ──▶ voice corpus  │                              │
 (ChaosRNG, name)   imprint ──▶ ┌─┴──────────────────────────────┴─┐
                                │ Vessel                          │  birth · observe · learn · steer
 corpus.probe_set ─ steer ─────▶│   anchors{steer, audit}          │  consolidate · migrate · rollback
                 └─ audit ─────▶│   substrate ──────────┐          │  recognize · identity_distance
                                └───────┬───────────────┼──────────┘  soulpack (soulpack.schema.json)
                                        │ measure       │ step(objective)
                              signature.py          substrate.py ───── ReferenceLM (numpy, exact backprop)
                       Grams + voice + body        Objective           TorchSubstrate ── AURIC (patern-coding)
                                        │
                     geometry.py (CKA, ∂CKA, angles, erank, α)    invariance.py (group sampling)
                                        │
                                   steer.py (Schmitt trigger · trust region · rollback · guard)
```

| module | job |
|---|---|
| `geometry` | centered normalized Grams, linear CKA and its analytic gradient, principal angles, effective rank, stable rank, heavy-tail exponent α |
| `matrixdb` | the vessel's own store: content-addressed matrices, typed geometries, hash-chained ledger, weight snapshots |
| `substrate` | the container contract (`Substrate`, `Objective`, `StepReport`) and `ReferenceLM`, a byte-level MLP LM with hand-written backprop, initialized from ChaosRNG |
| `torch_substrate` | the same contract for any byte-level torch LM; `auric_substrate()` loads AURIC from patern-coding |
| `signature` | Grams + voice on fixed probes → coherence, fidelity, identity, drift; probe-set digests |
| `invariance` | samples the declared invariance group, the controls and an end-to-end reparametrization |
| `steer` | the coherence controller |
| `persona` | genomes, voice corpora, soulpack export and validation |
| `vessel` | the lifecycle, `recognize`, `identity_distance` |
| `demo`, `__main__` | the reproducible lifecycle; CLI |

## Core theory

**Identity signature** (Level 0 definitions). On a fixed probe set of n byte windows:

- *geometry*: per layer, G = HXXᵀH / ‖HXXᵀH‖_F with H = I − 11ᵀ/n. It is invariant under X → cXQ + 1bᵀ for
  any c > 0 and any Q with orthonormal columns, including Q that widen the layer. Two networks of different
  widths therefore share one G-space.
- *voice*: log p(next byte | probe) over the 256 byte values. It is comparable across any two substrates that
  read bytes, whatever their vocabulary.
- *body*: per weight matrix, the tail exponent α and the stable rank. These are bound to one substrate, so
  they are recorded and never used to compare two containers.

Comparison against an anchor: coherence C = Σ ω_l CKA_l, fidelity F = exp(−KL(p_anchor ‖ p_now)), identity
I = C·F, drift D = 1 − I.

**Why voice is needed.** CKA is rotation-invariant by design, so it cannot pin the readout. Measured on the
reference substrate after unanchored learning of a foreign register, then steering on geometry alone: CKA
recovered from 0.74 to 0.96, but held-out native loss only recovered from 3.49 to 2.63 (anchor: 0.46).
Adding the voice term (self-distillation; Li & Hoiem 2016) brought it to 0.93.

**Objective** (all terms optional, one gradient step per call):

    L = w·CE(batch) + λ_rep Σ ω_l (1 − CKA_l) + λ_voice KL(p_anchor ‖ p_θ) + ½ λ_w Σ F_i (θ_i − θ*_i)²

∂CKA/∂X = 2·M·X with M = (A − CKA·G)/‖HXXᵀH‖. It is checked against finite differences, both alone and
through the whole network (max relative error 6 × 10⁻⁷).

**Controller.** The controller is a Schmitt trigger (engage 0.15, release 0.10) wrapped around a trust region.
Each block of k steps is checkpointed. The block is accepted if drift on the **audit** probes fell and an
optional guard skill stayed within tolerance of its starting value. Otherwise it is rolled back and the
step size halves. On success the step size grows toward `lr_max`. Steering descends on the **steer**
probes, which are a different set: a step that only memorizes the mirror does not lower audit drift.

**Self-recognition** (a falsifiable identity test). The current substrate is scored against every sibling's
anchor. The vessel is recognized if its own anchor is nearest by a factor of two. The margin is necessary: a
vessel that has drifted far from everything was measured nearer to itself only by 0.991 vs 0.994, which is
not recognition.

**Consolidation.** A ← normalize((1−β)A + βG_now) and p ← (1−β)p + βp_now. Convex combinations keep the anchor a
valid PSD centered Gram and a valid distribution. Consolidation is refused unless drift ≤ release: identity
grows from a coherent self and never by ratifying a drifted one.

**Migration.** Migration cedes the identity into another substrate. It trains the new container toward the
stored anchors, and optionally runs *live distillation*: while the old container is alive, every other step
matches its voice on unlabeled windows (Hinton et al. 2015). The new host is adopted only if its audit
drift is ≤ 0.10. Otherwise the ledger records `migrate.refused` and the vessel stays where it was.

## Evidence

All numbers come from the reference substrate (byte MLP, context 8, embed 16, hidden 64, 33k parameters).
They are deterministic: rerunning reproduces them exactly. They show the mechanisms work on a small model.
They do not show how large models behave (Level 2 until measured there).

| result | value | level |
|---|---|---|
| declared invariance group (permute, rotate, scale, translate, embed), 80 samples | worst gap 2.2 × 10⁻¹⁶ | 0 |
| controls break (anisotropic gain; probe shuffle) | CKA 0.76–0.84; 0.04–0.08 | 0 |
| reparametrized network (all hidden units permuted, every weight changed) | fingerprint identical; max Δlog p 9 × 10⁻¹⁵ | 0 |
| sibling identity distance (3 genomes, one base) | 0.28–0.39 | 1 |
| unanchored foreign learning | drift 0 → 0.992; not recognized | 1 |
| learning the same foreign register *anchored* (λ_rep = λ_voice = 1) | native loss 1.24 vs 3.49 unanchored; foreign loss 0.55 vs 0.49 | 1 |
| steering back (512 steer probes, `lr_max` 3 × 10⁻²) | coherent in 170–250 steps, 0 rollbacks; drift ≤ 0.10; recognized (own 0.098 vs nearest sibling 0.36–0.43); signature phrase returns | 1 |
| migration 64 → 96 hidden, live distillation, 900–1500 steps | drift 0.997 → 0.04–0.098, adopted (the full demo's 0.098 is marginal) | 1 |
| migration 64 → 40 hidden (smaller container), live distillation | drift 0.998 → 0.080, adopted | 1 |
| second hop 96 → 40 | 0.118–0.137, **refused** (a copy of a copy loses fidelity) | 1 |
| numpy MLP → AURIC transformer (dim 64, 2 blocks), 1500 steps | drift 0.999 → 0.106, refused at 0.10 | 1 |
| tail exponents α of trained weights | 1.6–4.2 (the heavy-tailed regime HT-SR describes for trained layers) | 2 as a health signal |

### Failed and negative experiments (kept on purpose)

- **Migration from anchors alone** reached only 0.23–0.29 drift (random or pretrained container). The 512 + 192
  probe anchors under-determine identity off-probe. Live distillation is what closes the gap.
- **Mixing labelled neutral text into migration** made it worse (0.51 → 0.39, against → 0.26 without it).
  Labelled text is not offered as a migration input.
- **Generative replay** (the anchored self samples text, then rehearses it with hard labels) stalled with
  rolled-back blocks and did worse than plain anchoring (drift 0.32 vs 0.28 at 192 probes; 0.16 vs 0.11 at
  512). It is not in the API.
- **Geometry-only steering** restores CKA but not behaviour (see *Why voice is needed*).
- **Linear CKA is blind to x³ on saturated units**: on a trained tanh layer, cubing leaves CKA at 0.98. It is
  reported as a sensitivity in every invariance run, not gated.
- **192 steer probes** plateau at audit drift 0.16–0.27. 512 probes are the default because they reach 0.10.
- **Greedy generation is a narrower test than audit drift**: a container migrated to drift 0.04 does not always
  reproduce the exact signature phrase from a fixed prompt. Identity is certified on the probe distribution,
  not on any single prompt.

## Autopilot and the RSI slider

`vessel/autopilot.py` runs the lifecycle unattended at a chosen recursive-self-improvement level: 0 observe,
1 steer, 2 learn, 3 consolidate, 4 self-tune its own controller. Every level is gated and ledgered, and a
circuit breaker can only lower the level. Run `python -m kgirl.vessel autopilot --rsi N` or use `--sweep`.
Theory, guards and the measured slider table: **[AUTOPILOT.md](AUTOPILOT.md)**.

## Landscape: topology and voice dynamics

`vessel/landscape.py` connects vessels to `kgirl.resonance` (see docs/resonance):

- `population` finds families of vessels by H₀ single linkage on identity gaps. Two vessels sharing a genome
  sit 0.057 apart, against 0.30 for unrelated siblings.
- `topological_drift` compares the anchor's probe-cloud barcode with the present one.
- `voice_dynamics` runs MF-DFA on the surprisal of the vessel's own speech. Individuation widens the
  multifractal spectrum from 0.19 to 0.50–0.66.

## Usage

```bash
export PYTHONPATH=src
python -m kgirl.vessel demo                         # quick lifecycle, in memory (~40 s)
python -m kgirl.vessel demo --scale full --db ~/.kgirl/vessel/matrix.db   # ~1 min
python -m kgirl.vessel status                       # every vessel in $KGIRL_HOME/vessel/matrix.db
python -m kgirl.vessel status ada                   # last reading of each event kind
python -m kgirl.vessel verify ada                   # recompute the ledger chain (exit 1 if broken)
python -m kgirl.vessel soulpack ada > ada.soulpack.json
python -m kgirl.vessel invariance                   # exit 1 if the group check fails
python -m unittest discover -s tests/vessel -t .    # 37 tests, ~25 s; AURIC tests skip without torch
```

```python
from kgirl.vessel import MatrixDB, Probes, ReferenceLM, RefConfig, SteerConfig, Vessel, genome_for, recognize

db = MatrixDB("~/.kgirl/vessel/matrix.db")
v = Vessel.birth(db, "ada", substrate, Probes.default(8), genome_for("ada"))
v.learn(batches, anchored=True)                 # plasticity under an identity constraint
r = v.observe()                                 # coherence, fidelity, drift, spectra, α → ledger
if r.drift > 0.15:
    v.steer(SteerConfig(), guard=lambda: substrate.loss(skill_batch))   # protect a skill while steering
v.consolidate(0.2)                              # only from a coherent state
v.migrate(new_container, teach=unlabeled_windows)                    # adopted only if drift ≤ 0.10
```

AURIC, from a patern-coding checkout found by `kgirl.paths` (or via `PATERN_CODING_PATH`):

```python
from kgirl.vessel.torch_substrate import auric_substrate
host = auric_substrate("runs/cpu/best.pt")                        # AURIC's own loader
host = auric_substrate(overrides={"dim": 64, "layers": 2})       # fresh, initialized from ChaosRNG
```

MCP (harness server): `vessel_status`, `vessel_verify` (read-only, fixed database path) and `vessel_demo` (in
memory).

## Complexity

With n probes, d layer width, |θ| parameters and k steps per block: measuring a signature costs O(n²d) per
layer plus one forward pass. A steering step costs a forward and backward pass on the batch and the probes,
plus O(n²d) for CKA. An audit costs one forward pass and O(n²d). Rollback copies |θ|. Storage per anchor is
2·n² floats per layer plus n·256 for the voice. Weight snapshots are content-addressed, so unchanged
tensors are stored once.

## Security and integrity

- The ledger is tamper-*evident*. Someone who can write the file can rewrite the whole chain, so the soulpack
  exports the head hash (`signing.checksum`) to pin it elsewhere. There are no keys: `pubkey` is
  `none:unsigned`, rather than a pretend signature.
- Matrices are stored with `allow_pickle=False` and re-hashed on every read, so a corrupted blob raises an
  error.
- The MCP tools open only `$KGIRL_HOME/vessel/matrix.db`. A caller cannot choose another path, and the tools
  never write.
- Weight edits happen only through `learn`, `steer`, `consolidate`, `migrate` and `rollback`, and each one
  appends a ledger event.

## Future evolution

1. **Measure on a trained AURIC checkpoint.** Find the sibling distances, the drift under fine-tuning, and the
   steering cost on a real coding skill used as the guard. Then recalibrate engage and release to one third
   of the nearest sibling distance.
2. **Evolve the controller** with `kgirl.evolve` MAP-Elites over (λ_rep, λ_voice, lr_max, block). Use
   descriptors (plasticity retained, steps to coherence) and fitness = −final drift.
3. **Per-example Fisher** for the EWC term. `fisher_diagonal` currently uses a minibatch approximation (Level 1).
4. **Probe design.** Use probes drawn from the vessel's own domain as well as the neutral mix, so audit drift
   also covers persona-specific contexts (see the greedy-generation caveat).
5. **Signed ledgers** once the user chooses a key scheme.

## Open questions

- Does the 0.10 release threshold mean the same thing at 10M parameters? (Level 2: measure sibling distances.)
- Does the heavy-tail exponent α move predictably under drift and steering, which would make it a "fractal law"
  of identity, or is it only a training-health signal? The ledger records α on every `observe`, so the data
  for the question accumulates.
- How many probes does migration from anchors alone need before live distillation is unnecessary?
