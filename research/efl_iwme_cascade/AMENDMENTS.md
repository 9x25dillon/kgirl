# Experiment A — amendments to the pre-registration

Changes made after `PREREGISTRATION.md` was committed and before any registered ground truth, family
comparison, closure statistic or intervention was run. Each entry gives the change, the reason, and the
evidence, which comes only from test cases outside the registered set.


## Amendment 1 — estimator for HMM2

**Was:** expectation–maximization, 4 restarts, 80 iterations each, pseudo-count ½, best training
likelihood kept.

**Now:** the same objective (log-likelihood plus the pseudo-count ½ prior) and the same 4 deterministic
ChaosRNG starts. Each start runs 25 EM steps and then quasi-Newton (BFGS) ascent with exact gradients
from the forward–backward pass, to convergence. Best objective kept.

**Reason.** A convergence check on an unregistered hidden Markov cascade,
p = (0.8, 0.3), P = (((0.6, 0.4), (0.1, 0.9)), ((0.5, 0.5), (0.7, 0.3))), π0 = (0.3, 0.7),
fitted to its own exact string law at depth 12:

    EM steps      regret of the fit, bits per symbol (four starts)
          80      2.1e-04 to 3.7e-04
         500      2.2e-06 to 2.6e-05
        1000      6.8e-09 to 1.2e-06
        3000      below 2e-12

The optimum is 0. Eighty steps would have left the conventional hidden-state baseline with a fixed
excess regret of about 3 × 10⁻⁴ at every sample size, which is larger than the asymptotic regret of
every family at N = 3200. That would have biased A2, A3 and A4 in favour of the attention–memory
family. The amended estimator reaches regret 3 × 10⁻¹⁷ on the same law at depth 16.

**Side observation, Level 0.** The four starts reached zero regret at four different parameter
vectors (left probabilities (0.399, 0.675), (0.743, 0.432), (0.424, 0.689), (0.425, 0.707)). The
string law does not identify the seven parameters of this family, even up to relabeling of the two
states. This is the reason §7 of the pre-registration defines no intervention handle for G1.

Nothing else in the pre-registration changes: objective, pseudo-counts, restarts, families,
parameter counts, design, scores, thresholds and decision rules are as committed.


## Amendment 2 — isolation, random streams, memory coupling, and one more intervention

Written before any registered result exists. A first attempt at the registered run was started on
2026-11-12 and was stopped by an environment shutdown during its second stage. It wrote no result
file, and none of its numbers were read.

**2a. Location.** The experiment is a self-contained research package in
`research/efl_iwme_cascade/`. Nothing under `src/` changes. The package reads `kgirl.iwme.cascade`,
`kgirl.iwme.attention`, `kgirl.iwme.core` and `kgirl.evolve.chaos` through one adapter module
(`host.py`). The two registered documents moved here unchanged. The commands quoted in
`PREREGISTRATION.md` §9–§10 are now `python -m efl_iwme_cascade data FILE` and `--fast`.

**2b. Random streams.** The registered run keeps its ChaosRNG seeds exactly as registered. Every
family always receives the same training strings; no family draws its own data. All streams come
from one adapter (`streams.py`) that records generator, version and seed for every purpose.

Added, **replication R1:** the complete design is repeated with independent numpy PCG64 streams,
`SeedSequence(20261112, spawn_key=(purpose, index, replicate))`, including the starting points of
the HMM2 fits. The same decision rules are applied. R1 counts as a replication if its pattern of
MET and NOT MET equals that of the registered run. R1 exists to show that no verdict depends on the
ChaosRNG convention.

**2c. Memory coupling, stated in the terms of the repaired core.** With H the memory before the
latest symbol s is written, the attention–memory family is

    H′ = (1 − ρ) H + ρ s                         memory M
    logit of the next "left" = C_E(H) + (ρ s + bias) / T,   C_E(H) = κ (1 − ρ) H / T

with κ = 1. The memory acts on the dynamics; it is the actuation map of the core, and κ = 0 is
hypothesis H1 (C_E ≡ 0). At κ = 0 the family is an order-1 Markov chain in scale, so the matched
comparison of AM with MC1 in A3 is a comparison of memory actuation against its absence at equal
parameter count. A unit test drives `kgirl.iwme.core.IWMECore` with these maps and requires symbol
for symbol agreement with the family. No registered quantity changes.

**2d. Added interventions on the memory substrate, and hypothesis A7.**

- **I4 on G2** and **I5 on G3:** the actuation is severed, κ 1 → 0. AM sets its own κ to 0. The
  other families have no corresponding term and stay as fitted.
- Post-intervention streams follow the registered formula with intervention indices 3 and 4.
- **A7, memory-substrate intervention.** In ≥ 11 of 12 replicates, for I4 and for I5, the edited
  AM has regret at most one tenth of the smallest regret among the five other families.

**2e. Added description, no pass or fail.** Total description length at N = 3200 for every family
on every truth: −log2 P(training strings) under the fit, plus (k/2) log2(N n) bits for the
parameters, mean over replicates.

**2f. Part 2.** The EFL arm on layered scale signals (adjacent coupling, skip coupling, nonlinear
layers) is registered separately in `PREREGISTRATION_PART2.md`, committed with this amendment.

Unchanged: everything in `PREREGISTRATION.md` §2–§8 and Amendment 1.
