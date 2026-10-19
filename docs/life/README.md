# kgirl.life — the Algorithmic Life Equation as a verified dynamical system

`src/kgirl/life/` rebuilds the mathematics of three uploaded manuscripts as one pure discrete map, its continuous
counterpart and closed-form analysis. All three are kept verbatim in `research/life/originals/`, with a triage in
`research/life/README.md`:
- *The Thermodynamics of Being* (revised manuscript)
- *Minimal Implementation of the Algorithmic Life Equation*
- its mathematical appendix

`src/kgirl/harness/metabolism.py` then runs the harness itself as an organism under that map.

Executive summary:
- Every equation of the appendix is implemented. The map passes its own invariants: the energy ledger closes to
  1e-12, entropy production is ≥ 0 at every step, and every state stays in its domain under violent input.
- The reference simulator has four measured defects, and the rebuild removes each one:
  - the self-model dies at step 49, so C_eff ≡ 0 for 88 % of steps
  - P never leaves log 2
  - the organization map runs at Euler multiplier −1.67
  - learning continues at zero free energy
- The appendix has two mathematical errors. Its coherence "barrier" is a constant. Its half-cycle replica offsets
  make the third replica redundant.
- The closed forms predict the simulation. Steady-state F and O agree to within 0.1 %. The critical inflow
  E_crit = 0.0664 marks the collapse: organization falls through O_min between Ē = 0.07 and Ē = 0.066.
- In the harness, the map sizes and staggers Jev swarms (replication criterion), sets Soup's forgetting rate,
  and reports vitals. At p = ½ it reproduces the old default of 3 agents. `KGIRL_LIFE=0` restores the old harness
  exactly.

```bash
PYTHONPATH=src python -m kgirl.life demo                  # one organism through a shock
PYTHONPATH=src python -m kgirl.life steady --energy 0.5   # closed form vs simulation, E_crit
PYTHONPATH=src python -m kgirl.life stress                # also: ablation, perturbation, compression
PYTHONPATH=src python -m kgirl.life stagger               # replica schedules, swarm plans
PYTHONPATH=src python -m kgirl.life triage                # the uploaded reference, measured (numpy)
PYTHONPATH=src python -m kgirl.evolve life -g 20          # MAP-Elites over the equation's parameters
PYTHONPATH=src python -m kgirl.harness vitals             # the harness's own metabolism
```

## Architecture

```
                           kgirl.life  (standard library only)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ core.step(p, s, I, E, u; X, reset, ablate) ──▶ (s', StepRecord)      pure, frozen state       │
 │   predict ─▶ ε ─▶ Var ε ─▶ energy ledger (W, D, deficit, paid, σ) ─▶ O, C, A (ẋ = α − βx)     │
 │          ─▶ M (NLMS + ridge/L1) ─▶ H (EMA) ─▶ self-model (S, q ─▶ G, L) ─▶ R ─▶ P ─▶ J ─▶ Q   │
 │ analysis   steady_state · critical_free_energy · critical_energy · early_warning · rigidity   │
 │ continuous vector_field · RK4 · corrected barrier · entropy production                        │
 │ replication availability · simulate · best_replication · swarm_size · schedule · best_stagger │
 │ experiments ablation · stress · perturbation · compression · markers (multi-seed)             │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
          ▲ step per verb                       │ plan / forgetting / vitals
 ┌────────┴─────────────────────────────────────▼───────────────────────────────────────────────┐
 │ harness.metabolism  (Hermes subscriber on "*", state in $KGIRL_HOME/life.json)               │
 │   assistant.ask|task opens an episode · llm.complete.* and jev.done accumulate · answer|swarm.end │
 │   closes it ─▶ I = (success, latency, cost, errors), E = 1 budget, X = tokens / budget       │
 └─────┬──────────────────────────────┬───────────────────────────────┬────────────────────────┘
       ▼                              ▼                               ▼
 Assistant.task(size=None)      Curator.step()                 CLI `vitals`, MCP `life_vitals`,
 SwarmConfig(size=N*,           γ_eff = γ^(1 + Var ε / 0.05)    `life_experiment`
             stagger_s=Δ*)      (variable forgetting factor)
 Jev k waits kΔ; a verified win before then ─▶ never spawned
```

## Core theory

State s = (A, C, O, F, H, M, R, P, π). Exogenous: observation I ∈ ℝᵈ, inflow E ≥ 0, measured work X ≥ 0.

| quantity | update | domain |
|---|---|---|
| error | ε = I − M φ(H), φ(H) = [H; 1] | |
| work | W = w0 + wA A + wM ‖M‖_F + wu c(u) + w_ext X | ≥ 0 |
| dissipation | D = δ0 + δF F + δε ‖ε‖² | ≥ 0 |
| free energy | F' = [F + η_in E − W − D]₊, deficit = [W + D − F − η_in E]₊ | F ≥ 0 |
| paid fraction | paid = min(1, (F + η_in E)/(W + D)) | [0, 1] |
| organization | Ȯ = a F C (1 − O) − b (D + ‖ε‖²) O | [0, 1] |
| coherence | Ċ = η_C (C_* − C) − η_V (Var ε − v_ref) | (C_min, C_max) |
| processing | Ȧ = η_A (A_target·paid − A) − η_AV Var ε | [0, A_max] |
| world model | M' = (1 − 2η_M λ) M + 2η_M·paid·ε φᵀ/(κ + ‖φ‖²), then prox of l1‖M‖₁ | |
| memory | H' = (1 − ρ·paid) H + ρ·paid [I; ε] | |
| self-model | Ṙ = η_R (G (1 − R) − L R), G = (1 + skill)/2, L = tanh(‖z − ẑ‖/tol) + ξ·1{disc} | [0, 1] |
| potential | P = log(1 + #{u ∈ Uʰ : Σ (overhead + wu c(uᵢ)) ≤ O κ(C) F}), κ(C) = 1 − ((C − C_*)/h)² | ≥ 0 |
| objective | J = α(‖ε‖² + λ‖M‖²) + β W + γ V + δ (P − P')₊; Q_u ← Q_u + η_π(−J − Q_u) | |
| index | C_eff = (A/A_max) C O R F/(F + K_F) | [0, 1) |

**One invariant structure.** A, C, O and R each obey ẋ = α − βx with α, β ≥ 0 frozen over a step. The step is
integrated exactly: x' = x e^{−β} + (α/β)(1 − e^{−β}). It equals Euler to first order, is unconditionally stable,
and is a convex combination for O and R, so they stay in [0, 1] with no clip. `integrator="euler"` keeps the
appendix's literal map for comparison. *(Level 0)*

**Invariants checked every step in tests** *(Level 0)*:
- F' = F + η_in E − W − D + deficit, to 1e-12
- σ = [(1 − η_in)E + min(W + D, F + η_in E)]/kT ≥ 0
- F ≥ 0, O, R ∈ [0, 1], A ∈ [0, A_max], C ∈ (C_min, C_max)
- with paid = 0, M decays exactly as (1 − 2η_M λ)ᵗ M₀ and H does not move
- LifeParams.violations() enforces the conditions the map needs:
  - NLMS stability: η_M < 1
  - ridge sign: 2η_M λ < 1
  - η_in ≤ 1
  - the Landauer floor: w0 ≥ kT ln 2 · bits erased per step

### Closed forms *(Level 0 algebra on a Level 2 model)*

Under stationary statistics (Ē, e² = E‖ε‖², v = Var ε, m = ‖M‖, c̄, X̄):

```
A* = A_target − (η_AV/η_A) v        C* = C_* − (η_V/η_C)(v − v_ref)
F* = [η_in Ē − (w0 + wA A* + wM m + wu c̄ + w_ext X̄) − δ0 − δε e²]₊ / δF          τ_F = 1/δF
O* = g/(g + ℓ),  g = a F* C*,  ℓ = b(δ0 + δF F* + (1 + δε) e²)                     τ_O = 1/(g + ℓ)
R* = G/(G + L)
F_crit = b O_min (δ0 + (1 + δε) e²) / (a C* (1 − O_min) − b δF O_min)      (O* ≥ O_min  ⇔  F* ≥ F_crit)
E_crit = [δF F_crit + w0 + wA A* + wM m + wu c̄ + w_ext X̄ + δ0 + δε e²] / η_in
```

| at Ē = 0.5, default parameters | closed form | simulated (mean of steps 1000–1500) |
|---|---|---|
| F | 34.83 | 34.80 |
| O | 0.9756 | 0.9755 |
| C | 0.4886 | 0.4872 |
| A | 0.6891 | 0.6877 |

The RK4 integration of the continuous system also settles on these values, to within 2e-3.

## What was fixed, and how it was found

| # | reference behaviour (measured by `python -m kgirl.life triage`) | rebuild |
|---|---|---|
| 1 | F settles at 35.4, so g + ℓ = 2.67: the Euler organization map has multiplier **−1.67**. Only the clip at O = 1 hides a period-2 chatter. | Exact step. With a constant world, the Euler map's O swings by > 0.01 every step; the exact map's swing is < 1e-6 (tested). |
| 2 | The self-model gain is G = −tanh‖ε‖ ≤ 0 (PDF listing) or 0.1(1 − \|F − 2\|/5), which is negative for F > 7 (.py). **R = 0 from step 49**, and C_eff = 0 on 88 % of steps. | G is the appendix's "predictive usefulness of internal state for the next error", as an online R² skill score. L is the self-prediction error of a learned linear self-model plus ξ at a discontinuity. Logistic form, with an interior equilibrium. C_eff is never 0 while fed. |
| 3 | P = log(1 + max(1, int(4·min(1, F/2)·O·C))), which is **log 2 for every state**, fed or starving. | N_reach counts affordable action sequences under the usable budget O κ(C) F. P falls monotonically with F, from log 17 to 0 (≥ 6 distinct levels). |
| 4 | Var ε = `np.var` over a list of vectors (time and channels pooled). | Channel mean of an exponentially weighted per-channel variance. |
| 5 | Ring-buffer memory; ρ unused. | The appendix's EMA, with ρ. |
| 6 | Nothing can push C above C_*, so the "too rigid" side of the window cannot be reached. | v_ref > 0 makes the window two-sided: a world that is too predictable raises C toward C_max. |
| 7 | π never changes. | Bandit on the appendix's constrained objective J. It learns the cheaper action when costs differ (tested). |
| 8 | With F = 0 the model, memory and self-model keep learning. Stationary error was identical (0.447) at every inflow from 0.6 down to 0.02. | Work is done only as far as it is paid for. Under starvation the error now rises (0.447 → 0.844 at Ē = 0.03) and A falls. |

Two errors in the mathematics itself *(Level 0)*:
- **The coherence confinement term** −γ_C(C − C_min)₊ − γ_C(C_max − C)₊ equals −γ_C(C_max − C_min) everywhere
  inside the window, because the two brackets sum to a constant. It is a constant downward bias. Under strong
  variance, C leaves the window with it (tested). `continuous.barrier` uses a log-barrier gradient, which keeps C
  inside.
- **Staggered replicas** T_k(t) = T(t − (k − 1)τ/2), k = 1, 2, 3 have offsets 0, τ/2 and τ ≡ 0. Replica 3 is in
  phase with replica 1. Three replicas then get exactly the availability of two and pay for three (tested for
  r = 0.3–0.8τ). Uniform phases (k − 1)τ/N are optimal for equal resets. Downtime is min(1, [N r/τ − (N − 1)]₊),
  so N replicas give full availability iff r ≤ (N − 1)τ/N.

## Experiments (the manuscript's five axes)

3 seeds. World: 8 channels of sinusoid + 0.1 noise, Ē = 0.5 ± 20 %, distribution shift at t = 600 (offset +1,
phases reversed). Level 1 measurements of a Level 2 model.

**Ablation** (pre-shock means; adaptation = steps until the smoothed error is back within 1.5× its pre-shock mean):

| ablated | error | C_eff | R | adaptation steps | overshoot |
|---|---|---|---|---|---|
| none | 0.704 ± 0.006 | 0.239 | 0.757 | 6.3 ± 0.6 | 1.95 |
| memory H | 0.993 ± 0.009 | 0.216 | 0.718 | 24.3 ± 1.5 | 1.95 |
| model M (frozen) | 1.567 ± 0.001 | 0.202 | 0.714 | never | 3.04 |
| self R (frozen) | 0.704 | 0.095 | 0.300 | 6.3 | 1.95 |
| coherence C (frozen) | 0.704 | 0.247 | 0.760 | 6.3 | 1.95 |

Finding: memory and the world model carry the functional capacities. **R and C are diagnostic in these
equations.** Freezing them changes C_eff and nothing else, because R enters no other update and C enters only
aFC and κ(C). The manuscripts claim that the self-model matters for capacities, and the equations do not
instantiate that claim. See Open questions.

**Resource stress** (late-run means, one row per inflow):

| Ē | 0.6 | 0.2 | 0.1 | 0.08 | 0.07 | **0.066** | 0.06 | 0.05 | 0.03 |
|---|---|---|---|---|---|---|---|---|---|
| O | 0.978 | 0.946 | 0.840 | 0.699 | 0.444 | **0.189** | 0.039 | 0.002 | 0 |
| C_eff | 0.252 | 0.201 | 0.140 | 0.089 | 0.032 | 0.005 | 0.0002 | 0 | 0 |
| A | 0.687 | 0.687 | 0.687 | 0.687 | 0.687 | 0.687 | 0.642 | 0.541 | 0.315 |
| error² | 0.447 | 0.447 | 0.447 | 0.447 | 0.447 | 0.449 | 0.485 | 0.576 | 0.844 |

Analytic E_crit = **0.0665**, and O first falls below O_min = 0.2 at Ē = 0.066. Order of degradation (inflow at
which a factor loses half its unstressed value): C_eff at 0.08, O at 0.07, A at 0.03. C and R never lose half.
R even rises under starvation, because a system that has stopped changing is easy to predict. R measures
self-*predictability*, not self-maintenance.

**Perturbation** (shift size → adaptation steps, overshoot, min R/R_before, min O after):
0.25 → 1, 1.21, 0.84, 0.93 · 0.5 → 1, 1.38, 0.81, 0.89 · 1 → 6.3, 1.95, 0.77, 0.78 · 2 → 15, 3.23, 0.72, 0.49 ·
4 → 20, 5.70, 0.63, 0.20. Every measure degrades monotonically with shock size.

**Compression** (ridge λ → ‖M‖, error, adaptation steps):
0 → 2.10, 0.671, 6.3 · 0.01 → 1.32, 0.704, 6.3 · 0.1 → 0.85, 0.829, 9 · **0.3 → 0.69, 0.919, 17.7** ·
1 → 0.41, 1.08, never · 3 → 0.19, 1.27, never. The knee is at λ = 0.3, where compression starts to destroy what
recovery needs. From λ = 1 the organism never re-adapts. This is the manuscript's "compress what is redundant
while preserving the differences needed for adaptation", located numerically.

**Markers.** `experiments.markers(run, observed)` correlates C_eff and each factor with external markers, beside
the manuscript's baselines (pure energy, pure error). No marker data exists here, so none was fabricated.

**Evolution.** `python -m kgirl.evolve life` runs MAP-Elites over (a, b, η_C, η_V, η_M, λ, ρ, δF) at Ē = 0.12.
The behaviour grid is ‖M‖ × adaptation steps. In 12 generations: 32 % coverage, every elite survived the shock,
and the champion's mean C_eff was 0.245 against 0.152 for the published defaults. The search pushes b and δF to
their lower bounds (slower decay, lower leak). That is what this fitness rewards, not a discovery about life.

## The "99.9 % ceiling", replaced by observables

The manuscript (§9.3) asks for "an observable criterion, such as a measured loss of adaptability or rising
recovery time". `analysis.rigidity(series, shocks)` combines two signals:
- **critical slowing down** (Scheffer et al. 2009; Dakos et al. 2012): Kendall τ of rolling lag-1
  autocorrelation and of variance
- **recovery time** after each perturbation, and its trend

The verdict is `adaptive`, `slowing` or `rigidifying`, and it is None while there are too few samples.
Validation: an AR(1) process ramped toward φ = 0.97 gives τ(lag-1) > 0.6, flagged. A stationary AR(1) is not
flagged. An unrecovered shock reads `rigidifying`.

Note *(Level 1)*: the life map has no bifurcation. Its relaxation rate g + ℓ shrinks only toward ℓ as F → 0, so
recovery slows smoothly and does not diverge. The early warning is for the harness, where the dynamics are not
known in advance.

## Harness integration *(mechanics Level 0, benefit Level 2)*

| harness layer | what the life equation does there | default / switch |
|---|---|---|
| Hermes bus | `Metabolism.on_envelope` on `*` is the sensorium. Every `ask` or `task` is one life step: success, latency, cost and error rate in, tokens charged as measured work, one budget of inflow. | on; `KGIRL_LIFE=0` |
| Jev swarm size | Replication criterion N* = ⌈ln(c/(pV))/ln(1 − p)⌉. p is the decayed Beta posterior of agent outcomes (cancelled agents are censored), and V/c = `KGIRL_LIFE_VALUE` (10). N* is capped by the affordable sizes N_reach(F, O, C), the same count that defines P. | used when no `size` is given (CLI `-n`, MCP `size`) |
| Jev launch | Stagger Δ* = argmin E[cost] + w·E[latency] over [0, τ̂] (exact expectations, `replication.schedule`). Agent k waits kΔ, and a verified win before then means it never spawns (`SwarmResult.unspawned`). | Δ = 0 until an agent run time is measured; w = `KGIRL_LIFE_LATENCY` (2) |
| Soup curator | γ_eff = γ^(1 + Var ε/0.05), capped at γ⁴: evidence fades faster while outcomes surprise the harness's own world model (variable forgetting factor). | `Curator(gamma_fn=…)`; `CurationReport.gamma` |
| state | $KGIRL_HOME/life.json, written atomically. A corrupt or incompatible file restarts the organism and marks the next step a discontinuity. | |
| observability | `python -m kgirl.harness vitals [--json]`, MCP `life_vitals`, `life_experiment`; `life.plan` envelopes in the trace | |

Swarm plans at V/c = 10 for 60 s agents with w = 2:

| p | N* | Δ* | E[cost] (agent·s) | E[latency] (s) | P(success) |
|---|---|---|---|---|---|
| 0.1 | 1 | 0 | 60 | 60 | 0.10 |
| 0.3 | 4 | 0 | 240 | 60 | 0.76 |
| 0.5 | 3 | 30 | 127.5 (180 unstaggered) | 82.5 | 0.875 |
| 0.7 | 2 | 60 | 78 | 78 | 0.91 |
| 0.9 | 1 | 0 | 60 | 60 | 0.90 |

Replica schedules, N = 3, reset r = 0.4τ (availability: exact / jittered phase-locked / jittered free-running):
- single: 0.60 / 0.60 / 0.60
- sync: 0.60 / 0.70 / 0.93
- half-cycle (manuscript): 1.00 / 0.998 / 0.92
- uniform: 1.00 / 1.000 / 0.95
- random: 0.76 / 0.77 / 0.93

Free-running replicas lose their offsets and converge to 1 − (r/τ)³ = 0.936. **Holding the phases is what the
coordination cost buys.** That makes the manuscript's criterion "availability gain > coordination cost +
replication cost" quantitative.

## Testing

- `tests/life/test_core.py` covers:
  - the invariants (ledger, entropy, domains, determinism, serialization)
  - every fix, measured against the reference: chatter, self-model, P, paid work, the two-sided window, policy
  - ablations and the triage
- `tests/life/test_analysis.py` covers:
  - closed form vs simulation, and E_crit bracketing the collapse (±15 %)
  - early-warning detection on AR(1) and the recovery time of an exponential
  - the confinement finding, the barrier, ODE vs closed form
  - exact availability vs closed form, the half-cycle finding, free-running convergence
  - N* vs brute force, schedule vs Monte Carlo, determinism of the evolve problem
- `tests/harness/test_metabolism.py` covers:
  - one bus episode = one step, persistence, corrupt-state recovery
  - plans that follow p and the reserve
  - adaptive forgetting, staggered swarms that skip spawns
  - Assistant, MCP and CLI wiring, and `KGIRL_LIFE=0`

## Failure modes

- Two processes (CLI and MCP) share `life.json`. The last writer wins and one episode can be lost. This is
  bounded: the state is a summary, not a ledger.
- Cost is counted per verb, not per correlation id: model calls carry no task parent on the bus. Verbs are
  sequential per Assistant, so this is exact today. It would break if verbs ran concurrently.
- With a single local GPU, parallel agents serialize. The schedule model then overstates the latency cost of
  staggering, which makes Δ* conservative.
- The swarm model assumes independent agents with equal run times (Level 1). Correlated failures, where all
  agents share one bad memory slice, make N* too large.

## Open questions

1. Give R and C causal roles, so that the self-model ablation can fail in a capacity and not only in C_eff. One
   option is metaplasticity η_M ∝ (1 − R) (learn faster when the self-model fails). Another is exploration
   τ_π ∝ (C − C_*) (explore more when rigid). Each must earn its place in the ablation table.
2. Is R, as self-predictability, the right reading of "self-model strength"? A dead system scores high. A
   self-*maintenance* term (prediction of one's own viability) may be what the manuscript means.
3. Fit the metabolism's V/c and latency weight from the user's own history (`vitals` over weeks), not from
   defaults.
4. The markers axis needs real data: behavioural or physiological markers of conscious processing.

## Evidence vs speculation

| claim | level |
|---|---|
| the map, its invariants, the exact integrator, the closed forms, the availability formulas, N*, schedule expectations | 0 |
| measured fixes, ablation/stress/perturbation/compression tables, the swarm and replica numbers | 1 (measured, this model) |
| every functional form, C_eff as a diagnostic, the harness mappings (tokens as free energy, outcomes as observations) | 2 |
| C_eff as a measure of experience; a universal optimization ceiling; consciousness as "the first artificial intelligence" | 3 — not claimed here |
