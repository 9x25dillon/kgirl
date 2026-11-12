# Experiment A — pre-registration

Phase 4 (A) of IWME-PRIME-01: the multifractal experiment with pre-registered symbolization and
parameter-count-matched model comparison, extended with an EFL closure test of memory sufficiency.

This file was committed before any model comparison was run. The commit that adds it contains no
experiment code. Anything that changes after results exist is recorded under "Deviations" in
`RESULTS.md`, never edited in here.

Evidence levels follow the paper: 0 established, 1 extrapolation, 2 hypothesis, 3 speculative.


## 1. What is tested, and what cannot be

The working hypothesis of Phases 3–4 (paper §Hypothesis, Level 2) is parametric: a low-complexity
attention–memory class predicts held-out behaviour better than hidden-state models of matched
dimension and identical information access. Proposition P1 rules out every other route through
observables.

A simulation leaves that hypothesis about nature untested. Its job here is to test the instrument
that will later be pointed at data:

- whether the model comparison identifies the generating mechanism when the truth is known,
  including the cases where the attention–memory class must lose;
- how much data that takes;
- whether an EFL closure statistic detects insufficient memory;
- whether a model that generates the observed law also predicts the law after a change of conditions.

Every hypothesis below is about that instrument (Level 0–1 once computed). The Level 2 claim stays
open until the same pipeline runs on measured cascades (§9).


## 2. System and symbolization

- Dyadic tree of depth n = 16. A sample is one point of [0, 1) drawn from the cascade measure.
- Symbolization σ, fixed: a point maps to its first n binary digits c_1 … c_n. c_k = 0 is the left
  child at level k, c_k = 1 the right child. s_k = +1 for left and −1 for right.
- The hidden state of every generator is unobserved. Every model receives the same symbol strings
  and nothing else.
- Scale range: all 16 levels. No level is dropped or pooled.


## 3. Ground truths

Five generators. Probabilities of all 2^16 strings are computed exactly for each.

- **G0, no memory.** Binomial cascade, p = 0.7.
- **G1, hidden Markov.** `cascade.MarkovCascade()` with its default parameters:
  p = (0.7, 0.4), P = (((0.9, 0.1), (0.3, 0.7)), ((0.2, 0.8), (0.6, 0.4))), π0 = (0.5, 0.5).
- **G2, attention–memory above threshold.** `cascade.EMACascade()` with its defaults:
  ρ = 0.4, T = 0.8, bias = 0.3, h0 = 0. Mean-field gain 1/(2T) = 0.625.
- **G3, attention–memory below threshold.** `cascade.EMACascade(rho=0.15, T=0.4, bias=0.0)`.
  Mean-field gain 1/(2T) = 1.25, the lock-in side of the pitchfork.
- **G4, delay window.** Order-3 Markov chain in scale. Probability of left given the last three
  symbols (oldest first), contexts in the order LLL, LLR, LRL, LRR, RLL, RLR, RRL, RRR:
  0.80, 0.35, 0.60, 0.25, 0.70, 0.45, 0.30, 0.65.
  Start: 0.50 for the first symbol; 0.60, 0.40 after L, R; 0.70, 0.50, 0.45, 0.35 after LL, LR, RL, RR.

The cascade temperature T and the attention temperature of `attention.two_point_memory` are related by
T_attention = 2T, so G2 sits above T_c = 1 (1.6) and G3 below it (0.8). The cascade memory
h′ = (1 − ρ)h ± ρ is the sampled form of the paper's memory map: its conditional mean is Φ(h).


## 4. Model families

All are fitted to the same training strings. k counts free parameters.

- **B**, binomial. k = 1.
- **MC1, MC2, MC3**, Markov chains of order 1, 2, 3 in scale with a full set of start contexts.
  k = 3, 7, 15. These are the conventional memories: a window of the last m scales.
- **HMM2**, two-state hidden Markov cascade (the `MarkovCascade` family). k = 7. This is the
  generic hidden-state model.
- **AM**, the attention–memory family: `EMACascade` with h0 = 0. k = 3 (ρ, T, bias).

Matched pairs for the comparison: AM (3) against MC1 (3); HMM2 (7) against MC2 (7).

Estimators, fixed:

- B and MC: Krichevsky–Trofimov (add ½) estimates of every conditional probability.
- HMM2: expectation–maximization, 4 restarts from deterministic ChaosRNG starts, 80 iterations each,
  pseudo-count ½ on every expected count, best training likelihood kept.
- AM: maximum likelihood. For fixed ρ the model is a logistic regression of the next symbol on the
  memory h, fitted by Newton's method with ridge 10⁻³ on (1/T, bias/T) and the constraint 1/T ≥ 0.
  ρ is profiled on a grid of 49 values from 0.02 to 0.98 and refined by golden-section search.


## 5. Design

- Training sizes N: 50, 100, 200, 400, 800, 1600, 3200 strings.
- 12 replicates per ground truth. Replicate r of ground truth g draws 3200 strings by inverse CDF
  from the exact string distribution with `ChaosRNG(0.1 + 0.01 g + 0.001 r)` and uses the first N.
- Score: exact regret r(F) = KL(truth ‖ fitted F) / n in bits per symbol, summed over all 2^16
  strings. No test set is sampled.
- Also reported: the N = ∞ floor of each family on each truth (the KL projection, fitted to the
  exact string distribution).
- "≥ 11 of 12" below is a paired sign test across replicates (one-sided p < 0.004).


## 6. EFL closure statistic

For a fitted model F and truth P:

- residual e_k = s_k − E_F[s_k given c_1 … c_{k−1}], residual walk Y_k = e_1 + … + e_k;
- adjacent-scale coherence ψ_k = Corr_P²(Y_{k−1}, Y_k);
- closure ratio R = Corr_P²(Y_1, Y_16) / (ψ_2 ψ_3 … ψ_16): end-to-end coherence over the product
  of all fifteen adjacent coherences, the EFL recursion Φ_{n+1} = Φ_n ψ_{n+1} read as an identity;
- closure defect D = abs(log2 R).

All moments are exact under P. The raw walk uses the symbols themselves.

Level 0 fact used: a walk with uncorrelated increments has R = 1 exactly, and the residuals of a
model whose conditional mean is correct are uncorrelated. Calibration on generic walks whose
increments are AR(1) with coefficient φ (no ground truth or model of this experiment involved):
D = 0.60 at φ = 0.1 and 1.93 at φ = 0.5. Negative φ gives R above 1.

Secondary, descriptive: the local profile abs(log2 R(k−1, k+1)) for k = 2 … 15.


## 7. Interventions

Fits at N = 3200 on pre-intervention data. The condition is then changed in the generator. No
post-intervention data is given for the zero-shot score.

- **I1 on G2:** T 0.8 → 0.4 (attention sharpened through the threshold).
  AM edits its fitted T to half its value. Other families have no corresponding term and stay as fitted.
- **I2 on G3:** ρ 0.15 → 0.60 (memory shortened).
  AM multiplies its fitted ρ by 4, capped at 0.98. Others stay as fitted.
- **I3 on G4:** table entry LLL 0.80 → 0.20.
  MC3 sets that entry to 0.20. Others stay as fitted.

Score: exact regret against the intervened truth.

No intervention is run on G1. A fitted hidden-state model carries no label saying which latent state
is which, so an edit to "state 0" has no defined target. That gap is reported as a finding about
handles, with no pass or fail attached.

Descriptive only: each family is refitted from scratch on N_post = 25, 50, 100, 200, 400, 800
post-intervention strings (6 replicates), and the smallest N_post at which its mean regret falls to
the zero-shot regret of the edited model is reported.


## 8. Hypotheses and decision rules

Each is MET or NOT MET by the stated rule. Nothing is re-scored after the fact.

- **A1, recovery.** At N = 3200 the family that contains the truth has mean regret at most
  5 k / (2 N n ln 2) bits per symbol (five times the asymptotic expectation). Applies to
  B on G0, HMM2 on G1, AM on G2 and G3, MC3 on G4.
- **A2, specificity.** At N = 3200 the family that contains the truth has the lowest mean regret of
  the six, and beats each other family in ≥ 11 of 12 replicates. On G0 this family is B.
- **A3, matched size.** On G2 and G3, AM beats MC1 in ≥ 11 of 12 replicates at every N from 100 to
  3200, and beats MC2 and HMM2 in ≥ 11 of 12 at N = 3200. On G1 and G4, at N = 3200, the truth's own
  family beats AM in ≥ 11 of 12. The advantage is specific to the mechanism, as P1 requires.
- **A4, sample efficiency.** On G2 and G3, AM has lower mean regret than MC2, MC3 and HMM2 at every
  N in the design.
- **A5, closure as memory sufficiency.**
  (i) On G0 the raw walk has D < 10⁻⁹.
  (ii) For every memory truth whose raw walk has D > 0.10, the residual walk of the truth's own
  family fitted at N = 3200 (replicate 0) has D < 0.10 and less than one fifth of the raw D.
  A memory truth with raw D ≤ 0.10 is declared uninformative for this test.
  (iii) On G2 and G3 the residual walks of B and of MC1 each have D more than five times the D of AM.
- **A6, change of conditions.** In ≥ 11 of 12 replicates:
  for I1 and I2 the edited AM has regret at most one tenth of the smallest regret among the five
  other families; for I3 the edited MC3 has regret at most one tenth of AM's.

Predicted outcome if the instrument works: all six MET. A3 and A6 include the cases where the
attention–memory class is predicted to lose.


## 9. What would count against the theory, later

On measured cascades with the same symbolization (`python -m kgirl.iwme experiment-a --data FILE`):

- AM fails to beat MC1 on held-out bits per symbol at equal parameter count: the parametric
  hypothesis fails on that system.
- A conventional family restores closure (D below the surrogate band) with fewer parameters than AM:
  the memory is conventional there.
- No family restores closure: the symbolization or the scale range is wrong for that system, and the
  result is inconclusive.


## 10. Fixed in advance

Depth, symbolization, generators, families, estimators, sizes, seeds, scores, statistic, thresholds
and decision rules above. A reduced configuration (`--fast`: depth 10, 2 replicates, two sizes)
exists for unit tests only and supports no claim.
