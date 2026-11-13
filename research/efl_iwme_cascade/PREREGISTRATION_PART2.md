# Part 2 — the EFL arm on layered scale signals: pre-registration

Committed with Amendment 2, before any run of Part 1 or Part 2 produced a result.

Part 1 tests memory under partial observation and uses EFL closure as its sufficiency test. Part 2
tests the EFL structure itself where every scale is observed: whether assuming adjacent-scale
coupling improves model selection when it holds, whether the closure statistic flags the cases
where it fails, and whether a nonlinear layer is told apart from a coupling between non-adjacent
scales.


## 1. System

Eight scale signals X_0 … X_7, zero mean, independent across samples. Three ground truths.

- **ADJ, adjacent coupling.** Var(X_0) = 1 and X_k = g_k X_{k−1} + E_k for k = 1 … 7, with
  gains g = (1.0, 2.0, 0.5, 1.0, 3.0, 0.7, 1.5) and independent Gaussian E_k whose variances are
  chosen so that the adjacent coherences are ψ = (0.95, 0.90, 0.80, 0.97, 0.60, 0.85, 0.75):
  Var(E_k) = g_k² Var(X_{k−1}) (1/ψ_k − 1).
- **SKIP, coupling between non-adjacent scales.** X_1 as in ADJ; for k ≥ 2,
  X_k = g_k X_{k−1} + 0.6 X_{k−2} + E_k, with the same gains and the same Var(E_k) as ADJ.
- **NL, nonlinear layers with adjacent coupling.** Z is the ADJ chain with every layer scaled to unit
  variance. X_k = Z_k³ for odd k and X_k = Z_k for even k.

All covariances are exact: linear algebra for ADJ and SKIP, Gaussian moments for NL
(E[z³w] = 3ρ, E[z³w³] = 9ρ + 6ρ³, Var(z³) = 15 for unit normals with correlation ρ).


## 2. Models

Zero-mean Gaussian laws on the eight signals.

- **chain**, the EFL model: Markov in scale. Correlation between scales i < j is the product of the
  adjacent correlations between them. k = 15 (8 variances, 7 adjacent correlations).
- **full**: unstructured covariance. k = 36.
- **diagonal**: no cross-scale structure. k = 8.

Estimators: maximum likelihood from second moments about zero. For the chain this is the sample
variances and the seven sample adjacent correlations.


## 3. Design and score

- Sample sizes N: 20, 40, 80, 160, 320, 640, 1280, 2560.
- 12 replicates per truth. Replicate r of truth t (ADJ = 0, SKIP = 1, NL = 2) draws 2560 × 8 standard
  normals from `ChaosRNG(0.3 + 0.01 t + 0.001 r, lanes=4096)` and uses the first N rows. All models
  receive the same samples. Replication with PCG64 streams as in Amendment 2b (purpose 4).
- Score: regret against the best Gaussian description,
  r = ½ [tr(Σ̂⁻¹Σ) − 8 − ln det(Σ̂⁻¹Σ)] / ln 2 bits per sample, with Σ the exact covariance of the
  truth. No test set is sampled. Asymptotic expectation for a correct model: k / (2 N ln 2).


## 4. Closure

- Closure ratio R over all seven links and defect D = abs(log2 R), as in Part 1 §6.
- Two versions: **raw**, from second moments, and **normal-score**, after each signal is replaced by
  the normal scores of its ranks. Normal scores undo any monotone distortion of a layer. For a
  Gaussian copula the normal-score correlation is the maximal correlation, the measure for which
  the product rule along a Markov chain is guaranteed.
- Population values are exact. For NL the normal-score population is the latent chain.
- Sample values at N = 2560, each with the 95 % point of the same statistic under the fitted chain
  model: 200 surrogate samples of the same size from
  `ChaosRNG(0.45 + 0.01 t + 0.001 r, lanes=4096)`. A statistic above its 95 % point is a violation.
- Classification, fixed: no violation → "adjacent, linear"; raw violates and normal-score does
  not → "nonlinear layers, adjacent coupling"; normal-score violates → "non-adjacent coupling".

Reported with every truth: adjacent coherences, end-to-end coherence, the closure ratio in both
versions, and for ADJ the per-frequency Welch coherence of one long realization against the
broadband value.


## 5. Hypotheses and decision rules

- **B1, adjacent coupling helps model selection.** On ADJ the chain has lower mean regret than
  the full model at every N, wins the paired comparison in ≥ 11 of 12 replicates at N = 80 and at
  N = 2560, and at N = 2560 has mean regret at most 5 × 15 / (2 N ln 2).
- **B2, a skip coupling is identified.** On SKIP the population defect exceeds 0.10 in both
  versions; ≥ 11 of 12 replicates are classified "non-adjacent coupling"; and at N = 2560 the full
  model has lower mean regret than the chain.
- **B3, a nonlinear layer is detected and attributed.** On NL the population raw defect exceeds
  0.10 and the normal-score defect is below 10⁻⁹; ≥ 9 of 12 replicates are classified "nonlinear
  layers, adjacent coupling".
- **B4, no false alarm.** On ADJ, ≥ 9 of 12 replicates are classified "adjacent, linear".

The thresholds of 9 allow for the 5 % false-alarm rate of each band.

Negative outcomes are results. If SKIP or NL is misclassified, that is where the EFL closure test
does not apply as built, and it is reported as such.
