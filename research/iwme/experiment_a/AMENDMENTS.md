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
