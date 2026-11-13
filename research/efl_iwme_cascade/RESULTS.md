# Results

Runs of 2026-11-12. Every number below is program output in `results/`; nothing was re-scored.

Evidence levels: Level 0, computed fact about these synthetic systems and this instrument. Level 1,
extrapolation about the instrument beyond the tested points. Level 2, a claim about measured systems.
All results here are Level 0. No Level 2 claim has been tested.


## 1. Verdicts

Registered run (ChaosRNG streams as registered), then replication (independent PCG64 streams).

    hypothesis                                   registered   replication
    A1  recovery                                 MET          MET
    A2  specificity                              MET          MET
    A3  matched size                             MET          MET
    A4  sample efficiency                        MET          MET
    A5  closure as memory sufficiency            MET          NOT MET
    A6  change of conditions                     MET          MET
    A7  memory-substrate intervention            MET          MET
    B1  adjacent coupling helps model selection  MET          MET
    B2  a skip coupling is identified            MET          MET
    B3  a nonlinear layer is detected            MET          MET
    B4  no false alarm                           MET          MET

Ten of eleven hypotheses are met in both runs. A5 is met once and fails once. By the criterion fixed
in Amendment 2b (the pattern of MET and NOT MET must be equal), Part 2 is replicated and Part 1 is
not: it differs at A5. Section 4 shows that the registered A5 verdict rested on one draw.

The exploratory study of section 5 (not registered) finds that the Part 2 classifier gives the right
label on the three registered truths and on a heavy-tailed linear chain, and the wrong label on
four other chains. B2, B3 and B4 hold for the registered cases and do not generalize as built.


## 2. Part 1, partial observability (registered Experiment A)

Design: depth 16; truths G0 binomial, G1 hidden Markov, G2 attention memory above its threshold,
G3 attention memory below it, G4 order-3 window; families B (1), MC1 (3), MC2 (7), MC3 (15),
HMM2 (7), AM (3); N from 50 to 3200; 12 replicates; exact regret over all 65,536 strings.

Mean regret at N = 3200, bits per symbol × 10⁻⁴. Registered run:

    truth        B       MC1       MC2       MC3      HMM2        AM
    G0        0.20      0.63      1.39      2.32      0.83      0.40
    G1       46.23     10.82      3.16      2.68      0.80     10.44
    G2      798.25    237.86     75.85     25.95     18.83      0.46
    G3     1460.76    781.50    446.37    263.72    137.72      0.29
    G4     1111.36    822.41    395.98      2.31    444.44    667.16

Replication:

    truth        B       MC1       MC2       MC3      HMM2        AM
    G0        0.09      0.42      0.94      2.32      0.58      0.25
    G1       46.11     10.64      3.23      2.53      0.81     10.37
    G2      797.98    237.83     75.82     25.94     19.08      0.46
    G3     1460.58    781.17    446.32    263.72    137.34      0.32
    G4     1111.30    822.33    395.93      1.96    444.45    667.13

- **A1.** The family that contains the truth reaches its bound on all five truths in both runs
  (largest ratio to the bound 0.28).
- **A2.** The family that contains the truth has the lowest mean regret on all five truths and wins
  12 of 12 paired comparisons, except B over AM on G0 in the registered run (11 of 12).
- **A3.** On G2 and G3, AM beats MC1 (three parameters each) in 12 of 12 replicates at every N
  from 100 to 3200, and beats MC2 and HMM2 in 12 of 12 at N = 3200. On G1, HMM2 beats AM in
  12 of 12; on G4, MC3 beats AM in 12 of 12. Both runs. The advantage follows the mechanism: the
  attention-memory class wins where the truth is attention memory and loses where it is not.
- **A4.** On G2 and G3, AM has lower mean regret than MC2, MC3 and HMM2 at every N in both runs.
  On G3, AM at N = 50 (23.9 registered, 38.9 replication) is already below the N = ∞ floor of every
  other family (lowest floor 136.7, HMM2). On G2 the same holds from N = 200 (5.0 and 8.7 against a
  lowest floor of 18.1).
- **A6.** Zero-shot regret against the changed generator, × 10⁻⁴, registered then replication.
  I1 (G2, T 0.8 → 0.4; the law moves by 906): edited AM 0.97, 0.91; best other family 1031.6, 1036.0.
  I2 (G3, ρ 0.15 → 0.60; 1272): edited AM 1.29, 1.72; best other 1070.9, 1065.9.
  I3 (G4, one table entry 0.80 → 0.20; 976): edited MC3 2.34, 2.03; AM 1335.2, 1332.4.
  Rule met in 12 of 12 replicates for each, both runs.
- **A7.** Severing the actuation (κ 1 → 0) moves the law by 257.5 (G2) and 581.0 (G3). The AM model
  fitted before the change, with its κ set to 0, has regret 0.32, 0.36 (I4) and 0.19, 0.14 (I5);
  the best other family 49.9, 49.1 and 122.8, 125.8 (MC1). Rule met in 12 of 12, both runs.
- In all five interventions no family refitted from scratch on up to 800 post-change strings
  reaches the regret of the edited model, the correct family included.

Total description length at N = 3200 (Amendment 2e, descriptive): bits above the shortest, mean over
replicates, registered then replication.

    truth   shortest   then
    G0      B          MC1 13.4, 14.0;  AM 14.6, 14.8;  MC2 40.8, 42.6;  HMM2 43.6, 44.4
    G1      HMM2       MC2 6.9, 7.5;  MC1 17.2, 14.6;  AM 18.1, 14.0;  MC3 57.2, 57.8
    G2      AM         HMM2 124.8, 121.6;  MC3 206.5, 203.5;  MC2 406.5, 401.3;  MC1 1199.0, 1192.6
    G3      AM         HMM2 737.8, 735.4;  MC3 1437.8, 1437.3;  MC2 2305.1, 2340.5;  MC1 4005.5, 4000.0
    G4      MC3        MC2 2000.4, 1949.2;  HMM2 2267.7, 2199.6;  AM 3378.7, 3315.9

HMM2 is charged for seven parameters. Check C14 shows five are identifiable; charged for five it is
15.6 bits shorter in every row, which changes no ranking (on G2 it is then 109.2 and 106.0 bits above
AM, on G3 722.2 and 719.8).

What Part 1 establishes: on synthetic cascades the comparison identifies the generating mechanism in
both directions, at equal parameter count and against larger hidden-state and window models, in
regret and in total description length; and a model whose terms correspond to mechanisms predicts
the law after a change of conditions that it was told about, where models that fit the old law
equally well or worse do not. I3 shows that this last property belongs to whichever family owns the
changed mechanism: there the window model has it and the attention-memory model does not.


## 3. Part 2, layered coupling

Design: eight scale signals; ADJ adjacent linear chain, SKIP the same with +0.6 from two scales up,
NL the adjacent chain with odd layers cubed; models chain (15), full (36), diagonal (8); N from 20
to 2560; 12 replicates; 200 surrogates per replicate.

- **B1.** On ADJ the chain has lower mean regret than the full model at every N in both runs:
  138.8 against 383.9 at N = 80 and 3.55 against 9.18 at N = 2560 (× 10⁻³ bits per sample;
  replication 131.3 against 355.7 and 4.39 against 10.42). Paired wins 12 of 12 at both sizes. Chain
  regret at N = 2560 is below its bound of 21.1.
- **B2.** SKIP has population defect 0.201 in both versions. All 12 replicates are classified
  "non-adjacent coupling" in both runs (sample defects 0.18 to 0.23 against bands of 0.02 to 0.03).
  The full model has regret 10.4 and 10.7 against 1493.6 and 1493.9 for the chain, whose floor
  is 1489.3.
- **B3.** NL has population raw defect 4.42 (closure ratio 21.4) and normal-score defect 0. All 12
  replicates are classified "nonlinear layers, adjacent coupling" in both runs.
- **B4.** ADJ is classified "adjacent, linear" in 11 of 12 replicates (registered; the twelfth has
  raw defect 0.133 against a band of 0.132) and 12 of 12 (replication).
- Per-frequency against broadband coherence (ADJ, 256 segments of 256): the two agree within 0.006
  on all seven links, as they must, because these signals are white in time. The case where they
  differ, a self-similar cascade, is computed exactly in check C26: pooled coherence falls as 1/n
  while fixed-frequency coherence keeps a floor of 0.3127.


## 4. A5 was not replicated

A5 has three parts. Parts (i) and (ii) hold in both runs: the binomial walk closes (defect
2.4 × 10⁻¹⁴), and on each memory truth the residual walk of the truth's own family, replicate 0, has
defect 0.019 to 0.066 against raw defects of 0.52 to 1.33.

Part (iii) asks that on G2 and G3 the residual defects of B and of MC1 each exceed five times that
of AM, read from replicate 0. On G3 this holds by a wide margin in both runs (MC1 0.83 and 0.86
against AM 0.028 and 0.035). On G2 the registered run has MC1 0.101 against AM 0.019, a factor of
5.2; the replication has 0.088 against 0.024, a factor of 3.7. That is the NOT MET.

`limits.py` reads the same statistic from all 12 fits of each run (exploratory):

    G2, rule (iii) holds in      6 of 12 replicates (registered),  4 of 12 (replication)
    G3, rule (iii) holds in     12 of 12,                         12 of 12
    G2, MC1 residual defect     median 0.120 and 0.099; its N = ∞ value is 0.115
    G2, AM residual defect      median 0.024 and 0.038; its N = ∞ value is 0
    own family restores closure, rule (ii) applied to every replicate:
        G1 12 and 12 of 12,  G2 12 and 12,  G3 11 and 11,  G4 11 and 11

Reading (Level 0 for these systems). The registered MET on A5 was one favourable draw of a rule
that holds about half the time on G2. The rule divides by a quantity whose true value is zero and
whose fitted value is estimation noise. The closure statistic is a second-order diagnostic. It
detects an absent memory on every memory truth (B keeps the full raw defect), and it detects windows
that are too short for the long memory of G3 (0.77 to 0.91 against at most 0.11 for AM). It does not
separate an order-1 window from the attention memory on G2, where the window already removes
nine tenths of the raw defect while its regret stays 500 times that of AM. On G1 it does not
separate AM from HMM2 (medians 0.037 and 0.029). A5 as registered is not established.


## 5. Limits of the Part 2 classifier, measured (exploratory, not registered)

`python -m efl_iwme_cascade limits`; N = 2560, 12 replicates, 200 surrogates, PCG64 streams outside
every registered index. No hypothesis was fixed in advance for these cases.

    case      what it is                                           right label given in
    HEAVY     linear adjacent chain, Laplace innovations           12 of 12
    QUAD      adjacent only, one link partly quadratic              0 of 12  (12 "non-adjacent coupling")
    CUBE      adjacent only, one cubic link, fresh noise after it   1 of 12  (11 "non-adjacent coupling")
    NEGSKIP   skip coupling of opposite sign, −0.6                  0 of 12  (12 "adjacent, linear")
    NLSKIP    registered +0.6 skip under cubed odd layers           0 of 12  (12 "nonlinear layers")

Why, from the same output:

- QUAD and CUBE keep second-order closure (large-sample raw ratio 1.006 and 0.998; the chain's
  Gaussian regret floor is 2 × 10⁻⁵ bits) and break normal-score closure (ratio 1.53 and 1.30). The
  registered rule reads a normal-score violation as a non-adjacent coupling. For a nonlinear
  coupling between adjacent scales that reading is wrong.
- NEGSKIP has a large population defect (3.04, closure ratio 8.2) and is never flagged. The skip
  nearly cancels two adjacent links (adjacent coherences 0.15 and 0.08), the product of adjacent
  coherences is 0.001, and the surrogate band of the end-to-end statistic is 4.1 to 8.3. The
  end-to-end ratio has no power once the product of adjacent coherences approaches 1/N. The chain's
  regret floor there is 2.48 bits per sample, so a comparison of code lengths still shows the
  mismatch, and the three-scale profile shows it locally (5.5 at the affected scale).
- NLSKIP has the same normal-score defect as the registered SKIP truth (0.20) and is never flagged,
  because both bands come from surrogates of the chain fitted to the distorted signals and the
  normal-score band is then 0.6 to 1.2 instead of 0.02 to 0.03. The attribution in B3 has the
  same weakness: its normal-score band was 1.1 to 1.6.

Reading. The classification is validated for a linear adjacent chain, a positive skip coupling on a
Gaussian chain, and monotone read-out distortions of a Gaussian adjacent chain. Outside these it
either mislabels or stays silent. The product rule is an identity only for Gaussian-copula chains;
for a general chain that is Markov in scale, maximal correlation satisfies the inequality
ρ(X, Z) ≤ ρ(X, Y) ρ(Y, Z), and normal-score correlation is maximal correlation only under a
Gaussian copula. The sentence in `PREREGISTRATION_PART2.md` §4 that calls the product rule
"guaranteed" for maximal correlation should be read with that restriction.


## 6. Acceptance rows

    row                                                    outcome
    Adjacent coupling: EFL improves model selection        supported: B1 and B4 met in both runs
    Skip coupling: EFL identifies the mismatch             supported for the registered skip (B2, both runs);
                                                           an opposite-sign skip is missed by the closure
                                                           test 12 of 12 (exploratory); code length still shows it
    Nonlinear coupling: closure failure is detected        detected and attributed for monotone layer
                                                           distortion (B3, both runs); nonlinear couplings are
                                                           flagged and mislabelled 23 of 24 (exploratory)
    Partial observability: measurable memory advantage     supported: A1 to A4 met in both runs
    Memory intervention changes predicted behaviour        supported: A7 and A6 met in both runs
    Real multiscale data                                   open: no measured data set has been analysed
    Reproducibility                                        Part 2 replicated; Part 1 replicated on six of
                                                           seven hypotheses, A5 not; clean-checkout reruns
                                                           reproduce the committed output (section 7)

The package is not ready for promotion under the rule in `README.md` §9.


## 7. Reproducibility and provenance

- Commits: registrations fbc7239, f34b7cb, af758b4; code 8180617; Part 2 registered output
  ee2934e; Part 1 registered output 5e658a5; replication outputs afc5d1d. The modules that compute
  registered numbers (`models/`, `metrics/`, `experiments/`, `streams.py`, `host.py`) are unchanged
  since 8180617: `git diff 8180617 -- research/efl_iwme_cascade/{models,metrics,experiments,streams.py,host.py}`
  is empty. `checks.py`, `limits.py`, the tests and the documents were added afterwards.
- Clean checkout: the registered Part 2 design rerun from a fresh clone of afc5d1d gives output
  identical to the committed files except for the recorded host commit and the elapsed time.
  The registered Part 1 design rerun from the same clone gives the same: all 108 kB of numbers are
  identical, and the only differing lines are the host commit and the elapsed time.
- Independent streams: the PCG64 replications of section 1.
- Every results file records generator, numpy version (2.5.3), host commit, and that `src/` was
  unmodified.
- `python -m efl_iwme_cascade checks`: 30 of 30. Unit tests: 119 pass. The host's own suites
  still pass with `src/` untouched: `tests/harness` 146 (2 skipped), `tests/iwme` 83 (2 skipped).


## 8. Deviations and defects

- **Stream reuse in the registered ChaosRNG formulas.** Found after registration. The seed step is
  0.01 per truth and 0.001 per replicate, so with 12 replicates the streams of replicates 10 and 11
  of one truth equal those of replicates 0 and 1 of the next: 16 reused pairs across Part 1 training
  and Part 2 data and surrogates. Replicates inside a truth are distinct and every decision rule
  compares inside one truth, so no verdict depends on it; samples of neighbouring truths are
  coupled in those replicates. The PCG replications have no reuse and give the same verdicts except
  A5. `tests/test_streams.py` pins the defect.
- **First attempt at the Part 1 run** was stopped by an environment shutdown before any result
  existed (Amendment 2). The registered run reported here is the first to complete.
- **Order.** Part 2 was run before Part 1. Neither registration fixes an order.
- **HMM2 parameter count.** Registered as seven; five are identifiable (C14). The registered count
  was kept in every rule; the effect on description length is stated in section 2.
- **Catch-up N** is "> 800" in every row of the intervention tables, the edited family included:
  the column compares refits from scratch with the edited model, and none reaches it by 800.


## 9. What these results do not show

- That any measured system is an attention-memory system, or that EFL closure holds in any
  measured system. Level 2, untested.
- That the attention-memory class beats hidden-state models in general. It wins on its own truths
  with three parameters. A hidden-state model with enough states can approach the same law
  (Proposition P1); only two states were tried.
- That the results hold away from the tested parameter points: one depth, binary symbolization, two
  attention-memory truths, one skip size, one kind of nonlinearity.
- That handles exist in real systems. A6 and A7 show that the comparison detects which family owns
  a mechanism when the change is known and the generator is in the family.
- Independence from the author: registration, code and runs come from one source. The decision
  rules could fail (A5 did, and `tests/test_verdicts.py` exercises each failure), but an outside
  rerun from a clean checkout is still owed.


## 10. For the next registration (proposals, untested)

1. Replace A5 (iii) by a rule that does not divide by a noise-level quantity: compare each family's
   residual defect with its own surrogate band, over all replicates, as `real_data.py` already does.
2. Read closure locally (three-scale ratios, maximum over scales, with a band for the maximum) so
   that weak links do not remove all power (NEGSKIP).
3. Draw the normal-score band from a chain fitted to the normal scores (NLSKIP, B3).
4. Read the two statistics as four outcomes instead of three: neither violates, raw only, score
   only, both. QUAD and CUBE fall in "score only". Whether that cell means "nonlinear coupling,
   adjacent" in general is a hypothesis.
5. One measured data set, with its own registration: system, scale construction, depth, and the
   criteria of `PREREGISTRATION.md` §9 fixed before the data are read.
