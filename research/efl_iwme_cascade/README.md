# efl_iwme_cascade

Integrated validation of EFL (cross-scale coherence closure) and IWME (attention memory with
actuation) on cascades. An isolated research package hosted in KGIRL.

Status: both registered designs have been run once and replicated once. Outcomes are in
`RESULTS.md`. Nothing here has been promoted into `src/kgirl`.


## 1. Boundary with the host

- Nothing under `src/` is changed by this package, and nothing under `src/` imports it.
- `host.py` is the only module that reads KGIRL. It takes five names: `EMACascade`, `MarkovCascade`
  (`kgirl.iwme.cascade`), `two_point_memory` (`kgirl.iwme.attention`), `IWMECore` (`kgirl.iwme.core`),
  `ChaosRNG` (`kgirl.evolve.chaos`).
- Requirements: Python 3.9 or later, numpy. No network, no database, no Julia service, no API key.
  Every run is deterministic and offline.
- Check C30 and `tests/test_streams.py` enforce the boundary on every test run.


## 2. Layout

    research/efl_iwme_cascade/
        host.py                       adapter to the host (read only)
        streams.py                    every random stream, by purpose, with provenance
        checks.py                     30 Level 0 claims, recomputed
        limits.py                     exploratory: where the registered tests stop working
        models/
            cascade.py                strings on the dyadic tree, exact laws, sampling
            baseline.py               B, MC1, MC2, MC3, HMM2 and their estimators
            iwme.py                   attention-memory family in the core's terms; IWMECore adapter
            efl.py                    chain, full, diagonal Gaussian models; ADJ, SKIP, NL truths
        metrics/
            coherence.py              per-frequency, broadband, normal-score
            closure.py                closure ratio, defect, surrogate band, classification
            description_length.py     regret, two-part code length
            interventions.py          handles and zero-shot scoring
        experiments/
            partial_observability.py  Part 1, the registered Experiment A
            layered_coupling.py       Part 2, adjacent, skip and nonlinear coupling
            real_data.py              the same analyses on measured data
        configs/                      registered designs, PCG replications, reduced variants
        results/                      program output of the registered and replication runs
        tests/                        unit tests
        PREREGISTRATION.md            Part 1, fixed before any run
        PREREGISTRATION_PART2.md      Part 2, fixed before any run
        AMENDMENTS.md                 changes made before any registered result, with reasons
        RESULTS.md                    verdicts, replication, deviations, limits

Part 2 keeps adjacent, skip and nonlinear coupling in one module because the three truths share one
design, one set of models and one classification rule. Splitting them would let each case be tuned
alone.


## 3. Commands

Run from `research/`.

    python -m efl_iwme_cascade checks
    python -m efl_iwme_cascade part1 --config part1_registered --jobs 2 --resume
    python -m efl_iwme_cascade part1 --config part1_replication_pcg --jobs 2 --resume
    python -m efl_iwme_cascade part2 --config part2_registered
    python -m efl_iwme_cascade part2 --config part2_replication_pcg
    python -m efl_iwme_cascade limits                      exploratory, not registered
    python -m efl_iwme_cascade data FILE [--depth 16]      measured points of [0, 1), one per line
    python -m efl_iwme_cascade layers FILE                 measured scale signals, one column per scale
    python -m unittest discover -s efl_iwme_cascade/tests -t .

Measured on 2 cores: checks 8 s, Part 1 about 10 minutes with `--jobs 2`, Part 2 about 25 s, limits
about 1 minute once the Part 1 stage caches exist, tests about 35 s. `--jobs` and `--resume` do not change any number (tested). `--resume` keeps finished
stages in `results/.cache`, keyed by a hash of the configuration and of every source file the numbers
depend on.

Exit status of `part1` and `part2` is 0 when every registered hypothesis is met and 1 otherwise.


## 4. Mathematical interfaces

**Strings.** A point of [0, 1) is its first n binary digits c_1 … c_n; 0 is the left child, 1 the
right, s = +1 or −1. A model is anything with `cond_left(strings)` (probability of "left" at each
level given the prefix) and `k` (free parameters). Laws over all 2^n strings are exact.

**Families (Part 1).**

- B, one probability, k = 1.
- MC1, MC2, MC3, a window over the last 1, 2, 3 scales with free start contexts, k = 3, 7, 15.
  Add-half (KT) estimates.
- HMM2, two hidden states, k = 7 nominal. Posterior mode with pseudo-count one half, EM warm-up then
  quasi-Newton, four deterministic starts (Amendment 1). Check C14: the string law depends on
  these seven parameters through five numbers, so five are identifiable.
- AM, attention memory, k = 3 (ρ, T, bias). Profile likelihood over ρ, logistic regression inside.

**AM in the terms of the repaired core.** With H the memory before the latest symbol s is written:

    memory       H' = (1 − ρ) H + ρ s                              core map M
    actuation    C_E(H) = κ (1 − ρ) H / T                          core map C_E, κ = 1 in the family
    dynamics     P(left) = σ( C_E(H) + (ρ s + bias) / T )          core map F_E

κ = 0 is hypothesis H1 (C_E ≡ 0): the memory still updates and no longer acts, and the law is an
order-1 Markov chain (check C06). `models/iwme.as_core` builds `kgirl.iwme.core.IWMECore` from these
maps and `core_rollout` runs it; check C05 requires symbol for symbol agreement with the family.

**Scores.**

- Regret on strings: KL(truth ‖ model) / n, bits per symbol, summed over all 2^n strings.
- Regret on scale signals: ½ [tr(Σ̂⁻¹Σ) − d − ln det(Σ̂⁻¹Σ)] / ln 2, bits per sample.
- Asymptotic regret of a correct family: k / (2 m ln 2) per observation, m observations.
- Total description length: −log2 P(data under the fit) + (k/2) log2(observations).

**Coherence, three versions kept apart.**

- per frequency: Welch magnitude-squared coherence, Hann window, non-overlapping segments. K segments
  put a floor of 1/K under zero coherence (`debias`); the 5 % null point is 1 − 0.05^(1/(K−1)).
- broadband: squared correlation at zero lag.
- normal-score: broadband after each signal is replaced by the normal scores of its ranks.

Coherence needs no normalization: it is unchanged by any gain or invertible linear filter on either
signal (C23, C25). Scales made by linearly filtering one signal have coherence 1 with each other
and carry no information; scale signals must be separate measurements or nonlinear functionals.

**Closure.** For scale signals Y_1 … Y_n with second-moment matrix C:

    ψ_k = Corr²(Y_{k−1}, Y_k)                 adjacent-scale coherence
    R   = Corr²(Y_1, Y_n) / (ψ_2 … ψ_n)       closure ratio (the EFL recursion read as an identity)
    D   = abs(log2 R)                          closure defect

In Part 1 the scale signals are the partial sums of a model's residuals s_k − E[s_k given the
prefix]; R = 1 there is a second-order test that the model's memory is sufficient. In Part 2 they
are the observed layers.

**Classification (Part 2, fixed).** Each of raw D and normal-score D is compared with the 95 %
point of the same statistic over 200 surrogate samples from the fitted chain. No violation:
"adjacent, linear". Raw violates and normal-score does not: "nonlinear layers, adjacent coupling".
Normal-score violates: "non-adjacent coupling".


## 5. Stream contract

`streams.stream(kind, purpose, index, replicate)` returns an object with `uniform(size)`,
`normal(size)` and a `provenance` record.

- `kind = "chaos"`: `kgirl.evolve.chaos.ChaosRNG` with the seeds written in the pre-registrations.
  The registered runs.
- `kind = "pcg"`: numpy PCG64 from `SeedSequence(20261112, spawn_key=(purpose, index, replicate))`.
  The replications.
- Purposes: `train`, `post` (data after an intervention), `hmm_init`, `layered`, `surrogate`,
  `welch`, `data_surrogate`.
- No family draws data. Training strings are drawn once per (truth, replicate) and handed to every
  family. The only stream a family touches is `hmm_init`. Regret is exact, so there is no validation
  or test sample to draw.
- Every results file records generator, numpy version, host commit and whether `src/` was unmodified.

Known defect of the registered ChaosRNG formulas, found after registration and pinned by
`tests/test_streams.py`: the seed step is 0.01 per truth and 0.001 per replicate, so with 12
replicates the streams of replicates 10 and 11 of one truth are the streams of replicates 0 and 1 of
the next (16 reused pairs across Part 1 training and Part 2 data and surrogates). Replicates inside
one truth are distinct, and every decision rule compares inside one truth. The PCG replication has
no reuse.


## 6. Experiments and acceptance criteria

Part 1 (`PREREGISTRATION.md`, `AMENDMENTS.md`): five ground truths at depth 16 (binomial, hidden
Markov, attention memory above and below its threshold, order-3 window), six families, seven sample
sizes, 12 replicates, five interventions. Hypotheses A1 to A7.

Part 2 (`PREREGISTRATION_PART2.md`): eight scale signals; truths ADJ, SKIP, NL; models chain (15),
full (36), diagonal (8); eight sample sizes, 12 replicates. Hypotheses B1 to B4.

Acceptance rows and the hypotheses that decide them:

- Adjacent coupling, EFL improves model selection: B1, B4.
- Skip coupling, EFL identifies the structural mismatch: B2.
- Nonlinear coupling, closure failure is detected: B3.
- Partial observability, measurable memory advantage: A2, A3, A4, with A1 and A5 as instrument checks.
- Memory intervention, changing the coupling changes predicted behaviour: A7, with A6.
- Real multiscale data: `data` and `layers` commands, criteria in `PREREGISTRATION.md` §9. Open
  until a measured data set is analysed.
- Reproducibility: the PCG replications (Amendment 2b, R1).

Negative outcomes are results and stay in `RESULTS.md`. `limits.py` measures where the registered
tests stop working: the margin of the closure rule A5 over all replicates, and the Part 2
classification on five chains outside the registered set. It is exploratory and carries no verdict.


## 7. Test matrix

- `checks.py`, C01 to C30: symbolization; laws and parameter counts; hidden-state forward recursion;
  AM equals the host's sampled two-point attention memory; IWMECore emits the family symbol for
  symbol; κ = 0 is an order-1 chain; a memory that does not actuate is unobservable; estimators
  (KT, recovery, projections, HMM2 gradient and EM, Amendment 1); HMM2 identifiability; closure
  identities and calibration; layered chain, skip and cubed layer; Gaussian moments; normal scores;
  chain model; Gaussian regret; Welch estimator; cascade pooled against fixed-frequency coherence;
  description length; handles; streams; isolation.
- `tests/test_checks.py`: every check passes; the check registry.
- `tests/test_models.py`: conventions, handles, core adapter, fits, layered truths.
- `tests/test_metrics.py`: closure, coherence, description length, interventions.
- `tests/test_streams.py`: registered seeds, the stream reuse defect, PCG keys, provenance, host
  defaults the registration relies on.
- `tests/test_registration.py`: code constants and configurations equal the registered designs; the
  registered documents are byte-identical to their registering commits; amendments are append-only.
- `tests/test_verdicts.py`: every decision rule can be met and can fail, on synthetic results.
- `tests/test_experiments.py`: reduced end-to-end runs; same numbers with two processes and through
  the cache; reports carry the "supports no claim" label.
- `tests/test_real_data.py`: measured-data entry points on stand-in data with a known answer.
- `tests/test_limits.py`: machinery of the exploratory limits module.


## 8. Measured data

`data FILE`: points of [0, 1), for example event times scaled to the record length. The first 80 %
train every family; the last 20 % give held-out code length and the closure defect of each family's
residual walk against that family's own surrogate band.

`layers FILE`: one column per scale, one row per sample. Adjacent coherence in three versions,
closure in two, the classification, and held-out code length of chain, full and diagonal.

What would count against the theory is fixed in `PREREGISTRATION.md` §9. Data requirements, measured
on stand-in data (`tests/test_real_data.py`): the closure test on strings has no power with a few
hundred test strings (95 % band near 0.65 at 400); it separates a missing memory at 1,600 test
strings and above. Timestamps must resolve 2^-depth of the record length.

A result on measured data is the only evidence this package can give about the Level 2 claims.
Everything else here is about the instrument.


## 9. Promotion rule

Nothing moves into `src/kgirl` until all of the following hold:

1. every acceptance row of section 6 has a recorded outcome, the measured-data row included;
2. an independent run reproduces the recorded outcomes from a clean checkout;
3. the change to the core that the experiment calls for is written down, with the result that
   requires it.

The experiment has not so far required any change to `kgirl.iwme.core`.
