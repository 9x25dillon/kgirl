# kgirl.neutronics — surrogate on real PWR lattice-physics data

```
NuRea_sim/lattice-physics+…/raw.csv  (24 000 sims)  ──► dataset.load/validate ──► Surrogate.fit (ridge, 5-fold CV over λ)
NuRea_sim/lattice-physics+…/test.csv (360 held out) ──► metrics vs. mean baseline ──► telemetry.json, surrogate.npz, limps_payload.json
```

```sh
git -C ../NuRea_sim lfs pull                 # the CSVs are Git LFS objects
PYTHONPATH=src python -m kgirl.neutronics info
PYTHONPATH=src python -m kgirl.neutronics fit --out runs/neutronics
PYTHONPATH=src python -m kgirl.neutronics cv --out runs/neutronics/cv.json   # poly vs maxtype, 5-fold x 3 chaos seeds
python -m pytest -q tests/integration/test_neutronics.py
```

Where the data is found: `--data DIR`, else `$NUREA_LATTICE_DIR`, else `kgirl.paths.find_repo("NuRea_sim")`
(`$NUREA_SIM_PATH`, `<kgirl>/NuRea_sim`, `../NuRea_sim`, `~/NuRea_sim`, `/home/kill/NuRea_sim`).

## Column interpretation (checked, not assumed)

| Column | Meaning | Range in raw.csv | Check |
|---|---|---|---|
| 0 | k_inf | 1.2464 – 1.3856 | corr(k_inf, mean enrichment) = 0.958 (train), 0.959 (test) |
| 1 | PPPF (pin power peaking factor) | 1.527 – 2.473 | corr(PPPF, max/mean enrichment) = 0.775 (train), 0.799 (test) |
| 2–40 | U-235 enrichment of 39 rods, wt% | 0.700 – 5.000 | |

`dataset.validate` raises if these checks fail, so a misread file can't produce a silent fit.

## Measured telemetry (L0)

Measured on 2026-10-17 on CPU: trained on all 24 000 rows of raw.csv and scored on the 360 rows of test.csv. The fit is
deterministic (folds are row index mod 5), so a rerun gives the same numbers. The degree-1 and degree-2 rows are
identical to the 2026-10-13 measurement.

| Model | Terms | λ (CV) | k_inf RMSE | pcm | max err (pcm) | R² | PPPF RMSE | max err | R² | Fit time |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean (baseline) | 1 | – | 0.019488 | 1471.8 | 5809 | −0.003 | 0.12374 | 0.408 | −0.008 | – |
| Degree 1 | 40 | 1 | 0.002205 | 166.5 | 1111 | 0.9872 | 0.07739 | 0.287 | 0.606 | 0.4 s |
| Degree 1 + max-type | 45 | 0.1 | 0.001221 | 92.2 | 593 | 0.9961 | 0.06392 | 0.265 | 0.731 | 0.4 s |
| Degree 2 | 820 | 1 | 0.000353 | 26.7 | 202 | 0.9997 | 0.06318 | 0.274 | 0.737 | 9.4 s |
| **Degree 2 + max-type** | 825 | 0.1 | 0.000330 | 24.9 | 193 | 0.9997 | **0.05730** | **0.223** | **0.784** | 8.2 s |

`fit` now reports `best` as the model with the lowest mean RMSE relative to the mean predictor over both targets;
that is degree 2 + max-type.

The max-type features are five peak-sensitive summaries of each design, standardized with training statistics:
`max`, the mean of the three largest enrichments, the spread (std), the smooth max ½·log Σ exp(2x), and max/mean.

### Cross-validated (L0): `python -m kgirl.neutronics cv`

5-fold CV repeated over 3 chaos seeds on all 24 000 rows; folds are ChaosGenerator permutations, λ is chosen inside
each outer fold, and the bootstrap resamples out-of-fold rows (1000 resamples, ChaosGenerator indices). The
acceptance rule, written before the run, was mean ΔR² ≥ 0.02 with a 95% interval that excludes 0.

| Degree 2 | PPPF R² | PPPF RMSE | k_inf RMSE | λ |
|---|---|---|---|---|
| poly | 0.7042 ± 0.0045 | 0.0653 | 23.1 pcm | 1 |
| + max-type | **0.7540 ± 0.0039** | **0.0595** | **21.7 pcm** | 0.1 |

ΔR²(PPPF) = **+0.0498** (min +0.042, max +0.055), positive in **15 of 15** folds; paired bootstrap ΔR² +0.0495,
95% CI **[+0.0463, +0.0526]**. The rule passes. k_inf improves too, so the extra columns do not trade one target for
the other.

Reading of these numbers:

- **k_inf (L0):**
  - The surrogates' held-out error is about 25–27 pcm, cross-validated about 22–23 pcm.
  - This is below the ~100s-of-pcm tolerances typically quoted for lattice codes against benchmarks. The surrogate reproduces *this code's* outputs; it has not been validated against reality.
- **PPPF (L0):**
  - PPPF is the harder target. Peaking is a local, max-type quantity, depending on which rod is hottest and on its neighbours, and a global polynomial captures it poorly (R² 0.70–0.74).
  - Five max-type summaries lift it to 0.75–0.78. They also compress: the 45-term degree-1 + max-type model matches the 820-term degree-2 model on PPPF (0.731 vs 0.737).
- **Candidate next models (L1):** pin-position-aware features, or a small net on the 2-D pin map. Both need the rod-index → position map, which is not yet confirmed.

## Correction to Eopiez's surrogate

The surrogate follows the approach of `Eopiez/refactored/neutronics_surrogate.py`: polynomial features, closed-form ridge, and the LIMPS payload shape.

**The bug in Eopiez:**

- It paired row *i* of `raw.csv` with row *i* of `test.csv` as (input, target).
- Both files are independent samples with the same 41 columns, so that pairing has no physical meaning.

**The fix here:** enrichments (columns 2–40) predict (k_inf, PPPF) (columns 0–1) within the same row.

## Evolving loading patterns (`python -m kgirl.evolve fuel`)

The validated surrogates become a design tool. MAP-Elites (`kgirl.evolve`, chaos-driven) searches the 39 rod
enrichments:

- **Fitness:** −PPPF, so flatter power wins.
- **Behaviour grid:** mean enrichment (fuel cost) × predicted k_inf (reactivity).
- **Result:** the archive holds the flattest design found for each cost/reactivity trade-off.

```sh
PYTHONPATH=src python -m kgirl.evolve fuel -g 1500 -b 128 --out runs/fuel_elites.json              # ensemble (default)
PYTHONPATH=src python -m kgirl.evolve fuel -g 300 --features poly                                   # the original search
```

### Modes

| `--features` | PPPF used for fitness | Penalties |
|---|---|---|
| `ensemble` (default) | the worse of the poly and max-type predictions | trust > 3 and support > 3 |
| `maxtype` | max-type model | trust > 3 and support > 3 |
| `poly` | poly model, as before this change | trust > 3 |

**Trust** is |mean − 2.851| / 0.198: how far the design's mean enrichment is from the training designs' mean, in
their standard deviations. The training designs draw each rod independently from U(0.7, 5.0), so their mean is
concentrated (observed range 2.14–3.63).

**Support** is the largest |z| of the design's five max-type features against the training designs. The same
independent draw keeps every training design's maximum at 4.0–5.0 wt% and its spread at 0.84–1.55 wt%. A
flattened design leaves that region even when its mean is ordinary, and trust cannot see it.

### Why the default changed: the better-fitting model was exploited (L0)

Making the max-type model the selector, with no other change, *looked* like a large gain:
- the champion's PPPF was 1.367;
- all 35 compared cells beat the simulated data.

The poly model scored that same champion at 1.502: an **optimism gap of +0.135**, against +0.019 for the original
search. The champion's max enrichment sat 8.7 SD below every training design. Search had found where the new
features extrapolate.

The same check applied to the original search (`poly`): **45 of its 59 elites** fall outside the training range of
some max-type feature, and its champion's support is 3.85 SD. Most of the earlier "17 of 35 cells flatter than
simulated" claims were extrapolations as well.

The rule, written before the measurement: the ensemble becomes the default only if its champion stays within
support (≤ 3 SD) and the two models agree on it within 0.02 PPPF. It met that in all four runs below.

### Measured runs (L0 for the numbers, L2 for the designs)

10 × 10 grid, batch 128. "In support" counts only elites with support ≤ 3 SD; the best simulated design in the
whole dataset has PPPF 1.527.

| Mode | Gens | Seed | Coverage | Champion PPPF | Other model (spread) | Support | Flatter than best simulated, same cell |
|---|---|---|---|---|---|---|---|
| poly | 300 | 0.4123 | 0.59 | 1.435 | max-type 1.453 (+0.019) | 3.85 | 17 of 35; **9 of 19 in support** |
| maxtype | 300 | 0.4123 | 0.54 | 1.425 | poly 1.464 (+0.038) | 2.84 | 19 of 35 |
| ensemble | 300 | 0.4123 | 0.48 | 1.446 | 0.009 | 2.52 | 13 of 33 |
| ensemble | 300 | 0.2718 | 0.53 | 1.436 | 0.015 | 2.32 | 13 of 35 |
| ensemble | 300 | 0.5772 | 0.51 | 1.477 | 0.001 | 2.91 | 12 of 33 |
| **ensemble** | **1500** | 0.4123 | 0.55 | **1.423** | 0.004 | 2.31 | 14 of 33 (median ΔPPPF +0.008) |

Reading the table:

- **The ensemble covers less.** Coverage is 0.48–0.55 against 0.59, because it is held to designs that resemble
  the training data. That is the price of credible predictions, not a regression in search.
- **Its champions are the most credible candidates so far.** They are inside the data's support, the two models
  agree on them, and the 1500-generation champion (1.423) is flatter than the original search's champion and
  0.10 below the best simulated design.
- **Most in-support elites do not beat the best simulated design in their cell** (median ΔPPPF > 0).
- **Everything here is a candidate for a lattice-code run, not a result.** The next step is to simulate the
  ensemble's elites in `--out` and refit on them.
