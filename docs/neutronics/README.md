# kgirl.neutronics — surrogate on real PWR lattice-physics data

```
NuRea_sim/lattice-physics+…/raw.csv  (24 000 sims)  ──► dataset.load/validate ──► Surrogate.fit (ridge, 5-fold CV over λ)
NuRea_sim/lattice-physics+…/test.csv (360 held out) ──► metrics vs. mean baseline ──► telemetry.json, surrogate.npz, limps_payload.json
```

```sh
git -C ../NuRea_sim lfs pull                 # the CSVs are Git LFS objects
PYTHONPATH=src python -m kgirl.neutronics info
PYTHONPATH=src python -m kgirl.neutronics fit --out runs/neutronics
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

Measured on 2026-10-13 in this container on CPU: trained on all 24 000 rows of raw.csv and scored on the 360 rows of test.csv. The fit is deterministic (folds are row index mod 5), so a rerun gives the same numbers.

| Model | Terms | λ (CV) | k_inf RMSE | pcm | max err (pcm) | R² | PPPF RMSE | max err | R² | Fit time |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean (baseline) | 1 | – | 0.019488 | 1471.8 | 5809 | −0.003 | 0.12374 | 0.408 | −0.008 | – |
| Degree 1 | 40 | 1 | 0.002205 | 166.5 | 1111 | 0.9872 | 0.07739 | 0.287 | 0.606 | 0.3 s |
| Degree 2 | 820 | 1 | 0.000353 | 26.7 | 202 | 0.9997 | 0.06318 | 0.274 | 0.737 | 8.4 s |

Reading of these numbers:

- **k_inf (L0):**
  - The degree-2 surrogate's held-out error is about 27 pcm.
  - This is below the ~100s-of-pcm tolerances typically quoted for lattice codes against benchmarks. The surrogate reproduces *this code's* outputs; it has not been validated against reality.
- **PPPF (L0):**
  - PPPF is the harder target: R² is only 0.74 at degree 2.
  - Peaking is a local, max-type quantity, depending on which rod is hottest and on its neighbours. A global polynomial captures it poorly.
- **Candidate next models (L1):** pin-position-aware features, or a small net on the 2-D pin map.

## Correction to Eopiez's surrogate

The surrogate follows the approach of `Eopiez/refactored/neutronics_surrogate.py`: polynomial features, closed-form ridge, and the LIMPS payload shape.

**The bug in Eopiez:**

- It paired row *i* of `raw.csv` with row *i* of `test.csv` as (input, target).
- Both files are independent samples with the same 41 columns, so that pairing has no physical meaning.

**The fix here:** enrichments (columns 2–40) predict (k_inf, PPPF) (columns 0–1) within the same row.

## Evolving loading patterns (`python -m kgirl.evolve fuel`)

The validated surrogate becomes a design tool. MAP-Elites (`kgirl.evolve`, chaos-driven) searches the 39 rod enrichments:

- **Fitness:** −PPPF, so flatter power wins.
- **Behaviour grid:** mean enrichment (fuel cost) × predicted k_inf (reactivity).
- **Result:** the archive holds the flattest design found for each cost/reactivity trade-off.

```sh
PYTHONPATH=src python -m kgirl.evolve fuel -g 1500 -b 128 --out runs/fuel_elites.json
```

**Trust region.** Each rod in the training designs is drawn independently from U(0.7, 5.0) wt%. The 39-rod mean is therefore concentrated at 2.851 ± 0.198 wt%: the observed range is 2.14–3.63. Every elite carries `trust_sd` = |mean − 2.851| / 0.198, and fitness is penalised beyond 3 SD. Each elite is also compared with the best design *actually simulated* in its behaviour cell (Level 0).

**Measured runs** (L0 for the numbers, L2 for the designs; 10 × 10 grid, batch 128):

| Generations | Wall | Coverage | Champion PPPF (pred.) | k_inf | Mean wt% | Trust | Elite flatter than best simulated, same cell |
|---|---|---|---|---|---|---|---|
| 300 | 5.8 s | 59 % | 1.435 | 1.376 | 3.44 | 2.98 | 17 of 35 |
| 1 500 | 30 s | 61 % | 1.420 | 1.373 | 3.44 | 3.00 | 20 of 35 (median ΔPPPF −0.028) |

The best design actually simulated in the dataset has PPPF 1.528.

Reading the table:

- The champion is predicted 0.11 flatter than the best simulated design. That is about 1.7 times the surrogate's PPPF error (RMSE 0.063, R² 0.74), so the elites are candidates for a lattice-code run, not results.
- The champion sits at the trust boundary. The optimiser uses the high-enrichment edge of the data, which is where the surrogate is least constrained.
- The next step is to simulate the elites in `--out`. Refitting on them closes the loop.
