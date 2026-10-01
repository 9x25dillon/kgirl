---
name: evolve-designs
description: "Use when asked to optimize, explore or evolve designs, parameters or recipes in kgirl (assay conditions, pulse schedules, any bounded parameter set with a simulator): runs chaos-driven MAP-Elites quality-diversity search and reports elites, coverage, trade-off fronts and emergence statistics."
---

# Evolve designs (kgirl.evolve)

`src/kgirl/evolve/` runs deterministic quality-diversity search. Variation comes from
r = 4 logistic-map chaos mapped exactly to uniform/normal variates, so there is no random
number generator, which matches the user's lab conventions.

## Run the built-in assay problem

- MCP: `evolve_assay_design` (`generations`, `batch`, `seed`, `top`).
- CLI: `PYTHONPATH=src python -m kgirl.evolve assay -g 40 --plot docs/evolve/figures/assay_run.png`

Report four things:
- the champion against `baseline` (the patent's recipe under continuous drive)
- coverage and QD-score
- the size of the Pareto front over (yield, viability, sensitivity)
- stepping stones: how many niches the champion's ancestry crossed

Designs that sit on a claim bound (for example probe 25 µg/mL) are predictions to test in the
wet lab, not conclusions.

## Add a new problem

1. Define a `Space([Param(name, lo, hi, log=...)])`. Genomes live in the unit cube.
2. Write `evaluate(X: ndarray[n, d]) -> list[Evaluation(fitness, behavior, info)]`. Vectorize across the batch where you can.
3. Pick 2 behavior `Descriptor`s that actually vary. Check coverage: if it stalls below about 20 %, the descriptor ranges are too wide.
4. `MapElites(space, evaluate, descriptors, seed=...).run(generations)`, then use `.elites()`, `.grid()`, `.ancestry()` and `.stepping_stones()`.
5. Add a test that runs a few generations (see `tests/evolve/`).
