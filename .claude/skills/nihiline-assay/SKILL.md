---
name: nihiline-assay
description: "Use for anything about the CN121933729A c-BPE-ECL dual tumor-marker assay (CEA/AFP on living MCF-7 cells) or the nihiline lab code: converting ECL intensities to concentrations or cell counts, detection-limit and scar-on-zero questions, per-cell amplification, pulse duty / viability, the Bloom sentinel or BPE array behaviour."
---

# nihiline assay lab

The lab lives in `src/kgirl/nihiline/`. The user's original scripts sit unchanged in
`research/nihiline/originals/`. Never modify or delete those; build on them instead.

## Quick answers (MCP tools)

- `ecl_quantify` (`cea`, `afp`, `mode=conc|cells`): ECL a.u. → ng/mL or MCF-7 cells/mL with the
  patent's curves. Flags: `ok`, `below_lod` (under the scar on zero), `below_range`, `above_range`.
- `ecl_protocol`: the nine-level assay run end to end for given cell densities. Deterministic.
- `ecl_gap`: the per-cell amplification chain and the predicted vs reported LOD.

## Deeper work (CLI, needs numpy for pulse/bloom/array)

```bash
PYTHONPATH=src python -m kgirl.nihiline scar          # LOD consistency across the four curves
PYTHONPATH=src python -m kgirl.nihiline pulse         # optimal nihil (off) fraction of the 5 V drive
PYTHONPATH=src python -m kgirl.nihiline bloom         # phase-before-amplitude viability warning
PYTHONPATH=src python -m kgirl.nihiline array         # crosstalk, lane RSD, defects, Lyapunov
PYTHONPATH=src python -m kgirl.nihiline plot -o docs/nihiline/figures
```

## Rules

- State confidence levels the way `docs/nihiline/README.md` does. Patent numbers are Level 0; the pulse, array and binding response curves are Level 2; THz phase-curvature sensing is Level 2–3.
- The AFP cell slope is 4473. The patent's "44703 4473" is an extraction artifact; `scar_report` shows why.
- Noise is deterministic logistic chaos. Do not introduce `random` or `np.random`.
- Tests: `python -m unittest discover -s tests/nihiline -t .`
