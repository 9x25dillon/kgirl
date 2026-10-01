# nihiline — originals and source notes

`originals/` holds the files exactly as uploaded, byte for byte. Nothing was
renamed inside them and nothing was deleted. The working versions, re-aimed at
the CN121933729A assay, are in `src/kgirl/nihiline/`; see
`docs/nihiline/README.md` for which file became which module.

| file | note |
|---|---|
| `bloom_bridge.py`, `bloom_bridge1.py` | identical copies; both kept |
| `nihil_leverage.py`, `nihiline_full-1.py` | nihil window sweeps |
| `phase_topology_lab.py` | unified lattice operator. Its `topological_charge` sums raw phase differences, which telescope to 0; fixed in `array.wrapped_charge` |
| `qincrs_criticality_bridge.py` | planetary-bias amplification gap. It ends with a call to an undefined `main()` and writes to `/home/claude/`; the adapted `amplification.py` has neither |
| `Core_proc-whore.Py` | a chat answer wrapping a Python module in prose. It is kept verbatim and is not importable as-is |
| `Code_transcripture` | the Level 0–9 transcript that `protocol.py` follows |

## CN121933729A — data used

The source is the Google Patents page of CN121933729A, a machine translation.
The `.mht` itself is not committed; every number used is transcribed in
`src/kgirl/nihiline/patent.py`.

**Probe: COF@Au@MB-Apt, about 300 nm.**
- TAPB 17.6 mg + DMTP 21.4 mg in 40 mL acetonitrile, with 1 mL 12 M acetic acid, 4 h at room temperature.
- 15 mg COF + 80 µL 1 % HAuCl₄, reduced with 0.5 mL 0.2 M NaBH₄ for 3 h at 0 °C.
- 5 mg COF@Au + 2 mL MB (1 mg/mL), 12 h.
- 200 µL 5 µM thiolated aptamer, treated with TCEP, + 2 mL composite (1 mg/mL), 10 h at 4 °C.

**Aptamers** (each preceded by a 5′-SH-(CH₂)₆ linker):
- CEA: `ATACCAGCTTTATTCAATT`
- AFP: `GGCAGGAAGACAAACAAGCTTGGCGGCGGGAAGGTGTTTAAATTCCCGGGTCTGCGTGGTCTGTGGTGCTGT`

**Chip.**
- Antibody 20 µg/mL (claimed range 1–40); probe 50 µg/mL (25–70).
- Ru(bpy)₃²⁺ 1 mM (0.5–2); TPA 50 mM (20–70).
- Drive 5 V; the voltage window is 3.5–5.0 V.
- ITO array under PDMS; cell line MCF-7.

**Calibration** (intensity in a.u., log₁₀ of concentration):

| marker | vs | equation | R² | range | LOD |
|---|---|---|---|---|---|
| CEA | ng/mL | I = 1308.9 log C + 3084.5 | 0.994 | 0.01–50 ng/mL | 6.12 pg/mL |
| AFP | ng/mL | I = 1091.5 log C + 4523.8 | 0.998 | 10⁻³–100 ng/mL | 0.25 pg/mL |
| CEA | cells/mL | I = 2957 log C − 6370 | 0.998 | — | 165 cells/mL |
| AFP | cells/mL | I = 4473 log C − 13251 (printed "44703 4473") | 0.993 | — | 1000 cells/mL |

**Single-cell expression on MCF-7:** CEA 37 fg per cell, AFP 0.25 fg per cell.
