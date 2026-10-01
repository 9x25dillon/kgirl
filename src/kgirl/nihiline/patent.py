"""CN121933729A as data: the c-BPE-ECL dual tumor-marker in-situ living-cell assay.

Everything in this module is *reported by the patent* (confidence Level 0 as a
record of what the document states), except fields explicitly marked
`literature` (Level 1: textbook values we add to make the numbers computable).

Platform (claims 1-10, examples 1-6):

    cathode chamber (bio)                     anode chamber (optics)
    ITO BPE cathode ─ antibody (20 µg/mL)      ITO BPE anode
      └ MCF-7 cell ─ CEA / AFP antigen           1 mM Ru(bpy)3²⁺ + 50 mM TPA
          └ COF@Au@MB-Apt probe (50 µg/mL)       ECL photons  ∝ charge passed
              MB + 2e⁻ → leuco-MB  ── closed BPE charge balance ──▶ Ru²⁺ → Ru³⁺ → Ru²⁺*
    drive electrodes: 5 V (window 3.5–5.0 V), PDMS microfluidic chip on ITO array

Note on extraction: the AFP cell-curve slope is printed as "44703 4473". Only
4473 is consistent with the patent's own LOD (1000 cells/mL gives 168 a.u.,
matching the ~170–190 a.u. LOD signal of the other curves); 44703 would give
1.2e5 a.u. at the LOD. We use 4473 and flag it here.
"""

from __future__ import annotations

from dataclasses import dataclass

PATENT_ID = "CN121933729A"
TITLE = ("Sensing method for detecting double tumor markers carcinoembryonic antigen CEA and "
         "alpha-fetoprotein AFP on surface of living cell by ultrasensitive in-situ electrochemiluminescence")

AVOGADRO = 6.02214076e23
FARADAY = 96485.33212


@dataclass(frozen=True)
class LogLinearCurve:
    """I = slope · log10(x) + intercept, valid on [lo, hi] (units of x given by `unit`)."""
    slope: float
    intercept: float
    unit: str
    lo: float
    hi: float
    lod: float          # reported limit of detection, same unit (S/N = 3)
    r2: float


@dataclass(frozen=True)
class Marker:
    name: str
    full_name: str
    aptamer: str                 # 5'→3' after the 5'-SH-(CH2)6- linker
    per_cell_fg: float           # reported mean surface expression on a single MCF-7 cell
    conc_curve: LogLinearCurve   # standard curve vs antigen concentration (ng/mL)
    cell_curve: LogLinearCurve   # standard curve vs MCF-7 cell concentration (cells/mL)
    mw_kda: float                # literature (Level 1)

    @property
    def molecules_per_cell(self) -> float:
        return self.per_cell_fg * 1e-15 / (self.mw_kda * 1e3) * AVOGADRO


CEA = Marker(
    name="CEA", full_name="carcinoembryonic antigen",
    aptamer="ATACCAGCTTTATTCAATT",
    per_cell_fg=37.0,
    conc_curve=LogLinearCurve(1308.9, 3084.5, "ng/mL", 0.01, 50.0, 6.12e-3, 0.994),
    cell_curve=LogLinearCurve(2957.0, -6370.0, "cells/mL", 165.0, 1e6, 165.0, 0.998),
    mw_kda=180.0,
)

AFP = Marker(
    name="AFP", full_name="alpha-fetoprotein",
    aptamer="GGCAGGAAGACAAACAAGCTTGGCGGCGGGAAGGTGTTTAAATTCCCGGGTCTGCGTGGTCTGTGGTGCTGT",
    per_cell_fg=0.25,
    conc_curve=LogLinearCurve(1091.5, 4523.8, "ng/mL", 1e-3, 100.0, 2.5e-4, 0.998),
    cell_curve=LogLinearCurve(4473.0, -13251.0, "cells/mL", 1000.0, 1e6, 1000.0, 0.993),
    mw_kda=69.0,
)

MARKERS = {"CEA": CEA, "AFP": AFP}

# (min, max, preferred) from claims 6-7 and the description.
CLAIM_RANGES = {
    "antibody_ug_per_ml": (1.0, 40.0, 20.0),
    "probe_ug_per_ml": (25.0, 70.0, 50.0),
    "ru_bpy3_mM": (0.5, 2.0, 1.0),
    "tpa_mM": (20.0, 70.0, 50.0),
    "drive_V": (3.5, 5.0, 5.0),
}

PROBE = {
    "name": "COF@Au@MB-Apt",
    "diameter_nm": 300.0,            # TEM, example 2
    "density_g_cm3": 1.3,            # literature-scale estimate for an imine COF with ~3 wt% Au (Level 1)
    "signal_molecule": "methylene blue (MB)",
    "mb_molar_mass": 319.85,         # MB chloride, literature
    "mb_electrons": 2,               # MB + 2e⁻ + H⁺ → leuco-MB, literature
}

# Example 1, verbatim quantities (masses mg, volumes mL, concentrations as stated).
SYNTHESIS = {
    "COF": {"TAPB_mg": 17.6, "TAPB_mmol": 0.05, "TAPB_mw": 351.45, "TAPB_amines": 3,
            "DMTP_mg": 21.4, "DMTP_mmol": 0.11, "DMTP_mw": 194.18, "DMTP_aldehydes": 2,
            "acetonitrile_mL": 40.0, "acetic_acid_12M_mL": 1.0, "time_h": 4.0},
    "COF@Au": {"COF_mg": 15.0, "HAuCl4_1pct_uL": 80.0, "HAuCl4_mw": 339.79, "Au_mw": 196.97,
                "NaBH4_0.2M_mL": 0.5, "temp_C": 0.0, "time_h": 3.0},
    "COF@Au@MB": {"COF_Au_mg": 5.0, "MB_mg_per_mL": 1.0, "MB_mL": 2.0, "time_h": 12.0},
    "COF@Au@MB-Apt": {"aptamer_uL": 200.0, "aptamer_uM": 5.0, "TCEP_mM": 1.0,
                      "composite_mL": 2.0, "composite_mg_per_mL": 1.0, "temp_C": 4.0, "time_h": 10.0},
}

CELL_LINE = {"name": "MCF-7", "diameter_um": 20.0}   # diameter: literature (Level 1)
