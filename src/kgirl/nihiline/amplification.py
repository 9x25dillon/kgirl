"""The amplification chain: femtograms of antigen on one cell → photons at the anode.

Adapted from `qincrs_criticality_bridge.py`. That script asked whether any known
amplifier could lift a 10⁻⁵⁰ bias to a measurable effect, and answered with an
honest orders-of-magnitude gap. Here the same method is aimed at the patent:
can the c-BPE-ECL chain, stage by stage, turn 37 fg (CEA) or 0.25 fg (AFP) per
MCF-7 cell into the reported LODs of 165 and 1000 cells/mL? Where the chain
falls short, the gap says which stage must be stronger than assumed.

    antigen molecules ──binding──▶ bound probes (capped by surface packing)
      ──MB per probe──▶ MB molecules ──electroactive fraction──▶ reducible MB
      ──×2 e⁻──▶ cathode electrons ══ closed-BPE charge balance ══▶ anode oxidations
      ──ECL yield──▶ photons ──collection × QE──▶ detected counts
    LOD: N cells with  S/√(S+B) = 3   →   cells/mL = N / (channel volume × capture)

Confidence: patent numbers Level 0; molecular weights, probe geometry, packing
limit Level 0–1; binding, MB retention, electroactive fraction, ECL yield,
optics, background Level 2 — they are the knobs, and `sensitivity()` ranks them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from .patent import AVOGADRO, CELL_LINE, MARKERS, PROBE, SYNTHESIS, Marker

RSA_DISC_JAMMING = 0.547  # random sequential adsorption limit for discs on a plane


@dataclass(frozen=True)
class ChainParams:
    binding_efficiency: float = 0.5         # antigens that end up holding a probe        (Level 2)
    mb_retention: float = 0.05              # fraction of Example-1 MB upper bound kept     (Level 2)
    electroactive_fraction: float = 0.01    # bound MB close enough to exchange e⁻ with ITO (Level 2)
    ecl_photons_per_electron: float = 0.02  # Ru(bpy)3²⁺/TPA anodic ECL yield               (Level 1)
    collection_efficiency: float = 0.05     # optics through PDMS/ITO                       (Level 1)
    detector_qe: float = 0.5                #                                               (Level 1)
    background_counts: float = 1.0e4        # anode blank (TPA background + dark) per read  (Level 2)
    channel_volume_ul: float = 10.0         #                                               (Level 2)
    capture_efficiency: float = 0.2         # cells in the sample that adhere to the cathode (Level 2)


@dataclass(frozen=True)
class Stage:
    name: str
    value: float
    unit: str
    gain: float      # multiplicative factor from the previous stage
    level: int       # confidence level of the assumption that sets this stage


@dataclass
class Chain:
    marker: str
    stages: list[Stage]
    counts_per_cell: float
    lod_cells_per_ml: float
    reported_lod_cells_per_ml: float
    limiting: str
    params: ChainParams

    @property
    def gap_orders(self) -> float:
        """log10(predicted LOD / reported LOD): >0 means the chain is weaker than the patent's result."""
        return math.log10(self.lod_cells_per_ml / self.reported_lod_cells_per_ml)

    def render(self) -> str:
        lines = [f"{self.marker}: amplification chain per MCF-7 cell (limiting: {self.limiting})"]
        for s in self.stages:
            lines.append(f"  {s.name:<26} {s.value:>11.3e} {s.unit:<14} ×{s.gain:>9.3g}   L{s.level}")
        verdict = (f"HEADROOM: chain beats the reported LOD by {-self.gap_orders:.1f} orders "
                   "(the patent's result survives weaker assumptions)" if self.gap_orders <= 0 else
                   f"GAP: chain is {self.gap_orders:.1f} orders of magnitude short of the reported LOD")
        lines.append(f"  predicted LOD {self.lod_cells_per_ml:.3g} cells/mL vs reported "
                     f"{self.reported_lod_cells_per_ml:.3g} cells/mL → {verdict}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"marker": self.marker, "stages": [asdict(s) for s in self.stages],
                "counts_per_cell": self.counts_per_cell, "lod_cells_per_ml": self.lod_cells_per_ml,
                "reported_lod_cells_per_ml": self.reported_lod_cells_per_ml, "gap_orders": self.gap_orders,
                "limiting": self.limiting, "params": asdict(self.params)}


def probe_mass_g() -> float:
    r_cm = PROBE["diameter_nm"] * 1e-7 / 2
    return PROBE["density_g_cm3"] * 4.0 / 3.0 * math.pi * r_cm ** 3


def mb_upper_bound_per_probe() -> float:
    """All MB from Example 1 retained: 2 mg MB on 5 mg COF@Au → MB mass fraction 2/7."""
    s = SYNTHESIS["COF@Au@MB"]
    mb_mg = s["MB_mg_per_mL"] * s["MB_mL"]
    frac = mb_mg / (mb_mg + s["COF_Au_mg"])
    return frac * probe_mass_g() / PROBE["mb_molar_mass"] * AVOGADRO


def probe_packing_cap() -> float:
    d_um = CELL_LINE["diameter_um"]
    area = math.pi * d_um ** 2                       # sphere surface 4πr² = πd²
    footprint = math.pi * (PROBE["diameter_nm"] * 1e-3 / 2) ** 2
    return RSA_DISC_JAMMING * area / footprint


def lod_signal(background: float, k: float = 3.0) -> float:
    """Smallest S with S/√(S+B) = k (Poisson counting)."""
    return (k * k + math.sqrt(k ** 4 + 4 * k * k * background)) / 2


def chain(marker: str | Marker, p: ChainParams | None = None) -> Chain:
    m = MARKERS[marker.upper()] if isinstance(marker, str) else marker
    p = p or ChainParams()
    st: list[Stage] = []

    def add(name, value, unit, level):
        prev = st[-1].value if st else value
        st.append(Stage(name, value, unit, value / prev if prev else 0.0, level))

    antigens = m.molecules_per_cell
    add("antigen molecules", antigens, "molecules", 0)
    cap = probe_packing_cap()
    wanted = antigens * p.binding_efficiency
    probes = min(wanted, cap)
    limiting = "probe packing on the membrane" if wanted > cap else "antigen count × binding"
    add("bound probes", probes, "probes", 2)
    mb = probes * mb_upper_bound_per_probe() * p.mb_retention
    add("MB on the cell", mb, "molecules", 2)
    reducible = mb * p.electroactive_fraction
    add("electroactive MB", reducible, "molecules", 2)
    electrons = reducible * PROBE["mb_electrons"]
    add("cathode electrons", electrons, "e⁻", 0)
    add("anode oxidations", electrons, "events", 0)      # closed-BPE charge balance: gain 1
    photons = electrons * p.ecl_photons_per_electron
    add("ECL photons", photons, "photons", 1)
    counts = photons * p.collection_efficiency * p.detector_qe
    add("detected counts", counts, "counts", 1)

    s_min = lod_signal(p.background_counts)
    n_cells = s_min / counts if counts > 0 else math.inf
    lod = n_cells / (p.channel_volume_ul * 1e-3 * p.capture_efficiency)
    return Chain(m.name, st, counts, lod, m.cell_curve.lod, limiting, p)


def sensitivity(marker: str, p: ChainParams | None = None, step: float = 1.1) -> list[tuple[str, float]]:
    """Elasticity d log(LOD) / d log(param) for every knob — which assumption matters most."""
    p = p or ChainParams()
    base = math.log10(chain(marker, p).lod_cells_per_ml)
    out = []
    for name, val in asdict(p).items():
        bumped = chain(marker, replace(p, **{name: val * step})).lod_cells_per_ml
        out.append((name, (math.log10(bumped) - base) / math.log10(step)))
    return sorted(out, key=lambda kv: -abs(kv[1]))


def required(marker: str, knob: str = "electroactive_fraction", p: ChainParams | None = None) -> float:
    """Value of one knob that makes the predicted LOD equal the reported LOD (others fixed)."""
    p = p or ChainParams()
    c = chain(marker, p)
    elasticity = dict(sensitivity(marker, p))[knob]
    if abs(elasticity) < 1e-9:
        return math.nan
    return getattr(p, knob) * 10 ** (-c.gap_orders / elasticity)
