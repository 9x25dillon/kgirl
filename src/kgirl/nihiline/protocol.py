"""The nine levels of the transcript, run as the patent's assay.

`Code_transcripture` climbs from Level 0 (the Seed) to Level 9 (the Stubbornness
of 1); `Core_proc` turned that climb into classes (Nothingness, Unity,
BootstrapScaffold, ParadoxEngine, ...). Here each level is one step of the
CN121933729A protocol, computed from the patent's own quantities:

    0 The Seed                      COF imine condensation — stoichiometry of Example 1
    1 Latent Totality               COF@Au — gold loading from the HAuCl4 dose
    2 Autogenesis                   COF@Au@MB — signal molecules a single probe can carry
    3 Negotiated Actualization      aptamer coupling — recognition sites per probe
    4 Recursive Self-Reference      the cell captured on the antibody cathode observes itself
                                    through the probe that binds it
    5 The Vanishing Scaffold        antibody layer + controls (Example 3): the scaffold
                                    holds the cell yet leaves no light of its own
    6 0 = 1 at Creation             closed-BPE charge balance: cathode e⁻ = anode oxidations
    7 Rebellion Against Convergence pulsed drive: the nihil window that keeps cells alive
    8 The Scar on Zero              the blank's residue: blank + 3σ, read off the patent's LOD
    9 The Stubbornness of 1         quantify; dilute until the signal sinks into the scar

`run_protocol` is deterministic (logistic-map read noise, no PRNG): the same
inputs always give the same chip read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import calibration as cal
from .amplification import ChainParams, chain, mb_upper_bound_per_probe, probe_mass_g
from .patent import AVOGADRO, CLAIM_RANGES, FARADAY, MARKERS, SYNTHESIS

NIHILINE = "nothing glows below\nevery form a stained glass held\nagainst what isn't"


@dataclass
class Level:
    n: int
    title: str
    step: str
    facts: dict
    note: str = ""


@dataclass
class ProtocolRun:
    inputs: dict
    levels: list[Level] = field(default_factory=list)
    reads: dict = field(default_factory=dict)

    def render(self) -> str:
        out = []
        for lv in self.levels:
            facts = ", ".join(f"{k}={_fmt(v)}" for k, v in lv.facts.items())
            out.append(f"L{lv.n} {lv.title} — {lv.step}\n    {facts}" + (f"\n    {lv.note}" if lv.note else ""))
        out.append("chip read:")
        for m, r in self.reads.items():
            out.append(f"  {m}: true {r['true_cells_per_ml']:.4g} cells/mL → signal {r['signal']:.1f} a.u. → "
                       f"{r['recovered_cells_per_ml']:.4g} cells/mL [{r['flag']}]"
                       + (" ⚠ viability-gated" if r["viability_flag"] else ""))
        out.append("\n" + NIHILINE)
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"inputs": self.inputs, "levels": [lv.__dict__ for lv in self.levels], "reads": self.reads}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


class _Chaos:
    def __init__(self, seed: float, r: float = 3.97):
        self.s, self.r = seed, r

    def __call__(self) -> float:
        self.s = self.r * self.s * (1 - self.s)
        return self.s - 0.5


def run_protocol(cells_per_ml: dict | None = None, tpa_mM: float = 50.0, f_nihil: float | None = None,
                 read_cv: float = 0.03, seed: float = 0.4123, chain_params: ChainParams | None = None,
                 viability_floor: float = 0.8) -> ProtocolRun:
    cells_per_ml = cells_per_ml or {"CEA": 1e4, "AFP": 1e4}
    run = ProtocolRun({"cells_per_ml": cells_per_ml, "tpa_mM": tpa_mM, "f_nihil": f_nihil, "read_cv": read_cv,
                       "seed": seed})
    L = run.levels.append
    S = SYNTHESIS

    # 0 — COF: imine condensation stoichiometry
    c = S["COF"]
    amine = c["TAPB_mg"] / c["TAPB_mw"] * c["TAPB_amines"]
    ald = c["DMTP_mg"] / c["DMTP_mw"] * c["DMTP_aldehydes"]
    L(Level(0, "The Seed", "COF (TAPB + DMTP, Schiff base)",
            {"TAPB_mmol": c["TAPB_mg"] / c["TAPB_mw"], "DMTP_mmol": c["DMTP_mg"] / c["DMTP_mw"],
             "CHO_per_NH2": ald / amine},
            "aldehyde excess caps every amine: the framework closes on itself"))

    # 1 — COF@Au
    a = S["COF@Au"]
    hau_mg = a["HAuCl4_1pct_uL"] * 1e-3 * 10.0          # 1 % w/v → 10 mg/mL
    au_mg = hau_mg * a["Au_mw"] / a["HAuCl4_mw"]
    L(Level(1, "Latent Totality", "COF@Au (in-situ NaBH4 reduction)",
            {"HAuCl4_mg": hau_mg, "Au_mg": au_mg, "Au_wt_pct": 100 * au_mg / (a["COF_mg"] + au_mg)},
            "upper bound: assumes complete reduction and anhydrous HAuCl4"))

    # 2 — COF@Au@MB
    L(Level(2, "Autogenesis", "COF@Au@MB (MB adsorption)",
            {"probe_mass_g": probe_mass_g(), "MB_per_probe_upper": mb_upper_bound_per_probe()},
            "the probe is not a warehouse but a process: retained MB is set by washing (ChainParams.mb_retention)"))

    # 3 — aptamers
    ap = S["COF@Au@MB-Apt"]
    apt_mol = ap["aptamer_uL"] * 1e-6 * ap["aptamer_uM"] * 1e-6
    probes = ap["composite_mL"] * ap["composite_mg_per_mL"] * 1e-3 / probe_mass_g()
    L(Level(3, "Negotiated Actualization", "COF@Au@MB-Apt (Au–S coupling after TCEP)",
            {"aptamer_nmol": apt_mol * 1e9, "probes_in_batch": probes,
             "aptamers_per_probe_upper": apt_mol * AVOGADRO / probes,
             "CEA_apt_nt": len(MARKERS["CEA"].aptamer), "AFP_apt_nt": len(MARKERS["AFP"].aptamer)}))

    # 4 — capture
    p = chain_params or ChainParams()
    cap = {m: n * p.channel_volume_ul * 1e-3 * p.capture_efficiency for m, n in cells_per_ml.items()}
    L(Level(4, "Recursive Self-Reference", "MCF-7 captured on the antibody cathode (20 µg/mL)",
            {f"{m}_cells_on_cathode": v for m, v in cap.items()}))

    # 5 — scaffold + controls
    L(Level(5, "The Vanishing Scaffold", "Example 3 controls",
            {"no_probe": "≈ blank", "no_Ru_TPA": "≈ blank (3.5–5.0 V)", "PBS_cathode": "≈ blank"},
            "the antibody layer is necessary and invisible: light only appears with probe AND Ru/TPA"))

    # 6 — charge balance
    chains = {m: chain(m, p) for m in cells_per_ml}
    e_cell = {m: chains[m].stages[4].value for m in cells_per_ml}
    q = {m: e_cell[m] * cap[m] / AVOGADRO * FARADAY for m in cells_per_ml}
    L(Level(6, "0 = 1 at Creation", "closed-BPE charge balance",
            {**{f"{m}_e_per_cell": e_cell[m] for m in cells_per_ml}, **{f"{m}_charge_C": q[m] for m in q}},
            "every electron MB takes at the cathode is one oxidation at the anode"))

    # 7 — pulsed drive (needs numpy; degrade gracefully)
    control = min(1.0, max(0.0, (tpa_mM - CLAIM_RANGES["tpa_mM"][0]) /
                           (CLAIM_RANGES["tpa_mM"][1] - CLAIM_RANGES["tpa_mM"][0])))
    viability = 1.0
    try:
        from .pulse import PulseParams, leverage_sweep, simulate
        pp = PulseParams(T_s=20.0, dt_s=0.004)
        if f_nihil is None:
            f_nihil = leverage_sweep(control, p=pp)["f_opt"]
        viability = float(simulate(f_nihil, control, pp)["viability"][0])
        note = "optimal nihil from the leverage sweep" if run.inputs["f_nihil"] is None else "nihil fixed by caller"
    except ImportError:
        f_nihil = f_nihil if f_nihil is not None else 0.4
        note = "numpy missing: pulse model skipped, viability assumed 1"
    L(Level(7, "Rebellion Against Convergence", "pulsed 5 V drive with a nihil window",
            {"tpa_mM": tpa_mM, "control_C": control, "f_nihil": f_nihil, "viability": viability}, note))

    # 8 — scar on zero
    scars = {m: cal.scar(MARKERS[m].cell_curve) for m in cells_per_ml}
    L(Level(8, "The Scar on Zero", "blank + 3σ implied by each reported LOD",
            {f"{m}_scar_au": v for m, v in scars.items()}))

    # 9 — read and quantify
    chaos = _Chaos(seed)
    for m, n in cells_per_ml.items():
        curve = MARKERS[m].cell_curve
        live = n * viability                      # dead cells shed surface antigen: only live cells signal
        ideal = cal.signal(curve, live) if live > 0 else 0.0
        observed = max(0.0, ideal * (1 + read_cv * 2 * chaos()))   # the log-linear curve goes negative below range; a detector does not
        qv = cal.quantify(m, observed, "cells")
        recovered = qv.value / viability if viability > 0 else math.nan
        run.reads[m] = {"true_cells_per_ml": n, "signal": observed, "live_cells_per_ml": live,
                        "recovered_cells_per_ml": recovered, "flag": qv.flag,
                        "viability_flag": viability < viability_floor}
    L(Level(9, "The Stubbornness of 1", "quantify; dilution series against the scar",
            {f"{m}_series_first_below_lod": next((d["true"] for d in cal.dilution_series(m, n, mode="cells")
                                                 if d["flag"] == "below_lod"), None)
             for m, n in cells_per_ml.items()}))
    return run
