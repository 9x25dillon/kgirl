"""Evolving c-BPE-ECL assay designs inside the CN121933729A claim ranges.

Genome (all bounded by claims 6–7 or the description):

    tpa_mM       20–70     coreactant: brighter ECL, faster anode fouling (pulse.py control C)
    ru_mM        0.5–2     Ru(bpy)3²⁺: ECL yield saturates as ru / (ru + 0.5)
    probe_ug_ml  25–70     COF@Au@MB-Apt: binding saturates as c / (c + 20); nonspecific
                           adsorption raises the blank ∝ (1 + c/50)
    drive_V      3.5–5.0   ECL turns on above ~3.2 V; cell stress grows ∝ V²
    f_nihil      0–0.9     off fraction of each drive pulse (the dose of nothing)
    period_s     0.5–5     pulse period (log scale)

Phenotype, computed per design:
    pulse model     yield, reproducibility, viability       (pulse.simulate, vectorized per generation)
    chain model     predicted LOD for CEA and AFP            (amplification.chain)
    sensitivity     mean over markers of clip(log10(reported / predicted LOD), 0, 3) / 3

Fitness = yield × reproducibility × viability × (0.25 + 0.75 · sensitivity).
Behavior grid = (viability, sensitivity): the archive keeps the best design for every
trade-off between keeping cells alive and seeing fewer of them.

Every response curve above is a Level 2 modelling choice (saturations, V² stress,
threshold voltage); the claim ranges are Level 0. The point is the *search*:
which corners of the claimed space are worth a wet-lab run.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ...nihiline.amplification import ChainParams, chain
from ...nihiline.patent import CLAIM_RANGES, MARKERS
from ...nihiline.pulse import PulseParams, simulate
from ..mapelites import Descriptor, Evaluation, MapElites
from ..space import Param, Space

SPACE = Space([
    Param("tpa_mM", *CLAIM_RANGES["tpa_mM"][:2], unit="mM"),
    Param("ru_mM", *CLAIM_RANGES["ru_bpy3_mM"][:2], unit="mM"),
    Param("probe_ug_ml", *CLAIM_RANGES["probe_ug_per_ml"][:2], unit="µg/mL"),
    Param("drive_V", *CLAIM_RANGES["drive_V"][:2], unit="V"),
    Param("f_nihil", 0.0, 0.9),
    Param("period_s", 0.5, 5.0, log=True, unit="s"),
])

DESCRIPTORS = [Descriptor("viability", 0.4, 1.0, 12), Descriptor("sensitivity", 0.35, 0.65, 12)]

PATENT_DEFAULT = {"tpa_mM": 50.0, "ru_mM": 1.0, "probe_ug_ml": 50.0, "drive_V": 5.0, "f_nihil": 0.0,
                  "period_s": 2.0}

V_ON = 3.2


def _ru_term(ru):
    return (ru / (ru + 0.5)) / (1.0 / 1.5)          # = 1 at the preferred 1 mM


def _v_term(v):
    return np.clip((v - V_ON) / (5.0 - V_ON), 0.0, None)   # = 1 at 5 V


def make_evaluator(pulse: PulseParams | None = None, base: ChainParams | None = None):
    pulse = pulse or PulseParams(T_s=16.0, dt_s=0.004)
    base = base or ChainParams()

    def evaluate(X: np.ndarray) -> list[Evaluation]:
        g = SPACE.decode_batch(X)
        control = (g["tpa_mM"] - CLAIM_RANGES["tpa_mM"][0]) / (CLAIM_RANGES["tpa_mM"][1] - CLAIM_RANGES["tpa_mM"][0])
        gain = _ru_term(g["ru_mM"]) * _v_term(g["drive_V"])
        stress = (g["drive_V"] / 5.0) ** 2
        r = simulate(g["f_nihil"], control, pulse, period_s=g["period_s"], gain_scale=gain, stress_scale=stress)
        out = []
        for i in range(len(X)):
            c = g["probe_ug_ml"][i]
            p = replace(base,
                        binding_efficiency=0.8 * c / (c + 20.0),
                        ecl_photons_per_electron=base.ecl_photons_per_electron * gain[i],
                        background_counts=base.background_counts * (1.0 + c / 50.0))
            lods = {m: chain(m, p).lod_cells_per_ml for m in MARKERS}
            sens = np.mean([min(3.0, max(0.0, math.log10(MARKERS[m].cell_curve.lod / lods[m]))) / 3.0
                            if np.isfinite(lods[m]) and lods[m] > 0 else 0.0 for m in MARKERS])
            prod = float(r["product"][i])
            fitness = prod * (0.25 + 0.75 * sens)
            info = {**{k: float(v[i]) for k, v in g.items()},
                    "yield": float(r["yield_"][i]), "reproducibility": float(r["reproducibility"][i]),
                    "viability": float(r["viability"][i]), "sensitivity": float(sens),
                    "lod_CEA": float(lods["CEA"]), "lod_AFP": float(lods["AFP"]), "product": prod}
            out.append(Evaluation(fitness, (info["viability"], info["sensitivity"]), info))
        return out

    return evaluate


def baseline(evaluate=None) -> Evaluation:
    """The patent's preferred recipe under continuous drive (f_nihil = 0)."""
    evaluate = evaluate or make_evaluator()
    return evaluate(SPACE.encode(PATENT_DEFAULT)[None, :])[0]


def evolve(generations: int = 40, batch: int = 48, seed: float = 0.4123, init: int = 96) -> MapElites:
    me = MapElites(SPACE, make_evaluator(), DESCRIPTORS, seed=seed, batch=batch, init=init)
    return me.run(generations)
