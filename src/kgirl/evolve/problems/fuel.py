"""Evolving PWR fuel-assembly enrichment patterns on the NuRea_sim lattice-physics surrogate.

Genome: the U-235 enrichment of each of the 39 fuel rods, 0.7–5.0 wt% (the dataset's range).

Phenotype (kgirl.neutronics degree-2 ridge surrogate, fitted on the 24 000 simulations):
    k_inf    infinite multiplication factor       held-out RMSE ≈ 27 pcm (R² 0.9997)
    PPPF     pin power peaking factor             held-out RMSE ≈ 0.063  (R² 0.74)

Fitness   = −PPPF − 10 · max(0, trust − 3)        flatter power, inside the surrogate's support
Behaviour = (mean enrichment, k_inf)              fuel cost × reactivity

The archive therefore holds, for each (cost, reactivity) cell, the flattest-power loading pattern found.
That is the classic loading-pattern trade-off, searched as quality-diversity instead of one weighted sum.

Trust region. The training designs draw each rod independently from U(0.7, 5.0), so their mean
enrichment is concentrated: 2.85 ± ~0.2 wt%. A quadratic surrogate is only credible near the designs it
was fitted on, so `trust` = |mean − μ̄| / σ̄ (in training-mean standard deviations) is reported with every
elite and penalised beyond 3. The baseline for each cell is the best design actually simulated in the
dataset that falls in it (Level 0). Evolved designs are surrogate predictions (Level 2) until a lattice
code simulates them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mapelites import Descriptor, Evaluation, MapElites
from ..space import Param, Space

N_RODS = 39
E_LO, E_HI = 0.7, 5.0
SPACE = Space([Param(f"e{i + 1:02d}", E_LO, E_HI, unit="wt%") for i in range(N_RODS)])


@dataclass
class FuelModel:
    surrogate: object
    mean_mu: float
    mean_sd: float
    train_mean_range: tuple[float, float]
    k_range: tuple[float, float]
    data_best: dict           # (cell) → best simulated design in that cell


def _descriptors(model: FuelModel, bins: int) -> list[Descriptor]:
    return [Descriptor("mean_enrichment", *model.train_mean_range, bins), Descriptor("k_inf", *model.k_range, bins)]


def load_model(data_dir: str | None = None, bins: int = 10) -> FuelModel:
    from ...neutronics import dataset
    from ...neutronics.surrogate import Surrogate
    train = dataset.load("train", data_dir)
    s = Surrogate(degree=2).fit(train)
    means = train.enrichment.mean(axis=1)
    m = FuelModel(s, float(means.mean()), float(means.std()), (float(means.min()), float(means.max())),
                  (float(train.k_inf.min()), float(train.k_inf.max())), {})
    descs = _descriptors(m, bins)
    for i in range(train.n):                        # best real (simulated) design per behaviour cell
        cell = (descs[0].cell(means[i]), descs[1].cell(train.k_inf[i]))
        cur = m.data_best.get(cell)
        if cur is None or train.pppf[i] < cur["pppf"]:
            m.data_best[cell] = {"pppf": float(train.pppf[i]), "k_inf": float(train.k_inf[i]),
                                 "mean_enrichment": float(means[i]), "row": i}
    return m


def make_evaluator(model: FuelModel):
    def evaluate(X: np.ndarray) -> list[Evaluation]:
        e = E_LO + np.clip(np.asarray(X, dtype=np.float64), 0.0, 1.0) * (E_HI - E_LO)   # all parameters linear
        pred = model.surrogate.predict(e)
        k, pppf = pred[:, 0], pred[:, 1]
        mean = e.mean(axis=1)
        trust = np.abs(mean - model.mean_mu) / model.mean_sd
        fit = -pppf - 10.0 * np.maximum(0.0, trust - 3.0)
        return [Evaluation(float(fit[i]), (float(mean[i]), float(k[i])),
                           {"k_inf": float(k[i]), "pppf": float(pppf[i]), "mean_enrichment": float(mean[i]),
                            "max_enrichment": float(e[i].max()), "trust_sd": float(trust[i]),
                            "enrichments": [round(float(v), 4) for v in e[i]]})
                for i in range(len(X))]
    return evaluate


def evolve(generations: int = 60, batch: int = 64, seed: float = 0.4123, init: int = 128, bins: int = 10,
           model: FuelModel | None = None, data_dir: str | None = None) -> tuple[MapElites, FuelModel]:
    model = model or load_model(data_dir, bins)
    me = MapElites(SPACE, make_evaluator(model), _descriptors(model, bins), seed=seed, batch=batch, init=init)
    return me.run(generations), model


def compare_to_data(me: MapElites, model: FuelModel) -> dict:
    """Per cell holding both an elite and a simulated design: predicted elite PPPF − best simulated PPPF."""
    rows = []
    for el in me.elites():
        real = model.data_best.get(tuple(el.cell))
        if real is not None and el.info["trust_sd"] <= 3.0:
            rows.append(el.info["pppf"] - real["pppf"])
    d = np.array(rows)
    return {"cells_compared": int(d.size),
            "cells_where_elite_flatter": int((d < 0).sum()) if d.size else 0,
            "median_pppf_delta": float(np.median(d)) if d.size else None,
            "best_pppf_delta": float(d.min()) if d.size else None}
