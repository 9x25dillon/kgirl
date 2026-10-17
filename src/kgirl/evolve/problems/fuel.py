"""Evolving PWR fuel-assembly enrichment patterns on the NuRea_sim lattice-physics surrogate.

Genome: the U-235 enrichment of each of the 39 fuel rods, 0.7–5.0 wt% (the dataset's range).

Phenotype: kgirl.neutronics degree-2 ridge surrogates fitted on the 24 000 simulations (5-fold CV, docs/neutronics):
    poly       degree 2                       PPPF CV R² ≈ 0.704, k_inf ≈ 23 pcm
    maxtype    degree 2 + max-type features   PPPF CV R² ≈ 0.754, k_inf ≈ 22 pcm

Modes (`features`):
    poly       fitness = −PPPF_poly − 10·max(0, trust − 3)                         the original behaviour, unchanged
    maxtype    fitness = −PPPF_max  − 10·max(0, trust − 3) − 10·max(0, support − 3)
    ensemble   fitness = −max(PPPF_poly, PPPF_max) − the same two penalties        pessimistic: no single model's
               (default)                                                           extrapolation can carry a design
Behaviour = (mean enrichment, k_inf)   fuel cost × reactivity (k_inf is the models' mean in ensemble mode)

Two support checks, both reported with every elite:
    trust      |mean − μ̄| / σ̄ in training-mean standard deviations. The training designs draw each rod
               independently from U(0.7, 5.0), so their mean is concentrated: 2.85 ± ~0.2 wt%.
    support    max_j |z_j| over the five max-type features (max, top-3 mean, spread, smooth max, max/mean). The
               same i.i.d. draw keeps every training design's maximum at 4.0–5.0 wt% and its spread at 0.84–1.55.
               A "flat" evolved design leaves that region even when its mean is ordinary. Measured on 300
               generations: 45 of 59 poly-mode elites and 51 of 61 maxtype elites without the penalty fell outside
               the training range of some feature, and a maxtype champion found without it disagreed with the poly
               model by 0.135 PPPF. That is extrapolation, and the trust check alone cannot see it.

The baseline for each cell is the best design actually simulated in the dataset that falls in it (Level 0).
Evolved designs are surrogate predictions (Level 2) until a lattice code simulates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mapelites import Descriptor, Evaluation, MapElites
from ..space import Param, Space

N_RODS = 39
E_LO, E_HI = 0.7, 5.0
SPACE = Space([Param(f"e{i + 1:02d}", E_LO, E_HI, unit="wt%") for i in range(N_RODS)])
MODES = ("ensemble", "maxtype", "poly")
SUPPORT_SD = 3.0


@dataclass
class FuelModel:
    surrogate: object                                # the mode's primary model (maxtype, or poly in poly mode)
    mean_mu: float
    mean_sd: float
    train_mean_range: tuple[float, float]
    k_range: tuple[float, float]
    data_best: dict                                  # (cell) → best simulated design in that cell
    mode: str = "poly"
    surrogates: dict = field(default_factory=dict)   # name → Surrogate used for the PPPF score
    feat_mu: np.ndarray | None = None                # max-type feature statistics of the training designs
    feat_sd: np.ndarray | None = None

    def support(self, e: np.ndarray) -> np.ndarray:
        from ...neutronics.surrogate import max_type_features
        return np.abs((max_type_features(e) - self.feat_mu) / self.feat_sd).max(axis=1)


def _descriptors(model: FuelModel, bins: int) -> list[Descriptor]:
    return [Descriptor("mean_enrichment", *model.train_mean_range, bins), Descriptor("k_inf", *model.k_range, bins)]


def load_model(data_dir: str | None = None, bins: int = 10, features: str = "ensemble") -> FuelModel:
    from ...neutronics import dataset
    from ...neutronics.surrogate import Surrogate, max_type_features
    if features not in MODES:
        raise ValueError(f"features must be one of {MODES}, got {features!r}")
    train = dataset.load("train", data_dir)
    names = {"poly": ("poly",), "maxtype": ("maxtype",), "ensemble": ("poly", "maxtype")}[features]
    surrogates = {n: Surrogate(degree=2, features=n).fit(train) for n in names}
    means = train.enrichment.mean(axis=1)
    f = max_type_features(train.enrichment)
    sd = f.std(axis=0)
    sd[sd == 0] = 1.0
    primary = surrogates["poly" if features == "poly" else "maxtype"]
    m = FuelModel(primary, float(means.mean()), float(means.std()), (float(means.min()), float(means.max())),
                  (float(train.k_inf.min()), float(train.k_inf.max())), {}, features, surrogates, f.mean(axis=0), sd)
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
        preds = {n: s.predict(e) for n, s in model.surrogates.items()}
        k = np.mean([p[:, 0] for p in preds.values()], axis=0)
        pppf = np.max([p[:, 1] for p in preds.values()], axis=0)                      # pessimistic over models
        spread = np.ptp([p[:, 1] for p in preds.values()], axis=0) if len(preds) > 1 else np.zeros(len(e))
        mean = e.mean(axis=1)
        trust = np.abs(mean - model.mean_mu) / model.mean_sd
        support = model.support(e)
        fit = -pppf - 10.0 * np.maximum(0.0, trust - 3.0)
        if model.mode != "poly":                                                       # poly: the original fitness
            fit = fit - 10.0 * np.maximum(0.0, support - SUPPORT_SD)
        return [Evaluation(float(fit[i]), (float(mean[i]), float(k[i])),
                           {"k_inf": float(k[i]), "pppf": float(pppf[i]), "mean_enrichment": float(mean[i]),
                            "max_enrichment": float(e[i].max()), "trust_sd": float(trust[i]),
                            "support_sd": float(support[i]), "pppf_spread": float(spread[i]),
                            "pppf_by_model": {n: float(p[i, 1]) for n, p in preds.items()},
                            "enrichments": [round(float(v), 4) for v in e[i]]})
                for i in range(len(X))]
    return evaluate


def evolve(generations: int = 60, batch: int = 64, seed: float = 0.4123, init: int = 128, bins: int = 10,
           model: FuelModel | None = None, data_dir: str | None = None,
           features: str = "ensemble") -> tuple[MapElites, FuelModel]:
    model = model or load_model(data_dir, bins, features)
    me = MapElites(SPACE, make_evaluator(model), _descriptors(model, bins), seed=seed, batch=batch, init=init)
    return me.run(generations), model


def compare_to_data(me: MapElites, model: FuelModel) -> dict:
    """Per cell holding both an elite and a simulated design: predicted elite PPPF − best simulated PPPF.

    `cells_*` keep the original rule (trust ≤ 3). `*_in_support` also require support ≤ 3 SD: only those elites
    resemble the designs the surrogate was fitted on.
    """
    def summary(rows):
        d = np.array(rows)
        return d.size, int((d < 0).sum()) if d.size else 0, float(np.median(d)) if d.size else None, \
            float(d.min()) if d.size else None

    rows, rows_in = [], []
    for el in me.elites():
        real = model.data_best.get(tuple(el.cell))
        if real is not None and el.info["trust_sd"] <= 3.0:
            rows.append(el.info["pppf"] - real["pppf"])
            if el.info["support_sd"] <= SUPPORT_SD:
                rows_in.append(el.info["pppf"] - real["pppf"])
    n, flatter, med, best = summary(rows)
    n_in, flatter_in, med_in, _ = summary(rows_in)
    return {"cells_compared": n, "cells_where_elite_flatter": flatter, "median_pppf_delta": med,
            "best_pppf_delta": best, "cells_compared_in_support": n_in,
            "cells_where_elite_flatter_in_support": flatter_in, "median_pppf_delta_in_support": med_in}
