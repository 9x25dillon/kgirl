"""Evolving parameters of the Algorithmic Life Equation (kgirl.life) for viable, adaptive regimes.

Genome (log scales unless noted): construction a, decay b, coherence pull η_C, variance drive η_V, model
rate η_M, complexity penalty λ, memory rate ρ (linear) and free-energy leak δF. Every decoded genome passes
LifeParams.check(), so the search never leaves the map's admissible region (NLMS and ridge stability).

Protocol: the experiments' World with d = 4 channels at a lean inflow Ē = 0.12 (twice the default parameters'
critical inflow), a distribution shift halfway, 500 steps, one seed.

    fitness   = mean C_eff over the run × (1 if O never falls below O_min after the shock, else ½)
    behavior  = (‖M‖_F at the end, adaptation steps after the shock)
              a compression axis and a recovery axis: the archive keeps the best organism for every trade-off
              between how much it models and how fast it recovers, the manuscript's compression experiment
              turned into a map

Level 1: it measures this model under this protocol. Which corners are "alive" says nothing beyond that.
Pure Python per evaluation (about 25 ms); numpy only through MapElites.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ...life.core import LifeParams
from ...life.experiments import World, adaptation, run
from ..mapelites import Descriptor, Evaluation, MapElites
from ..space import Param, Space

SPACE = Space([
    Param("a", 0.02, 0.5, log=True),
    Param("b", 0.01, 0.3, log=True),
    Param("eta_C", 0.02, 0.5, log=True),
    Param("eta_V", 0.01, 0.5, log=True),
    Param("eta_M", 0.01, 0.5, log=True),
    Param("lambda_reg", 1e-3, 3.0, log=True),
    Param("rho", 0.05, 0.8),
    Param("deltaF", 0.002, 0.1, log=True),
])

DESCRIPTORS = [Descriptor("model_norm", 0.0, 2.0, 10), Descriptor("adapt_steps", 0.0, 60.0, 10)]
WORLD = World(d=4, energy=0.12, shock_at=250, shock=1.0, steps=500)
MAX_ADAPT = 60.0


def evaluate_one(values: dict[str, float], world: World = WORLD, seed: float = 0.4123) -> Evaluation:
    p = replace(LifeParams(), **values)
    if p.violations():
        return Evaluation(float("nan"), (0.0, 0.0), {"violations": p.violations()})
    r = run(p, world, seed)
    ad = adaptation(r, world.shock_at, pre=100)
    steps = float(ad["adapt_steps"]) if ad["adapt_steps"] is not None else MAX_ADAPT
    survived = min(r.O[world.shock_at:]) >= p.O_min
    ceff = sum(r.ceff) / len(r.ceff)
    fitness = ceff * (1.0 if survived else 0.5)
    info = {**values, "C_eff": ceff, "survived": survived, "adapt_steps": steps, "model_norm": r.model_norm,
            "F_end": r.F[-1], "O_end": r.O[-1], "R_end": r.R[-1]}
    return Evaluation(fitness, (r.model_norm, min(steps, MAX_ADAPT)), info)


def make_evaluator(world: World = WORLD):
    def evaluate(X: np.ndarray) -> list[Evaluation]:
        return [evaluate_one(SPACE.decode(x), world) for x in X]
    return evaluate


def baseline(world: World = WORLD) -> Evaluation:
    """The published default parameters under the same protocol."""
    d = LifeParams()
    return evaluate_one({k: getattr(d, k) for k in SPACE.names}, world)


def evolve(generations: int = 20, batch: int = 32, seed: float = 0.4123, init: int = 48,
           world: World = WORLD) -> MapElites:
    me = MapElites(SPACE, make_evaluator(world), DESCRIPTORS, seed=seed, batch=batch, init=init)
    return me.run(generations)
