"""kgirl.evolve — deterministic quality-diversity evolution (logistic-chaos variation, no PRNG).

    ChaosRNG     uniform/normal streams from the r = 4 logistic map (arcsine-conjugated)
    Space/Param  unit-cube genomes with bounds and log scales
    MapElites    Iso+LineDD MAP-Elites with lineage, coverage, QD-score, stepping stones
    pareto       non-dominated sorting for multi-objective readouts
    problems     domain adapters (assay: CN121933729A c-BPE-ECL design)
"""

from .chaos import ChaosRNG
from .mapelites import Descriptor, Elite, Evaluation, GenStats, MapElites
from .pareto import front, ranks
from .space import Param, Space

__all__ = ["ChaosRNG", "Descriptor", "Elite", "Evaluation", "GenStats", "MapElites", "front", "ranks",
           "Param", "Space"]
