"""kgirl.life — the Algorithmic Life Equation, rebuilt as a verified dynamical system and wired into the harness.

    core          the discrete map Φ (pure), LifeParams / LifeState, Organism (chaos-driven policy sampler)
    analysis      closed-form steady states, critical free energy and inflow, Euler stability, early-warning
                  statistics and `rigidity`, the observable replacement for the "99.9% ceiling"
    continuous    the ODE counterpart with RK4, the corrected coherence barrier, entropy production
    replication   staggered replicas (exact availability, jittered simulation) and swarm sizing / staggering
    experiments   the manuscript's five axes: ablation, stress, perturbation, compression, markers
    chaos         ChaosStream, a standard-library twin of kgirl.evolve.ChaosRNG (no PRNG)

The harness uses it through `kgirl.harness.metabolism`. Standard library only, except the optional
evolve problem (kgirl.evolve.problems.life), which needs numpy like the rest of kgirl.evolve.
Sources and triage: research/life/. Guide, findings and evidence levels: docs/life/README.md.
"""

from .analysis import (EarlyWarning, Rigidity, SteadyState, critical_energy, critical_free_energy, early_warning,
                       recovery_time, rigidity, steady_state)
from .chaos import ChaosStream
from .core import ABLATIONS, LifeParams, LifeState, Organism, StepRecord, step
from .replication import availability, best_replication, best_stagger, schedule, swarm_size

__all__ = ["ABLATIONS", "ChaosStream", "EarlyWarning", "LifeParams", "LifeState", "Organism", "Rigidity",
           "SteadyState", "StepRecord", "availability", "best_replication", "best_stagger", "critical_energy",
           "critical_free_energy", "early_warning", "recovery_time", "rigidity", "schedule", "steady_state",
           "step", "swarm_size"]
