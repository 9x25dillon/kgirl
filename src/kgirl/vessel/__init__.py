"""kgirl.vessel — identity as a measured, defended, portable invariant of a model's geometry.

    geometry     CKA (+ analytic gradient), principal angles, effective rank, heavy-tail exponent α
    matrixdb     SQLite: content-addressed matrices, typed geometries, hash-chained ledger
    substrate    the container contract + ReferenceLM (numpy, hand-written backprop, ChaosRNG init)
    signature    Grams + voice on fixed probes → coherence, fidelity, drift
    invariance   group-sampling verification of the signature's invariance (after substrate-comm)
    steer        Schmitt-triggered, trust-region, rollback-on-failure coherence controller
    persona      chaos-seeded genomes, voice corpora, soulpack export
    vessel       birth · observe · learn · steer · consolidate · migrate · recognize
    torch_substrate  any byte-level torch LM; AURIC (patern-coding) via auric_substrate()   [needs torch]

Run `python -m kgirl.vessel demo` for the whole lifecycle; docs/vessel/README.md for theory and evidence.
"""

from .matrixdb import MatrixDB
from .persona import Genome, genome_for
from .signature import Comparison, Signature, compare, measure
from .steer import SteerConfig, SteerResult
from .substrate import Objective, ReferenceLM, RefConfig, StepReport, Substrate
from .vessel import Probes, Reading, Vessel, identity_distance, recognize

__all__ = ["MatrixDB", "Genome", "genome_for", "Comparison", "Signature", "compare", "measure", "SteerConfig",
           "SteerResult", "Objective", "ReferenceLM", "RefConfig", "StepReport", "Substrate", "Probes", "Reading",
           "Vessel", "identity_distance", "recognize"]
