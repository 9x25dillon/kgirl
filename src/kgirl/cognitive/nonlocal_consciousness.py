"""
nonlocal_consciousness.py

Conceptual Python translation of:

Chris H. Hardy, Ph.D.
"NONLOCAL CONSCIOUSNESS IN THE UNIVERSE:
PANPSYCHISM, PSI, & MIND OVER MATTER ALLOWED IN A
HYPERDIMENSIONAL PHYSICS"

This module models the paper's key concepts as Python classes:
- Semantic Fields Theory (SFT)
- Infinite-Spiral-Staircase Theory (ISST)
- CSR hyperdimension: Center, Syg, Rhythm
- sygons / syg energy
- psi, telepathy, bio-PK, experimenter effect
- law of syg-influence on matter systems

It is a conceptual ontology, not a physical simulation.

Evidence level (kgirl convention): L3. Every class and rule below encodes the paper's hypotheses as data
structures; nothing here is a measurement or a prediction tested against data, and psi effects are not
established by the evidence base this repo relies on. The measurable counterpart in kgirl is
`kgirl.cognitive.coupled_manifolds`: two lattices coupled only through their macroscopic voices, whose
pair-only observables (phase lock, cross-channel information, A → B bit rate) are computed, with γ = 0
controls.

kgirl changes to the original: `nonlocal_connect` now records which psi process and which fields a sygon
links (the `process` argument was accepted but unused), and the module exposes `demo()` returning its results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set

# ----------------------------------------------------------------------
# Constants from the paper / cosmology
# ----------------------------------------------------------------------

PHI = (1.0 + 5.0 ** 0.5) / 2.0          # 1.618033988749895
PLANCK_LENGTH_CM = 1.616e-33
PLANCK_TIME_S = 1e-43
INFLATION_TIME_S = 1e-36
SPEED_OF_LIGHT_M_S = 299_792_458

FRACTION_ORDINARY_MATTER = 0.05
FRACTION_DARK_MATTER = 0.26
FRACTION_DARK_ENERGY = 0.69


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------

class FocusState(Enum):
    """The two focus-states of a wave-particle in ISST."""
    CSR = auto()   # Center-Syg-Rhythm: wave / frequency domain / hyperdimensional
    QST = auto()   # Quantum-Spacetime: particle / matter / causal


class Anomaly(Enum):
    """Five nonlocal anomalies pointing to a hyperdimension."""
    ENTANGLEMENT = auto()
    PSI = auto()
    SUB_PLANCKIAN = auto()
    DARK_ENERGY = auto()
    INFLATION = auto()


class PsiProcess(Enum):
    """Psi processes discussed in the paper."""
    TELEPATHY = auto()
    REMOTE_VIEWING = auto()
    PRECOGNITION = auto()
    RETROCOGNITION = auto()
    BIO_PK = auto()
    MICRO_PK = auto()
    SYNCHRONICITY = auto()


# ----------------------------------------------------------------------
# Sygon: faster-than-light semantic-energy particle
# ----------------------------------------------------------------------

@dataclass
class Sygon:
    """
    A sygon is a sub-Planckian, faster-than-light particle of pure
    semantic energy (syg energy) in the CSR hyperdimension.
    """
    frequency: float
    semantic_intensity: float = 1.0
    faster_than_light: bool = True
    process: Optional[PsiProcess] = None
    endpoints: tuple = ()

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError("frequency must be > 0")
        if self.semantic_intensity < 0:
            raise ValueError("semantic_intensity must be >= 0")

    @property
    def syg_energy(self) -> float:
        """Syg energy is modeled as frequency times semantic intensity."""
        return self.frequency * self.semantic_intensity


# ----------------------------------------------------------------------
# Semantic Field Theory
# ----------------------------------------------------------------------

@dataclass
class SemanticConstellation:
    """A meaningful cluster inside a semantic field."""
    name: str
    meaning: str
    intensity: float = 1.0
    links: Set[str] = field(default_factory=set)


@dataclass
class SemanticField:
    """
    A semantic field (syg-field) is a self-organizing, meaningful
    organizational layer of a system. It can range from proto-consciousness
    to a full mind.
    """
    name: str
    syg_intensity: float = 1.0
    focus: FocusState = FocusState.CSR
    constellations: List[SemanticConstellation] = field(default_factory=list)
    links: Dict[str, float] = field(default_factory=dict)  # field name -> semantic proximity
    matter_system: Optional[str] = None

    def add_constellation(self, constellation: SemanticConstellation) -> None:
        self.constellations.append(constellation)

    def link_to(self, other: "SemanticField", proximity: float) -> None:
        """Create a nonlocal semantic link to another field."""
        if not 0.0 <= proximity <= 1.0:
            raise ValueError("proximity must be in [0, 1]")
        self.links[other.name] = proximity
        other.links[self.name] = proximity

    def resonance_with(self, other: "SemanticField") -> float:
        """Resonance = semantic proximity weighted by lower syg intensity."""
        proximity = self.links.get(other.name, 0.0)
        return proximity * min(self.syg_intensity, other.syg_intensity)

    def shift_focus(self, state: FocusState) -> None:
        self.focus = state

    def update_intensity(self, delta: float) -> None:
        self.syg_intensity = max(0.0, self.syg_intensity + delta)


# ----------------------------------------------------------------------
# Infinite Spiral Staircase and CSR Hyperdimension
# ----------------------------------------------------------------------

@dataclass
class InfiniteSpiralStaircase:
    """
    The ISS is a phi-based golden spiral at the origin of the universe.
    Its bows are virtual strings vibrating at discrete frequencies.
    """
    bows: List[float] = field(default_factory=list)
    cosmic: bool = True

    @classmethod
    def from_fibonacci(cls, n: int = 12, base_frequency: float = 1.0) -> "InfiniteSpiralStaircase":
        """
        Generate a finite approximation of the ISS frequency ladder
        using powers of phi, related to the Fibonacci sequence.
        """
        bows = [base_frequency * (PHI ** i) for i in range(n)]
        return cls(bows=bows, cosmic=True)

    @property
    def phi(self) -> float:
        return PHI


@dataclass
class Hyperdimension:
    """
    The CSR hyperdimension:
    - Center: hyperspace
    - Syg: consciousness / semantic energy
    - Rhythm: hypertime
    """
    name: str = "CSR"
    hyperspace: str = "Center"
    hypertime: str = "Rhythm"
    consciousness: str = "Syg"

    iss: InfiniteSpiralStaircase = field(
        default_factory=InfiniteSpiralStaircase.from_fibonacci
    )
    sygons: List[Sygon] = field(default_factory=list)
    fields: List[SemanticField] = field(default_factory=list)

    def emit_sygon(self, frequency: float, semantic_intensity: float,
                   process: Optional[PsiProcess] = None, endpoints: tuple = ()) -> Sygon:
        sygon = Sygon(frequency=frequency, semantic_intensity=semantic_intensity,
                      process=process, endpoints=endpoints)
        self.sygons.append(sygon)
        return sygon

    def register_field(self, semantic_field: SemanticField) -> None:
        self.fields.append(semantic_field)

    def nonlocal_connect(
        self,
        a: SemanticField,
        b: SemanticField,
        process: PsiProcess,
    ) -> bool:
        """
        Establish a nonlocal semantic connection between two fields.
        Mediated by faster-than-light sygons.
        """
        resonance = a.resonance_with(b)
        if resonance <= 0.0:
            return False
        self.emit_sygon(
            frequency=resonance,
            semantic_intensity=resonance,
            process=process,
            endpoints=(a.name, b.name),
        )
        return True


# ----------------------------------------------------------------------
# Systems, Minds, and Cosmos
# ----------------------------------------------------------------------

@dataclass
class System:
    """Any particle, biosystem, or mind with a semantic field."""
    name: str
    kind: str  # "particle", "cell", "biosystem", "mind", ...
    syg_field: SemanticField
    matter_complexity: float = 1.0

    def __post_init__(self) -> None:
        if self.syg_field.matter_system is None:
            self.syg_field.matter_system = self.name

    @property
    def syg_intensity(self) -> float:
        return self.syg_field.syg_intensity

    def shift_focus(self, state: FocusState) -> None:
        self.syg_field.shift_focus(state)


@dataclass
class Mind(System):
    """A self-reflexive system with free will and intentionality."""
    self_reflexive: bool = True
    free_will: bool = True


@dataclass
class Cosmos:
    """
    A two-layer cosmos:
    - CSR hyperdimension (HD)
    - Quantum-Spacetime (QST) region
    """
    hd: Hyperdimension = field(default_factory=Hyperdimension)
    qst_systems: List[System] = field(default_factory=list)

    def add_system(self, system: System) -> None:
        self.qst_systems.append(system)
        self.hd.register_field(system.syg_field)

    def law_of_syg_influence(
        self,
        high: System,
        low: System,
        organization: str,
        proximity: float,
    ) -> bool:
        """
        Law of syg-influence:
        If two systems are in strong semantic proximity, the system with
        higher syg intensity imposes its semantic organization on the
        lower-intensity system.
        """
        if proximity < 0.5:
            return False
        if high.syg_intensity <= low.syg_intensity:
            return False

        imposed = SemanticConstellation(
            name=f"imposed:{organization}",
            meaning=organization,
            intensity=high.syg_intensity - low.syg_intensity,
        )
        low.syg_field.add_constellation(imposed)
        low.syg_field.update_intensity(0.1 * (high.syg_intensity - low.syg_intensity))
        return True

    def experimenter_effect(
        self,
        experimenter: Mind,
        target: System,
        intended_focus: FocusState,
    ) -> FocusState:
        """
        The experimenter's syg-field entangles with the target system.
        If strong enough, it biases the target's focus-state.
        """
        experimenter.syg_field.link_to(target.syg_field, proximity=0.9)

        if experimenter.syg_intensity > target.syg_intensity:
            target.shift_focus(intended_focus)

        return target.syg_field.focus

    def psi_connection(
        self,
        sender: System,
        receiver: System,
        process: PsiProcess,
    ) -> bool:
        """Psi as a nonlocal semantic connection mediated by the CSR HD."""
        return self.hd.nonlocal_connect(sender.syg_field, receiver.syg_field, process)

    def bio_pk(
        self,
        healer: Mind,
        patient: System,
        healing_organization: str,
    ) -> bool:
        """Bio-PK / healing as syg-influence from a stronger mind-field."""
        return self.law_of_syg_influence(
            high=healer,
            low=patient,
            organization=healing_organization,
            proximity=0.8,
        )


# ----------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------

def demo() -> dict:
    cosmos = Cosmos()

    experimenter = Mind(
        name="Experimenter",
        kind="mind",
        syg_field=SemanticField("Experimenter", syg_intensity=10.0),
    )
    photon_beam = System(
        name="Photon beam",
        kind="particle",
        syg_field=SemanticField("Photon beam", syg_intensity=1.0),
    )

    cosmos.add_system(experimenter)
    cosmos.add_system(photon_beam)

    focus = cosmos.experimenter_effect(
        experimenter=experimenter,
        target=photon_beam,
        intended_focus=FocusState.QST,
    )
    print(f"Photon focus after experimenter effect: {focus.name}")

    alice = Mind(
        name="Alice",
        kind="mind",
        syg_field=SemanticField("Alice", syg_intensity=5.0),
    )
    bob = Mind(
        name="Bob",
        kind="mind",
        syg_field=SemanticField("Bob", syg_intensity=4.0),
    )

    cosmos.add_system(alice)
    cosmos.add_system(bob)

    alice.syg_field.link_to(bob.syg_field, proximity=0.95)

    telepathy_ok = cosmos.psi_connection(
        sender=alice,
        receiver=bob,
        process=PsiProcess.TELEPATHY,
    )
    print(f"Telepathy connection established: {telepathy_ok}")

    healed = cosmos.bio_pk(
        healer=experimenter,
        patient=bob,
        healing_organization="healthier organization",
    )
    print(f"Bio-PK influence: {healed}")
    return {"focus": focus, "telepathy": telepathy_ok, "bio_pk": healed, "sygons": list(cosmos.hd.sygons),
            "bob_intensity": bob.syg_intensity}


if __name__ == "__main__":
    demo()
