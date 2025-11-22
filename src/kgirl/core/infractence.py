import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import hashlib
import asyncio
import time
from enum import Enum
from scipy.stats import entropy
import logging
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# 1. FRACTAL LATTICE + INFRASONOMANCY CORE
# ================================

@dataclass
class FractalLatticeConfig:
    width: int = 64
    height: int = 64
    max_iter: int = 100
    zoom: float = 1.0
    center: Tuple[float, float] = (0.0, 0.0)
    julia_c: complex = complex(-0.4, 0.6)  # default, will be overridden by seed


class FractalLattice:
    def __init__(self, config: FractalLatticeConfig):
        self.config = config

    def _make_grid(self) -> np.ndarray:
        w, h = self.config.width, self.config.height
        zx = np.linspace(-2.0, 2.0, w) / self.config.zoom + self.config.center[0]
        zy = np.linspace(-2.0, 2.0, h) / self.config.zoom + self.config.center[1]
        X, Y = np.meshgrid(zx, zy)
        return X + 1j * Y

    def generate_julia(self) -> np.ndarray:
        """Return normalized [0,1] lattice of escape times."""
        c = self.config.julia_c
        Z = self._make_grid()
        M = np.zeros(Z.shape, dtype=int)
        mask = np.ones(Z.shape, dtype=bool)

        for i in range(self.config.max_iter):
            Z[mask] = Z[mask] * Z[mask] + c
            escaped = np.abs(Z) > 2
            newly_escaped = escaped & mask
            M[newly_escaped] = i
            mask &= ~escaped
            if not mask.any():
                break

        M = M.astype(float) / max(self.config.max_iter - 1, 1)
        return M


@dataclass
class FrequencyBands:
    infrasonic: Tuple[float, float] = (0.1, 20.0)
    bass: Tuple[float, float] = (20.0, 200.0)
    mid: Tuple[float, float] = (200.0, 2000.0)
    high: Tuple[float, float] = (2000.0, 12000.0)


class InfrasonomancyMapper:
    def __init__(self, bands: Optional[FrequencyBands] = None):
        self.bands = bands or FrequencyBands()

    @staticmethod
    def _lerp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return lo + v * (hi - lo)

    def lattice_to_freq_layers(self, lattice: np.ndarray) -> Dict[str, np.ndarray]:
        layers: Dict[str, np.ndarray] = {}
        for name, band in vars(self.bands).items():
            lo, hi = band
            layers[name] = self._lerp(lattice, lo, hi)
        return layers

    def lattice_to_midi_grid(
        self,
        lattice: np.ndarray,
        note_range: Tuple[int, int] = (24, 96),
    ) -> np.ndarray:
        lo, hi = note_range
        notes = lo + lattice * (hi - lo)
        return np.round(notes).astype(int)


@dataclass
class DigiologyPattern:
    notes: List[Tuple[int, float, float]]  # (midi_note, start_time, duration)
    infrasonic_envelope: List[Tuple[float, float]]  # (time, hz)
    control_curves: Dict[str, List[Tuple[float, float]]]  # name -> (time, value)


class FractalInfrasonomancer:
    """
    Deterministic fractal pattern generator keyed by seed_text.
    """

    def __init__(self, seed_text: str, config: Optional[FractalLatticeConfig] = None):
        self.seed_text = seed_text
        self.seed_hash = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)

        # Local RNG so we don't stomp global np.random
        self.rng = np.random.default_rng(self.seed_hash & 0xFFFFFFFF)

        julia_real = -0.8 + (self.seed_hash % 1600) / 1000.0
        julia_imag = -0.8 + ((self.seed_hash >> 12) % 1600) / 1000.0

        cfg = config or FractalLatticeConfig(
            julia_c=complex(julia_real, julia_imag),
            zoom=1.0 + ((self.seed_hash >> 24) % 300) / 100.0,
        )
        self.lattice_engine = FractalLattice(cfg)
        self.mapper = InfrasonomancyMapper()

    def _time_grid(self, length: float, steps: int) -> np.ndarray:
        return np.linspace(0.0, length, steps, endpoint=False)

    def build_pattern(
        self,
        length_seconds: float = 16.0,
        note_density: float = 0.1,
    ) -> DigiologyPattern:
        lattice = self.lattice_engine.generate_julia()
        h, w = lattice.shape

        midi_grid = self.mapper.lattice_to_midi_grid(lattice)
        t_grid = self._time_grid(length_seconds, w)

        # density threshold
        thresh = np.quantile(lattice, 1.0 - min(max(note_density, 0.0), 1.0))

        notes: List[Tuple[int, float, float]] = []
        for x in range(w):
            for y in range(h):
                if lattice[y, x] >= thresh:
                    note = int(midi_grid[y, x])
                    start = float(t_grid[x])
                    dur = float(
                        (length_seconds / w) * self.rng.uniform(0.5, 1.5)
                    )
                    notes.append((note, start, dur))

        # infrasonic envelope: mean per column
        infrasonic_layer = self.mapper.lattice_to_freq_layers(lattice)["infrasonic"]
        infrasonic_mean = infrasonic_layer.mean(axis=0)
        infrasonic_env = [
            (float(t_grid[i]), float(infrasonic_mean[i]))
            for i in range(len(t_grid))
        ]

        # coherence curve: simple variance proxy
        coherence_curve: List[Tuple[float, float]] = []
        for i, t in enumerate(t_grid):
            col = lattice[:, i]
            coherence = float(col.std())
            coherence_curve.append((float(t), coherence))

        control_curves: Dict[str, List[Tuple[float, float]]] = {
            "coherence": coherence_curve
        }

        return DigiologyPattern(
            notes=sorted(notes, key=lambda n: n[1]),
            infrasonic_envelope=infrasonic_env,
            control_curves=control_curves,
        )


# ================================
# 2. FREQUENCY TRANSLATOR LAYER
# ================================

class FrequencyTranslator:
    """
    Base translator for mapping frequencies to symbolic signatures.
    """

    def encode_frequency_signature(self, freq: float) -> str:
        # simple deterministic hex-ish tag
        scaled = int(freq * 1000)
        return f"fsig_{scaled:08x}"


class EnhancedFrequencyTranslator(FrequencyTranslator):
    """
    Wraps a FractalInfrasonomancer and exposes 'spatial radiation patterns'
    as a higher-level interface.
    """

    def __init__(self):
        super().__init__()
        self.infrasonomancer: Optional[FractalInfrasonomancer] = None

    def initialize_infrasonomancer(self, seed_text: str) -> FractalInfrasonomancer:
        self.infrasonomancer = FractalInfrasonomancer(seed_text)
        return self.infrasonomancer

    def _midi_to_hz(self, midi_note: int) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def generate_spatial_radiation_pattern(
        self,
        length_seconds: float = 16.0,
        note_density: float = 0.1,
    ) -> Dict[str, Any]:
        if self.infrasonomancer is None:
            raise ValueError(
                "Infrasonomancer not initialized. Call initialize_infrasonomancer()."
            )

        pattern = self.infrasonomancer.build_pattern(
            length_seconds=length_seconds, note_density=note_density
        )

        radiation_signatures: List[Dict[str, Any]] = []
        for note, start, dur in pattern.notes:
            freq = self._midi_to_hz(note)
            radiation_signatures.append(
                {
                    "midi": note,
                    "frequency": freq,
                    "start_time": start,
                    "duration": dur,
                    "encoded_signature": self.encode_frequency_signature(freq),
                }
            )

        return {
            "pattern": pattern,
            "radiation_signatures": radiation_signatures,
            "infrasonic_envelope": pattern.infrasonic_envelope,
            "coherence_curve": pattern.control_curves.get("coherence", []),
        }


# ================================
# 3. NEURO-SYMBIOTIC COHERENCE SYSTEM
# ================================

class BiometricStream(Enum):
    BREATH = "respiratory"
    HEART = "cardiac"
    MOVEMENT = "locomotion"
    NEURAL = "eeg"


class CoherenceState(Enum):
    DEEP_SYNC = "deep_synchrony"
    HARMONIC = "harmonic_alignment"
    ADAPTIVE = "adaptive_coherence"
    FRAGMENTED = "fragmented"
    DISSOCIATED = "dissociated"


class LearningPhase(Enum):
    ATTUNEMENT = "initial_attunement"
    RESONANCE = "resonance_building"
    SYMBIOSIS = "symbiotic_maintenance"
    TRANSCENDENCE = "transcendent_coherence"


@dataclass
class BiometricSignature:
    stream: BiometricStream
    frequency: float  # Hz
    amplitude: float
    variability: float
    phase: float  # rad, 0-2π
    complexity: float
    timestamp: float

    def coherence_with(self, other: "BiometricSignature") -> float:
        if not (self.frequency > 0 and other.frequency > 0):
            return 0.0

        phase_coh = math.cos(self.phase - other.phase)
        freq_ratio = min(self.frequency, other.frequency) / max(
            self.frequency, other.frequency
        )
        amp_ratio = min(self.amplitude, other.amplitude) / max(
            self.amplitude, other.amplitude
        )
        complexity_coh = math.exp(-abs(self.complexity - other.complexity))

        return (phase_coh + freq_ratio + amp_ratio + complexity_coh) / 4.0


@dataclass
class ConsciousnessState:
    breath: BiometricSignature
    heart: BiometricSignature
    movement: BiometricSignature
    neural: BiometricSignature
    timestamp: float = field(default_factory=time.time)

    def overall_coherence(self) -> float:
        streams = [self.breath, self.heart, self.movement, self.neural]
        scores: List[float] = []
        for i, s1 in enumerate(streams):
            for s2 in streams[i + 1 :]:
                scores.append(s1.coherence_with(s2))
        return float(np.mean(scores)) if scores else 0.0

    def get_state(self) -> CoherenceState:
        coh = self.overall_coherence()
        if coh > 0.8:
            return CoherenceState.DEEP_SYNC
        if coh > 0.6:
            return CoherenceState.HARMONIC
        if coh > 0.4:
            return CoherenceState.ADAPTIVE
        if coh > 0.2:
            return CoherenceState.FRAGMENTED
        return CoherenceState.DISSOCIATED


class NSCTS:
    """
    NeuroSymbiotic Coherence Training System.

    Uses the EnhancedFrequencyTranslator as a generative "neuro-weather"
    and folds it into a multi-stream coherence metric.
    """

    def __init__(self):
        self.translator = EnhancedFrequencyTranslator()
        self.states: List[ConsciousnessState] = []
        self.current_phase: LearningPhase = LearningPhase.ATTUNEMENT
        self.coherence_history: List[float] = []

    def initialize_infrasonomancer(self, seed_text: str):
        self.translator.initialize_infrasonomancer(seed_text)

    def generate_simulated_biometrics(
        self,
        length_seconds: float = 32.0,
        note_density: float = 0.05,
    ) -> List[BiometricSignature]:
        """
        Convert radiation signatures into four biometric streams.
        One simple mapping: round-robin assignment across streams.
        """
        pattern = self.translator.generate_spatial_radiation_pattern(
            length_seconds=length_seconds, note_density=note_density
        )

        signatures: List[BiometricSignature] = []
        streams = [
            BiometricStream.BREATH,
            BiometricStream.HEART,
            BiometricStream.MOVEMENT,
            BiometricStream.NEURAL,
        ]

        for idx, sig in enumerate(pattern["radiation_signatures"]):
            stream = streams[idx % len(streams)]
            freq = sig["frequency"]
            duration = sig["duration"]
            start = sig["start_time"]

            base_amp = 1.0 + 0.2 * math.sin(start)
            amp_jitter = random.uniform(0.8, 1.2)
            amplitude = base_amp * amp_jitter

            variability = 1.0 / (duration + 1e-6)
            phase = (start * 2 * math.pi / max(length_seconds, 1e-6)) % (2 * math.pi)

            # Simple synthetic complexity: entropy of a windowed spectrum kernel
            complexity = float(
                entropy(np.abs(np.fft.rfft(np.hanning(16))) + 1e-9)
            )

            signatures.append(
                BiometricSignature(
                    stream=stream,
                    frequency=freq,
                    amplitude=amplitude,
                    variability=variability,
                    phase=phase,
                    complexity=complexity,
                    timestamp=start,
                )
            )

        return signatures

    async def training_loop(
        self, duration_minutes: float = 5.0, phase: LearningPhase = LearningPhase.SYMBIOSIS
    ):
        self.current_phase = phase
        end_time = time.time() + duration_minutes * 60.0

        while time.time() < end_time:
            biometrics = self.generate_simulated_biometrics()

            # pick the latest per stream
            def latest(stream: BiometricStream) -> BiometricSignature:
                candidates = [b for b in biometrics if b.stream == stream]
                if not candidates:
                    # fallback neutral signature
                    return BiometricSignature(
                        stream=stream,
                        frequency=1.0,
                        amplitude=1.0,
                        variability=0.1,
                        phase=0.0,
                        complexity=1.0,
                        timestamp=time.time(),
                    )
                return max(candidates, key=lambda b: b.timestamp)

            state = ConsciousnessState(
                breath=latest(BiometricStream.BREATH),
                heart=latest(BiometricStream.HEART),
                movement=latest(BiometricStream.MOVEMENT),
                neural=latest(BiometricStream.NEURAL),
            )

            self.states.append(state)
            coh = state.overall_coherence()
            self.coherence_history.append(coh)

            logger.info(
                "Phase=%s | Coherence=%.3f | State=%s",
                phase.value,
                coh,
                state.get_state().value,
            )
            await asyncio.sleep(1.0)

        avg = float(np.mean(self.coherence_history)) if self.coherence_history else 0.0
        logger.info("Training complete. Avg coherence = %.3f", avg)


# ================================
# 4. DEMONSTRATION
# ================================

async def demonstrate_nscts():
    print("=== Enhanced Frequency Translator Demo ===")
    translator = EnhancedFrequencyTranslator()
    translator.initialize_infrasonomancer("Spatial Membrane Radiation Mapping")
    radiation_pattern = translator.generate_spatial_radiation_pattern(
        length_seconds=8.0, note_density=0.08
    )
    freqs = [sig["frequency"] for sig in radiation_pattern["radiation_signatures"]]
    print(f"Total Radiation Signatures: {len(freqs)}")
    if freqs:
        print(f"Frequency Range: {min(freqs):.2f} Hz – {max(freqs):.2f} Hz")

    print("\n=== NSCTS Training Demo ===")
    nscts = NSCTS()
    nscts.initialize_infrasonomancer("NeuroSymbiotic Coherence Seed")
    await nscts.training_loop(duration_minutes=0.02, phase=LearningPhase.RESONANCE)

    print("\nDemo complete. System ready to be wired into audio/BCI backends.")


if __name__ == "__main__":
    asyncio.run(demonstrate_nscts())
