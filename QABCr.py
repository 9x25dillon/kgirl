"""
Adaptive Bi‑Coupled Coherence Recovery (ABCR) — Regenerated v2

What’s new in v2 (integrated + implied fixes)
- Percentile‑based significant‑position detection with absolute floor (noise‑hardening)
- Mode‑aware significance percentiles (tunable per SystemMode)
- Safer math (no SciPy dependency; custom sigmoid)
- Guarded audits (no div/empty issues), cleaner logging
- End‑to‑end demo + PNG/JSON export
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt

# ================================ LOGGING ================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ABCR")

# ================================ ENUMS ================================
class FrequencyBand(Enum):
    DELTA = 'delta'
    THETA = 'theta'
    ALPHA = 'alpha'
    BETA = 'beta'
    GAMMA = 'gamma'

class StreamType(Enum):
    STREAM_A = "Stream A: Hypo-coherence"
    STREAM_B = "Stream B: Hyper-coherence"

class SeamType(Enum):
    TYPE_I = "Type I: Perfect Recovery"
    TYPE_II = "Type II: Acceptable Loss"
    TYPE_III = "Type III: Failed Recovery"

class SystemMode(Enum):
    STANDARD = "standard"
    HIGH_SENSITIVITY = "high_sensitivity"
    STABILITY = "stability"
    RECOVERY = "recovery"
    ADAPTIVE = "adaptive"

class ChainState(Enum):
    HYPO = "hypo-coherent"
    HYPER = "hyper-coherent"
    INTACT = "intact"

# ================================ DATACLASSES ================================
@dataclass
class SpatialPosition:
    x: float
    y: float
    m: int
    n: int

    def distance_to(self, other: 'SpatialPosition') -> float:
        return np.hypot(self.x - other.x, self.y - other.y)

    def radius(self) -> float:
        return np.hypot(self.x, self.y)

@dataclass
class ChainComponent:
    band: FrequencyBand
    positions: List[SpatialPosition]
    coherence: float
    phase_std: float
    state: ChainState
    stream: StreamType

@dataclass
class DualAuditResult:
    delta_kappa_A: float
    s_A: float
    delta_kappa_B: float
    s_B: float
    s_composite: float
    tau_R: float
    D_C: float
    D_omega: float
    R: float
    I: float
    seam_type: SeamType
    audit_pass: bool
    active_streams: List[StreamType]
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdaptiveThresholds:
    tau_low: float
    tau_high: float
    tau_phase: float
    alpha: float
    stress: float
    mode: SystemMode

# ================================ CONFIG ================================
class ABCRConfig:
    SPATIAL_GRID_M = 8
    SPATIAL_GRID_N = 8
    SPATIAL_UNIT = 0.1
    PROPAGATION_SPEED = 1.0

    # Base coherence gates
    TAU_BASE = 0.3
    TAU_PHASE = 0.5

    # Mode params
    MODE_PARAMS = {
        SystemMode.STANDARD:      {'alpha_base': 0.60, 'alpha_mod': 0.10, 'rho': 0.70, 'novelty': 0.10, 'baseline': 0.60},
        SystemMode.HIGH_SENSITIVITY:{'alpha_base': 0.65, 'alpha_mod': 0.15, 'rho': 0.60, 'novelty': 0.12, 'baseline': 0.65, 'tau_low_factor': 0.8, 'tau_high_factor': 1.2},
        SystemMode.STABILITY:     {'alpha_base': 0.50, 'alpha_mod': 0.05, 'rho': 0.80, 'novelty': 0.05, 'baseline': 0.50},
        SystemMode.RECOVERY:      {'alpha_base': 0.65, 'alpha_mod': 0.15, 'rho': 0.60, 'novelty': 0.15, 'baseline': 0.70},
        SystemMode.ADAPTIVE:      {'alpha_base': 0.60, 'alpha_mod': 0.12, 'rho': 0.65, 'novelty': 0.12, 'baseline': 0.60},
    }

    # Significance percentile per mode (for amplitude map) + absolute floor
    MODE_PERCENTILES = {
        SystemMode.STANDARD: 92.0,
        SystemMode.HIGH_SENSITIVITY: 85.0,
        SystemMode.STABILITY: 96.0,
        SystemMode.RECOVERY: 90.0,
        SystemMode.ADAPTIVE: 92.0,
    }
    ABS_NOISE_FLOOR = 1e-3

    # Coupling
    LAMBDA_CROSS_STREAM = 0.3

    # Audit
    AUDIT_TOLERANCE = 0.01
    TYPE_I_THRESHOLD = 1e-6

    # Emergency
    EMERGENCY_HYPO_THRESHOLD = 0.10
    EMERGENCY_HYPER_THRESHOLD = 0.90

    # Reconstruction
    MAX_RECONSTRUCTION_ITERATIONS = 100
    CONVERGENCE_TOLERANCE = 1e-3

    # Frequencies
    BAND_FREQUENCIES = {
        FrequencyBand.DELTA: 2.0,
        FrequencyBand.THETA: 6.0,
        FrequencyBand.ALPHA: 10.0,
        FrequencyBand.BETA: 20.0,
        FrequencyBand.GAMMA: 40.0,
    }

# ================================ UTILS ================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically safer sigmoid
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

# ================================ ENCODER ================================
class DualStreamEncoder:
    def __init__(self, M: int = ABCRConfig.SPATIAL_GRID_M, N: int = ABCRConfig.SPATIAL_GRID_N):
        self.M = M
        self.N = N
        self.positions = self._init_positions()
        self._spatial_cache: Dict[Any, float] = {}

    def _init_positions(self) -> List[SpatialPosition]:
        pos = []
        for m in range(-self.M, self.M + 1):
            for n in range(-self.N, self.N + 1):
                pos.append(SpatialPosition(m * ABCRConfig.SPATIAL_UNIT, n * ABCRConfig.SPATIAL_UNIT, m, n))
        return pos

    def _k(self, band: FrequencyBand) -> float:
        return 2 * np.pi * ABCRConfig.BAND_FREQUENCIES[band] / ABCRConfig.PROPAGATION_SPEED

    def encode_forward(self, kappa: Dict[FrequencyBand, float], phi: Dict[FrequencyBand, float]) -> np.ndarray:
        B = len(FrequencyBand)
        C = np.zeros((2*self.M+1, 2*self.N+1, B), dtype=complex)
        for p in self.positions:
            r = p.radius()
            G = np.exp(-r / (self.M * ABCRConfig.SPATIAL_UNIT))
            for b_idx, band in enumerate(FrequencyBand):
                total_phase = phi[band] - self._k(band) * r
                C[p.m + self.M, p.n + self.N, b_idx] = G * kappa[band] * np.exp(1j * total_phase)
        return C

    def encode_mirror(self, kappa: Dict[FrequencyBand, float], phi: Dict[FrequencyBand, float]) -> np.ndarray:
        B = len(FrequencyBand)
        C = np.zeros((2*self.M+1, 2*self.N+1, B), dtype=complex)
        for p in self.positions:
            r = p.radius()
            G = np.exp(-r / (self.M * ABCRConfig.SPATIAL_UNIT))
            for b_idx, band in enumerate(FrequencyBand):
                total_phase = (np.pi - phi[band]) + self._k(band) * r
                C[p.m + self.M, p.n + self.N, b_idx] = G * (1.0 - kappa[band]) * np.exp(1j * total_phase)
        return C

    def spatial_coupling(self, p1: SpatialPosition, p2: SpatialPosition, b1: FrequencyBand, b2: FrequencyBand) -> float:
        key = (p1.m, p1.n, p2.m, p2.n, b1.value, b2.value)
        if key in self._spatial_cache:
            return self._spatial_cache[key]
        d = p1.distance_to(p2)
        spatial = np.exp(-d / ABCRConfig.SPATIAL_UNIT)
        fdiff = abs(ABCRConfig.BAND_FREQUENCIES[b1] - ABCRConfig.BAND_FREQUENCIES[b2])
        freq_fac = np.exp(-fdiff / 10.0)
        val = float(spatial * freq_fac)
        self._spatial_cache[key] = val
        return val

# ================================ THRESHOLDS ================================
class AdaptiveThresholdManager:
    def compute_stress(self, kappa: Dict[FrequencyBand, float], history: List[Dict[FrequencyBand, float]]) -> float:
        if history:
            prev = history[-1]
            num = sum(abs(kappa[b] - prev[b]) for b in FrequencyBand)
            den = sum(kappa[b] for b in FrequencyBand) + 1e-9
            s = min(1.0, num / den)
        else:
            bal = 0.5
            s = np.mean([abs(k - bal) for k in kappa.values()]) * 2.0
        return float(np.clip(s, 0, 1))

    def compute(self, stress: float, mode: SystemMode) -> AdaptiveThresholds:
        p = ABCRConfig.MODE_PARAMS[mode]
        tau_low = ABCRConfig.TAU_BASE * (1 - 0.3 * stress)
        tau_high = 1 - ABCRConfig.TAU_BASE * (1 - 0.3 * stress)
        if mode == SystemMode.HIGH_SENSITIVITY:
            tau_low *= p.get('tau_low_factor', 1.0)
            tau_high *= p.get('tau_high_factor', 1.0)
        elif mode == SystemMode.STABILITY:
            tau_low *= 1.1
            tau_high *= 0.9
        alpha = np.clip(p['alpha_base'] + p['alpha_mod'] * stress, 0.3, 0.8)
        tau_phase = ABCRConfig.TAU_PHASE * (1 + 0.1 * stress)
        return AdaptiveThresholds(
            tau_low=float(np.clip(tau_low, 0.1, 0.5)),
            tau_high=float(np.clip(tau_high, 0.5, 0.9)),
            tau_phase=float(tau_phase),
            alpha=float(alpha),
            stress=float(stress),
            mode=mode,
        )

# ================================ PROCESSOR ================================
class DualStreamProcessor:
    def __init__(self, encoder: DualStreamEncoder, mode: SystemMode):
        self.encoder = encoder
        self.mode = mode

    def _phase_coherence(self, C: np.ndarray, b_idx: int) -> float:
        band_slice = C[:, :, b_idx]
        mask = np.abs(band_slice) > 1e-9
        if not np.any(mask):
            return 0.0
        phases = np.angle(band_slice[mask])
        mean_vec = np.mean(np.exp(1j * phases))
        return float(np.abs(mean_vec))

    def _significant_positions(self, C: np.ndarray, b_idx: int) -> List[SpatialPosition]:
        # --- Corrected logic: percentile + absolute floor, mode-aware ---
        amps = np.abs(C[:, :, b_idx]).ravel()
        perc = ABCRConfig.MODE_PERCENTILES.get(self.mode, 92.0)
        # Avoid empty or all-zeros
        if amps.size == 0:
            thr = ABCRConfig.ABS_NOISE_FLOOR
        else:
            thr = max(np.percentile(amps, perc), ABCRConfig.ABS_NOISE_FLOOR)
        positions = []
        for p in self.encoder.positions:
            a = np.abs(C[p.m + self.encoder.M, p.n + self.encoder.N, b_idx])
            if a >= thr:
                positions.append(p)
        return positions

    def detect_broken(self, kappa: Dict[FrequencyBand, float], C_F: np.ndarray, C_M: np.ndarray, thr: AdaptiveThresholds
                      ) -> Tuple[List[ChainComponent], List[ChainComponent], Dict[FrequencyBand, float]]:
        broken_A: List[ChainComponent] = []
        broken_B: List[ChainComponent] = []
        intact: Dict[FrequencyBand, float] = {}
        for b_idx, band in enumerate(FrequencyBand):
            kb = kappa[band]
            if kb < thr.tau_low:  # hypo side
                ph = self._phase_coherence(C_F, b_idx)
                if ph < thr.tau_phase:
                    comp = ChainComponent(
                        band=band,
                        positions=self._significant_positions(C_F, b_idx),
                        coherence=kb,
                        phase_std=float(np.sqrt(max(0.0, 1 - ph))),
                        state=ChainState.HYPO,
                        stream=StreamType.STREAM_A,
                    )
                    broken_A.append(comp)
                else:
                    intact[band] = kb
            elif kb > thr.tau_high:  # hyper side
                ph = self._phase_coherence(C_M, b_idx)
                if ph < thr.tau_phase:
                    comp = ChainComponent(
                        band=band,
                        positions=self._significant_positions(C_M, b_idx),
                        coherence=kb,
                        phase_std=float(np.sqrt(max(0.0, 1 - ph))),
                        state=ChainState.HYPER,
                        stream=StreamType.STREAM_B,
                    )
                    broken_B.append(comp)
                else:
                    intact[band] = kb
            else:
                intact[band] = kb
        return broken_A, broken_B, intact

# ================================ RECONSTRUCTOR ================================
class BiCoupledReconstructor:
    def __init__(self, encoder: DualStreamEncoder):
        self.encoder = encoder
        self.H_A: Dict[FrequencyBand, complex] = {}
        self.H_B: Dict[FrequencyBand, complex] = {}

    def compute_hamiltonians(self, broken_A: List[ChainComponent], broken_B: List[ChainComponent],
                              intact: Dict[FrequencyBand, float], C_F: np.ndarray, C_M: np.ndarray) -> None:
        # Stream A (hypo)
        for comp in broken_A:
            b = comp.band
            b_idx = list(FrequencyBand).index(b)
            hF = 0+0j
            hM = 0+0j
            for p in comp.positions:
                hF += C_F[p.m + self.encoder.M, p.n + self.encoder.N, b_idx]
                hM += C_M[p.m + self.encoder.M, p.n + self.encoder.N, b_idx]
            J = 0.0
            for intact_band, val in intact.items():
                for p1 in comp.positions:
                    for p2 in self.encoder.positions:
                        J += self.encoder.spatial_coupling(p1, p2, b, intact_band) * val
            self.H_A[b] = hF + ABCRConfig.LAMBDA_CROSS_STREAM * hM + J
        # Stream B (hyper)
        for comp in broken_B:
            b = comp.band
            b_idx = list(FrequencyBand).index(b)
            hM = 0+0j
            hF = 0+0j
            for p in comp.positions:
                hM += C_M[p.m + self.encoder.M, p.n + self.encoder.N, b_idx]
                hF += C_F[p.m + self.encoder.M, p.n + self.encoder.N, b_idx]
            J = 0.0
            for intact_band, val in intact.items():
                for p1 in comp.positions:
                    for p2 in self.encoder.positions:
                        J += self.encoder.spatial_coupling(p1, p2, b, intact_band) * (1 - val)
            self.H_B[b] = hM + ABCRConfig.LAMBDA_CROSS_STREAM * hF + J

    def reconstruct(self, broken_A: List[ChainComponent], broken_B: List[ChainComponent],
                    intact: Dict[FrequencyBand, float]) -> Dict[FrequencyBand, float]:
        kappa = intact.copy()
        for comp in broken_A:
            kappa[comp.band] = 0.30
        for comp in broken_B:
            kappa[comp.band] = 0.70
        for it in range(ABCRConfig.MAX_RECONSTRUCTION_ITERATIONS):
            conv = True
            for comp in broken_A:
                b = comp.band
                field = np.abs(self.H_A.get(b, 0.0))
                new = float(sigmoid(field))
                if abs(new - kappa[b]) > ABCRConfig.CONVERGENCE_TOLERANCE:
                    conv = False
                kappa[b] = new
            for comp in broken_B:
                b = comp.band
                field = np.abs(self.H_B.get(b, 0.0))
                new = float(1.0 - sigmoid(field))
                if abs(new - kappa[b]) > ABCRConfig.CONVERGENCE_TOLERANCE:
                    conv = False
                kappa[b] = new
            if conv:
                logger.info(f"Reconstruction converged in {it+1} iterations")
                break
        return kappa

# ================================ AUDITOR ================================
class DualStreamAuditor:
    def _stream_delta(self, orig: Dict[FrequencyBand, float], rec: Dict[FrequencyBand, float], comps: List[ChainComponent]) -> float:
        if not comps:
            return 0.0
        vals = [rec[c.band] - orig[c.band] for c in comps]
        return float(np.mean(vals)) if vals else 0.0

    def _curvature_change(self, orig: Dict[FrequencyBand, float], rec: Dict[FrequencyBand, float]) -> float:
        o = np.array(list(orig.values()))
        r = np.array(list(rec.values()))
        if o.size >= 3:
            oc = np.mean(np.abs(np.diff(o, n=2)))
            rc = np.mean(np.abs(np.diff(r, n=2)))
            return float(abs(rc - oc))
        return 0.0

    def _entropy_drift(self, orig: Dict[FrequencyBand, float], rec: Dict[FrequencyBand, float]) -> float:
        e = np.array([rec[b] - orig[b] for b in FrequencyBand])
        return float(np.std(e))

    def _return_credit(self, orig: Dict[FrequencyBand, float], rec: Dict[FrequencyBand, float]) -> float:
        ratios = []
        for b in FrequencyBand:
            if orig[b] > 0:
                r = np.clip(rec[b] / (orig[b] + 1e-12), 0, 2)
                ratios.append(1 - abs(1 - r))
        return float(np.mean(ratios)) if ratios else 0.0

    def audit(self, kappa_orig: Dict[FrequencyBand, float], kappa_rec: Dict[FrequencyBand, float],
              broken_A: List[ChainComponent], broken_B: List[ChainComponent],
              t0: float, t1: float) -> DualAuditResult:
        dkA = self._stream_delta(kappa_orig, kappa_rec, broken_A)
        dkB = self._stream_delta(kappa_orig, kappa_rec, broken_B)
        tau_R = abs(t1 - t0)
        D_C = self._curvature_change(kappa_orig, kappa_rec)
        D_w = self._entropy_drift(kappa_orig, kappa_rec)
        R = self._return_credit(kappa_orig, kappa_rec)
        s_A = R * tau_R - (dkA + D_w + D_C) if broken_A else 0.0
        s_B = R * tau_R - (dkB + D_w + D_C) if broken_B else 0.0
        active = []
        if broken_A: active.append(StreamType.STREAM_A)
        if broken_B: active.append(StreamType.STREAM_B)
        if broken_A and broken_B:
            wA = len(broken_A) / (len(broken_A) + len(broken_B))
            wB = 1 - wA
            s_comp = wA * s_A + wB * s_B
        elif broken_A:
            s_comp = s_A
        elif broken_B:
            s_comp = s_B
        else:
            s_comp = 0.0
        dk_avg = float(np.mean([kappa_rec[b] - kappa_orig[b] for b in FrequencyBand]))
        if abs(s_comp) < ABCRConfig.AUDIT_TOLERANCE:
            seam = SeamType.TYPE_I if abs(dk_avg) < ABCRConfig.TYPE_I_THRESHOLD else SeamType.TYPE_II
            ok = True
        else:
            seam = SeamType.TYPE_III
            ok = False
        I = float(np.exp(np.mean(list(kappa_rec.values()))))
        return DualAuditResult(
            delta_kappa_A=dkA, s_A=s_A, delta_kappa_B=dkB, s_B=s_B,
            s_composite=s_comp, tau_R=tau_R, D_C=D_C, D_omega=D_w, R=R, I=I,
            seam_type=seam, audit_pass=ok, active_streams=active,
            details={'delta_kappa_avg': dk_avg, 'broken_A_count': len(broken_A), 'broken_B_count': len(broken_B)}
        )

# ================================ RENEWAL ================================
class AdaptiveRenewalEngine:
    def __init__(self):
        self.Pi: Optional[Dict[FrequencyBand, float]] = None
        self.renewal_history: List[Dict[str, Any]] = []

    def init_field(self, kappa0: Dict[FrequencyBand, float]):
        self.Pi = kappa0.copy()
        logger.info(f"Invariant field initialized (mean κ={np.mean(list(self.Pi.values())):.3f})")

    def update_field(self, kappa: Dict[FrequencyBand, float], beta: float = 0.1):
        if self.Pi is None:
            self.init_field(kappa)
            return
        for b in FrequencyBand:
            self.Pi[b] = (1 - beta) * self.Pi[b] + beta * kappa[b]

    def renew(self, kappa_frag: Dict[FrequencyBand, float], mode: SystemMode) -> Dict[FrequencyBand, float]:
        if self.Pi is None:
            return kappa_frag
        p = ABCRConfig.MODE_PARAMS[mode]
        rho, novelty, base = p['rho'], p['novelty'], p['baseline']
        out: Dict[FrequencyBand, float] = {}
        for b in FrequencyBand:
            xi = float(np.random.normal(0.0, novelty))
            out[b] = float(np.clip(rho * self.Pi[b] + (1 - rho) * base + xi, 0.0, 1.0))
        self.renewal_history.append({'timestamp': datetime.now().isoformat(), 'mode': mode.value, 'kappa_after': out.copy()})
        return out

# ================================ SYSTEM ================================
class AdaptiveBiCoupledCoherenceSystem:
    def __init__(self, mode: SystemMode = SystemMode.STANDARD):
        self.mode = mode
        self.encoder = DualStreamEncoder()
        self.thr_mgr = AdaptiveThresholdManager()
        self.processor = DualStreamProcessor(self.encoder, mode)
        self.recon = BiCoupledReconstructor(self.encoder)
        self.audit = DualStreamAuditor()
        self.renew = AdaptiveRenewalEngine()
        self.capsules: Dict[str, Optional[np.ndarray]] = {'forward': None, 'mirror': None}
        self.kappa_history: List[Dict[FrequencyBand, float]] = []
        self.system_history: List[Dict[str, Any]] = []
        logger.info(f"ABCR initialized in mode={mode.value}")

    def set_mode(self, mode: SystemMode):
        self.mode = mode
        self.processor.mode = mode
        logger.info(f"Mode changed to {mode.value}")

    def process(self, kappa: Dict[FrequencyBand, float], phi: Dict[FrequencyBand, float], t: float) -> Optional[Dict[FrequencyBand, float]]:
        # thresholds
        stress = self.thr_mgr.compute_stress(kappa, self.kappa_history)
        thr = self.thr_mgr.compute(stress, self.mode)
        # encode
        C_F = self.encoder.encode_forward(kappa, phi)
        C_M = self.encoder.encode_mirror(kappa, phi)
        self.capsules = {'forward': C_F, 'mirror': C_M}
        # detect
        broken_A, broken_B, intact = self.processor.detect_broken(kappa, C_F, C_M, thr)
        # emergencies
        mn, mx = min(kappa.values()), max(kappa.values())
        if mn < ABCRConfig.EMERGENCY_HYPO_THRESHOLD or mx > ABCRConfig.EMERGENCY_HYPER_THRESHOLD or (len(broken_A) + len(broken_B) >= len(FrequencyBand)):
            logger.critical("EMERGENCY DECOUPLE")
            return None
        if not broken_A and not broken_B:
            self.kappa_history.append(kappa.copy())
            return kappa
        # reconstruct
        self.recon.compute_hamiltonians(broken_A, broken_B, intact, C_F, C_M)
        rec = self.recon.reconstruct(broken_A, broken_B, intact)
        # audit
        t0 = self.kappa_history[-1]['_t'] if (self.kappa_history and '_t' in self.kappa_history[-1]) else t
        ar = self.audit.audit(kappa, rec, broken_A, broken_B, t0, t)
        if ar.audit_pass:
            final = self.renew.renew(rec, self.mode)
            self.renew.update_field(final)
            self._record('successful_recovery', t, kappa, final, ar)
            k_with_t = final.copy(); k_with_t['_t'] = t  # store time in history entry
            self.kappa_history.append(k_with_t)
            return final
        else:
            self._record('failed_recovery', t, kappa, rec, ar)
            # fallback
            if self.renew.Pi is not None:
                fb = self.renew.Pi.copy()
                k_with_t = fb.copy(); k_with_t['_t'] = t
                self.kappa_history.append(k_with_t)
                return fb
            fb = {b: 0.5 for b in FrequencyBand}
            k_with_t = fb.copy(); k_with_t['_t'] = t
            self.kappa_history.append(k_with_t)
            return fb

    def _record(self, event: str, t: float, kb: Dict[FrequencyBand, float], ka: Dict[FrequencyBand, float], audit: DualAuditResult):
        self.system_history.append({
            'timestamp': t,
            'event': event,
            'kappa_before': {b.value: float(kb[b]) for b in FrequencyBand},
            'kappa_after': {b.value: float(ka[b]) for b in FrequencyBand},
            'audit': {
                'seam_type': audit.seam_type.value,
                's_composite': audit.s_composite,
                's_A': audit.s_A,
                's_B': audit.s_B,
                'active_streams': [s.value for s in audit.active_streams],
            }
        })

    # --------------- Simulation & Viz ---------------
    def simulate(self, duration: float = 10.0, dt: float = 0.1, scenario: str = 'dual_stress') -> List[Dict[str, Any]]:
        steps = int(duration / dt)
        hist: List[Dict[str, Any]] = []
        kappa = {FrequencyBand.DELTA: 0.72, FrequencyBand.THETA: 0.68, FrequencyBand.ALPHA: 0.75, FrequencyBand.BETA: 0.70, FrequencyBand.GAMMA: 0.65}
        phi = {FrequencyBand.DELTA: 0.1, FrequencyBand.THETA: 0.3, FrequencyBand.ALPHA: 0.5, FrequencyBand.BETA: 0.7, FrequencyBand.GAMMA: 0.9}
        if self.renew.Pi is None:
            self.renew.init_field(kappa)
        for i in range(steps):
            t = i * dt
            # scenario dynamics
            if scenario == 'dual_stress':
                if 2.0 <= t <= 4.0:
                    kappa[FrequencyBand.DELTA] = 0.15
                    kappa[FrequencyBand.THETA] = 0.20
                    kappa[FrequencyBand.BETA]  = 0.18
                elif 5.0 <= t <= 7.0:
                    kappa[FrequencyBand.ALPHA] = 0.92
                    kappa[FrequencyBand.GAMMA] = 0.88
                    kappa[FrequencyBand.BETA]  = 0.85
            elif scenario == 'oscillatory':
                for b in FrequencyBand:
                    f = ABCRConfig.BAND_FREQUENCIES[b]
                    kappa[b] = 0.5 + 0.4 * np.sin(2 * np.pi * f * t / 20.0)
            elif scenario == 'cascade':
                if t > 3.0:
                    fail = int((t - 3.0) / 1.5)
                    for idx, b in enumerate(FrequencyBand):
                        if idx < fail:
                            kappa[b] = 0.1
            # noise
            for b in FrequencyBand:
                kappa[b] = float(np.clip(kappa[b] + np.random.normal(0.0, 0.02), 0.0, 1.0))
            rec = self.process(kappa, phi, t)
            if rec is not None:
                kappa = rec
            hist.append({'timestamp': t, 'kappa_state': {b.value: kappa[b] for b in FrequencyBand}, 'recovered': rec is not None})
        return hist

    def visualize(self, history: List[Dict[str, Any]], save_path: Optional[str] = None):
        ts = [h['timestamp'] for h in history]
        bands = {b: [h['kappa_state'][b.value] for h in history] for b in FrequencyBand}
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        colors = {FrequencyBand.DELTA: 'blue', FrequencyBand.THETA: 'green', FrequencyBand.ALPHA: 'red', FrequencyBand.BETA: 'orange', FrequencyBand.GAMMA: 'purple'}
        ax1 = axes[0]
        for b in FrequencyBand:
            ax1.plot(ts, bands[b], label=b.value, color=colors[b], linewidth=2)
        ax1.axhspan(0, ABCRConfig.TAU_BASE, alpha=0.1, color='blue', label='Hypo zone')
        ax1.axhspan(1-ABCRConfig.TAU_BASE, 1, alpha=0.1, color='red', label='Hyper zone')
        ax1.axhline(0.5, color='gray', linestyle=':')
        ax1.set_ylabel('κ')
        ax1.set_title('ABCR Dual-Stream Coherence Dynamics')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        ax2 = axes[1]
        A_times, B_times = [], []
        for e in self.system_history:
            if 'audit' in e:
                streams = e['audit']['active_streams']
                t = e['timestamp']
                if StreamType.STREAM_A.value in streams:
                    A_times.append(t)
                if StreamType.STREAM_B.value in streams:
                    B_times.append(t)
        if A_times:
            ax2.scatter(A_times, [0.3]*len(A_times), color='blue', s=50, alpha=0.7, label='Stream A')
        if B_times:
            ax2.scatter(B_times, [0.7]*len(B_times), color='red', s=50, alpha=0.7, label='Stream B')
        ax2.set_ylabel('Stream Activity')
        ax2.set_title('Dual-Stream Activity')
        ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0,1)
        ax3 = axes[2]
        audit_t, s_vals, seams = [], [], []
        for e in self.system_history:
            if 'audit' in e:
                audit_t.append(e['timestamp']); s_vals.append(e['audit']['s_composite']); seams.append(e['audit']['seam_type'])
        if audit_t:
            seam_colors = ['green' if 'Type I' in s else 'orange' if 'Type II' in s else 'red' for s in seams]
            ax3.scatter(audit_t, s_vals, c=seam_colors, s=30, alpha=0.8)
            ax3.axhline(0, color='gray', linestyle='-')
            ax3.axhline(ABCRConfig.AUDIT_TOLERANCE, color='green', linestyle='--', alpha=0.6)
            ax3.axhline(-ABCRConfig.AUDIT_TOLERANCE, color='green', linestyle='--', alpha=0.6)
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Composite Residual'); ax3.set_title('Audit Results'); ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        plt.show()

    # --------------- Persistence ---------------
    def save_state(self, path: str):
        state = {
            'mode': self.mode.value,
            'invariant_field': {b.value: float(self.renew.Pi[b]) for b in FrequencyBand} if self.renew.Pi else None,
            'system_history': self.system_history,
            'kappa_history': self.kappa_history[-10:] if self.kappa_history else []
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"State saved to {path}")

    def load_state(self, path: str):
        with open(path, 'r') as f:
            s = json.load(f)
        self.mode = SystemMode(s['mode'])
        if s['invariant_field']:
            self.renew.Pi = {FrequencyBand(b): v for b, v in s['invariant_field'].items()}
        self.system_history = s['system_history']
        self.kappa_history = s['kappa_history']
        logger.info(f"State loaded from {path}")

# ================================ DEMO ================================
def demonstrate_abcr():
    print("=" * 70)
    print("ADAPTIVE BI-COUPLED COHERENCE RECOVERY (ABCR) — v2")
    print("=" * 70)
    scenarios = [("dual_stress", SystemMode.ADAPTIVE), ("oscillatory", SystemMode.HIGH_SENSITIVITY), ("cascade", SystemMode.RECOVERY)]
    for name, mode in scenarios:
        print(f"
Scenario: {name} — mode={mode.value}")
        sys = AdaptiveBiCoupledCoherenceSystem(mode)
        hist = sys.simulate(10.0, 0.1, name)
        succ = sum(1 for e in sys.system_history if e['event'] == 'successful_recovery')
        total = len(sys.system_history)
        print(f"  Steps: {len(hist)} | Recovery attempts: {total} | Success: {succ}")
        if total:
            print(f"  Success rate: {succ/total*100:.1f}%")
        sys.visualize(hist, f"abcr_{name}_{mode.value}.png")
        sys.save_state(f"abcr_state_{name}_{mode.value}.json")
    print("
ABCR demonstration complete.")

if __name__ == '__main__':
    demonstrate_abcr()
