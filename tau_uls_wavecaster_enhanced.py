#!/usr/bin/env python3
# tau_uls_wavecaster_enhanced.py
# SPDX-License-Identifier: MIT
"""
TAU-ULS Enhanced WaveCaster with Neuro-Symbolic Adaptive Reflective Engine
--------------------------------------------------------------------------
Combines:
1. TAU-ULS (Two-level Trans-Algorithmic Universal Learning System) neural architecture
2. Dual LLM orchestration (local final inference + remote resource-only summaries)
3. Neuro-Symbolic Adaptive Reflective Engine for intelligent modulation selection
4. Advanced modulation schemes with adaptive link planning

Architecture:
- KFP (Kinetic Force Principle) layers for stability-driven optimization
- Entropy regulation based on environmental stress
- Dual LLM orchestration for content generation
- Adaptive modulation selection using RL and neuro-symbolic fusion
- Support for BFSK/BPSK/QPSK/16QAM/AFSK/OFDM modulation

Dependencies:
  Minimum: pip install numpy scipy torch requests
  Optional: pip install matplotlib sounddevice pycryptodome

Usage:
  # Basic modulation with TAU-ULS analysis
  python tau_uls_wavecaster_enhanced.py modulate --text "hello world" --scheme qpsk --wav

  # Full TAU-ULS enhanced casting with adaptive planning
  python tau_uls_wavecaster_enhanced.py tau-cast --prompt "technical analysis" \
      --resource-file data.txt --local-url http://127.0.0.1:8080 --adaptive --wav

  # TAU-ULS neural analysis of content
  python tau_uls_wavecaster_enhanced.py tau-analyze --text "complex data stream" --plot
"""

from __future__ import annotations
import argparse, base64, binascii, hashlib, json, logging, math, os, struct, sys, time, warnings, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime

# ---------- Hard requirements ----------
try:
    import numpy as np
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq
except Exception as e:
    raise SystemExit("numpy and scipy are required: pip install numpy scipy") from e

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

# ---------- Optional dependencies ----------
try:
    import requests
except Exception:
    requests = None  # HTTP backends disabled if missing

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tau_wavecaster")

# =========================================================
# TAU-ULS Neural Architecture Components
# =========================================================

class KFPLayer:
    """
    Kinetic Force Principle Layer - implements gradient-based parameter optimization
    following the principle that parameters move toward states of minimal fluctuation intensity
    """
    def __init__(self, dim: int, stability_weight: float = 0.1):
        self.dim = dim
        self.stability_weight = stability_weight

        # Fluctuation intensity tracking (Lyapunov function approximation)
        self.fluctuation_history = np.zeros(dim)
        self.momentum = 0.9

        # Kinetic force computation (simplified without PyTorch)
        self.force_weights = np.random.normal(0, 0.1, (dim, dim))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Compute current fluctuation intensity (variance across batch)
        current_fluctuation = np.var(x, axis=0)

        # Update fluctuation history with momentum
        self.fluctuation_history = (
            self.momentum * self.fluctuation_history +
            (1 - self.momentum) * current_fluctuation
        )

        # Apply kinetic force to push toward stability (simplified)
        kinetic_force = np.dot(x, self.force_weights.T)
        stability_term = -self.stability_weight * kinetic_force

        return x + stability_term, self.fluctuation_history

class TAULSControlUnit:
    """
    Two-level Trans-Algorithmic Universal Learning System
    Higher level: Learning and adaptation
    Lower level: Automatic control
    """
    def __init__(self, input_dim: int, hidden_dim: int, control_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        # Higher level: Learning system (meta-control) - simplified without PyTorch
        self.meta_weights1 = np.random.normal(0, 0.1, (hidden_dim, input_dim + control_dim))
        self.meta_weights2 = np.random.normal(0, 0.1, (control_dim, hidden_dim))

        # Add KFP layer for stability
        self.meta_kfp = KFPLayer(hidden_dim)

        # Lower level: Automatic control - simplified without PyTorch
        self.control_weights1 = np.random.normal(0, 0.1, (hidden_dim // 2, input_dim))
        self.control_weights2 = np.random.normal(0, 0.1, (control_dim, hidden_dim // 2))

        # Add KFP layer for stability
        self.auto_kfp = KFPLayer(hidden_dim // 2)

        # Control integration
        self.control_mixer = 0.5  # Simple mixing factor

    def forward(self, x: np.ndarray, prev_control: Optional[np.ndarray] = None) -> Dict:
        batch_size = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if prev_control is None:
            prev_control = np.zeros((batch_size, self.control_dim))

        # Higher level processing (learning) - simplified
        meta_input = np.concatenate([x, prev_control], axis=-1)
        meta_hidden = np.tanh(np.dot(meta_input, self.meta_weights1.T))
        meta_stable, meta_fluctuation = self.meta_kfp.forward(meta_hidden)
        meta_control = np.tanh(np.dot(meta_stable, self.meta_weights2.T))

        # Lower level processing (automatic control) - simplified
        auto_hidden = np.tanh(np.dot(x, self.control_weights1.T))
        auto_stable, auto_fluctuation = self.auto_kfp.forward(auto_hidden)
        auto_control = np.tanh(np.dot(auto_stable, self.control_weights2.T))

        # Integrate control signals using simple mixing
        integrated_control = self.control_mixer * meta_control + (1 - self.control_mixer) * auto_control

        return {
            'control_output': integrated_control,
            'meta_stability': meta_fluctuation,
            'auto_stability': auto_fluctuation,
            'control_mixing': self.control_mixer
        }

class EntropyRegulationModule:
    """
    Implements entropy regulation based on environmental stress
    Modulates parameter modification intensity to maintain active stability
    """
    def __init__(self, dim: int, max_entropy_target: float = 0.8):
        self.dim = dim
        self.max_entropy_target = max_entropy_target

        # Entropy estimation (simplified without PyTorch)
        self.entropy_weights = np.random.normal(0, 0.1, (1, dim))

        # Modification intensity controller (simplified)
        self.intensity_weights = np.random.normal(0, 0.1, (dim, 1))

    def compute_entropy(self, x: np.ndarray) -> float:
        """Approximate entropy using simple statistical method"""
        # Simple entropy estimation based on variance
        variance = np.var(x, axis=0)
        entropy = np.mean(np.log(1 + variance + 1e-12))
        return float(entropy)

    def forward(self, x: np.ndarray, environmental_stress: float) -> Tuple[np.ndarray, Dict]:
        current_entropy = self.compute_entropy(x)

        # Compute required entropy adjustment
        entropy_error = current_entropy - self.max_entropy_target
        stress_factor = environmental_stress

        # Adjust modification intensity based on stress and entropy (simplified)
        target_intensity = 1.0 / (1.0 + np.exp(-(entropy_error + stress_factor)))
        intensity_modulation = target_intensity * np.ones(self.dim)

        # Apply intensity modulation
        modulated_output = x * intensity_modulation

        return modulated_output, {
            'current_entropy': current_entropy,
            'target_intensity': target_intensity,
            'entropy_error': entropy_error
        }

class TAULSAnalyzer:
    """
    Complete TAU-ULS analyzer for text/data processing
    Provides stability metrics, entropy analysis, and control recommendations
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Text embedding (simple ASCII mapping)
        self.embedding_weights = np.random.normal(0, 0.1, (256, input_dim))

        # TAU-ULS control unit
        self.control_unit = TAULSControlUnit(input_dim, hidden_dim, hidden_dim // 2)

        # Entropy regulation
        self.entropy_regulator = EntropyRegulationModule(hidden_dim // 2)

        # KFP-based stability layer
        self.stability_layer = KFPLayer(hidden_dim // 2)

        # Output projection for analysis scores (simplified)
        self.output_weights1 = np.random.normal(0, 0.1, (hidden_dim, hidden_dim // 2))
        self.output_weights2 = np.random.normal(0, 0.1, (4, hidden_dim))

    def forward(self, text: str) -> Dict[str, Any]:
        # Convert text to embedding (simple ASCII encoding)
        text_indices = np.array([ord(c) % 256 for c in text[:512]])
        if len(text_indices) == 0:
            text_indices = np.array([0])

        # Embed text using simple lookup
        embedded = np.mean(self.embedding_weights[text_indices], axis=0, keepdims=True)

        # TAU-ULS control processing
        control_results = self.control_unit.forward(embedded)
        controlled = control_results['control_output']

        # Estimate environmental stress from text complexity
        stress = len(set(text)) / max(1, len(text))

        # Apply entropy regulation
        regulated, entropy_info = self.entropy_regulator.forward(controlled, stress)

        # Apply KFP-based stability
        stable, fluctuation = self.stability_layer.forward(regulated)

        # Generate analysis scores (simplified)
        hidden = np.tanh(np.dot(stable, self.output_weights1.T))
        scores = np.tanh(np.dot(hidden, self.output_weights2.T)).flatten()

        return {
            'stability_score': float(1.0 / (1.0 + np.exp(-scores[0]))),  # sigmoid
            'entropy_score': float(1.0 / (1.0 + np.exp(-scores[1]))),
            'complexity_score': float(1.0 / (1.0 + np.exp(-scores[2]))),
            'coherence_score': float(1.0 / (1.0 + np.exp(-scores[3]))),
            'control_mixing': float(control_results['control_mixing']),
            'meta_stability': float(np.mean(control_results['meta_stability'])),
            'auto_stability': float(np.mean(control_results['auto_stability'])),
            'entropy_info': {
                'current': float(entropy_info['current_entropy']),
                'target_intensity': float(entropy_info['target_intensity']),
                'error': float(entropy_info['entropy_error'])
            },
            'fluctuation_intensity': float(np.mean(fluctuation)),
            'text_length': len(text),
            'unique_chars': len(set(text))
        }

# =========================================================
# Polynomial KFP utilities
# =========================================================

def create_kfp_polynomial_basis(degree: int, dim: int) -> torch.Tensor:
    """
    Create polynomial basis functions for KFP approximation
    Based on the mathematical foundation that KFP follows gradient descent
    on fluctuation intensity functions
    """
    # Generate polynomial coefficients for stability landscape
    coefficients = torch.randn(degree + 1, dim, dim) * 0.1
    
    # Ensure stability (negative definite quadratic terms)
    coefficients[2] = -torch.abs(coefficients[2])  # Quadratic terms negative
    
    return coefficients

def kfp_polynomial_update(x: torch.Tensor, coefficients: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    """
    Polynomial-based KFP update rule
    Implements: dx/dt = -∇f(x) where f(x) is the fluctuation intensity
    """
    degree = coefficients.shape[0] - 1
    gradient = torch.zeros_like(x)
    
    # Compute polynomial gradient
    for d in range(1, degree + 1):
        power_term = torch.pow(x.unsqueeze(-1), d - 1)
        grad_term = d * torch.sum(coefficients[d] * power_term, dim=-1)
        gradient += grad_term
    
    # KFP update: move opposite to gradient
    return x - learning_rate * gradient

# =========================================================
# Enhanced Neuro-Symbolic Components (from mirror_cast)
# =========================================================

class EntropyAnalyzer:
    def measure(self, data: Any) -> float:
        s = str(data)
        if not s:
            return 0.0
        counts: Dict[str, int] = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
        n = len(s)
        ent = 0.0
        for cnt in counts.values():
            p = cnt / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

class DianneReflector:
    def reflect(self, data: Any) -> Dict[str, Any]:
        patterns = self._detect_patterns(data)
        head = str(data)[:40].replace("\n", " ")
        if "high_repetition" in patterns:
            insight = f"Cyclical resonance detected in Reflecting essence of: {head}..."
        elif "hierarchical_structure" in patterns:
            insight = f"Nested reality layers within Reflecting essence of: {head}..."
        else:
            insight = f"Linear transformation potential in Reflecting essence of: {head}..."
        return {"insight": insight, "patterns": patterns, "symbolic_depth": self._depth(data)}
    
    def _detect_patterns(self, data: Any) -> List[str]:
        s = str(data)
        patterns = []
        if len(s) > 100 and len(set(s)) < 20:
            patterns.append("high_repetition")
        if s.count('\n') > 5 and any(c in s for c in ['{', '[', '(', '<']):
            patterns.append("hierarchical_structure")
        return patterns
    
    def _depth(self, data: Any) -> int:
        s = str(data)
        return min(10, len(s) // 100)

class MatrixTransformer:
    def project(self, data: Any) -> Dict[str, Any]:
        dims = self._analyze(data)
        h = hash(str(data)) & 0xFFFFFFFF
        rank = int(dims["rank"])
        eivals = [math.sin(h * 0.001 * i) for i in range(max(1, min(3, rank)))]
        return {
            "projected_rank": dims["rank"],
            "structure": dims["structure"],
            "eigenvalues": eivals,
            "determinant": math.cos(h * 0.0001),
            "trace": (math.tan(h * 0.00001) if (h % 100) else 0.0),
        }
    
    def _analyze(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        return {
            "rank": min(10, len(s) // 50),
            "structure": "sparse" if len(set(s)) < 20 else "dense"
        }

class TAUEnhancedMirrorCast:
    """
    Mirror Cast engine enhanced with TAU-ULS neural analysis
    """
    def __init__(self):
        self.entropy = EntropyAnalyzer()
        self.reflector = DianneReflector()
        self.matrix = MatrixTransformer()
        self.tau_analyzer = TAULSAnalyzer()
        
    def cast(self, data: Any) -> Dict[str, Any]:
        # Traditional analysis
        base_analysis = {
            "entropy": self.entropy.measure(data),
            "reflection": self.reflector.reflect(data),
            "matrix": self.matrix.project(data),
            "timestamp": time.time()
        }
        
        # TAU-ULS neural analysis
        tau_analysis = self.tau_analyzer.forward(str(data))
        
        # Combine analyses
        return {
            **base_analysis,
            "tau_uls": tau_analysis,
            "combined_stability": (
                base_analysis["entropy"] * 0.3 + 
                tau_analysis["stability_score"] * 0.7
            ),
            "recommendation": self._recommend_modulation(base_analysis, tau_analysis)
        }
    
    def _recommend_modulation(self, base: Dict, tau: Dict) -> str:
        """Recommend modulation based on combined analysis"""
        stability = tau["stability_score"]
        entropy = tau["entropy_score"]
        complexity = tau["complexity_score"]
        
        if stability > 0.8 and complexity < 0.3:
            return "bpsk"  # Simple, stable
        elif stability > 0.6 and complexity < 0.6:
            return "qpsk"  # Moderate
        elif complexity > 0.7 or entropy > 0.8:
            return "ofdm"  # Complex, high entropy
        else:
            return "qam16"  # Default high-capacity

# =========================================================
# Modulation and Communication Components
# =========================================================

class ModulationScheme(Enum):
    BFSK = auto()
    BPSK = auto()
    QPSK = auto()
    QAM16 = auto()
    AFSK = auto()
    OFDM = auto()
    DSSS_BPSK = auto()

class FEC(Enum):
    NONE = auto()
    HAMMING74 = auto()
    REED_SOLOMON = auto()
    LDPC = auto()
    TURBO = auto()

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class OrchestratorSettings:
    temperature: float = 0.7
    max_tokens: int = 512
    style: str = "concise"
    max_context_chars: int = 8000

@dataclass
class ModConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0
    f1: float = 2200.0
    fc: float = 1800.0
    clip: bool = True
    ofdm_subc: int = 64
    cp_len: int = 16
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8
    version: int = 1

@dataclass
class SecurityConfig:
    password: Optional[str] = None
    watermark: Optional[str] = None
    hmac_key: Optional[str] = None

# =========================================================
# Utility Functions
# =========================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def crc32_bytes(data: bytes) -> bytes:
    return binascii.crc32(data).to_bytes(4, "big")

def crc16_ccitt(data: bytes) -> bytes:
    poly, crc = 0x1021, 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc.to_bytes(2, "big")

def to_bits(data: bytes) -> List[int]:
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]

def from_bits(bits: Sequence[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | (1 if b else 0)
        out.append(byte)
    return bytes(out)

def chunk_bits(bits: Sequence[int], n: int) -> List[List[int]]:
    return [list(bits[i:i+n]) for i in range(0, len(bits), n)]

def safe_json(obj: Any) -> str:
    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        if isinstance(x, datetime):
            return x.isoformat()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        return str(x)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=enc)

def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int):
    import wave
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

def write_iq_f32(path: Path, iq: np.ndarray):
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be 1-D complex array")
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    path.write_bytes(interleaved.tobytes())

def plot_wave_and_spectrum(path_png: Path, x: np.ndarray, sr: int, title: str):
    if not HAS_MPL: 
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5))
    t = np.arange(len(x))/sr
    ax1.plot(t[:min(len(t), int(0.05*sr))], x[:min(len(x), int(0.05*sr))])
    ax1.set_title(f"{title} (first 50ms)")
    ax1.set_xlabel("s")
    ax1.set_ylabel("amplitude")
    spec = np.abs(rfft(x)) + 1e-12
    freqs = rfftfreq(len(x), 1.0/sr)
    ax2.semilogy(freqs, spec/spec.max())
    ax2.set_xlim(0, min(8000, sr//2))
    ax2.set_xlabel("Hz")
    ax2.set_ylabel("norm |X(f)|")
    plt.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)

def plot_tau_analysis(path_png: Path, tau_analysis: Dict[str, Any], title: str = "TAU-ULS Analysis"):
    if not HAS_MPL:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stability metrics
    metrics = ['stability_score', 'entropy_score', 'complexity_score', 'coherence_score']
    values = [tau_analysis[m] for m in metrics]
    ax1.bar(metrics, values)
    ax1.set_title("TAU-ULS Scores")
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels([m.replace('_score', '') for m in metrics], rotation=45)
    
    # Control mixing visualization
    ax2.pie([tau_analysis['control_mixing'], 1 - tau_analysis['control_mixing']], 
            labels=['Meta Control', 'Auto Control'],
            autopct='%1.1f%%')
    ax2.set_title("Control Mixing Ratio")
    
    # Stability comparison
    stabilities = ['meta_stability', 'auto_stability', 'fluctuation_intensity']
    stab_values = [tau_analysis[s] for s in stabilities]
    ax3.bar(stabilities, stab_values)
    ax3.set_title("Stability Metrics")
    ax3.set_xticklabels(['Meta', 'Auto', 'Fluctuation'], rotation=45)
    
    # Entropy info
    entropy_data = tau_analysis['entropy_info']
    ax4.plot(['Current', 'Target\nIntensity', 'Error'], 
             [entropy_data['current'], entropy_data['target_intensity'], abs(entropy_data['error'])],
             'o-')
    ax4.set_title("Entropy Regulation")
    ax4.set_ylabel("Value")
    
    plt.suptitle(f"{title} - Text Length: {tau_analysis['text_length']}, Unique Chars: {tau_analysis['unique_chars']}")
    plt.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)

def play_audio(x: np.ndarray, sr: int):
    if not HAS_AUDIO:
        log.warning("sounddevice not installed; cannot play audio")
        return
    sd.play(x, sr)
    sd.wait()

# =========================================================
# FEC Implementation
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    out = []
    for i in range(0, len(data_bits), 4):
        d0, d1, d2, d3 = data_bits[i:i+4]
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        out += [p1, p2, d0, p3, d1, d2, d3]
    return out

def fec_encode(bits: List[int], scheme: FEC) -> List[int]:
    if scheme == FEC.NONE:
        return list(bits)
    if scheme == FEC.HAMMING74:
        return hamming74_encode(bits)
    if scheme in (FEC.REED_SOLOMON, FEC.LDPC, FEC.TURBO):
        raise NotImplementedError(f"{scheme.name} encoding not implemented in this minimal build")
    raise ValueError("Unknown FEC")

# =========================================================
# Security Functions
# =========================================================

def aes_gcm_encrypt(plaintext: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return b"AGCM" + salt + nonce + tag + ct

def apply_hmac(data: bytes, hkey: str) -> bytes:
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def add_watermark(data: bytes, wm: str) -> bytes:
    return hashlib.sha256(wm.encode("utf-8")).digest()[:8] + data

def frame_payload(payload: bytes, fcfg: FrameConfig) -> bytes:
    header = struct.pack(">BBI", 0xA5, fcfg.version, now_ms() & 0xFFFFFFFF)
    core = header + payload
    tail = b""
    if fcfg.use_crc32:
        tail += crc32_bytes(core)
    if fcfg.use_crc16:
        tail += crc16_ccitt(core)
    return fcfg.preamble + core + tail

def encode_text(
    text: str,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
) -> List[int]:
    data = text.encode("utf-8")
    if sec.watermark:
        data = add_watermark(data, sec.watermark)
    if sec.password:
        data = aes_gcm_encrypt(data, sec.password)
    framed = frame_payload(data, fcfg)
    if sec.hmac_key:
        framed = apply_hmac(framed, sec.hmac_key)
    bits = to_bits(framed)
    bits = fec_encode(bits, fec_scheme)
    return bits

# =========================================================
# Modulators
# =========================================================

class Modulators:
    @staticmethod
    def bfsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        sr, rb = cfg.sample_rate, cfg.symbol_rate
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        s = []
        a = cfg.amplitude
        for b in bits:
            f = cfg.f1 if b else cfg.f0
            s.append(a * np.sin(2*np.pi*f*t))
        y = np.concatenate(s) if s else np.zeros(0, dtype=np.float64)
        return np.clip(y, -1, 1).astype(np.float32) if cfg.clip else y.astype(np.float32)

    @staticmethod
    def bpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        a = cfg.amplitude
        audio_blocks, iq_blocks = [], []
        for b in bits:
            phase = 0.0 if b else np.pi
            audio_blocks.append(a * np.sin(2*np.pi*fc*t + phase))
            iq_blocks.append(a * (np.cos(phase) + 1j*np.sin(phase)) * np.ones_like(t, dtype=np.complex64))
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float64)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq

    @staticmethod
    def qpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        pairs = chunk_bits(bits, 2)
        syms = []
        for p in pairs:
            b0, b1 = (p + [0,0])[:2]
            if (b0, b1) == (0,0): s = 1 + 1j
            elif (b0, b1) == (0,1): s = -1 + 1j
            elif (b0, b1) == (1,1): s = -1 - 1j
            else: s = 1 - 1j
            syms.append(s / math.sqrt(2))
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def qam16(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        quads = chunk_bits(bits, 4)
        def map2(b0,b1):
            val = (b0<<1) | b1
            return [-3,-1,1,3][val]
        syms = []
        for q in quads:
            b0,b1,b2,b3 = (q+[0,0,0,0])[:4]
            I = map2(b0,b1)
            Q = map2(b2,b3)
            syms.append((I + 1j*Q)/math.sqrt(10))
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def _psk_qam_to_audio_iq(syms: np.ndarray, cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        a = cfg.amplitude
        i = np.repeat(syms.real.astype(np.float32), spb)
        q = np.repeat(syms.imag.astype(np.float32), spb)
        t = np.arange(len(i)) / sr
        audio = a * (i*np.cos(2*np.pi*fc*t) - q*np.sin(2*np.pi*fc*t))
        iq = (a * i) + 1j*(a * q)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq.astype(np.complex64)

    @staticmethod
    def afsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        return Modulators.bfsK(bits, cfg)

    @staticmethod
    def dsss_bpsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        pn = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)
        sr = cfg.sample_rate
        spb = int(sr / (cfg.dsss_chip_rate))
        base = []
        for b in bits:
            bit_val = 1.0 if b else -1.0
            ch = bit_val * pn
            ch = np.repeat(ch, spb)
            base.append(ch)
        baseband = np.concatenate(base) if base else np.zeros(0, dtype=np.float32)
        t = np.arange(len(baseband))/sr
        audio = cfg.amplitude * baseband * np.sin(2*np.pi*cfg.fc*t)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32)

    @staticmethod
    def ofdm(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        N = cfg.ofdm_subc
        spb_sym = int(cfg.sample_rate / cfg.symbol_rate)
        chunks = chunk_bits(bits, 2*N)
        a = cfg.amplitude
        wave = []
        iq = []
        for ch in chunks:
            qsyms = []
            pairs = chunk_bits(ch, 2)
            for p in pairs:
                b0,b1 = (p+[0,0])[:2]
                if (b0,b1)==(0,0): s = 1+1j
                elif (b0,b1)==(0,1): s = -1+1j
                elif (b0,b1)==(1,1): s = -1-1j
                else: s = 1-1j
                qsyms.append(s/math.sqrt(2))
            if len(qsyms) < N:
                qsyms += [0j]*(N-len(qsyms))
            Xk = np.array(qsyms, dtype=np.complex64)
            xt = np.fft.ifft(Xk)
            cp = xt[-cfg.cp_len:]
            sym = np.concatenate([cp, xt])
            reps = max(1, int(spb_sym/len(sym)))
            sym_up = np.repeat(sym, reps)
            t = np.arange(len(sym_up))/cfg.sample_rate
            audio = a*(sym_up.real*np.cos(2*np.pi*cfg.fc*t) - sym_up.imag*np.sin(2*np.pi*cfg.fc*t))
            wave.append(audio.astype(np.float32))
            iq.append((a*sym_up).astype(np.complex64))
        audio = np.concatenate(wave) if wave else np.zeros(0, dtype=np.float32)
        iqc = np.concatenate(iq) if iq else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio, iqc

# =========================================================
# LLM Backends
# =========================================================

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str: 
        raise NotImplementedError

class LocalLLM(BaseLLM):
    def __init__(self, configs: List[HTTPConfig]):
        if requests is None:
            raise RuntimeError("LocalLLM requires 'requests' (pip install requests)")
        self.configs = configs
        self.idx = 0

    def generate(self, prompt: str, **kwargs) -> str:
        last = None
        for _ in range(len(self.configs)):
            cfg = self.configs[self.idx]
            try:
                out = self._call(cfg, prompt, **kwargs)
                return out
            except Exception as e:
                last = e
                self.idx = (self.idx + 1) % len(self.configs)
        raise last or RuntimeError("All local LLM configs failed")

    def _post(self, cfg: HTTPConfig, url: str, headers: dict, body: dict) -> dict:
        s = requests.Session()
        for attempt in range(cfg.max_retries):
            try:
                r = s.post(url, headers=headers, json=body, timeout=cfg.timeout, verify=cfg.verify_ssl)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < cfg.max_retries-1:
                    time.sleep(cfg.retry_delay*(2**attempt))
                else:
                    raise

    def _call(self, cfg: HTTPConfig, prompt: str, **kwargs) -> str:
        mode = cfg.mode
        if mode == "openai-chat":
            url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-4o-mini",
                "messages": [{"role":"user","content":prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["message"]["content"]
        if mode == "openai-completions":
            url = f"{cfg.base_url.rstrip('/')}/v1/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["text"]
        if mode == "llama-cpp":
            url = f"{cfg.base_url.rstrip('/')}/completion"
            body = {"prompt": prompt, "temperature": kwargs.get("temperature",0.7), "n_predict": kwargs.get("max_tokens",512)}
            data = self._post(cfg, url, {}, body)
            if "content" in data: return data["content"]
            if "choices" in data and data["choices"]: return data["choices"][0].get("text","")
            return data.get("text","")
        if mode == "textgen-webui":
            url = f"{cfg.base_url.rstrip('/')}/api/v1/generate"
            body = {"prompt": prompt, "max_new_tokens": kwargs.get("max_tokens",512), "temperature": kwargs.get("temperature",0.7)}
            data = self._post(cfg, url, {}, body)
            return data.get("results",[{}])[0].get("text","")
        raise ValueError(f"Unsupported mode: {mode}")

class ResourceLLM(BaseLLM):
    def __init__(self, cfg: Optional[HTTPConfig] = None):
        self.cfg = cfg

    def generate(self, prompt: str, **kwargs) -> str:
        if self.cfg is None or requests is None:
            return LocalSummarizer().summarize(prompt)
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type":"application/json"}
        if self.cfg.api_key: headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        system = ("You are a constrained assistant. ONLY summarize/structure the provided INPUT RESOURCES. "
                  "Do not add external knowledge.")
        body = {
            "model": self.cfg.model or "gpt-4o-mini",
            "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
            "temperature": kwargs.get("temperature",0.2),
            "max_tokens": kwargs.get("max_tokens",512),
        }
        s = requests.Session()
        r = s.post(url, headers=headers, json=body, timeout=self.cfg.timeout, verify=self.cfg.verify_ssl)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class LocalSummarizer:
    def __init__(self):
        self.stop = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are",
            "was","were","be","been","being","have","has","had","do","does","did","will","would",
            "could","should","from","that","this","it","as"
        }
    
    def summarize(self, text: str) -> str:
        txt = " ".join(text.split())
        if not txt: return "No content to summarize."
        sents = [s.strip() for s in txt.replace("?",".").replace("!",".").split(".") if s.strip()]
        if not sents: return txt[:300] + ("..." if len(txt)>300 else "")
        words = [w.lower().strip(",;:()[]") for w in txt.split()]
        freq: Dict[str,int] = {}
        for w in words:
            if w and w not in self.stop: freq[w] = freq.get(w,0)+1
        scored = []
        for s in sents:
            sw = [w.lower().strip(",;:()[]") for w in s.split()]
            score = len(s) * 0.1 + sum(freq.get(w,0) for w in sw)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [s for s,_ in scored[: min(6,len(scored))]]
        keep.sort(key=lambda k: sents.index(k))
        out = " ".join(keep)
        return out[:800] + ("..." if len(out)>800 else "")

# =========================================================
# Orchestrator
# =========================================================

class DualLLMOrchestrator:
    def __init__(self, local: LocalLLM, resource: ResourceLLM, settings: OrchestratorSettings):
        self.local, self.resource, self.set = local, resource, settings

    def _load_resources(self, paths: List[str], inline: List[str]) -> str:
        parts = []
        for p in paths:
            pa = Path(p)
            if pa.exists() and pa.is_file():
                try:
                    parts.append(pa.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    parts.append(f"[[UNREADABLE_FILE:{pa.name}]]")
            else:
                parts.append(f"[[MISSING_FILE:{pa}]]")
        parts += [str(x) for x in inline]
        blob = "\n\n".join(parts)
        return blob[: self.set.max_context_chars]

    def compose(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Tuple[str,str]:
        res_text = self._load_resources(resource_paths, inline_resources)
        res_summary = self.resource.generate(
            f"INPUT RESOURCES:\n{res_text}\n\nTASK: Summarize/structure ONLY the content above.",
            temperature=0.2, max_tokens=self.set.max_tokens
        )
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{res_summary}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.set.style}. Be clear and directly actionable."
        )
        return final_prompt, res_summary

    def run(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str,str]:
        fp, summary = self.compose(user_prompt, resource_paths, inline_resources)
        ans = self.local.generate(fp, temperature=self.set.temperature, max_tokens=self.set.max_tokens)
        return {"summary": summary, "final": ans, "prompt": fp}

# =========================================================
# TAU-ULS Enhanced Adaptive Link Planner
# =========================================================

class TAUAdaptiveLinkPlanner:
    """
    Adaptive link planner enhanced with TAU-ULS neural analysis
    """
    def __init__(self):
        self.tau_caster = TAUEnhancedMirrorCast()
        
    def plan(self, text: str, base_config: ModConfig) -> Tuple[ModConfig, Dict[str, Any]]:
        # Get TAU-ULS enhanced analysis
        analysis = self.tau_caster.cast(text)
        
        # Extract recommendation
        recommended_mod = analysis["recommendation"]
        
        # Create new config based on TAU-ULS analysis
        new_config = ModConfig(
            sample_rate=base_config.sample_rate,
            symbol_rate=base_config.symbol_rate,
            amplitude=base_config.amplitude,
            f0=base_config.f0,
            f1=base_config.f1,
            fc=base_config.fc,
            clip=base_config.clip,
            ofdm_subc=base_config.ofdm_subc,
            cp_len=base_config.cp_len,
            dsss_chip_rate=base_config.dsss_chip_rate
        )
        
        # Adjust parameters based on TAU-ULS scores
        tau_scores = analysis["tau_uls"]
        
        # Stability affects symbol rate
        if tau_scores["stability_score"] > 0.8:
            new_config.symbol_rate = min(4800, base_config.symbol_rate * 2)
        elif tau_scores["stability_score"] < 0.4:
            new_config.symbol_rate = max(600, base_config.symbol_rate // 2)
        
        # Complexity affects modulation order
        if tau_scores["complexity_score"] > 0.7:
            new_config.ofdm_subc = 128  # More subcarriers for complex data
        
        # Entropy affects amplitude (power control)
        if tau_scores["entropy_score"] > 0.8:
            new_config.amplitude = min(0.9, base_config.amplitude * 1.1)
        
        return new_config, {
            "tau_analysis": analysis["tau_uls"],
            "recommended_modulation": recommended_mod,
            "stability_adjusted": tau_scores["stability_score"] != 0.5,
            "config_changes": {
                "symbol_rate": f"{base_config.symbol_rate} -> {new_config.symbol_rate}",
                "amplitude": f"{base_config.amplitude:.2f} -> {new_config.amplitude:.2f}",
                "ofdm_subc": f"{base_config.ofdm_subc} -> {new_config.ofdm_subc}"
            }
        }

# =========================================================
# End-to-end casting
# =========================================================

@dataclass
class OutputPaths:
    wav: Optional[Path] = None
    iq: Optional[Path] = None
    meta: Optional[Path] = None
    png: Optional[Path] = None
    tau_png: Optional[Path] = None

def bits_to_signals(bits: List[int], scheme: ModulationScheme, mcfg: ModConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if scheme == ModulationScheme.BFSK:
        return Modulators.bfsK(bits, mcfg), None
    if scheme == ModulationScheme.AFSK:
        return Modulators.afsK(bits, mcfg), None
    if scheme == ModulationScheme.BPSK:
        return Modulators.bpsK(bits, mcfg)
    if scheme == ModulationScheme.QPSK:
        return Modulators.qpsK(bits, mcfg)
    if scheme == ModulationScheme.QAM16:
        return Modulators.qam16(bits, mcfg)
    if scheme == ModulationScheme.OFDM:
        return Modulators.ofdm(bits, mcfg)
    if scheme == ModulationScheme.DSSS_BPSK:
        return Modulators.dsss_bpsK(bits, mcfg), None
    raise ValueError("Unknown modulation scheme")

def full_tau_cast_and_save(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
    want_wav: bool,
    want_iq: bool,
    tau_analysis: Optional[Dict[str, Any]] = None,
    title: str = "TAU-WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"tau_cast_{scheme.name.lower()}_{ts}"
    
    # Encode text
    bits = encode_text(text, fcfg, sec, fec_scheme)
    
    # Generate signals
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    
    paths = OutputPaths()
    
    # Save audio
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav")
        write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    
    # Save IQ
    if want_iq:
        if iq is None and audio is not None:
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32")
            write_iq_f32(paths.iq, iq)
    
    # Visualizations
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix("_signal.png")
        plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    
    if tau_analysis is not None and HAS_MPL:
        paths.tau_png = base.with_suffix("_tau_analysis.png")
        plot_tau_analysis(paths.tau_png, tau_analysis, title)
    
    # Metadata
    meta = {
        "timestamp": ts,
        "scheme": scheme.name,
        "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate,
        "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
        "fec": fec_scheme.name,
        "encrypted": bool(sec.password),
        "watermark": bool(sec.watermark),
        "hmac": bool(sec.hmac_key),
        "tau_analysis": tau_analysis
    }
    paths.meta = base.with_suffix(".json")
    paths.meta.write_text(safe_json(meta), encoding="utf-8")
    
    return paths

# =========================================================
# CLI Commands
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tau_uls_wavecaster_enhanced",
        description="TAU-ULS Enhanced WaveCaster with Neuro-Symbolic Adaptive Engine"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_mod_args(sp):
        sp.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme], default="bfsk")
        sp.add_argument("--sample-rate", type=int, default=48000)
        sp.add_argument("--symbol-rate", type=int, default=1200)
        sp.add_argument("--amplitude", type=float, default=0.7)
        sp.add_argument("--f0", type=float, default=1200.0)
        sp.add_argument("--f1", type=float, default=2200.0)
        sp.add_argument("--fc", type=float, default=1800.0)
        sp.add_argument("--no-clip", action="store_true")
        sp.add_argument("--outdir", type=str, default="tau_casts")
        sp.add_argument("--wav", action="store_true")
        sp.add_argument("--iq", action="store_true")
        sp.add_argument("--play", action="store_true", help="Play audio to soundcard")
        sp.add_argument("--ofdm-subc", type=int, default=64)
        sp.add_argument("--cp-len", type=int, default=16)
        sp.add_argument("--dsss-chip-rate", type=int, default=4800)

    # tau-cast: TAU-ULS enhanced 2-LLM orchestration then modulate
    sp_tau_cast = sub.add_parser("tau-cast", help="TAU-ULS enhanced dual LLM composition and modulation")
    sp_tau_cast.add_argument("--prompt", type=str, required=True)
    sp_tau_cast.add_argument("--resource-file", nargs="*", default=[])
    sp_tau_cast.add_argument("--resource-text", nargs="*", default=[])
    sp_tau_cast.add_argument("--local-url", type=str, default="http://127.0.0.1:8080")
    sp_tau_cast.add_argument("--local-mode", choices=["openai-chat","openai-completions","llama-cpp","textgen-webui"], default="llama-cpp")
    sp_tau_cast.add_argument("--local-model", type=str, default="local-gguf")
    sp_tau_cast.add_argument("--local-key", type=str, default=None)
    sp_tau_cast.add_argument("--remote-url", type=str, default=None)
    sp_tau_cast.add_argument("--remote-model", type=str, default="gpt-4o-mini")
    sp_tau_cast.add_argument("--remote-key", type=str, default=None)
    sp_tau_cast.add_argument("--style", type=str, default="concise")
    sp_tau_cast.add_argument("--max-tokens", type=int, default=512)
    sp_tau_cast.add_argument("--temperature", type=float, default=0.7)
    sp_tau_cast.add_argument("--password", type=str, default=None)
    sp_tau_cast.add_argument("--watermark", type=str, default=None)
    sp_tau_cast.add_argument("--hmac-key", type=str, default=None)
    sp_tau_cast.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="hamming74")
    sp_tau_cast.add_argument("--adaptive", action="store_true", help="Use TAU-ULS adaptive planning")
    add_mod_args(sp_tau_cast)

    # modulate: direct text to waveform
    sp_mod = sub.add_parser("modulate", help="Modulate text with TAU-ULS analysis")
    sp_mod.add_argument("--text", type=str, required=True)
    sp_mod.add_argument("--password", type=str, default=None)
    sp_mod.add_argument("--watermark", type=str, default=None)
    sp_mod.add_argument("--hmac-key", type=str, default=None)
    sp_mod.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="none")
    sp_mod.add_argument("--adaptive", action="store_true", help="Use TAU-ULS adaptive planning")
    add_mod_args(sp_mod)

    # tau-analyze: TAU-ULS neural analysis
    sp_tau = sub.add_parser("tau-analyze", help="TAU-ULS neural analysis of text")
    sp_tau.add_argument("--text", type=str, required=True)
    sp_tau.add_argument("--plot", action="store_true", help="Generate analysis plots")
    sp_tau.add_argument("--outdir", type=str, default="tau_analysis")

    # visualize existing WAV
    sp_vis = sub.add_parser("visualize", help="Plot waveform + spectrum from WAV")
    sp_vis.add_argument("--wav", type=str, required=True)
    sp_vis.add_argument("--out", type=str, default=None)

    # analyze: basic metrics
    sp_an = sub.add_parser("analyze", help="Basic audio metrics of WAV")
    sp_an.add_argument("--wav", type=str, required=True)

    # tau-demo: demonstrate TAU-ULS components
    sp_demo = sub.add_parser("tau-demo", help="Demonstrate TAU-ULS neural components")
    sp_demo.add_argument("--text", type=str, default="Example text for TAU-ULS demonstration")
    sp_demo.add_argument("--iterations", type=int, default=10)

    return p

def parse_scheme(s: str) -> ModulationScheme:
    return ModulationScheme[s.upper()]

def parse_fec(s: str) -> FEC:
    return FEC[s.upper()]

def make_modcfg(args: argparse.Namespace) -> ModConfig:
    return ModConfig(
        sample_rate=args.sample_rate,
        symbol_rate=args.symbol_rate,
        amplitude=args.amplitude,
        f0=args.f0,
        f1=args.f1,
        fc=args.fc,
        clip=not args.no_clip,
        ofdm_subc=getattr(args, "ofdm_subc", 64),
        cp_len=getattr(args,"cp_len",16),
        dsss_chip_rate=getattr(args,"dsss_chip_rate",4800),
    )

def cmd_tau_cast(args: argparse.Namespace) -> int:
    """TAU-ULS enhanced dual LLM casting"""
    # Build LLMs
    local = LocalLLM([HTTPConfig(
        base_url=args.local_url,
        model=args.local_model,
        mode=args.local_mode,
        api_key=args.local_key
    )])
    
    rcfg = HTTPConfig(
        base_url=args.remote_url,
        model=args.remote_model,
        api_key=args.remote_key
    ) if args.remote_url else None
    
    resource = ResourceLLM(rcfg)
    
    orch = DualLLMOrchestrator(local, resource, OrchestratorSettings(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        style=args.style
    ))
    
    # Generate content
    result = orch.run(args.prompt, args.resource_file, args.resource_text)
    
    # Get base config
    mcfg = make_modcfg(args)
    scheme = parse_scheme(args.scheme)
    
    # TAU-ULS analysis and adaptive planning
    tau_analysis = None
    if args.adaptive:
        planner = TAUAdaptiveLinkPlanner()
        mcfg, plan_info = planner.plan(result["final"], mcfg)
        tau_analysis = plan_info["tau_analysis"]
        
        # Use recommended modulation if adaptive
        recommended = plan_info["recommended_modulation"]
        if recommended in [s.name.lower() for s in ModulationScheme]:
            scheme = parse_scheme(recommended)
            log.info(f"TAU-ULS recommended modulation: {recommended}")
    else:
        # Still run TAU analysis for visualization
        analyzer = TAULSAnalyzer()
        tau_analysis = analyzer(result["final"])
    
    # Build frame and security configs
    fcfg = FrameConfig()
    sec = SecurityConfig(
        password=args.password,
        watermark=args.watermark,
        hmac_key=args.hmac_key
    )
    fec_s = parse_fec(args.fec)
    
    # Cast with TAU analysis
    paths = full_tau_cast_and_save(
        text=result["final"],
        outdir=Path(args.outdir),
        scheme=scheme,
        mcfg=mcfg,
        fcfg=fcfg,
        sec=sec,
        fec_scheme=fec_s,
        want_wav=args.wav or (not args.iq),
        want_iq=args.iq,
        tau_analysis=tau_analysis,
        title=f"TAU-{scheme.name} | Enhanced Wave"
    )
    
    # Play audio if requested
    if args.play and paths.wav and HAS_AUDIO:
        try:
            import wave
            with wave.open(str(paths.wav), "rb") as w:
                sr = w.getframerate()
                n = w.getnframes()
                data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
            play_audio(data, sr)
        except Exception as e:
            log.warning(f"Could not play audio: {e}")
    
    # Output results
    output = {
        "files": {
            "wav": str(paths.wav) if paths.wav else None,
            "iq": str(paths.iq) if paths.iq else None,
            "meta": str(paths.meta) if paths.meta else None,
            "signal_png": str(paths.png) if paths.png else None,
            "tau_analysis_png": str(paths.tau_png) if paths.tau_png else None
        },
        "content_preview": result["final"][:400] + "..." if len(result["final"]) > 400 else result["final"],
        "summary_preview": result["summary"][:400] + "..." if len(result["summary"]) > 400 else result["summary"],
        "tau_scores": {
            "stability": tau_analysis["stability_score"],
            "entropy": tau_analysis["entropy_score"],
            "complexity": tau_analysis["complexity_score"],
            "coherence": tau_analysis["coherence_score"]
        } if tau_analysis else None,
        "modulation": scheme.name,
        "adaptive_planning": args.adaptive
    }
    
    print(safe_json(output))
    return 0

def cmd_modulate(args: argparse.Namespace) -> int:
    """Direct modulation with TAU-ULS analysis"""
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(
        password=args.password,
        watermark=args.watermark,
        hmac_key=args.hmac_key
    )
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    
    # TAU-ULS analysis
    tau_analysis = None
    if args.adaptive:
        planner = TAUAdaptiveLinkPlanner()
        mcfg, plan_info = planner.plan(args.text, mcfg)
        tau_analysis = plan_info["tau_analysis"]
        
        # Use recommended modulation
        recommended = plan_info["recommended_modulation"]
        if recommended in [s.name.lower() for s in ModulationScheme]:
            scheme = parse_scheme(recommended)
            log.info(f"TAU-ULS recommended modulation: {recommended}")
    else:
        analyzer = TAULSAnalyzer()
        tau_analysis = analyzer(args.text)
    
    paths = full_tau_cast_and_save(
        text=args.text,
        outdir=Path(args.outdir),
        scheme=scheme,
        mcfg=mcfg,
        fcfg=fcfg,
        sec=sec,
        fec_scheme=fec_s,
        want_wav=args.wav or (not args.iq),
        want_iq=args.iq,
        tau_analysis=tau_analysis,
        title=f"TAU-{scheme.name} | Direct Mod"
    )
    
    if args.play and paths.wav:
        try:
            import wave
            with wave.open(str(paths.wav), "rb") as w:
                sr = w.getframerate()
                n = w.getnframes()
                data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
            play_audio(data, sr)
        except Exception:
            log.warning("Could not play audio")
    
    output = {
        "files": {
            "wav": str(paths.wav) if paths.wav else None,
            "iq": str(paths.iq) if paths.iq else None,
            "meta": str(paths.meta) if paths.meta else None,
            "signal_png": str(paths.png) if paths.png else None,
            "tau_analysis_png": str(paths.tau_png) if paths.tau_png else None
        },
        "tau_scores": {
            "stability": tau_analysis["stability_score"],
            "entropy": tau_analysis["entropy_score"],
            "complexity": tau_analysis["complexity_score"],
            "coherence": tau_analysis["coherence_score"]
        } if tau_analysis else None,
        "modulation": scheme.name
    }
    
    print(safe_json(output))
    return 0

def cmd_tau_analyze(args: argparse.Namespace) -> int:
    """Pure TAU-ULS neural analysis"""
    analyzer = TAULSAnalyzer()
    analysis = analyzer(args.text)
    
    # Also run enhanced mirror cast for comparison
    tau_caster = TAUEnhancedMirrorCast()
    full_analysis = tau_caster.cast(args.text)
    
    output = {
        "tau_uls_analysis": analysis,
        "combined_analysis": {
            "entropy": full_analysis["entropy"],
            "matrix": full_analysis["matrix"],
            "reflection": full_analysis["reflection"],
            "recommendation": full_analysis["recommendation"],
            "combined_stability": full_analysis["combined_stability"]
        }
    }
    
    if args.plot and HAS_MPL:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # TAU analysis plot
        tau_png = outdir / "tau_analysis.png"
        plot_tau_analysis(tau_png, analysis, "TAU-ULS Neural Analysis")
        output["tau_plot"] = str(tau_png)
        
        # Combined visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Entropy comparison
        ax1.bar(['Classic', 'Neural'], 
                [full_analysis["entropy"], analysis["entropy_score"]])
        ax1.set_title("Entropy Analysis Comparison")
        ax1.set_ylabel("Score")
        
        # Modulation recommendation visualization
        mods = ['bpsk', 'qpsk', 'qam16', 'ofdm']
        scores = [0.2, 0.3, 0.3, 0.2]  # Example distribution
        if full_analysis["recommendation"] in mods:
            idx = mods.index(full_analysis["recommendation"])
            scores[idx] = 0.7
        ax2.bar(mods, scores)
        ax2.set_title(f"Modulation Recommendation: {full_analysis['recommendation'].upper()}")
        ax2.set_ylabel("Confidence")
        
        plt.tight_layout()
        combined_png = outdir / "combined_analysis.png"
        fig.savefig(combined_png)
        plt.close(fig)
        output["combined_plot"] = str(combined_png)
    
    print(safe_json(output))
    return 0

def cmd_tau_demo(args: argparse.Namespace) -> int:
    """Demonstrate TAU-ULS components"""
    print("TAU-ULS Component Demonstration")
    print("=" * 50)
    
    # Create components
    kfp = KFPLayer(dim=64)
    control = TAULSControlUnit(input_dim=64, hidden_dim=128, control_dim=32)
    entropy_reg = EntropyRegulationModule(dim=32)
    
    # Create sample data
    x = torch.randn(1, 64)
    
    print("\n1. KFP Layer Demo:")
    for i in range(args.iterations):
        x_stable, fluctuation = kfp(x)
        if i % 3 == 0:
            print(f"   Iteration {i}: Fluctuation intensity = {fluctuation.mean().item():.4f}")
        x = x_stable
    
    print("\n2. TAU-ULS Control Unit Demo:")
    control_out = control(x)
    print(f"   Control mixing: {control_out['control_mixing'].item():.3f}")
    print(f"   Meta stability: {control_out['meta_stability'].mean().item():.4f}")
    print(f"   Auto stability: {control_out['auto_stability'].mean().item():.4f}")
    
    print("\n3. Entropy Regulation Demo:")
    stress = torch.tensor([0.7])
    regulated, entropy_info = entropy_reg(control_out['control_output'], stress)
    print(f"   Current entropy: {entropy_info['current_entropy'].item():.4f}")
    print(f"   Target intensity: {entropy_info['target_intensity'].item():.4f}")
    print(f"   Entropy error: {entropy_info['entropy_error'].item():.4f}")
    
    print("\n4. Full TAU-ULS Analysis:")
    analyzer = TAULSAnalyzer()
    analysis = analyzer(args.text)
    print(f"   Text: '{args.text[:50]}...'")
    print(f"   Stability: {analysis['stability_score']:.3f}")
    print(f"   Entropy: {analysis['entropy_score']:.3f}")
    print(f"   Complexity: {analysis['complexity_score']:.3f}")
    print(f"   Coherence: {analysis['coherence_score']:.3f}")
    
    print("\n5. Polynomial KFP Basis:")
    poly_coeffs = create_kfp_polynomial_basis(degree=3, dim=8)
    print(f"   Polynomial shape: {poly_coeffs.shape}")
    print(f"   Quadratic terms (should be negative): {poly_coeffs[2].diagonal()[:4].tolist()}")
    
    return 0

def cmd_visualize(args: argparse.Namespace) -> int:
    if not HAS_MPL:
        print("matplotlib is not installed.")
        return 1
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    out = Path(args.out or (Path(args.wav).with_suffix(".png")))
    plot_wave_and_spectrum(out, s, sr, f"Visualize: {Path(args.wav).name}")
    print(safe_json({"png": str(out), "sample_rate": sr, "seconds": len(s)/sr}))
    return 0

def cmd_analyze(args: argparse.Namespace) -> int:
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    dur = len(s)/sr
    rms = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(np.abs(s)))
    spec = np.abs(rfft(s))
    spec /= (spec.max()+1e-12)
    snr = 10*np.log10(np.mean(s**2) / (np.var(s - np.mean(s)) + 1e-12))
    print(safe_json({
        "sample_rate": sr,
        "seconds": dur,
        "rms": rms,
        "peak": peak,
        "snr_db": float(snr)
    }))
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    
    if args.cmd == "tau-cast": return cmd_tau_cast(args)
    if args.cmd == "modulate": return cmd_modulate(args)
    if args.cmd == "tau-analyze": return cmd_tau_analyze(args)
    if args.cmd == "tau-demo": return cmd_tau_demo(args)
    if args.cmd == "visualize": return cmd_visualize(args)
    if args.cmd == "analyze": return cmd_analyze(args)
    
    p.print_help()
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
  #!/usr/bin/env python3
"""
TA ULS (Two-level Trans-Algorithmic Universal Learning System) Transformer
=========================================================================

This module implements the core TA ULS architecture with:
- Kinetic Force Principle (KFP) layers for gradient-based parameter optimization
- Two-level control system (meta-control + automatic control)
- Entropy regulation based on environmental stress
- Enhanced transformer blocks with stability monitoring

Author: Assistant
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KFPLayer(nn.Module):
    """
    Kinetic Force Principle Layer - implements gradient-based parameter optimization
    following the principle that parameters move toward states of minimal fluctuation intensity
    """
    def __init__(self, dim: int, stability_weight: float = 0.1):
        super().__init__()
        self.dim = dim
        self.stability_weight = stability_weight
        
        # Fluctuation intensity tracking (Lyapunov function approximation)
        self.register_buffer('fluctuation_history', torch.zeros(dim))
        self.momentum = 0.9
        
        # Kinetic force computation
        self.force_projection = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Compute current fluctuation intensity (variance across batch)
        current_fluctuation = torch.var(x, dim=0, keepdim=False)
        
        # Update fluctuation history with momentum
        self.fluctuation_history.data = (
            self.momentum * self.fluctuation_history.data + 
            (1 - self.momentum) * current_fluctuation.detach()
        )
        
        # Compute kinetic force (gradient toward minimal fluctuation)
        if self.force_projection.weight.requires_grad:
            try:
                force_gradient = torch.autograd.grad(
                    outputs=self.fluctuation_history.sum(),
                    inputs=[self.force_projection.weight],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
            except RuntimeError:
                force_gradient = torch.zeros_like(self.force_projection.weight)
        else:
            force_gradient = torch.zeros_like(self.force_projection.weight)
        
        # Apply kinetic force to push toward stability
        kinetic_force = self.force_projection(x)
        stability_term = -self.stability_weight * kinetic_force
        
        return x + stability_term, self.fluctuation_history.clone()

class TAULSControlUnit(nn.Module):
    """
    Two-level Trans-Algorithmic Universal Learning System
    Higher level: Learning and adaptation
    Lower level: Automatic control
    """
    def __init__(self, input_dim: int, hidden_dim: int, control_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        
        # Higher level: Learning system (meta-control)
        self.meta_controller = nn.Sequential(
            nn.Linear(input_dim + control_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            KFPLayer(hidden_dim),
            nn.Linear(hidden_dim, control_dim)
        )
        
        # Lower level: Automatic control
        self.controller = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            KFPLayer(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, control_dim)
        )
        
        # Control integration
        self.control_mixer = nn.Parameter(torch.tensor(0.5))  # Learnable mixing
        
    def forward(self, x: torch.Tensor, prev_control: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len = x.shape[:2]
        
        if prev_control is None:
            prev_control = torch.zeros(batch_size, seq_len, self.control_dim, device=x.device)
        
        # Higher level processing (learning)
        meta_input = torch.cat([x, prev_control], dim=-1)
        meta_input_flat = meta_input.reshape(-1, meta_input.shape[-1])
        
        # Process through meta-controller layers
        meta_hidden = meta_input_flat
        for i, layer in enumerate(self.meta_controller[:-1]):
            if isinstance(layer, KFPLayer):
                meta_hidden, meta_stability = layer(meta_hidden)
            else:
                meta_hidden = layer(meta_hidden)
        
        meta_control = self.meta_controller[-1](meta_hidden).reshape(batch_size, seq_len, -1)
        
        # Lower level processing (automatic control)
        auto_input_flat = x.reshape(-1, x.shape[-1])
        auto_hidden = auto_input_flat
        for i, layer in enumerate(self.controller[:-1]):
            if isinstance(layer, KFPLayer):
                auto_hidden, auto_stability = layer(auto_hidden)
            else:
                auto_hidden = layer(auto_hidden)
        
        auto_control = self.controller[-1](auto_hidden).reshape(batch_size, seq_len, -1)
        
        # Integrate control signals using learnable mixing
        alpha = torch.sigmoid(self.control_mixer)
        integrated_control = alpha * meta_control + (1 - alpha) * auto_control
        
        return {
            'control_output': integrated_control,
            'meta_stability': meta_stability if 'meta_stability' in locals() else torch.zeros(self.hidden_dim),
            'auto_stability': auto_stability if 'auto_stability' in locals() else torch.zeros(self.hidden_dim // 2),
            'control_mixing': alpha
        }

class EntropyRegulationModule(nn.Module):
    """
    Implements entropy regulation based on environmental stress
    Modulates parameter modification intensity to maintain active stability
    """
    def __init__(self, dim: int, max_entropy_target: float = 0.8):
        super().__init__()
        self.dim = dim
        self.max_entropy_target = max_entropy_target
        
        # Entropy estimation network
        self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Modification intensity controller
        self.intensity_controller = nn.Linear(1, dim)
        
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate entropy using neural estimator"""
        batch_size = x.shape[0]
        entropy_est = self.entropy_estimator(x).squeeze(-1)
        return entropy_est.mean()
    
    def forward(self, x: torch.Tensor, environmental_stress: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        current_entropy = self.compute_entropy(x)
        
        # Compute required entropy adjustment
        entropy_error = current_entropy - self.max_entropy_target
        stress_factor = environmental_stress.mean()
        
        # Adjust modification intensity based on stress and entropy
        target_intensity = torch.sigmoid(entropy_error + stress_factor).unsqueeze(0)
        intensity_modulation = self.intensity_controller(target_intensity)
        
        # Apply intensity modulation
        modulated_output = x * intensity_modulation.unsqueeze(0)
        
        return modulated_output, {
            'current_entropy': current_entropy,
            'target_intensity': target_intensity,
            'entropy_error': entropy_error
        }

class TAULSTransformerBlock(nn.Module):
    """
    Transformer block enhanced with TA ULS control structure
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        
        # Standard attention mechanism
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # TA ULS control unit
        self.control_unit = TAULSControlUnit(d_model, d_ff, d_model)
        
        # Entropy regulation
        self.entropy_regulator = EntropyRegulationModule(d_model)
        
        # KFP-based stability layer
        self.stability_layer = KFPLayer(d_model)
        
        # Standard components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Estimate environmental stress from attention patterns
        environmental_stress = torch.var(attn_weights, dim=-1).mean(dim=-1, keepdim=True)
        
        # Apply entropy regulation
        regulated_x, entropy_info = self.entropy_regulator(x, environmental_stress)
        
        # TA ULS control processing
        control_results = self.control_unit(regulated_x)
        controlled_x = control_results['control_output']
        
        # Apply KFP-based stability
        stable_x, fluctuation_intensity = self.stability_layer(controlled_x)
        
        # Final normalization and residual
        output = self.norm2(x + self.dropout(stable_x))
        
        return {
            'output': output,
            'attention_weights': attn_weights,
            'control_info': control_results,
            'entropy_info': entropy_info,
            'stability_info': fluctuation_intensity
        }

class TAULSLanguageModel(nn.Module):
    """
    Complete language model implementing TA ULS architecture
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        
        # Standard embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # TA ULS transformer blocks
        self.blocks = nn.ModuleList([
            TAULSTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Global stability monitoring
        self.global_stability_tracker = KFPLayer(d_model)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        # Create embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=device).unsqueeze(0))
        x = token_embeds + pos_embeds
        
        # Track stability metrics across layers
        layer_outputs = []
        stability_metrics = []
        
        # Process through TA ULS blocks
        for i, block in enumerate(self.blocks):
            block_results = block(x, attention_mask)
            x = block_results['output']
            
            layer_outputs.append(x)
            stability_metrics.append({
                'layer': i,
                'control_info': block_results['control_info'],
                'entropy_info': block_results['entropy_info'],
                'stability_info': block_results['stability_info']
            })
        
        # Global stability check
        stable_x, global_stability = self.global_stability_tracker(x)
        
        # Generate logits
        logits = self.output_projection(stable_x)
        
        return {
            'logits': logits,
            'hidden_states': layer_outputs,
            'stability_metrics': stability_metrics,
            'global_stability': global_stability
        }

# Polynomial matrix formulation for KFP
def create_kfp_polynomial_basis(degree: int, dim: int) -> torch.Tensor:
    """
    Create polynomial basis functions for KFP approximation
    Based on the mathematical foundation that KFP follows gradient descent
    on fluctuation intensity functions
    """
    # Generate polynomial coefficients for stability landscape
    coefficients = torch.randn(degree + 1, dim, dim) * 0.1
    
    # Ensure stability (negative definite quadratic terms)
    coefficients[2] = -torch.abs(coefficients[2])  # Quadratic terms negative
    
    return coefficients

def kfp_polynomial_update(x: torch.Tensor, coefficients: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    """
    Polynomial-based KFP update rule
    Implements: dx/dt = -∇f(x) where f(x) is the fluctuation intensity
    """
    degree = coefficients.shape[0] - 1
    gradient = torch.zeros_like(x)
    
    # Compute polynomial gradient
    for d in range(1, degree + 1):
        power_term = torch.pow(x.unsqueeze(-1), d - 1)
        grad_term = d * torch.sum(coefficients[d] * power_term, dim=-1)
        gradient += grad_term
    
    # KFP update: move opposite to gradient
    return x - learning_rate * gradient

def demo_tauls_model():
    """Demonstration of the TA ULS model"""
    # Model parameters
    vocab_size = 50000
    d_model = 512
    n_heads = 8
    n_layers = 6
    max_seq_len = 2048
    
    # Create TA ULS model
    model = TAULSLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    
    # Example input
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    results = model(input_ids)
    
    logger.info(f"Model output shape: {results['logits'].shape}")
    logger.info(f"Number of stability metrics: {len(results['stability_metrics'])}")
    logger.info(f"Global stability shape: {results['global_stability'].shape}")
    
    # Demonstrate polynomial KFP basis
    poly_coeffs = create_kfp_polynomial_basis(degree=3, dim=d_model)
    logger.info(f"Polynomial coefficients shape: {poly_coeffs.shape}")
    
    return model, results

if __name__ == "__main__":
    demo_tauls_model()
