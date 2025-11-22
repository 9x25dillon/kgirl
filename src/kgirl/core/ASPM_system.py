#!/Aur/bin/env python3
"""
Advanced Signal Processing and Modulation System
===============================================

This module implements comprehensive digital signal processing including:
- Multiple modulation schemes (BFSK, BPSK, QPSK, QAM16, OFDM, DSSS)
- Forward Error Correction (FEC) coding
- Framing, security, and watermarking
- Audio and IQ signal generation
- Visualization and analysis tools

Author: Assistant
License: MIT
"""

import binascii
import hashlib
import math
import struct
import time
import wave
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# Enums and Configuration
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
    REED_SOLOMON = auto()   # stub
    LDPC = auto()           # stub
    TURBO = auto()          # stub

@dataclass
class ModConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0     # BFSK 0
    f1: float = 2200.0     # BFSK 1
    fc: float = 1800.0     # PSK/QAM audio carrier (for WAV)
    clip: bool = True
    # OFDM parameters
    ofdm_subc: int = 64
    cp_len: int = 16
    # DSSS parameters
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8  # 01010101 * 8
    version: int = 1

@dataclass
class SecurityConfig:
    password: Optional[str] = None           # AES-GCM if provided
    watermark: Optional[str] = None          # prepended SHA256[0:8]
    hmac_key: Optional[str] = None           # HMAC-SHA256 appended

@dataclass
class OutputPaths:
    wav: Optional[Path] = None
    iq: Optional[Path] = None
    meta: Optional[Path] = None
    png: Optional[Path] = None

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
    import json
    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        return str(x)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=enc)

# =========================================================
# FEC Implementation
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    """Hamming (7,4) encoding"""
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

def hamming74_decode(coded_bits: List[int]) -> Tuple[List[int], int]:
    """Hamming (7,4) decoding with error correction"""
    if len(coded_bits) % 7 != 0:
        coded_bits = coded_bits + [0] * (7 - len(coded_bits) % 7)
    
    decoded = []
    errors_corrected = 0
    
    for i in range(0, len(coded_bits), 7):
        r = coded_bits[i:i+7]  # received codeword
        p1, p2, d0, p3, d1, d2, d3 = r
        
        # Calculate syndrome
        s1 = p1 ^ d0 ^ d1 ^ d3
        s2 = p2 ^ d0 ^ d2 ^ d3
        s3 = p3 ^ d1 ^ d2 ^ d3
        
        syndrome = s1 + 2*s2 + 4*s3
        
        # Correct single-bit errors
        if syndrome != 0:
            errors_corrected += 1
            if syndrome <= 7:
                r[syndrome - 1] ^= 1  # flip the error bit
        
        # Extract data bits
        decoded.extend([r[2], r[4], r[5], r[6]])  # d0, d1, d2, d3
    
    return decoded, errors_corrected

def fec_encode(bits: List[int], scheme: FEC) -> List[int]:
    if scheme == FEC.NONE:
        return list(bits)
    elif scheme == FEC.HAMMING74:
        return hamming74_encode(bits)
    elif scheme in (FEC.REED_SOLOMON, FEC.LDPC, FEC.TURBO):
        raise NotImplementedError(f"{scheme.name} encoding not implemented")
    else:
        raise ValueError("Unknown FEC scheme")

def fec_decode(bits: List[int], scheme: FEC) -> Tuple[List[int], Dict[str, Any]]:
    if scheme == FEC.NONE:
        return list(bits), {"errors_corrected": 0}
    elif scheme == FEC.HAMMING74:
        decoded, errors = hamming74_decode(bits)
        return decoded, {"errors_corrected": errors}
    else:
        raise NotImplementedError(f"{scheme.name} decoding not implemented")

# =========================================================
# Security and Framing
# =========================================================

def aes_gcm_encrypt(plaintext: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    
    return b"AGCM" + salt + nonce + tag + ciphertext

def aes_gcm_decrypt(encrypted: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for decryption")
    
    if not encrypted.startswith(b"AGCM"):
        raise ValueError("Invalid encrypted format")
    
    data = encrypted[4:]  # skip "AGCM" header
    salt = data[:16]
    nonce = data[16:28]
    tag = data[28:44]
    ciphertext = data[44:]
    
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    
    return cipher.decrypt_and_verify(ciphertext, tag)

def apply_hmac(data: bytes, hkey: str) -> bytes:
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def verify_hmac(data: bytes, hkey: str) -> Tuple[bytes, bool]:
    if not data.endswith(b"HMAC"):
        return data, False
    
    # Find HMAC marker
    hmac_pos = data.rfind(b"HMAC")
    if hmac_pos == -1 or len(data) - hmac_pos != 36:  # 4 + 32 bytes
        return data, False
    
    payload = data[:hmac_pos]
    received_mac = data[hmac_pos + 4:]
    
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    expected_mac = hmac.new(key, payload, hashlib.sha256).digest()
    
    return payload, hmac.compare_digest(received_mac, expected_mac)

def add_watermark(data: bytes, wm: str) -> bytes:
    return hashlib.sha256(wm.encode("utf-8")).digest()[:8] + data

def check_watermark(data: bytes, wm: str) -> Tuple[bytes, bool]:
    if len(data) < 8:
        return data, False
    
    expected = hashlib.sha256(wm.encode("utf-8")).digest()[:8]
    received = data[:8]
    payload = data[8:]
    
    return payload, received == expected

def frame_payload(payload: bytes, fcfg: FrameConfig) -> bytes:
    header = struct.pack(">BBI", 0xA5, fcfg.version, now_ms() & 0xFFFFFFFF)
    core = header + payload
    
    tail = b""
    if fcfg.use_crc32:
        tail += crc32_bytes(core)
    if fcfg.use_crc16:
        tail += crc16_ccitt(core)
    
    return fcfg.preamble + core + tail

def unframe_payload(framed: bytes, fcfg: FrameConfig) -> Tuple[bytes, Dict[str, Any]]:
    if len(framed) < len(fcfg.preamble) + 7:  # minimum frame size
        return b"", {"error": "Frame too short"}
    
    # Check preamble
    if not framed.startswith(fcfg.preamble):
        return b"", {"error": "Invalid preamble"}
    
    data = framed[len(fcfg.preamble):]
    
    # Parse header
    if len(data) < 7:
        return b"", {"error": "Header too short"}
    
    sync, version, timestamp = struct.unpack(">BBI", data[:7])
    if sync != 0xA5:
        return b"", {"error": "Invalid sync byte"}
    
    # Calculate payload length
    tail_len = 0
    if fcfg.use_crc32:
        tail_len += 4
    if fcfg.use_crc16:
        tail_len += 2
    
    if len(data) < 7 + tail_len:
        return b"", {"error": "Frame too short for CRC"}
    
    payload = data[7:-tail_len] if tail_len > 0 else data[7:]
    
    # Verify CRCs
    info = {"version": version, "timestamp": timestamp}
    
    if fcfg.use_crc32:
        expected_crc32 = crc32_bytes(data[:-tail_len])
        received_crc32 = data[-tail_len:-tail_len+4] if fcfg.use_crc16 else data[-4:]
        info["crc32_ok"] = expected_crc32 == received_crc32
    
    if fcfg.use_crc16:
        expected_crc16 = crc16_ccitt(data[:-2])
        received_crc16 = data[-2:]
        info["crc16_ok"] = expected_crc16 == received_crc16
    
    return payload, info

def encode_text(text: str, fcfg: FrameConfig, sec: SecurityConfig, fec_scheme: FEC) -> List[int]:
    """Complete encoding pipeline"""
    data = text.encode("utf-8")
    
    # Apply watermark
    if sec.watermark:
        data = add_watermark(data, sec.watermark)
    
    # Apply encryption
    if sec.password:
        data = aes_gcm_encrypt(data, sec.password)
    
    # Frame the data
    framed = frame_payload(data, fcfg)
    
    # Apply HMAC
    if sec.hmac_key:
        framed = apply_hmac(framed, sec.hmac_key)
    
    # Convert to bits and apply FEC
    bits = to_bits(framed)
    bits = fec_encode(bits, fec_scheme)
    
    return bits

def decode_bits(bits: List[int], fcfg: FrameConfig, sec: SecurityConfig, fec_scheme: FEC) -> Tuple[str, Dict[str, Any]]:
    """Complete decoding pipeline"""
    info = {}
    
    try:
        # Apply FEC decoding
        decoded_bits, fec_info = fec_decode(bits, fec_scheme)
        info.update(fec_info)
        
        # Convert bits to bytes
        framed = from_bits(decoded_bits)
        
        # Verify HMAC
        if sec.hmac_key:
            framed, hmac_ok = verify_hmac(framed, sec.hmac_key)
            info["hmac_ok"] = hmac_ok
            if not hmac_ok:
                return "", {**info, "error": "HMAC verification failed"}
        
        # Unframe
        data, frame_info = unframe_payload(framed, fcfg)
        info.update(frame_info)
        
        if "error" in frame_info:
            return "", info
        
        # Decrypt
        if sec.password:
            data = aes_gcm_decrypt(data, sec.password)
            info["decrypted"] = True
        
        # Check watermark
        if sec.watermark:
            data, wm_ok = check_watermark(data, sec.watermark)
            info["watermark_ok"] = wm_ok
            if not wm_ok:
                return "", {**info, "error": "Watermark verification failed"}
        
        # Decode text
        text = data.decode("utf-8", errors="replace")
        return text, info
        
    except Exception as e:
        return "", {**info, "error": str(e)}

# =========================================================
# Modulation Schemes
# =========================================================

class Modulators:
    @staticmethod
    def bfsk(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        """Binary Frequency Shift Keying"""
        sr, rb = cfg.sample_rate, cfg.symbol_rate
        spb = int(sr / rb)  # samples per bit
        t = np.arange(spb) / sr
        
        signal_blocks = []
        for bit in bits:
            freq = cfg.f1 if bit else cfg.f0
            signal_blocks.append(cfg.amplitude * np.sin(2 * np.pi * freq * t))
        
        if not signal_blocks:
            return np.zeros(0, dtype=np.float32)
        
        signal = np.concatenate(signal_blocks)
        
        if cfg.clip:
            signal = np.clip(signal, -1, 1)
        
        return signal.astype(np.float32)
    
    @staticmethod
    def bpsk(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Binary Phase Shift Keying"""
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        
        audio_blocks = []
        iq_blocks = []
        
        for bit in bits:
            phase = 0.0 if bit else np.pi
            
            # Audio signal (upconverted)
            audio_blocks.append(cfg.amplitude * np.sin(2 * np.pi * fc * t + phase))
            
            # IQ signal (baseband)
            iq_symbol = cfg.amplitude * (np.cos(phase) + 1j * np.sin(phase))
            iq_blocks.append(iq_symbol * np.ones(spb, dtype=np.complex64))
        
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float32)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        
        if cfg.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio.astype(np.float32), iq
    
    @staticmethod
    def qpsk(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Quadrature Phase Shift Keying"""
        pairs = chunk_bits(bits, 2)
        symbols = []
        
        # Gray mapping: 00→(1+1j), 01→(-1+1j), 11→(-1-1j), 10→(1-1j)
        for pair in pairs:
            b0, b1 = (pair + [0, 0])[:2]
            if (b0, b1) == (0, 0):
                symbol = 1 + 1j
            elif (b0, b1) == (0, 1):
                symbol = -1 + 1j
            elif (b0, b1) == (1, 1):
                symbol = -1 - 1j
            else:  # (1, 0)
                symbol = 1 - 1j
            
            symbols.append(symbol / math.sqrt(2))  # normalize for unit energy
        
        return Modulators._psk_qam_to_audio_iq(np.array(symbols, dtype=np.complex64), cfg)
    
    @staticmethod
    def qam16(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        """16-QAM modulation"""
        quads = chunk_bits(bits, 4)
        
        def gray_map_2bit(b0, b1):
            # Gray mapping for 2 bits to {-3, -1, 1, 3}
            val = (b0 << 1) | b1
            return [-3, -1, 1, 3][val]
        
        symbols = []
        for quad in quads:
            b0, b1, b2, b3 = (quad + [0, 0, 0, 0])[:4]
            I = gray_map_2bit(b0, b1)
            Q = gray_map_2bit(b2, b3)
            symbol = (I + 1j * Q) / math.sqrt(10)  # normalize for unit average power
            symbols.append(symbol)
        
        return Modulators._psk_qam_to_audio_iq(np.array(symbols, dtype=np.complex64), cfg)
    
    @staticmethod
    def _psk_qam_to_audio_iq(symbols: np.ndarray, cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Convert PSK/QAM symbols to audio and IQ signals"""
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        
        # Upsample symbols (rectangular pulse shaping)
        i_data = np.repeat(symbols.real.astype(np.float32), spb)
        q_data = np.repeat(symbols.imag.astype(np.float32), spb)
        
        # Generate time vector
        t = np.arange(len(i_data)) / sr
        
        # Generate audio signal (upconverted)
        audio = cfg.amplitude * (i_data * np.cos(2 * np.pi * fc * t) - 
                                q_data * np.sin(2 * np.pi * fc * t))
        
        # Generate IQ signal (baseband)
        iq = (cfg.amplitude * i_data) + 1j * (cfg.amplitude * q_data)
        
        if cfg.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio.astype(np.float32), iq.astype(np.complex64)
    
    @staticmethod
    def afsk(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        """Audio Frequency Shift Keying (same as BFSK)"""
        return Modulators.bfsk(bits, cfg)
    
    @staticmethod
    def dsss_bpsk(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        """Direct Sequence Spread Spectrum BPSK"""
        # Simple PN sequence for spreading
        pn_sequence = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)
        
        sr = cfg.sample_rate
        chip_rate = cfg.dsss_chip_rate
        samples_per_chip = int(sr / chip_rate)
        
        baseband_signal = []
        
        for bit in bits:
            bit_value = 1.0 if bit else -1.0
            
            # Spread with PN sequence
            spread_chips = bit_value * pn_sequence
            
            # Upsample chips
            for chip in spread_chips:
                baseband_signal.extend([chip] * samples_per_chip)
        
        baseband = np.array(baseband_signal, dtype=np.float32)
        
        # Upconvert to carrier frequency
        t = np.arange(len(baseband)) / sr
        audio = cfg.amplitude * baseband * np.sin(2 * np.pi * cfg.fc * t)
        
        if cfg.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio.astype(np.float32)
    
    @staticmethod
    def ofdm(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Orthogonal Frequency Division Multiplexing"""
        N = cfg.ofdm_subc
        cp_len = cfg.cp_len
        
        # Group bits for QPSK mapping on each subcarrier
        symbol_chunks = chunk_bits(bits, 2 * N)
        
        audio_blocks = []
        iq_blocks = []
        
        for chunk in symbol_chunks:
            # Map bits to QPSK symbols
            qpsk_symbols = []
            bit_pairs = chunk_bits(chunk, 2)
            
            for pair in bit_pairs:
                b0, b1 = (pair + [0, 0])[:2]
                if (b0, b1) == (0, 0):
                    symbol = 1 + 1j
                elif (b0, b1) == (0, 1):
                    symbol = -1 + 1j
                elif (b0, b1) == (1, 1):
                    symbol = -1 - 1j
                else:
                    symbol = 1 - 1j
                qpsk_symbols.append(symbol / math.sqrt(2))
            
            # Pad to N subcarriers
            while len(qpsk_symbols) < N:
                qpsk_symbols.append(0j)
            
            # IFFT to get time domain signal
            freq_domain = np.array(qpsk_symbols[:N], dtype=np.complex64)
            time_domain = np.fft.ifft(freq_domain)
            
            # Add cyclic prefix
            cyclic_prefix = time_domain[-cp_len:]
            ofdm_symbol = np.concatenate([cyclic_prefix, time_domain])
            
            # Scale to fit symbol rate timing
            symbol_duration = int(cfg.sample_rate / cfg.symbol_rate)
            repeat_factor = max(1, symbol_duration // len(ofdm_symbol))
            upsampled = np.repeat(ofdm_symbol, repeat_factor)
            
            # Generate audio (upconverted)
            t = np.arange(len(upsampled)) / cfg.sample_rate
            audio = cfg.amplitude * (upsampled.real * np.cos(2 * np.pi * cfg.fc * t) -
                                   upsampled.imag * np.sin(2 * np.pi * cfg.fc * t))
            
            audio_blocks.append(audio.astype(np.float32))
            iq_blocks.append((cfg.amplitude * upsampled).astype(np.complex64))
        
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float32)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        
        if cfg.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio, iq

def bits_to_signals(bits: List[int], scheme: ModulationScheme, cfg: ModConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert bits to modulated signals"""
    if scheme == ModulationScheme.BFSK:
        return Modulators.bfsk(bits, cfg), None
    elif scheme == ModulationScheme.AFSK:
        return Modulators.afsk(bits, cfg), None
    elif scheme == ModulationScheme.BPSK:
        return Modulators.bpsk(bits, cfg)
    elif scheme == ModulationScheme.QPSK:
        return Modulators.qpsk(bits, cfg)
    elif scheme == ModulationScheme.QAM16:
        return Modulators.qam16(bits, cfg)
    elif scheme == ModulationScheme.OFDM:
        return Modulators.ofdm(bits, cfg)
    elif scheme == ModulationScheme.DSSS_BPSK:
        return Modulators.dsss_bpsk(bits, cfg), None
    else:
        raise ValueError(f"Unknown modulation scheme: {scheme}")

# =========================================================
# File I/O and Visualization
# =========================================================

def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int):
    """Write mono WAV file"""
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

def write_iq_f32(path: Path, iq: np.ndarray):
    """Write IQ data as interleaved float32"""
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be 1-D complex array")
    
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    
    path.write_bytes(interleaved.tobytes())

def plot_wave_and_spectrum(path_png: Path, x: np.ndarray, sr: int, title: str):
    """Plot waveform and spectrum"""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain plot (first 50ms)
    samples_to_plot = min(len(x), int(0.05 * sr))
    t = np.arange(samples_to_plot) / sr
    ax1.plot(t, x[:samples_to_plot])
    ax1.set_title(f"{title} - Time Domain (first 50ms)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain plot
    spectrum = np.abs(rfft(x)) + 1e-12
    freqs = rfftfreq(len(x), 1.0 / sr)
    ax2.semilogy(freqs, spectrum / spectrum.max())
    ax2.set_xlim(0, min(8000, sr // 2))
    ax2.set_title(f"{title} - Frequency Domain")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Normalized |X(f)|")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_constellation(symbols: np.ndarray, title: str = "Constellation", save_path: Optional[str] = None):
    """Plot constellation diagram"""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping constellation plot")
        return
    
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.7, s=20)
    plt.title(title)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def play_audio(x: np.ndarray, sr: int):
    """Play audio through soundcard"""
    if not HAS_AUDIO:
        logger.warning("sounddevice not installed; cannot play audio")
        return
    
    try:
        sd.play(x, sr)
        sd.wait()
    except Exception as e:
        logger.error(f"Audio playback failed: {e}")

# =========================================================
# Complete Processing Pipeline
# =========================================================

def full_process_and_save(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
    want_wav: bool,
    want_iq: bool,
    title: str = "SignalProcessor"
) -> OutputPaths:
    """Complete processing pipeline from text to files"""
    
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    base_name = f"signal_{scheme.name.lower()}_{timestamp}"
    base_path = outdir / base_name
    
    # Encode text to bits
    bits = encode_text(text, fcfg, sec, fec_scheme)
    logger.info(f"Encoded {len(text)} characters to {len(bits)} bits")
    
    # Modulate bits to signals
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    
    paths = OutputPaths()
    
    # Save WAV file
    if want_wav and audio is not None and len(audio) > 0:
        paths.wav = base_path.with_suffix(".wav")
        write_wav_mono(paths.wav, audio, mcfg.sample_rate)
        logger.info(f"Saved WAV: {paths.wav}")
    
    # Save IQ file
    if want_iq:
        if iq is None and audio is not None:
            # Generate IQ from audio using Hilbert transform
            try:
                analytic = sp_signal.hilbert(audio)
                iq = analytic.astype(np.complex64)
            except Exception as e:
                logger.warning(f"Failed to generate IQ from audio: {e}")
                iq = audio.astype(np.float32) + 1j * np.zeros_like(audio, dtype=np.float32)
        
        if iq is not None:
            paths.iq = base_path.with_suffix(".iqf32")
            write_iq_f32(paths.iq, iq)
            logger.info(f"Saved IQ: {paths.iq}")
    
    # Generate visualization
    if audio is not None and len(audio) > 0:
        paths.png = base_path.with_suffix(".png")
        plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
        logger.info(f"Saved plot: {paths.png}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "scheme": scheme.name,
        "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate,
        "duration_sec": len(audio) / mcfg.sample_rate if audio is not None else 0,
        "fec": fec_scheme.name,
        "encrypted": bool(sec.password),
        "watermark": bool(sec.watermark),
        "hmac": bool(sec.hmac_key),
        "text_length": len(text),
        "bits_length": len(bits)
    }
    
    paths.meta = base_path.with_suffix(".json")
    paths.meta.write_text(safe_json(metadata), encoding="utf-8")
    logger.info(f"Saved metadata: {paths.meta}")
    
    return paths

def demo_signal_processing():
    """Demonstration of signal processing capabilities"""
    
    # Test configuration
    text = "Hello, World! This is a test of the signal processing system. 🚀"
    
    schemes_to_test = [
        ModulationScheme.BFSK,
        ModulationScheme.QPSK,
        ModulationScheme.QAM16,
        ModulationScheme.OFDM
    ]
    
    mcfg = ModConfig(sample_rate=48000, symbol_rate=1200)
    fcfg = FrameConfig()
    sec = SecurityConfig(watermark="test_watermark")
    fec_scheme = FEC.HAMMING74
    
    results = []
    
    for scheme in schemes_to_test:
        logger.info(f"Testing {scheme.name}...")
        
        try:
            paths = full_process_and_save(
                text=text,
                outdir=Path("demo_output"),
                scheme=scheme,
                mcfg=mcfg,
                fcfg=fcfg,
                sec=sec,
                fec_scheme=fec_scheme,
                want_wav=True,
                want_iq=True,
                title=f"{scheme.name} Demo"
            )
            
            results.append({
                "scheme": scheme.name,
                "success": True,
                "paths": paths
            })
            
        except Exception as e:
            logger.error(f"Failed to process {scheme.name}: {e}")
            results.append({
                "scheme": scheme.name,
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    logger.info("=== Signal Processing Demo Complete ===")
    for result in results:
        status = "✓" if result["success"] else "✗"
        logger.info(f"{status} {result['scheme']}")
    
    return results

if __name__ == "__main__":
    demo_signal_processing()