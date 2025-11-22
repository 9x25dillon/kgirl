#!/usr/bin/env python3
"""
Dianne PolyServe - Phase 1 Core

Safe, runnable subset of the larger system focusing on:
 - TextEncoder (framing, CRC, optional AES/HMAC if Crypto available)
 - BPSKModulator
 - AudioProcessor (writes WAV using wave/stdlib)
 - WaveCaster orchestration
 - Minimal LLMClient stub (synchronous fallback)

This file is intentionally dependency-tolerant so it can be imported
and run inside the development workspace without optional packages.
"""

import time
import json
import hashlib
import binascii
import wave
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import asyncio
from llm_orchestrator import DualLLMOrchestrator

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    carrier_freq: float = 1800.0
    clip_signal: bool = True


@dataclass
class SystemConfig:
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    audio: AudioConfig = field(default_factory=AudioConfig)
    max_text_length: int = 2000

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TextEncoder:
    def __init__(self, config: SystemConfig):
        self.config = config

    def encode_text(self, text: str) -> (np.ndarray, Dict[str, Any]):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if len(text) > self.config.max_text_length:
            raise ValueError("text too long")

        payload = text.encode('utf-8')
        preamble = b'\x55' * 8
        header = b'\xA5' + len(payload).to_bytes(4, 'big') + int(time.time()).to_bytes(8, 'big')
        crc = binascii.crc32(header[1:] + payload).to_bytes(4, 'big')
        frame = preamble + header + payload + crc

        bits = []
        for b in frame:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)

        return np.array(bits, dtype=np.uint8), {"frame_bytes": len(frame)}


class BPSKModulator:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.sps = int(self.config.audio.sample_rate / self.config.audio.symbol_rate)

    def modulate(self, bits: np.ndarray) -> Dict[str, Any]:
        if bits.dtype != np.uint8:
            bits = bits.astype(np.uint8)
        n = len(bits)
        samples = n * self.sps
        baseband = np.zeros(samples, dtype=np.float32)
        for i, bit in enumerate(bits):
            v = self.config.audio.amplitude if bit == 1 else -self.config.audio.amplitude
            baseband[i*self.sps:(i+1)*self.sps] = v

        t = np.arange(len(baseband)) / self.config.audio.sample_rate
        carrier = np.cos(2 * np.pi * self.config.audio.carrier_freq * t)
        audio = baseband * carrier
        if self.config.audio.clip_signal:
            audio = np.clip(audio, -1.0, 1.0)

        return {"audio": audio, "baseband": baseband}


class AudioProcessor:
    def __init__(self, config: SystemConfig):
        self.config = config

    def save_wav(self, audio: np.ndarray, filename: str) -> str:
        # Convert to 16-bit PCM mono
        audio_clipped = np.clip(audio, -1.0, 1.0)
        pcm = (audio_clipped * 32767).astype(np.int16)
        out_path = self.config.output_dir / f"{filename}.wav"
        with wave.open(str(out_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.config.audio.sample_rate)
            wf.writeframes(pcm.tobytes())
        return str(out_path)


class LLMClientStub:
    def __init__(self, config: SystemConfig):
        self.config = config

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        # Simple echo-like stub
        out = prompt
        if len(out) > 200:
            out = out[:200] + '...'
        return out


class WaveCaster:
    def __init__(self, config: SystemConfig, llm_adapter=None):
        self.config = config
        self.encoder = TextEncoder(config)
        self.mod = BPSKModulator(config)
        self.audio_proc = AudioProcessor(config)
        # adapter can be provided or auto-created from environment
        if llm_adapter is not None:
            self.llm = llm_adapter
        else:
            try:
                from llm_adapters import create_adapter_from_env
                self.llm = create_adapter_from_env()
            except Exception:
                self.llm = LLMClientStub(config)

    def process_text(self, text: str, output_name: str = 'demo') -> Dict[str, Any]:
        # Optional LLM step: adapters may be async or sync. Support both.
        if hasattr(self.llm, 'generate'):
            try:
                # if it's an async function, run it
                if asyncio.iscoroutinefunction(self.llm.generate):
                    processed = asyncio.get_event_loop().run_until_complete(self.llm.generate(text))
                else:
                    processed = self.llm.generate(text)
            except RuntimeError:
                # no event loop, use asyncio.run
                processed = asyncio.run(self.llm.generate(text)) if asyncio.iscoroutinefunction(self.llm.generate) else self.llm.generate(text)
        else:
            processed = text

        bits, meta = self.encoder.encode_text(processed)
        sig = self.mod.modulate(bits)
        wav_path = self.audio_proc.save_wav(sig['audio'], output_name)

        return {"wav": wav_path, "bits": len(bits), "meta": meta}


class DianneSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.wavecaster = WaveCaster(config)

    def initialize(self):
        # placeholder for async initialization
        return True


def demo_run_short():
    cfg = SystemConfig()
    cfg.audio.sample_rate = 48000
    cfg.audio.symbol_rate = 1200
    ds = DianneSystem(cfg)
    text = "Hello Dianne PolyServe - test transmission"

    # Use the DualLLMOrchestrator to generate a processed candidate and critic score
    orch = DualLLMOrchestrator()
    orch_res = asyncio.run(orch.generate_and_critique(text, max_tokens=256))
    candidate = orch_res.get('candidate') or text

    print('Orchestrator candidate (truncated):', candidate[:200])
    print('Orchestrator critic score:', orch_res.get('score'))

    # Feed the candidate into the WaveCaster pipeline (don't request additional LLM processing)
    res = ds.wavecaster.process_text(candidate, output_name='polyserve_demo')
    print('Demo result:', res)


if __name__ == '__main__':
    demo_run_short()
