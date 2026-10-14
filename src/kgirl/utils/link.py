"""Measured link outcome for modulation choices: Hamming(7,4) over a binary symmetric channel.

The planners in kgirl (BasicAdaptivePlanner, NeuroSymbolicPlanner) used to learn from a coin flip
(`random() > 0.3`). This module replaces that with a measurement: encode the payload, flip each coded
bit with the mode's uncoded AWGN bit-error rate (closed forms below, Level 0), decode, and report
whether the payload came back exactly. Bit flips are drawn from logistic-map chaos (deterministic).

    BPSK, QPSK (Gray)          Pb = ½·erfc(√(Eb/N0))            1, 2 bits/symbol
    "ofdm" (16-QAM subcarriers) Pb ≈ ⅜·erfc(√(0.4·Eb/N0))       4 bits/symbol
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from kgirl.evolve.chaos import ChaosGenerator

MODES = {
    "bpsk": (1, lambda g: 0.5 * math.erfc(math.sqrt(g))),
    "qpsk": (2, lambda g: 0.5 * math.erfc(math.sqrt(g))),
    "ofdm": (4, lambda g: 0.375 * math.erfc(math.sqrt(0.4 * g))),
}

_G = np.array([[1, 1, 1, 0, 0, 0, 0],      # generator for the [p1, p2, d0, p3, d1, d2, d3] layout
               [1, 0, 0, 1, 1, 0, 0],
               [0, 1, 0, 1, 0, 1, 0],
               [1, 1, 0, 1, 0, 0, 1]], dtype=np.uint8)
_H = np.array([[1, 0, 1, 0, 1, 0, 1],
               [0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
_DATA_POS = np.array([2, 4, 5, 6])


def ber(mode: str, ebn0_db: float) -> float:
    return MODES[mode][1](10 ** (ebn0_db / 10.0))


def to_bits(payload: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8))


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    b = np.concatenate([bits, np.zeros((-len(bits)) % 4, dtype=np.uint8)]).reshape(-1, 4)
    return (b @ _G % 2).astype(np.uint8).ravel()


def hamming74_decode(code: np.ndarray) -> np.ndarray:
    c = code[: len(code) - len(code) % 7].reshape(-1, 7).copy()
    syndrome = (c @ _H.T % 2) @ np.array([1, 2, 4])          # 1-based error position, 0 = none
    rows = np.nonzero(syndrome)[0]
    c[rows, syndrome[rows] - 1] ^= 1
    return c[:, _DATA_POS].ravel()


def transmit(payload: str | bytes, mode: str, ebn0_db: float = 6.0,
             rng: Optional[ChaosGenerator] = None) -> Dict[str, float]:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    bits = to_bits(data)
    coded = hamming74_encode(bits)
    p = ber(mode, ebn0_db)
    flips = ((rng or ChaosGenerator(seed=0.6180)).random(coded.size) < p).astype(np.uint8)
    decoded = hamming74_decode(coded ^ flips)[: bits.size]
    return {"mode": mode, "ber": p, "bits_per_symbol": MODES[mode][0],
            "channel_bit_errors": int(flips.sum()), "residual_bit_errors": int((decoded != bits).sum()),
            "success": bool(np.array_equal(decoded, bits))}
