#!/usr/bin/env python3
"""
Simple Holographic Memory System (subset)

Provides a lightweight implementation of the HolographicAssociativeMemory
used by the recursive cognitive demo. This is a practical, classical
approximation suitable for testing and development.

Author: Assistant
License: MIT
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional
from numpy.fft import fft2, ifft2


class HolographicMemorySystem:
    """Lightweight holographic memory for storing and recalling arrays/text.

    This implementation stores complex-valued holograms (FFT domain) and
    supports simple associative recall based on phase-conjugation interference.
    """

    def __init__(self, hologram_dim: int = 64):
        self.hologram_dim = hologram_dim
        # accumulated hologram in frequency domain
        self.hologram = np.zeros((hologram_dim, hologram_dim), dtype=np.complex128)
        self.index: Dict[str, Dict[str, Any]] = {}

    def _ensure_array(self, data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=float)
        elif isinstance(data, str):
            # simple text -> numeric mapping (byte values)
            arr = np.frombuffer(data.encode('utf-8'), dtype=np.uint8).astype(float)
        else:
            arr = np.array([float(data)])

        # fit into square hologram
        size = self.hologram_dim * self.hologram_dim
        flat = arr.ravel()
        if flat.size < size:
            padded = np.zeros(size, dtype=float)
            padded[: flat.size] = flat
            flat = padded
        else:
            flat = flat[:size]

        return flat.reshape(self.hologram_dim, self.hologram_dim)

    def store(self, key: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """Store data under a key. Returns the key."""
        arr2d = self._ensure_array(data)
        H = fft2(arr2d)
        # use random phase reference to simulate holographic interference
        phase = np.exp(1j * 2 * np.pi * np.random.random(H.shape))
        hologram = H * phase
        self.hologram += hologram
        self.index[key] = {
            "metadata": metadata or {},
            "shape": arr2d.shape,
            "hologram": hologram,
        }
        return key

    def recall(self, query: Any, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Recall stored items similar to query. Returns list of dicts.

        Similarity is computed by measuring the cross-correlation in frequency
        domain between the query FFT and each stored hologram.
        """
        qarr = self._ensure_array(query)
        Q = fft2(qarr)
        results = []
        for k, v in self.index.items():
            H = v["hologram"]
            # similarity: magnitude of inner product between Q and conj(H)
            sim = abs((Q.conj() * H).sum())
            if sim >= threshold:
                # reconstruct candidate via inverse transform using phase of accumulated hologram
                recon_freq = np.abs(Q) * np.exp(1j * np.angle(self.hologram))
                recon = np.real(ifft2(recon_freq))
                results.append({"key": k, "similarity": float(sim), "reconstruction": recon})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

