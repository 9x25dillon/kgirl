"""Identity signatures: what a vessel *is*, measured on fixed probes, independent of the clay that holds it.

A Signature has two invariant parts and one substrate-bound part:

    geometry   per-layer normalized centered Gram G_l (n × n over probes). Invariant under feature rotation,
               permutation, isotropic scale, translation and change of width (see invariance.py, which verifies
               this by group sampling rather than asserting it).
    voice      log next-token distribution on the probes (n × V). Invariant under any reparametrization that
               preserves the function; comparable across substrates that share a vocabulary.
    body       weight-spectrum statistics (tail exponent α, stable rank) per 2-D weight. Bound to one substrate:
               observed and recorded, never used to compare two different containers.

Comparison (anchor vs now, same probes):

    coherence  C = Σ_l ω_l · CKA_l                         geometric agreement, in [−1, 1]
    fidelity   F = exp(−KL(p_anchor ‖ p_now))               behavioural agreement, in (0, 1]
    identity   I = C · F,     drift  D = 1 − I
    overlap    mean cos² of principal angles between the top-k probe subspaces (eigenvectors of G)

The probe set's sha256 is part of the signature; comparing signatures taken on different probes is an error.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .geometry import effective_rank, gram_alignment, gram_entropy, normalized_gram, stable_rank, tail_exponent


def probe_digest(probes: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(probes))
    h = hashlib.sha256(str(arr.shape).encode() + str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Signature:
    grams: dict
    voice: Optional[np.ndarray]
    spectra: dict
    body: dict = field(default_factory=dict)
    probes: str = ""

    def fingerprint(self, decimals: int = 3) -> str:
        """Stable hash of the invariant parts, rounded so float noise below 10^−decimals does not change it."""
        h = hashlib.sha256(self.probes.encode())
        for layer in sorted(self.grams):
            h.update(layer.encode())
            h.update(np.round(self.grams[layer], decimals).astype(np.float32).tobytes())
        if self.voice is not None:
            h.update(np.round(np.exp(self.voice), decimals).astype(np.float32).tobytes())
        return h.hexdigest()


def measure(substrate, probes: Any, *, body: bool = True) -> Signature:
    acts = substrate.activations(probes)
    grams = {layer: normalized_gram(X) for layer, X in acts.items()}
    spectra = {}
    for layer, X in acts.items():
        sv = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False)
        spectra[layer] = {"effective_rank": effective_rank(sv), "gram_entropy": gram_entropy(grams[layer]),
                          "width": int(X.shape[1])}
    voice = substrate.log_probs(probes) if hasattr(substrate, "log_probs") else None
    stats = {}
    if body:
        for name, W in substrate.parameters().items():
            if W.ndim == 2 and min(W.shape) >= 8:
                t = tail_exponent(W)
                stats[name] = {"alpha": t["alpha"], "tail": t["tail"], "ks": t["ks"], "stable_rank": stable_rank(W)}
    return Signature(grams, voice, spectra, stats, probe_digest(probes))


def top_subspace(G: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of a normalized Gram: the probe-space principal subspace, without the activations."""
    w, V = np.linalg.eigh((G + G.T) / 2.0)
    return V[:, np.argsort(w)[::-1][:k]]


def voice_kl(anchor: np.ndarray, now: np.ndarray) -> float:
    return float(np.sum(np.exp(anchor) * (anchor - now)) / len(anchor))


@dataclass(frozen=True)
class Comparison:
    cka: dict
    overlap: dict
    kl: float
    coherence: float
    fidelity: float

    @property
    def identity(self) -> float:
        return self.coherence * self.fidelity

    @property
    def drift(self) -> float:
        return 1.0 - self.identity

    def as_dict(self) -> dict:
        return {"cka": {k: round(v, 6) for k, v in self.cka.items()},
                "overlap": {k: round(v, 6) for k, v in self.overlap.items()},
                "kl": round(self.kl, 6), "coherence": round(self.coherence, 6), "fidelity": round(self.fidelity, 6),
                "identity": round(self.identity, 6), "drift": round(self.drift, 6)}


def compare(anchor_grams: Mapping[str, np.ndarray], anchor_voice: Optional[np.ndarray], now: Signature, *,
            anchor_probes: str = "", weights: Optional[Mapping[str, float]] = None, k: int = 8) -> Comparison:
    if anchor_probes and now.probes and anchor_probes != now.probes:
        raise ValueError("signatures were taken on different probe sets; identity is only defined on shared probes")
    layers = [layer for layer in anchor_grams if layer in now.grams]
    if not layers:
        raise ValueError("anchor and signature share no layers")
    cka = {layer: gram_alignment(anchor_grams[layer], now.grams[layer]) for layer in layers}
    overlap = {}
    for layer in layers:
        P, Q = top_subspace(anchor_grams[layer], k), top_subspace(now.grams[layer], k)
        overlap[layer] = float(np.mean(np.linalg.svd(P.T @ Q, compute_uv=False) ** 2))
    w = {layer: (weights or {}).get(layer, 1.0) for layer in layers}
    total = sum(w.values()) or 1.0
    coherence = sum(w[layer] * cka[layer] for layer in layers) / total
    kl = voice_kl(anchor_voice, now.voice) if anchor_voice is not None and now.voice is not None else 0.0
    return Comparison(cka, overlap, kl, coherence, math.exp(-kl))


def self_comparison(sig: Signature) -> Comparison:
    return compare(sig.grams, sig.voice, sig, anchor_probes=sig.probes)
