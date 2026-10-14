# domain_mapping.py
"""Thought ↔ EEG domain mapping: a NewThought coherence score ↔ per-band coherence κ.

Forward  coherence c, metadata {depth, entropy} → κ per band: the thought's recursion depth picks a
         dominant band (0 → gamma … 4+ → delta); the others fall off as exp(−distance / (1 + entropy)),
         scaled by entropy; the result is normalised to sum to c (each band clipped to [0, 1]).
Inverse  κ per band → c: the sum, clipped to [0, 1] (exact inverse while no band was clipped).

Same forward rule as kgirl.core.bridge_newthought_crd.distribute_coherence_to_bands. This file used to
register a zero "placeholder — real one below" with nothing below it, and initialised only two of its
five tables.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np

BANDS = ("delta", "theta", "alpha", "beta", "gamma")
DEPTH_TO_BAND = {0: "gamma", 1: "beta", 2: "alpha", 3: "theta", 4: "delta"}


@dataclass
class DomainMapping:
    entity_mappings: Dict[str, str] = field(default_factory=dict)
    value_transformations: Dict[str, Callable] = field(default_factory=dict)
    inverse_transformations: Dict[str, Callable] = field(default_factory=dict)
    aggregation_functions: Dict[str, Callable] = field(default_factory=dict)
    distribution_functions: Dict[str, Callable] = field(default_factory=dict)


def coherence_to_bands(coherence: float, metadata: dict) -> Dict[str, float]:
    depth = int(metadata.get("depth", 2))
    entropy = float(metadata.get("entropy", 0.3))
    dom = BANDS.index(DEPTH_TO_BAND.get(min(depth, 4), "alpha"))
    raw = {}
    for i, b in enumerate(BANDS):
        d = abs(i - dom)
        raw[b] = coherence * (1.0 - entropy * 0.5) if d == 0 else coherence * np.exp(-d / (1 + entropy)) * entropy
    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total * coherence for k, v in raw.items()}
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in raw.items()}


def bands_to_coherence(kappa: Dict[str, float], metadata: dict = None) -> float:
    return float(np.clip(sum(kappa.get(b, 0.0) for b in BANDS), 0.0, 1.0))


# === ACTUAL MAPPING ===
domain_mapping = DomainMapping()
domain_mapping.entity_mappings["Thought"] = "EEGBandState"
# Forward: Thought → EEG
domain_mapping.distribution_functions["coherence_to_bands"] = coherence_to_bands
# Inverse: EEG → Thought
domain_mapping.aggregation_functions["bands_to_coherence"] = bands_to_coherence
domain_mapping.inverse_transformations["coherence_to_bands"] = bands_to_coherence
